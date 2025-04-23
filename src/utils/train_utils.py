import sys
import os

from tqdm import tqdm

from invariance_constrained import primal_dual_augmentation
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import albumentations as A
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from torch.utils.data import DataLoader

from dataset import TrainValDataset, OBAValDataset
from config import (
    EPSILON, ETA_D, ETA_P, GAMMA, M_SAMPLES, N_MH_STEPS, SEED, EPOCHS,
    BATCH_SIZE_TRAIN, BATCH_SIZE_VAL, NUM_SAMPLE_INDICIES,
    NUM_WORKERS_TRAIN, NUM_WORKERS_VAL, PIN_MEMORY, PERSISTNAT_WORKERS,
    BACKGROUND_PROB, EXTRACT_FROM_SAME_IMAGE, OBA_PROB, NUM_OBA_OBJECTS
)
from src.utils.global_paths import (
    DATASET_PATH, TRAIN_OUTPUT_DIR, TRAIN_ANNOTATIONS_PATH, SEPARATE_BACKGROUND_IMAGES
)
from model import Model


sample_indices = list(range(NUM_SAMPLE_INDICIES))
train_indices, val_indices = train_test_split(
    sample_indices, test_size=0.2, random_state=SEED
)

def get_augmentations():
    return A.Compose([
        A.ShiftScaleRotate(
            p=0.5,
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            interpolation=2,
        ),
        A.RandomCrop(p=1, width=512, height=512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2)
    ])

def prepare_dataloaders():
    train_dataset = TrainValDataset(
        data_root=DATASET_PATH,
        sample_indices=train_indices,
        augmentations=get_augmentations()
    )
    val_dataset = TrainValDataset(
        data_root=DATASET_PATH,
        sample_indices=val_indices,
        augmentations=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        num_workers=NUM_WORKERS_TRAIN,
        shuffle=True,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTNAT_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        num_workers=NUM_WORKERS_VAL,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTNAT_WORKERS,
    )
    return train_loader, val_loader

def get_trainer():
    seed_everything(SEED)

    checkpoint_callback = ModelCheckpoint(
        dirpath=TRAIN_OUTPUT_DIR,
        filename="best_f1_05",
        save_weights_only=True,
        save_top_k=1,
        monitor="val/f1",
        mode="max",
        save_last=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(leave=True)
    tb_logger = TensorBoardLogger(save_dir=TRAIN_OUTPUT_DIR, name=None)

    trainer = Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar],
        logger=[tb_logger],
        precision="16-mixed",
        deterministic=True,
        benchmark=False,
        sync_batchnorm=False,
        check_val_every_n_epoch=1,
        default_root_dir=".",
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
    )
    return trainer


def train_model(use_oba=False, use_icl=False):
    if use_oba:
        train_loader, val_loader = prepare_dataloaders_oba()
    else:
        train_loader, val_loader = prepare_dataloaders()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = Model()
    trainer = get_trainer()

    if use_icl:
        optimizer_schedulers = model.configure_optimizers()
        optimizer = optimizer_schedulers["optimizer"]
        scheduler = optimizer_schedulers["lr_scheduler"]["scheduler"]     
        model = invariance_constrained_fit(model,train_loader,val_loader,optimizer, scheduler, EPOCHS, "cuda")

    else:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

    return model, train_loader, val_loader


def prepare_dataloaders_oba():
    # Use the OBA dataset for training and the original for validation
    train_dataset = OBAValDataset(
        data_root=DATASET_PATH,
        sample_indices=train_indices,
        annotations_path=TRAIN_ANNOTATIONS_PATH,
        background_root=SEPARATE_BACKGROUND_IMAGES,
        background_prob=BACKGROUND_PROB,
        extract_from_same_image=EXTRACT_FROM_SAME_IMAGE,
        augmentations=get_augmentations(),
        use_oba=True,
        oba_prob=OBA_PROB,
        visualize=False,
        num_oba_objects=NUM_OBA_OBJECTS
    )

    val_dataset = TrainValDataset(
        data_root=DATASET_PATH,
        sample_indices=val_indices,
        augmentations=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        num_workers=NUM_WORKERS_TRAIN,
        shuffle=True,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTNAT_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        num_workers=NUM_WORKERS_VAL,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTNAT_WORKERS,
    )
    return train_loader, val_loader



def invariance_constrained_fit(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    """
    Custom fit-function for training a model using invariance-constrained learning with a primal-dual optimization approach.

    Args:
    model (torch.nn.Module): The segmentation model to be trained.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for adjusting the learning rate.
    num_epochs (int): Number of epochs to train the model.
    device (str): Device to use for training (e.g., "cuda" or "cpu").
    
    Returns:
        torch.nn.Module: The trained model.
    """
      
    model.to(device)
    # Dual variable for primal-dual updates
    gamma = GAMMA

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):

            # Move data to device
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Wrap batch as a list of (image, mask) pairs
            data_batch = list(zip(images, masks))

            # Perform one update step using primal-dual
            batch_loss, gamma = primal_dual_augmentation(
                model, data_batch, get_augmentations(), optimizer, gamma, EPSILON,
                ETA_P, ETA_D, n_mh_steps=N_MH_STEPS, m_samples=M_SAMPLES, device=device
            )
            train_loss += batch_loss 

        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                loss = model.dice_loss_fn(logits, masks) + model.bce_loss_fn(logits, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step(epoch)

    return model

from itertools import product

def hyperparameter_tuning():
    """
    Perform hyperparameter tuning for invariance_constrained_fit.
    """
    # Define hyperparameter search space
    learning_rates = [1e-3]
    batch_sizes = [16, 32]
    gamma_values = [0.1, 0.5]
    epsilon_values = [0.01, 0.05]
    eta_p_values = [0.001, 0.01]
    eta_d_values = [0.001, 0.01]
    n_mh_steps = [2]

    # Store the best configuration and its validation loss
    best_config = None
    best_val_loss = float("inf")

    # Iterate over all combinations of hyperparameters
    for lr, batch_size, gamma, epsilon, eta_p, eta_d, n_mh in product(
        learning_rates, batch_sizes, gamma_values, epsilon_values, eta_p_values, eta_d_values, n_mh_steps
    ):
        print(f"Testing configuration: lr={lr}, batch_size={batch_size}, gamma={gamma}, epsilon={epsilon}, eta_p={eta_p}, eta_d={eta_d}, n_mh={n_mh}")

        # Update global hyperparameters
        global GAMMA, EPSILON, ETA_P, ETA_D, BATCH_SIZE_TRAIN
        GAMMA = gamma
        EPSILON = epsilon
        ETA_P = eta_p
        ETA_D = eta_d
        BATCH_SIZE_TRAIN = batch_size
        N_MH_STEPS = n_mh

        # Prepare data loaders with the current batch size
        train_loader, val_loader = prepare_dataloaders()

        # Initialize model, optimizer, and scheduler
        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Train the model
        trained_model = invariance_constrained_fit(
            model, train_loader, val_loader, optimizer, scheduler, num_epochs=5, device="cuda"
        )

        # Evaluate validation loss
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to("cuda")
                masks = batch["mask"].to("cuda")
                logits = trained_model(images)
                loss = model.dice_loss_fn(logits, masks) + model.bce_loss_fn(logits, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Validation Loss: {val_loss:.4f}")

        # Update the best configuration if the current one is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = {
                "learning_rate": lr,
                "batch_size": batch_size,
                "gamma": gamma,
                "epsilon": epsilon,
                "eta_p": eta_p,
                "eta_d": eta_d,
            }

    print("\nBest Configuration:")
    print(best_config)
    print(f"Best Validation Loss: {best_val_loss:.4f}")

