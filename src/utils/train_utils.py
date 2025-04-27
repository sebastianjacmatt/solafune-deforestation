import sys
import os
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks import EarlyStopping

from torch.utils.data import DataLoader
from dataset import TrainValDataset, OBAValDataset

from config import (
    EPSILON, ETA_D, ETA_P, GAMMA, M_SAMPLES, N_MH_STEPS, SEED, EPOCHS,
    BATCH_SIZE_TRAIN, BATCH_SIZE_VAL, NUM_SAMPLE_INDICIES,
    NUM_WORKERS_TRAIN, NUM_WORKERS_VAL, PIN_MEMORY, PERSISTNAT_WORKERS,
    BACKGROUND_PROB, EXTRACT_FROM_SAME_IMAGE, OBA_PROB, NUM_OBA_OBJECTS
)
from src.ir_model import IRModel
from src.utils.global_paths import (
    DATASET_PATH, TRAIN_OUTPUT_DIR, TRAIN_ANNOTATIONS_PATH, SEPARATE_BACKGROUND_IMAGES
)
from model import Model
from invariance_constrained import primal_dual_augmentation

# Append project paths
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

sample_indices = list(range(NUM_SAMPLE_INDICIES))
train_indices, val_indices = train_test_split(
    sample_indices, test_size=0.2, random_state=SEED
)

def get_augmentations():
    """
    get function for retrieving the set of augmentations to be applied to the training data.
    
    These augmentations include transformations such as shifting, scaling, rotating, 
    cropping, flipping, and brightness/contrast adjustments. If the Object-Based 
    Augmentation (OBA) implementation is enabled, these augmentations will be applied 
    on top of the OBA-generated augmentations.

    Returns:
        albumentations.Compose
    """
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

def get_augmentations_invariance():
    """
    get function for retrieving the set of augmentations to be applied to the training data.
    Only used with invarianced constrained learning. The augmentations are equal to get_augmentations, with p=1 for all.

    Returns:
        albumentations.Compose
    """
    return A.Compose([
        A.ShiftScaleRotate(
            p=1,
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            interpolation=2,
        ),
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.Transpose(p=1),
        A.RandomRotate90(p=1),
        A.RandomBrightnessContrast(p=1)
    ])
  
def random_crop_icl():
        return A.Compose([
            A.RandomCrop(p=1, width=512, height=512)
               ])
  
def prepare_dataloaders(augmentation, use_ir=False):
    """
    Prepares and returns PyTorch DataLoaders for training and validation datasets.

    This function creates dataset instances for training and validation using the
    TrainValDataset class

    Args:
        augmentation (callable): A set of augmentations to apply to
                                 the training dataset. Only getAugmentations or None should be used

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - val_loader (DataLoader): DataLoader for the validation dataset.
    """

    train_dataset = TrainValDataset(
        data_root=DATASET_PATH,
        sample_indices=train_indices,
        augmentations=augmentation,
        use_ir=use_ir
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

def ir_get_trainer():
    """
    Configures and returns a PyTorch Trainer spesific for Interpolation Robustness.

    Includes Checkpoint, learning rate monitor,
    progress bar and tensor board logger.

    Returns:
        pytorch_lightning.Trainer: A configured Trainer primed for training.
    """
    seed_everything(SEED)

    checkpoint_callback = ModelCheckpoint(
        dirpath=TRAIN_OUTPUT_DIR, filename="best_f1_05",
        save_weights_only=True, save_top_k=1,
        monitor="val/f1", mode="max", save_last=False,
    )
    lr_monitor   = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(leave=True)
    early_stop   = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=2,
        verbose=True,
        strict=False,
    )
    tb_logger = TensorBoardLogger(save_dir=TRAIN_OUTPUT_DIR, name=None)

    trainer = Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar, early_stop],  # include it here
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

def get_trainer():
    """
    Configures and returns a PyTorch Trainer.

    Includes Checkpoint, learning rate monitor,
    progress bar and tensor board logger.

    Returns:
        pytorch_lightning.Trainer: A configured Trainer primed for training.
    """
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

def train_model(use_oba=False, use_icl=False, use_ir=False):
    """
    Runs the training loop for the model
    The function prepares the dataloader, initializes the model and trainer and runs a fit function.
    Contains flags for oba and invariance-constrained learning implementations.

    Args:
        use_oba (bool): If True, uses the OBA dataset for training and applies 
                        object-based augmentations to the training samples.
        use_icl (bool): If True, trains the model using the invariance-constrained 
                        learning approach with a primal-dual optimization strategy.
        use_ir (bool): If True, trains the model using Interpolation Robustness
                         as a regularization method, works with or without OBA. 
    Returns:
        model (torch.nn.Module): The trained model.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    """

    if use_icl and use_ir:
        raise ValueError("Cannot use both use_icl and use_ir at the same time.")

    if use_oba:
        train_loader, val_loader = prepare_dataloaders_oba(get_augmentations())
    elif use_icl:
        train_loader, val_loader = prepare_dataloaders(random_crop_icl())
    elif use_oba & use_icl:
        train_loader, val_loader = prepare_dataloaders_oba(random_crop_icl())
    else:
        train_loader, val_loader = prepare_dataloaders(get_augmentations())
    
    trainer = get_trainer()
    
    if use_ir:
        train_loader, val_loader = prepare_dataloaders(get_augmentations(), use_ir=True)
        model = IRModel(use_ir=True)
        trainer = ir_get_trainer()
    elif use_ir & use_oba:
        train_loader, val_loader = prepare_dataloaders_oba(get_augmentations(), use_ir=True)
        model = IRModel(use_ir=True)
        trainer = ir_get_trainer()
    else: 
        model = Model()

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


def prepare_dataloaders_oba(augmentation, use_ir=False):
    """
    Prepares PyTorch DataLoaders using the OBA dataset for training and the original dataset for validation.

    Args:
        augmentation (callable): A set of transformations/augmentations to apply to the training dataset.
                                Only getAugmentations() or icl_resize() should be called.
    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the OBA-based training dataset.
            - val_loader (DataLoader): DataLoader for the original validation dataset.
    """
    train_dataset = OBAValDataset(
        data_root=DATASET_PATH,
        sample_indices=train_indices,
        annotations_path=TRAIN_ANNOTATIONS_PATH,
        background_root=SEPARATE_BACKGROUND_IMAGES,
        background_prob=BACKGROUND_PROB,
        extract_from_same_image=EXTRACT_FROM_SAME_IMAGE,
        augmentations=augmentation,
        use_oba=True,
        oba_prob=OBA_PROB,
        visualize=False,
        num_oba_objects=NUM_OBA_OBJECTS,
        use_ir=use_ir
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
        train_tp, train_fp, train_fn, train_tn = 0, 0, 0, 0

        for batch in tqdm(train_loader, desc="Training"):

            # Move data to device
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Wrap batch as a list of (image, mask) pairs
            data_batch = list(zip(images, masks))

            # Perform one update step using primal-dual
            batch_loss, gamma = primal_dual_augmentation(
                model, data_batch, get_augmentations_invariance(), optimizer, gamma, EPSILON,
                ETA_P, ETA_D, n_mh_steps=N_MH_STEPS, m_samples=M_SAMPLES, device=device
            )
            train_loss += batch_loss 

            logits = model(images)
            prob_mask = logits.sigmoid()
            threshold = 0.5
            tp, fp, fn, tn = smp.metrics.get_stats(
                (prob_mask > threshold).long(),
                masks.long(),
                mode=smp.losses.MULTILABEL_MODE,
            )
            train_tp += tp.sum().item()
            train_fp += fp.sum().item()
            train_fn += fn.sum().item()
            train_tn += tn.sum().item()

        train_loss /= len(train_loader)
        train_f1 = smp.metrics.f1_score(
            torch.tensor(train_tp), torch.tensor(train_fp), torch.tensor(train_fn), torch.tensor(train_tn)
        )
        print(f"Training Loss: {train_loss:.4f}, Training F1: {train_f1:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                loss = model.dice_loss_fn(logits, masks) + model.bce_loss_fn(logits, masks)
                val_loss += loss.item()

                prob_mask = logits.sigmoid()
                tp, fp, fn, tn = smp.metrics.get_stats(
                    (prob_mask > threshold).long(),
                    masks.long(),
                    mode=smp.losses.MULTILABEL_MODE,
                )
                val_tp += tp.sum().item()
                val_fp += fp.sum().item()
                val_fn += fn.sum().item()
                val_tn += tn.sum().item()

        val_loss /= len(val_loader)
        val_f1 = smp.metrics.f1_score(
            torch.tensor(val_tp), torch.tensor(val_fp), torch.tensor(val_fn), torch.tensor(val_tn)
        )
        print(f"Validation Loss: {val_loss:.4f}, Validation F1: {val_f1:.4f}")

        if scheduler:
            scheduler.step(epoch)

    return model

from itertools import product

def hyperparameter_tuning():
    """
    Perform hyperparameter tuning for invariance_constrained_fit.
    Updates the relevant values from config.py globally
    """
    # Define hyperparameter search space
    learning_rates = [1e-3]
    gamma_values = [0.1, 0.5]
    epsilon_values = [0.01, 0.05]
    eta_p_values = [0.001, 0.01]
    eta_d_values = [0.001, 0.01]
    n_mh_steps = [2]

    # Store the best configuration and its validation loss
    best_config = None
    best_val_loss = float("inf")

    # Iterate over all combinations of hyperparameters
    for lr, gamma, epsilon, eta_p, eta_d, n_mh in product(
        learning_rates, gamma_values, epsilon_values, eta_p_values, eta_d_values, n_mh_steps
    ):
        print(f"Testing configuration: lr={lr}, gamma={gamma}, epsilon={epsilon}, eta_p={eta_p}, eta_d={eta_d}, n_mh={n_mh}")

        # Update global hyperparameters
        global GAMMA, EPSILON, ETA_P, ETA_D
        GAMMA = gamma
        EPSILON = epsilon
        ETA_P = eta_p
        ETA_D = eta_d
        N_MH_STEPS = n_mh

        # Prepare data loaders
        train_loader, val_loader = prepare_dataloaders(random_crop_icl())

        # Initialize model, optimizer, and scheduler
        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Train the model
        trained_model = invariance_constrained_fit(
            model, train_loader, val_loader, optimizer, scheduler, num_epochs=3, device="cuda"
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
                "gamma": gamma,
                "epsilon": epsilon,
                "eta_p": eta_p,
                "eta_d": eta_d,
            }

    print("\nBest Configuration:")
    print(best_config)
    print(f"Best Validation Loss: {best_val_loss:.4f}")

