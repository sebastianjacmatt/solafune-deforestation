import sys
import os

from tqdm import tqdm

from invariance_constrained import primal_dual_augmentation
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import albumentations as A
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from torch.utils.data import DataLoader

from dataset import TrainValDataset, OBAValDataset
from config import EPSILON, ETA_D, ETA_P, GAMMA, INVARIANCE_CONSTRAINED_LEARNING, M_SAMPLES, N_MH_STEPS, SEED, EPOCHS, BATCH_SIZE_TRAIN, BATCH_SIZE_VAL, NUM_SAMPLE_INDICIES, NUM_WORKERS_TRAIN, NUM_WORKERS_VAL, PIN_MEMORY, PERSISTNAT_WORKERS
from global_paths import DATASET_PATH, TRAIN_OUTPUT_DIR, TRAIN_ANNOTATIONS_PATH
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
    tb_logger = TensorBoardLogger(save_dir=TRAIN_OUTPUT_DIR, name=None)

    trainer = Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=[tb_logger],
        precision="16-mixed",
        deterministic=True,
        benchmark=False,
        sync_batchnorm=False,
        check_val_every_n_epoch=1,
        default_root_dir=".",
        accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
    )
    return trainer


def train_model(use_oba=False):
    if use_oba:
        train_loader, val_loader = prepare_dataloaders_oba()
    else:
        train_loader, val_loader = prepare_dataloaders()
    model = Model()
    trainer = get_trainer()

    if INVARIANCE_CONSTRAINED_LEARNING:
        optimizer_schedulers = model.configure_optimizers()
        optimizer = optimizer_schedulers["optimizer"]
        scheduler = optimizer_schedulers["lr_scheduler"]["scheduler"]     
        model = custom_fit(model,train_loader,val_loader,optimizer, scheduler, EPOCHS, "cpu")

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
        augmentations=get_augmentations(),
        use_oba=True,
        oba_prob=1.0 # Set to 100% for testing, then reduce later
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



def custom_fit(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):

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

