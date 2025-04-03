import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import albumentations as A
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from torch.utils.data import DataLoader

from dataset import TrainValDataset
from config import SEED, EPOCHS, BATCH_SIZE_TRAIN, BATCH_SIZE_VAL
from global_paths import DATASET_PATH, TRAIN_OUTPUT_DIR
from model import Model

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
    sample_indices = list(range(176))
    train_indices, val_indices = train_test_split(
        sample_indices, test_size=0.2, random_state=SEED
    )

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
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader, train_indices, val_indices

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


def train_model():
    train_loader, val_loader, train_indices, val_indices = prepare_dataloaders()
    model = Model()
    trainer = get_trainer()

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    return model, train_loader, val_loader, train_indices, val_indices
