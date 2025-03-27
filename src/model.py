import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from config import EPOCHS, CLASS_NAMES
# or import something like LR, WEIGHT_DECAY if we keep them in config.py

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # create segmentation model
        self.model = smp.create_model(
            arch="unet",
            encoder_name="tu-tf_efficientnetv2_s",
            encoder_weights="imagenet",
            in_channels=12,
            classes=4,
        )

        # define loss functions
        self.dice_loss_fn = smp.losses.DiceLoss(
            mode=smp.losses.MULTILABEL_MODE, from_logits=True
        )
        self.bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)  # logits

    def shared_step(self, batch, stage):
        image = batch["image"]
        mask = batch["mask"]

        logits_mask = self.forward(image)

        loss = self.dice_loss_fn(logits_mask, mask) + \
               self.bce_loss_fn(logits_mask, mask)

        # compute stats
        prob_mask = logits_mask.sigmoid()
        threshold = 0.5
        tp, fp, fn, tn = smp.metrics.get_stats(
            (prob_mask > threshold).long(),
            mask.long(),
            mode=smp.losses.MULTILABEL_MODE,
        )

        output = {
            "loss": loss.detach().cpu(),
            "tp": tp.detach().cpu(),
            "fp": fp.detach().cpu(),
            "fn": fn.detach().cpu(),
            "tn": tn.detach().cpu(),
        }

        if stage == "train":
            self.training_step_outputs.append(output)
        else:
            self.validation_step_outputs.append(output)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def shared_epoch_end(self, outputs, stage):
        def log(name, tensor, prog_bar=False):
            # Log a scalar metric
            self.log(f"{stage}/{name}", tensor.to(self.device), sync_dist=True, prog_bar=prog_bar)

        # aggregate losses
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log("loss", loss, prog_bar=True)

        # compute F1 for each class
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        f1_scores = {}
        for i, class_name in enumerate(CLASS_NAMES):
            f1_scores[class_name] = smp.metrics.f1_score(
                tp[:, i], fp[:, i], fn[:, i], tn[:, i], reduction="macro-imagewise"
            )
            log(f"f1/{class_name}", f1_scores[class_name])

        f1_avg = torch.stack(list(f1_scores.values())).mean()
        log("f1", f1_avg, prog_bar=True)

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt="adamw",
            lr=1e-4,
            weight_decay=1e-2,
            filter_bias_and_bn=True
        )
        scheduler, _ = create_scheduler_v2(
            optimizer,
            sched="cosine",
            num_epochs=EPOCHS,
            min_lr=0.0,
            warmup_lr=1e-5,
            warmup_epochs=0,
            warmup_prefix=False,
            step_on_epochs=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        # Timm's scheduler needs the current epoch
        scheduler.step(epoch=self.current_epoch)
