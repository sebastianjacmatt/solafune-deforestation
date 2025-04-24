import sys
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

# Append project paths
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

from config import EPOCHS, CLASS_NAMES, IR_LAMBDA, T_PSI

class Model(pl.LightningModule):
    """
    PyTorch Lightning module for multi-label image segmentation using a U-Net architecture.
    """
    def __init__(self, use_ir=False):
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

        self.encoder = self.model.encoder #TODO: naive way of getting encoder
        self.decoder = self.model.decoder #TODO: naive way of getting decoder
        self.segmentation_head = self.model.segmentation_head #TODO: naive way of getting segmentation head
        self.T_psi = torch.nn.Sequential(  # this is T_ψ #TODO naive way of defining T_psi
            torch.nn.Conv2d(320, 320, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(320, 320, kernel_size=3, padding=1),
        )

        self.use_ir = use_ir

    def forward(self, x):
        return self.model(x)  # logits

    def shared_step(self, batch, stage):
        """
        Shared logic for training and validation steps.

        Args:
            batch (dict): A dictionary with "image" and "mask" tensors.
            stage (str): Either "train" or "val".

        Returns:
            torch.Tensor: Combined loss for the batch.
        """
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

    def interpolation_robustness_step(self, batch):
        """
        The interpolation robustness, where L_int is added as regularization parameter
        Args:
            batch: A dictionary with "image" and "mask" tensors.
        Return:
            Loss: L_dice + \lambda * L_int.
        """

        image_d1 = batch["domains"][0]["image"]
        image_d2 = batch["domains"][1]["image"]
        mask_d1 = batch["domains"][0]["mask"]
        mask_d2 = batch["domains"][1]["mask"]

        logits_mask_d1 = self.forward(image_d1)
        logits_mask_d2 = self.forward(image_d2)
        
        # compute the Dice and BCE losses as well as the IoU metrics
        base_loss_d1 = self.dice_loss_fn(logits_mask_d1, mask_d1) + self.bce_loss_fn(logits_mask_d1, mask_d1)
        base_output_d1 = self.IoU_out(base_loss_d1, logits_mask_d1, mask_d1)

        base_loss_d2 = self.dice_loss_fn(logits_mask_d2, mask_d2) + self.bce_loss_fn(logits_mask_d2, mask_d2)
        base_output_d2 = self.IoU_out(base_loss_d2, logits_mask_d1, mask_d2)
        
        # Combine outputs and loss
        combined_output = {
            "loss": (base_output_d1["loss"] + base_output_d2["loss"]) / 2,
            "tp": (base_output_d1["tp"] + base_output_d2["tp"]) // 2,
            "fp": (base_output_d1["fp"] + base_output_d2["fp"]) // 2,
            "fn": (base_output_d1["fn"] + base_output_d2["fn"]) // 2,
            "tn": (base_output_d1["tn"] + base_output_d2["tn"]) // 2,
        }
        combined_loss = (base_loss_d1 + base_loss_d2) / 2

        loss_int = self.int_loss(
            batch["domains"][0]["image"],
            batch["domains"][1]["image"],
            batch["domains"][0]["mask"], #TODO: use two masks instead of one and interpolate between them
            ir_lambda=IR_LAMBDA,
            w=0.5, # interpolate in the middle the images
            )
            
        self.training_step_outputs.append(combined_output)
        
        # interpolation loss added as regularization parameter
        return combined_loss + IR_LAMBDA * loss_int

    def int_loss(self,x, x_prime, y, ir_lambda=IR_LAMBDA, w=None):
        """
        Computes the interpolation loss based on equation (4) in the paper.
        Args:
            x (torch.Tensor): Input image tensor.
            x_prime (torch.Tensor): Perturbed image tensor.
            y (torch.Tensor): Ground truth mask tensor.
            ir_lambda (float): Weight for the interpolation loss.
            w (float): Interpolation weight(currently only interpolates at one point).

        Returns:
            torch.Tensor: Interpolation loss.
        """

        if w is None:
            w = 0.5

        # encode x and x'
        z = self.encoder(x)[-1]
        z_prime = self.encoder(x_prime)[-1]

        # compute interpolation representation Z(x, x', w)
        delta = z_prime - z
        z_interp = z + w * self.T_psi(delta)

        # decode z'' from Z(x, x', w)
        logits_interp = self.segmentation_head(self.decoder([z_interp]))

        # compute Dice and BCE losses of interpolated logits
        loss_interp = self.dice_loss_fn(logits_interp, y) + self.bce_loss_fn(logits_interp, y)

        # Step 5: regularization term ||Z(x,x',1) - E(x')||^2
        z_w1 = z + self.T_psi(delta)
        l2_loss = F.mse_loss(z_w1, z_prime)

        return loss_interp

    def IoU_out(self, loss, logits_mask, mask) -> dict:
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
        return output

    def training_step(self, batch, batch_idx):
        """
        Training step wrapper.
        """
        if self.use_ir:
            return self.interpolation_robustness_step(batch)
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step wrapper.
        """
        return self.shared_step(batch, "val")

    def shared_epoch_end(self, outputs, stage):
        """
        Aggregates and logs metrics at the end of an epoch.

        Args:
            outputs (list): List of dictionaries containing loss and confusion matrix stats.
            stage (str): Either "train" or "val".
        """
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
        """
        Handles logging and cleanup at the end of the training epoch.
        """
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        """
        Handles logging and cleanup at the end of the validation epoch.
        """
        self.shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configures optimizer and learning rate scheduler.
        """
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
        """
        Steps the learning rate scheduler.
        """
        scheduler.step(epoch=self.current_epoch)
