import sys
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch.utils.checkpoint import checkpoint

# Append project paths
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

from config import EPOCHS, CLASS_NAMES, OPTIMIZER, LEARNING_RATE_OPT, WEIGHT_DECAY, SCHEDULER, MIN_LEARNING_RATE, WARMUP_LEARNING_RATE, IR_LAMBDA, T_PSI

class IRModel(pl.LightningModule):
    """
    PyTorch Lightning module for multi-label image segmentation with interpolation robustness.
    """
    def __init__(self, use_ir=True):
        super().__init__()
        self.save_hyperparameters()

        # Create segmentation model
        self.model = smp.create_model(
            arch="unet",
            encoder_name="tu-tf_efficientnetv2_s",
            encoder_weights="imagenet",
            in_channels=12,
            classes=4,
        )


        # Define loss functions
        self.dice_loss_fn = smp.losses.DiceLoss(
            mode=smp.losses.MULTILABEL_MODE, from_logits=True
        )
        self.bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()

        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Model components
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
        self.T_psi = torch.jit.script(T_PSI)

        self.use_ir = use_ir

    def _checkpointed_encoder(self, x):
        return self.encoder(x)

    def _checkpointed_decoder(self, feats):
        return self.decoder(feats)

    def forward(self, x):
        """feats = checkpoint(self._checkpointed_encoder, x, use_reentrant=False)
        decoded = checkpoint(self._checkpointed_decoder, feats, use_reentrant=False)
        logits = self.segmentation_head(decoded)
        return logits"""
        return self.model(x)

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

        # Compute stats
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

    def _to_feature_list(self, encoder_out):
        """
        Convert encoder output to a list, handling both OrderedDict and list inputs.
        """
        return list(encoder_out.values()) if isinstance(encoder_out, dict) else encoder_out

    def interpolation_robustness_step(self, batch):
        """
        Interpolation robustness step with L_int regularization.

        Args:
            batch: A dictionary with "domains" containing "image" and "mask" tensors.

        Returns:
            torch.Tensor: Combined loss (L_dice + λ * L_int).
        """
        image_d1 = batch["domains"][0]["image"]
        image_d2 = batch["domains"][1]["image"]
        mask_d1 = batch["domains"][0]["mask"]
        mask_d2 = batch["domains"][1]["mask"]

        # Compute logits and encoder outputs
        with torch.no_grad():
            feats_d1 = self._to_feature_list(self.encoder(image_d1))
            feats_d2 = self._to_feature_list(self.encoder(image_d2))
        
        logits_d1 = self.segmentation_head(self.decoder(feats_d1))
        logits_d2 = self.segmentation_head(self.decoder(feats_d2))
        
        # Compute Dice and BCE losses for both domains
        base_loss_d1 = self.dice_loss_fn(logits_d1, mask_d1) + self.bce_loss_fn(logits_d1, mask_d1)
        base_loss_d2 = self.dice_loss_fn(logits_d2, mask_d2) + self.bce_loss_fn(logits_d2, mask_d2)
        base_loss = 0.5 * (base_loss_d1 + base_loss_d2)
        
        # Combine outputs and loss average across domains
        out_d1 = self.IoU_out(base_loss_d1, logits_d1, mask_d1)
        out_d2 = self.IoU_out(base_loss_d2, logits_d2, mask_d2)

        avg_loss = 0.5 * (out_d1["loss"] + out_d2["loss"])
        avg_tp = 0.5 * (out_d1["tp"] + out_d2["tp"])
        avg_fp = 0.5 * (out_d1["fp"] + out_d2["fp"])
        avg_fn = 0.5 * (out_d1["fn"] + out_d2["fn"])
        avg_tn = 0.5 * (out_d1["tn"] + out_d2["tn"])

        output = {
            "loss": avg_loss.detach().cpu(),
            "tp": avg_tp.detach().cpu(),
            "fp": avg_fp.detach().cpu(),
            "fn": avg_fn.detach().cpu(),
            "tn": avg_tn.detach().cpu(),
        }

        self.training_step_outputs.append(output)

        # Symmetric interpolation loss
        loss_int = 0.5 * (
            self.int_loss(feats_d1, feats_d2, mask_d1) + 
            self.int_loss(feats_d2, feats_d1, mask_d2)
        )
        
        return base_loss + IR_LAMBDA * loss_int

    def int_loss(self, feats, feats_prime, y):
        """
        Computes the interpolation loss.

        Args:
            feats: Encoder features for first domain.
            feats_prime: Encoder features for second domain.
            y: Ground truth mask tensor.

        Returns:
            torch.Tensor: Interpolation loss.
        """
        z = feats[-1]
        z_prime = feats_prime[-1]

        w = torch.rand(z.size(0), 1, 1, 1, device=z.device)
        delta = z_prime - z

        #with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
        z_interp = z + w * self.T_psi(delta)
        feats[-1] = z_interp
        logits_interp = self.segmentation_head(self.decoder(feats))
        
        loss_cls = self.dice_loss_fn(logits_interp, y) + \
            self.bce_loss_fn(logits_interp, y)

        z_w1 = z + self.T_psi(delta)
        l2_loss = F.mse_loss(z_w1, z_prime)

        #del z, z_prime, delta, z_interp, logits_interp, z_w1
        return loss_cls + l2_loss

    def IoU_out(self, loss, logits_mask, mask):
        """
        Computes IoU metrics for a given batch.

        Args:
            loss: Computed loss for the batch.
            logits_mask: Model output logits.
            mask: Ground truth mask.

        Returns:
            dict: Metrics including loss and confusion matrix stats.
        """
        prob_mask = logits_mask.sigmoid()
        threshold = 0.5
        tp, fp, fn, tn = smp.metrics.get_stats(
            (prob_mask > threshold).long(),
            mask.long(),
            mode=smp.losses.MULTILABEL_MODE,
        )
        return {
            "loss": loss.detach().cpu(),
            "tp": tp.detach().cpu(),
            "fp": fp.detach().cpu(),
            "fn": fn.detach().cpu(),
            "tn": tn.detach().cpu(),
        }

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
            outputs: List of dictionaries containing loss and confusion matrix stats.
            stage: Either "train" or "val".
        """
        def log(name, tensor, prog_bar=False):
            self.log(f"{stage}/{name}", tensor.to(self.device), sync_dist=True, prog_bar=prog_bar)

        # Aggregate losses
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log("loss", loss, prog_bar=True)

        # Compute F1 for each class
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
            opt=OPTIMIZER,
            lr=LEARNING_RATE_OPT,
            weight_decay=WEIGHT_DECAY,
            filter_bias_and_bn=True
        )
        scheduler, _ = create_scheduler_v2(
            optimizer,
            sched=SCHEDULER,
            num_epochs=EPOCHS,
            min_lr=MIN_LEARNING_RATE,
            warmup_lr=WARMUP_LEARNING_RATE,
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