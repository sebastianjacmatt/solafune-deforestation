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

from config import EPOCHS, CLASS_NAMES, OPTIMIZER, LEARNING_RATE_OPT, WEIGHT_DECAY, SCHEDULER, MIN_LEARNING_RATE, WARMUP_LEARNING_RATE, IR_LAMBDA, T_PSI

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
        self.T_psi = T_PSI

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

        # compute logits for both domains
        logits_d1 = self.forward(image_d1)
        logits_d2 = self.forward(image_d2)
        
        # compute Dice and BCE losses for both domains
        base_loss_d1 = self.dice_loss_fn(logits_d1, mask_d1) + self.bce_loss_fn(logits_d1, mask_d1)
        base_loss_d2 = self.dice_loss_fn(logits_d2, mask_d2) + self.bce_loss_fn(logits_d2, mask_d2)
        base_loss = 0.5 * (base_loss_d1 + base_loss_d2)
        
        # Combine outputs and loss avrage across domains
        out_d1 = self.IoU_out(base_loss_d1, logits_d1, mask_d1)
        out_d2 = self.IoU_out(base_loss_d2, logits_d2, mask_d2)

        avg_loss = 0.5 * (out_d1["loss"] + out_d2["loss"])
        avg_tp = 0.5 * (out_d1["tp"] + out_d2["tp"])
        avg_fp = 0.5 * (out_d1["fp"] + out_d2["fp"])
        avg_fn = 0.5 * (out_d1["fn"] + out_d2["fn"])
        avg_tn = 0.5 * (out_d1["tn"] + out_d2["tn"])

        output = {
            "loss" : avg_loss.detach().cpu(),
            "tp" : avg_tp.detach().cpu(),
            "fp" : avg_fp.detach().cpu(),
            "fn" : avg_fn.detach().cpu(),
            "tn" : avg_tn.detach().cpu(),
        }

        self.training_step_outputs.append(output)

        # symmetric interpolation loss
        loss_int = 0.5 * (
            self.int_loss(image_d1, image_d2, mask_d1) + 
            self.int_loss(image_d2, image_d1, mask_d2)
        )
        
        # interpolation loss added as regularization parameter
        return base_loss + IR_LAMBDA * loss_int

    def _to_feature_list(self, encoder_out):
        """
        SMP encoders sometimes return an OrderedDict (EfficientNet-V2) and
        sometimes a plain list.  Convert anything to a list whose order is
        [shallow … deep].  You only need this little util once.
        """
        return list(encoder_out.values()) if isinstance(encoder_out, dict) else encoder_out

    def int_loss(self,x, x_prime, y):
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
        # TODO: Preemtive fix for the decoder crashing, will fix after test runs
        # TODO: decoder will crash as it expects a list of features from encoding step, z_interp is not a list and we therefor need 
        feats       = self._to_feature_list(self.encoder(x))        # [f0 … fL]
        feats_prime = self._to_feature_list(self.encoder(x_prime))

        z, z_prime = feats[-1], self.encoder(x_prime)[-1] #TODO: no need for features for x_prime as it should already have been encoded and can be decoded properly

        w        = torch.rand(z.size(0), 1, 1, 1, device=z.device)
        delta    = z_prime - z
        z_interp = z + w * self.T_psi(delta)                        # ← IR step

        feats[-1] = z_interp                                        # ← **replace deepest**
        logits_interp = self.segmentation_head(self.decoder(feats)) # ← **decode**
        
        loss_cls = self.dice_loss_fn(logits_interp, y) + \
            self.bce_loss_fn (logits_interp, y)

        z_w1   = z + self.T_psi(delta)                              # w = 1
        l2_loss = F.mse_loss(z_w1, z_prime)

        return loss_cls + l2_loss

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
