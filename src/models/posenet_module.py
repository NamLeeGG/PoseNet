from typing import Any, Dict, Tuple, Optional

import hydra
import torch
import rootutils
import numpy as np
from omegaconf import DictConfig
from lightning import LightningModule
from src.loss.lossmodule import ICLoss
from torchmetrics import MinMetric, MeanMetric, Metric

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

### Failure Rate Metric ###
class FailureRate(Metric):
    """Failure Rate (FR) when NME exceeds threshold"""
    def __init__(self, threshold=0.08):
        super().__init__()
        self.threshold = threshold
        self.add_state("count_failures", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Compute normalizing distance (with epsilon to avoid div by zero)
        d = torch.norm(target[:, 0, :] - target[:, 4, :], dim=1) + 1e-6  # Ref: Eq.18, Page 59422

        # Compute normalized error per landmark and average over all landmarks
        error = torch.norm(preds - target, dim=2) / d.unsqueeze(1)
        error = error.mean(dim=1)  # Mean across landmarks

        self.count_failures += torch.sum(error > self.threshold)
        self.total += preds.size(0)

    def compute(self):
        return self.count_failures / self.total


### Cumulative Error Distribution (CED) and AUC Metric ###
class CED_AUC(Metric):
    """Computes the Area Under the Cumulative Error Distribution (CED) Curve"""
    def __init__(self, max_threshold=0.1, num_bins=100):
        super().__init__()
        self.max_threshold = max_threshold
        self.num_bins = num_bins
        self.add_state("errors", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        d = torch.norm(target[:, 0, :] - target[:, 4, :], dim=1) + 1e-6
        error = torch.norm(preds - target, dim=2).mean(dim=1) / d
        self.errors = torch.cat([self.errors, error], dim=0)

    def compute(self):
        thresholds = torch.linspace(0, self.max_threshold, self.num_bins)
        ced_curve = (self.errors.unsqueeze(0) < thresholds.unsqueeze(1)).float().mean(dim=1)
        return torch.trapz(ced_curve, thresholds)


### Normalized Mean Error (NME) Metric ###
class NME(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_nme", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        nme = torch.norm(preds - target, dim=2).mean(dim=1).sum()  # Ref: Eq.18, Page 59422
        self.sum_nme += nme
        self.total += preds.size(0)

    def compute(self):
        return self.sum_nme / self.total

def get_keypoints_from_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Extract keypoints from heatmaps using argmax.
    :param heatmaps: (B, N, H, W) tensor where N is number of landmarks.
    :return: (B, N, 2) tensor with keypoint coordinates (x, y).
    """
    B, N, H, W = heatmaps.shape  
    idxs = torch.argmax(heatmaps.reshape(B, N, -1), dim=2)  # Use PyTorch argmax
    y, x = torch.div(idxs, W, rounding_mode='floor'), idxs % W  # Convert to (x, y)
    keypoints_tensor = torch.stack((x, y), dim=2).float()  # Stack into (B, N, 2)
    return keypoints_tensor

def normalize_keypoints(keypoints: torch.Tensor) -> torch.Tensor:
    """
    Normalize keypoints using Min-Max scaling per sample.
    :param keypoints: (B, N, 2) tensor of keypoints.
    :return: Normalized keypoints.
    """
    x_min = keypoints[:, :, 0].min(dim=1, keepdim=True)[0]
    x_max = keypoints[:, :, 0].max(dim=1, keepdim=True)[0]
    y_min = keypoints[:, :, 1].min(dim=1, keepdim=True)[0]
    y_max = keypoints[:, :, 1].max(dim=1, keepdim=True)[0]

    keypoints[:, :, 0] = 2 * (keypoints[:, :, 0] - x_min) / (x_max - x_min + 1e-6) - 1
    keypoints[:, :, 1] = 2 * (keypoints[:, :, 1] - y_min) / (y_max - y_min + 1e-6) - 1
    return keypoints

class PoseNetModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        # self.criterion = torch.nn.MSELoss()
        self.criterion = ICLoss()


        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = NME()
        self.val_acc = NME()
        self.test_acc = NME()
        self.test_fr = FailureRate(threshold=0.08)
        self.test_auc = CED_AUC(max_threshold=0.1, num_bins=100)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MinMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        preds = normalize_keypoints(get_keypoints_from_heatmaps(logits))
        y = normalize_keypoints(get_keypoints_from_heatmaps(y))

        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_fr(preds, targets)
        self.test_auc(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/nme", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/failure_rate", self.test_fr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        self.log("test/nme_final", self.test_acc.compute(), sync_dist=True, prog_bar=True)
        self.log("test/failure_rate_final", self.test_fr.compute(), sync_dist=True, prog_bar=True)
        self.log("test/auc_final", self.test_auc.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/acc_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


@hydra.main(version_base="1.3", config_path="../../configs/model", config_name="model.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    model: LightningModule = hydra.utils.instantiate(cfg)

if __name__ == "__main__":
    main()
