from typing import Any, Dict, Tuple, Optional

import hydra
import torch
import rootutils
import numpy as np
from omegaconf import DictConfig
from lightning import LightningModule
from src.loss.lossmodule import NME
from torchmetrics import MinMetric, MeanMetric

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def get_keypoints_from_heatmaps(heatmaps):
    B, N, H, W = heatmaps.shape  # (batch_size, 23, 64, 32)
    idxs = np.argmax(heatmaps.reshape(B, N, -1).detach().cpu(), axis=2)
    y, x = np.unravel_index(idxs, (H, W))  # (B, N), (B, N)
    keypoints_np = np.stack((x, y), axis=2)
    keypoints_tensor = torch.tensor(keypoints_np, dtype=torch.float32)
    return keypoints_tensor  # (batch_size, 23, 2)

def normalize_keypoints(keypoints):    
    # Tính min và max cho x và y
    x_min = keypoints[:, :, 0].min()
    x_max = keypoints[:, :, 0].max()
    y_min = keypoints[:, :, 1].min()
    y_max = keypoints[:, :, 1].max()
    # Chuẩn hóa theo công thức Min-Max
    keypoints[:, :, 0] = 2 * (keypoints[:, :, 0] - x_min) / (x_max - x_min) - 1
    keypoints[:, :, 1] = 2 * (keypoints[:, :, 1] - y_min) / (y_max - y_min) - 1
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
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = NME()
        self.val_acc = NME()
        self.test_acc = NME()

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

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

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
