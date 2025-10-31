from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from lightning import LightningModule
from omegaconf import DictConfig

from src.models.losses import ICLoss
from src.utils.landmarks import (
    compute_ced_auc,
    compute_failure_rate,
    compute_normalized_mean_error,
    decode_heatmaps,
)


class PoseNetModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        compile: bool,
        heatmap_downsample: int = 4,
        theta_bg: float = 0.4,
        theta_fg: float = 0.9,
        use_ic_loss: bool = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.heatmap_downsample = heatmap_downsample
        self.theta_bg = theta_bg
        self.theta_fg = theta_fg

        self.use_ic_loss = use_ic_loss
        self.criterion = ICLoss(theta_bg=theta_bg, theta_fg=theta_fg) if use_ic_loss else torch.nn.MSELoss()

        self.val_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"].float()
        targets = batch["heatmap"].float()
        attention = batch.get("attention")
        preds = self.forward(images)
        if isinstance(self.criterion, ICLoss):
            loss = self.criterion(preds, targets, attention=attention)
        else:
            loss = self.criterion(preds, targets)
        return loss, preds

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _ = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_step_outputs = []

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        _, coords_image = decode_heatmaps(preds, self.heatmap_downsample)
        gt_coords = batch["landmarks"].to(coords_image.device)
        nme = compute_normalized_mean_error(coords_image, gt_coords)
        self.log("val/nme", nme.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_step_outputs.append(nme.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self.val_step_outputs:
            return
        nmes = torch.cat(self.val_step_outputs).to(self.device)
        fr_008 = compute_failure_rate(nmes, 0.08)
        fr_010 = compute_failure_rate(nmes, 0.10)
        _, _, auc = compute_ced_auc(nmes, max_threshold=0.2, step=0.001)

        self.log("val/fr_008", fr_008, prog_bar=False, sync_dist=True)
        self.log("val/fr_010", fr_010, prog_bar=False, sync_dist=True)
        self.log("val/auc", auc, prog_bar=True, sync_dist=True)
        self.val_step_outputs = []

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        _, coords_image = decode_heatmaps(preds, self.heatmap_downsample)
        gt_coords = batch["landmarks"].to(coords_image.device)
        nme = compute_normalized_mean_error(coords_image, gt_coords)
        self.log("test/nme", nme.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_step_outputs.append(nme.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs:
            return
        nmes = torch.cat(self.test_step_outputs).to(self.device)
        fr_008 = compute_failure_rate(nmes, 0.08)
        fr_010 = compute_failure_rate(nmes, 0.10)
        _, _, auc = compute_ced_auc(nmes, max_threshold=0.2, step=0.001)

        self.log("test/fr_008", fr_008, prog_bar=False, sync_dist=True)
        self.log("test/fr_010", fr_010, prog_bar=False, sync_dist=True)
        self.log("test/auc", auc, prog_bar=True, sync_dist=True)
        self.test_step_outputs = []

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/nme",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


@hydra.main(version_base="1.3", config_path="../../configs/model", config_name="model.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    _ = hydra.utils.instantiate(cfg)
    return None


if __name__ == "__main__":
    main()
