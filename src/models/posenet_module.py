from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from hydra.utils import instantiate
from lightning import LightningModule


class PoseNetModule(LightningModule):
    """LightningModule wrapping PoseNet with IC-Loss and evaluation metrics."""

    def __init__(
        self,
        model: Dict,
        loss: Optional[Dict],
        optimizer: Dict,
        scheduler: Optional[Dict] = None,
        loss_type: str = "ic_loss",
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = instantiate(model)
        self.loss_type = loss_type
        if loss_type == "ic_loss":
            if loss is None:
                raise ValueError("IC-Loss requested but no loss configuration provided.")
            self.criterion = instantiate(loss)
        elif loss_type == "mse":
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type '{loss_type}'.")

        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]], batch_idx: int) -> torch.Tensor:
        images, target_heatmaps, meta = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, target_heatmaps)

        pred_landmarks = self.decode_heatmaps(predictions, meta)
        gt_landmarks = meta["landmarks"].to(pred_landmarks.device)
        nme, fr_008, fr_010 = self.compute_metrics(pred_landmarks, gt_landmarks)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/nme", nme, on_epoch=True, prog_bar=True)
        self.log("train/fr_008", fr_008, on_epoch=True)
        self.log("train/fr_010", fr_010, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]], batch_idx: int) -> None:
        images, target_heatmaps, meta = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, target_heatmaps)

        pred_landmarks = self.decode_heatmaps(predictions, meta)
        gt_landmarks = meta["landmarks"].to(pred_landmarks.device)
        nme, fr_008, fr_010 = self.compute_metrics(pred_landmarks, gt_landmarks)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/nme", nme, on_epoch=True, prog_bar=True)
        self.log("val/fr_008", fr_008, on_epoch=True)
        self.log("val/fr_010", fr_010, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]], batch_idx: int) -> None:
        images, target_heatmaps, meta = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, target_heatmaps)

        pred_landmarks = self.decode_heatmaps(predictions, meta)
        gt_landmarks = meta["landmarks"].to(pred_landmarks.device)
        nme, fr_008, fr_010 = self.compute_metrics(pred_landmarks, gt_landmarks)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True)
        self.log("test/nme", nme, on_epoch=True, prog_bar=True)
        self.log("test/fr_008", fr_008, on_epoch=True)
        self.log("test/fr_010", fr_010, on_epoch=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        if self.scheduler_cfg:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def decode_heatmaps(self, heatmaps: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        batch_size, num_landmarks, hm_h, hm_w = heatmaps.shape
        flattened = heatmaps.view(batch_size, num_landmarks, -1)
        indices = flattened.argmax(dim=-1)
        ys = torch.div(indices, hm_w, rounding_mode="floor")
        xs = indices % hm_w

        img_size = meta.get("img_size")
        if isinstance(img_size, torch.Tensor):
            img_size = img_size.to(device=heatmaps.device, dtype=torch.float32)
            img_h, img_w = img_size[0], img_size[1]
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            img_h = torch.tensor(float(img_size[0]), device=heatmaps.device)
            img_w = torch.tensor(float(img_size[1]), device=heatmaps.device)
        else:
            img_h = torch.tensor(float(hm_h * 4), device=heatmaps.device)
            img_w = torch.tensor(float(hm_w * 4), device=heatmaps.device)

        scale_x = img_w / float(hm_w)
        scale_y = img_h / float(hm_h)
        coords_x = xs.float() * scale_x
        coords_y = ys.float() * scale_y
        return torch.stack([coords_x, coords_y], dim=-1)

    def compute_metrics(self, preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        errors = torch.norm(preds - target, dim=-1)
        ref = self.reference_distance(target)
        per_sample = errors.mean(dim=-1) / ref
        nme = per_sample.mean()
        fr_008 = (per_sample > 0.08).float().mean()
        fr_010 = (per_sample > 0.10).float().mean()
        return nme, fr_008, fr_010

    @staticmethod
    def reference_distance(landmarks: torch.Tensor) -> torch.Tensor:
        if landmarks.size(1) >= 5:
            ref = torch.norm(landmarks[:, 0] - landmarks[:, 4], dim=-1)
            fallback = torch.norm(landmarks[:, 0] - landmarks[:, -1], dim=-1)
            ref = torch.where(ref > 1e-6, ref, fallback)
        else:
            ref = torch.norm(landmarks[:, 0] - landmarks[:, -1], dim=-1)
        return ref.clamp_min(1e-6)


__all__ = ["PoseNetModule"]
