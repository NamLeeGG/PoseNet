from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from src.utils.heatmaps import build_attention_map, split_regions


def _compute_intensity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    diff = (pred - target) ** 2
    weighted = diff * attention
    weights_sum = attention.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
    loss_per_map = weighted.sum(dim=(-1, -2), keepdim=True) / weights_sum
    return loss_per_map.mean()


def _classification_logits(pred: torch.Tensor, theta_bg: float, theta_fg: float) -> torch.Tensor:
    mid_fg = 0.5 * (theta_bg + theta_fg)
    mid_cf = 0.5 * (theta_fg + 1.0)
    logits_bg = -((pred - theta_bg) ** 2)
    logits_fg = -((pred - mid_fg) ** 2)
    logits_cf = -((pred - mid_cf) ** 2)
    logits = torch.stack([logits_bg, logits_fg, logits_cf], dim=2)
    return logits


def _compute_classification_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    attention: torch.Tensor,
    theta_bg: float,
    theta_fg: float,
) -> torch.Tensor:
    bg_mask, fg_mask, cf_mask = split_regions(target, theta_bg, theta_fg)
    region_targets = torch.zeros_like(target, dtype=torch.long)
    region_targets[fg_mask] = 1
    region_targets[cf_mask] = 2

    logits = _classification_logits(pred, theta_bg, theta_fg)
    pred_classes = torch.argmax(logits, dim=2)

    logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, 3)
    targets_flat = region_targets.reshape(-1)
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    ce = ce.view_as(region_targets)

    mismatch = (pred_classes != region_targets).float()
    weighted_ce = ce * mismatch * attention

    denom = (mismatch * attention).sum()
    if denom <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return weighted_ce.sum() / denom


def compute_ic_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    attention: Optional[torch.Tensor] = None,
    theta_bg: float = 0.4,
    theta_fg: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = target.float()
    pred = pred.float()
    if attention is None:
        attention = build_attention_map(target, theta_bg=theta_bg, theta_fg=theta_fg)
    attention = attention.float()

    iloss = _compute_intensity_loss(pred, target, attention)
    closs = _compute_classification_loss(pred, target, attention, theta_bg, theta_fg)
    return iloss + closs, iloss, closs


class ICLoss(nn.Module):
    def __init__(self, theta_bg: float = 0.4, theta_fg: float = 0.9) -> None:
        super().__init__()
        self.theta_bg = theta_bg
        self.theta_fg = theta_fg

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total, _, _ = compute_ic_loss(
            pred,
            target,
            attention=attention,
            theta_bg=self.theta_bg,
            theta_fg=self.theta_fg,
        )
        return total


__all__ = ["ICLoss", "compute_ic_loss"]
