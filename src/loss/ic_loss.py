from __future__ import annotations

import torch
from torch import nn


class IntensityCategoricalLoss(nn.Module):
    """Implementation of the IC-Loss for heatmap regression."""

    def __init__(
        self,
        theta_bg: float = 0.4,
        theta_fg: float = 0.9,
        phi_bg: float = 0.5,
        phi_fg: float = 0.5,
        beta: float = 4.0,
        phi2_cf: float = 0.05,
    ) -> None:
        super().__init__()
        self.theta_bg = theta_bg
        self.theta_fg = theta_fg
        self.phi_bg = phi_bg
        self.phi_fg = phi_fg
        self.beta = beta
        self.phi2_cf = phi2_cf

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.shape != target.shape:
            raise ValueError(
                f"Predictions and targets must share the same shape, received {preds.shape} vs {target.shape}."
            )

        diff = torch.abs(preds - target)

        mask_bg = (target <= self.theta_bg).float()
        mask_fg = ((target > self.theta_bg) & (target <= self.theta_fg)).float()
        mask_cf = (target > self.theta_fg).float()

        bg_loss = self._region_loss(mask_bg, diff.pow(2)) * self.phi_bg
        fg_loss = self._region_loss(mask_fg, diff.pow(2)) * self.phi_fg
        cf_penalty = self.beta * torch.log1p(diff) + self.phi2_cf * diff.pow(2)
        cf_loss = self._region_loss(mask_cf, cf_penalty)

        total = bg_loss + fg_loss + cf_loss
        return total.mean()

    @staticmethod
    def _region_loss(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        region = (mask * value).sum(dim=(1, 2, 3))
        denom = mask.sum(dim=(1, 2, 3)).clamp_min(eps)
        return region / denom


__all__ = ["IntensityCategoricalLoss"]
