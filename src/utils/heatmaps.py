from typing import Tuple

import numpy as np
import torch


def generate_gaussian_heatmaps(
    landmarks: np.ndarray,
    heatmap_size: Tuple[int, int],
    sigma: float,
    downsample: int,
) -> np.ndarray:
    """Generate gaussian heatmaps centered at provided landmarks.

    Args:
        landmarks: Array with shape (K, 2) in resized image coordinates.
        heatmap_size: (height, width) for the generated heatmaps.
        sigma: Standard deviation of the gaussian in heatmap pixels.
        downsample: Factor relating image space to heatmap space.
    Returns:
        Heatmaps with shape (K, H, W).
    """

    height, width = heatmap_size
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    heatmaps = []
    for point in landmarks:
        center_x = float(point[0]) / float(downsample)
        center_y = float(point[1]) / float(downsample)
        gaussian = np.exp(
            -((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2.0 * sigma ** 2)
        )
        heatmaps.append(gaussian.astype(np.float32))

    return np.stack(heatmaps, axis=0)


def split_regions(
    heatmaps: torch.Tensor,
    theta_bg: float,
    theta_fg: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return boolean masks for BG, FG and CF regions."""

    bg_mask = heatmaps <= theta_bg
    fg_mask = (heatmaps > theta_bg) & (heatmaps <= theta_fg)
    cf_mask = heatmaps > theta_fg
    return bg_mask, fg_mask, cf_mask


def build_attention_map(
    heatmaps: torch.Tensor,
    theta_bg: float = 0.4,
    theta_fg: float = 0.9,
) -> torch.Tensor:
    """Compute the attention map described in Fard et al."""

    heatmaps = heatmaps.float()
    bg_mask, fg_mask, cf_mask = split_regions(heatmaps, theta_bg, theta_fg)

    area_bg = bg_mask.sum(dim=(-1, -2), keepdim=True).clamp(min=1.0)
    area_fg = fg_mask.sum(dim=(-1, -2), keepdim=True)
    area_cf = cf_mask.sum(dim=(-1, -2), keepdim=True)

    omega_fg = torch.where(area_fg > 0, area_bg / area_fg, torch.ones_like(area_bg))
    omega_cf = torch.where(area_cf > 0, area_bg / area_cf, torch.ones_like(area_bg))

    attention = (
        bg_mask.float()
        + fg_mask.float() * omega_fg
        + cf_mask.float() * omega_cf
    )
    return attention


__all__ = ["generate_gaussian_heatmaps", "build_attention_map", "split_regions"]
