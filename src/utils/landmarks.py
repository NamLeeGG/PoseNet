from typing import Optional, Tuple

import torch


def decode_heatmaps(heatmaps: torch.Tensor, downsample: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode argmax coordinates from heatmaps."""

    b, k, h, w = heatmaps.shape
    flat = heatmaps.view(b, k, -1)
    indices = torch.argmax(flat, dim=-1)
    ys = indices // w
    xs = indices % w
    coords_heatmap = torch.stack([xs.float(), ys.float()], dim=-1)
    coords_image = coords_heatmap * float(downsample)
    return coords_heatmap, coords_image


def restore_landmarks_to_original(
    coords_image: torch.Tensor,
    crop_box: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Map coordinates from resized crop back to the original image space."""

    scale_x = scale[:, 0].view(-1, 1)
    scale_y = scale[:, 1].view(-1, 1)
    min_x = crop_box[:, 0].view(-1, 1)
    min_y = crop_box[:, 1].view(-1, 1)

    scale_x = torch.where(scale_x == 0, torch.ones_like(scale_x), scale_x)
    scale_y = torch.where(scale_y == 0, torch.ones_like(scale_y), scale_y)

    orig_x = coords_image[..., 0] / scale_x + min_x
    orig_y = coords_image[..., 1] / scale_y + min_y
    return torch.stack([orig_x, orig_y], dim=-1)


def _valid_landmark_mask(landmarks: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(landmarks).all(dim=-1)


def _normalization_distance(landmarks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size, num_landmarks, _ = landmarks.shape
    distances = torch.empty(batch_size, dtype=landmarks.dtype, device=landmarks.device)
    for i in range(batch_size):
        if num_landmarks >= 5 and mask[i, 0] and mask[i, 4]:
            ref_a = landmarks[i, 0]
            ref_b = landmarks[i, 4]
        else:
            valid_indices = torch.nonzero(mask[i], as_tuple=False).view(-1)
            if valid_indices.numel() >= 2:
                ref_a = landmarks[i, valid_indices[0]]
                ref_b = landmarks[i, valid_indices[-1]]
            elif valid_indices.numel() == 1:
                ref_a = landmarks[i, valid_indices[0]]
                ref_b = ref_a
            else:
                distances[i] = torch.tensor(1.0, device=landmarks.device, dtype=landmarks.dtype)
                continue
        distances[i] = torch.norm(ref_a - ref_b, p=2)
    return distances.clamp(min=1e-6)


def compute_normalized_mean_error(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute per-image normalized mean error."""

    mask = _valid_landmark_mask(targets)
    diffs = torch.norm(preds - targets, dim=-1)
    masked_diffs = diffs * mask.float()
    counts = mask.float().sum(dim=-1).clamp(min=1.0)
    per_image_error = masked_diffs.sum(dim=-1) / counts
    distances = _normalization_distance(targets, mask)
    return per_image_error / distances


def compute_failure_rate(nmes: torch.Tensor, threshold: float) -> torch.Tensor:
    return (nmes > threshold).float().mean()


def compute_ced_auc(
    nmes: torch.Tensor,
    max_threshold: float = 0.2,
    step: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    thresholds = torch.arange(0.0, max_threshold + step, step, device=nmes.device)
    ced = torch.stack([(nmes <= thr).float().mean() for thr in thresholds])
    auc = torch.trapz(ced, thresholds) / max_threshold
    return thresholds, ced, auc


__all__ = [
    "decode_heatmaps",
    "restore_landmarks_to_original",
    "compute_normalized_mean_error",
    "compute_failure_rate",
    "compute_ced_auc",
]
