from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    """Container describing a single study entry from the JSON file."""

    image_id: str
    image_path: Path
    landmarks: np.ndarray  # (num_landmarks, 2) in the original image space
    orig_size: Optional[Tuple[int, int]]  # (height, width)


def _load_coco_style(payload: Dict, images_dir: Path) -> List[SampleRecord]:
    id_to_info: Dict[str, Tuple[str, int, int]] = {}
    for image_info in payload.get("images", []):
        file_name = image_info.get("file_name")
        if not file_name:
            continue
        image_id = str(image_info.get("id", Path(file_name).stem))
        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        id_to_info[image_id] = (file_name, height, width)

    grouped_annotations: Dict[str, List[Dict]] = {}
    for ann in payload.get("annotations", []):
        image_id = str(ann.get("image_id"))
        if image_id not in id_to_info:
            continue
        grouped_annotations.setdefault(image_id, []).append(ann)

    samples: List[SampleRecord] = []
    for image_id, annotations in grouped_annotations.items():
        file_name, height, width = id_to_info[image_id]
        image_path = (images_dir / file_name).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image '{file_name}' referenced in JSON not found under '{images_dir}'.")

        # Assume a single annotation per image. If multiple exist, take the first.
        annotation = annotations[0]
        keypoints = annotation.get("keypoints", [])
        if len(keypoints) % 3 != 0:
            raise ValueError(f"Annotation for image '{image_id}' does not contain valid keypoints triplets.")

        coords: List[Tuple[float, float]] = []
        for idx in range(0, len(keypoints), 3):
            x, y, _ = keypoints[idx : idx + 3]
            coords.append((float(x), float(y)))
        landmarks = np.asarray(coords, dtype=np.float32)

        orig_size: Optional[Tuple[int, int]] = None
        if height > 0 and width > 0:
            orig_size = (height, width)

        samples.append(
            SampleRecord(
                image_id=image_id,
                image_path=image_path,
                landmarks=landmarks,
                orig_size=orig_size,
            )
        )

    samples.sort(key=lambda record: record.image_id)
    return samples


def _load_simple_style(payload: Dict, images_dir: Path) -> List[SampleRecord]:
    samples: List[SampleRecord] = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        file_name = value.get("file_name", key)
        image_path = (images_dir / file_name).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image '{file_name}' referenced in JSON not found under '{images_dir}'.")

        raw_landmarks: Iterable[Iterable[float]] = value.get("landmarks", [])
        coords = [tuple(map(float, pair)) for pair in raw_landmarks]
        if not coords:
            raise ValueError(f"No landmarks provided for image '{file_name}'.")
        landmarks = np.asarray(coords, dtype=np.float32)

        image_id = str(value.get("image_id", Path(file_name).stem))
        samples.append(
            SampleRecord(
                image_id=image_id,
                image_path=image_path,
                landmarks=landmarks,
                orig_size=None,
            )
        )

    samples.sort(key=lambda record: record.image_id)
    return samples


def load_cervical_json_annotations(data_dir: Path, json_file: str, images_subdir: str) -> List[SampleRecord]:
    """Load annotations from a JSON file supporting both COCO-style and simple mappings."""

    json_path = (data_dir / json_file).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"Annotation file '{json_path}' does not exist.")

    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images_dir = (data_dir / images_subdir).resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory '{images_dir}' does not exist.")

    if "images" in payload and "annotations" in payload:
        samples = _load_coco_style(payload, images_dir)
    else:
        samples = _load_simple_style(payload, images_dir)

    if not samples:
        raise RuntimeError(f"No samples discovered in '{json_path}'.")

    # Ensure all samples share the same number of landmarks
    num_landmarks = {record.landmarks.shape[0] for record in samples}
    if len(num_landmarks) != 1:
        raise ValueError("Inconsistent number of landmarks across annotations.")

    return samples


def _compute_crop_box(landmarks: np.ndarray, img_height: int, img_width: int, margin: float) -> Tuple[int, int, int, int]:
    """Compute a crop box around the landmarks with the given margin."""

    min_xy = landmarks.min(axis=0)
    max_xy = landmarks.max(axis=0)
    min_x, min_y = min_xy
    max_x, max_y = max_xy

    box_width = max_x - min_x
    box_height = max_y - min_y
    pad_x = box_width * margin
    pad_y = box_height * margin

    left = int(np.floor(min_x - pad_x))
    top = int(np.floor(min_y - pad_y))
    right = int(np.ceil(max_x + pad_x))
    bottom = int(np.ceil(max_y + pad_y))

    left = max(left, 0)
    top = max(top, 0)
    right = min(right, img_width - 1)
    bottom = min(bottom, img_height - 1)

    if right <= left:
        right = min(img_width - 1, left + 1)
    if bottom <= top:
        bottom = min(img_height - 1, top + 1)

    return top, bottom, left, right


def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    height, width = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _prepare_channels(gray: np.ndarray, use_hist_eq: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray_01 = gray.astype(np.float32) / 255.0
    if use_hist_eq:
        equalized = cv2.equalizeHist(gray)
        equalized = equalized.astype(np.float32) / 255.0
    else:
        equalized = gray_01
    inverted = 1.0 - gray_01
    return gray_01, equalized, inverted


def _build_heatmaps(
    landmarks: np.ndarray,
    img_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    sigma: float,
) -> np.ndarray:
    img_h, img_w = img_size
    hm_h, hm_w = heatmap_size
    num_landmarks = landmarks.shape[0]

    xs = np.arange(hm_w, dtype=np.float32)
    ys = np.arange(hm_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    heatmaps = np.zeros((num_landmarks, hm_h, hm_w), dtype=np.float32)
    for idx, (x, y) in enumerate(landmarks):
        mu_x = x / img_w * hm_w
        mu_y = y / img_h * hm_h
        heatmaps[idx] = np.exp(-((grid_x - mu_x) ** 2 + (grid_y - mu_y) ** 2) / (2 * sigma**2))
    return heatmaps


class CervicalJSONDataset(Dataset):
    """Dataset producing PoseNet-ready samples from cervical spine JSON annotations."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        split: str,
        img_size: Sequence[int],
        heatmap_size: Sequence[int],
        sigma: float,
        use_flip: bool = True,
        use_hist_eq: bool = True,
        crop_margin: float = 0.15,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of 'train', 'val', or 'test'.")

        self.samples = list(samples)
        self.split = split
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.heatmap_size = (int(heatmap_size[0]), int(heatmap_size[1]))
        self.sigma = float(sigma)
        self.use_hist_eq = use_hist_eq
        self.crop_margin = crop_margin
        self.include_flipped = use_flip and split == "train"
        self.num_variants = 2 if self.include_flipped else 1

    def __len__(self) -> int:
        return len(self.samples) * self.num_variants

    def __getitem__(self, idx: int) -> Dict[str, object]:
        base_index = idx // self.num_variants
        variant_index = idx % self.num_variants
        record = self.samples[base_index]

        gray = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Unable to load image '{record.image_path}'.")
        orig_h, orig_w = gray.shape

        if record.orig_size is None:
            orig_size = (orig_h, orig_w)
        else:
            orig_size = record.orig_size

        gray_norm, equalized, inverted = _prepare_channels(gray, self.use_hist_eq)

        top, bottom, left, right = _compute_crop_box(record.landmarks, orig_h, orig_w, self.crop_margin)
        bottom_exclusive = bottom + 1
        right_exclusive = right + 1
        crop_slice = np.s_[top:bottom_exclusive, left:right_exclusive]
        gray_crop = gray_norm[crop_slice]
        eq_crop = equalized[crop_slice]
        inv_crop = inverted[crop_slice]

        crop_h = max(bottom_exclusive - top, 1)
        crop_w = max(right_exclusive - left, 1)
        scale_x = self.img_size[1] / float(crop_w)
        scale_y = self.img_size[0] / float(crop_h)

        resized_gray = _resize_image(gray_crop, self.img_size)
        resized_eq = _resize_image(eq_crop, self.img_size)
        resized_inv = _resize_image(inv_crop, self.img_size)

        # Adjust landmarks into the resized crop space
        adjusted_landmarks = record.landmarks.copy()
        adjusted_landmarks[:, 0] = (adjusted_landmarks[:, 0] - left) * scale_x
        adjusted_landmarks[:, 1] = (adjusted_landmarks[:, 1] - top) * scale_y

        if variant_index == 0:
            channels = np.stack([resized_gray, resized_eq, resized_inv], axis=0)
            landmarks = adjusted_landmarks
            variant_name = "original"
        else:
            flipped_gray = np.flip(resized_gray, axis=1)
            flipped_eq = np.flip(resized_eq, axis=1)
            flipped_inv = np.flip(resized_inv, axis=1)
            flipped_landmarks = adjusted_landmarks.copy()
            flipped_landmarks[:, 0] = self.img_size[1] - 1 - flipped_landmarks[:, 0]
            channels = np.stack([flipped_gray, flipped_inv, flipped_eq], axis=0)
            landmarks = flipped_landmarks
            variant_name = "flipped"

        heatmaps = _build_heatmaps(landmarks, self.img_size, self.heatmap_size, self.sigma)

        image_tensor = torch.from_numpy(channels).float()
        heatmap_tensor = torch.from_numpy(heatmaps).float()
        landmarks_tensor = torch.from_numpy(landmarks).float()

        meta = {
            "image_id": record.image_id,
            "orig_size": orig_size,
            "crop_box": (top, left, bottom, right),
            "variant": variant_name,
            "img_size": self.img_size,
            "heatmap_size": self.heatmap_size,
        }

        return {
            "image": image_tensor,
            "heatmaps": heatmap_tensor,
            "landmarks": landmarks_tensor,
            "meta": meta,
        }


__all__ = ["CervicalJSONDataset", "SampleRecord", "load_cervical_json_annotations"]
