import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.heatmaps import generate_gaussian_heatmaps, build_attention_map


@dataclass
class ImageSample:
    image_path: Path
    landmarks: np.ndarray  # (K, 2) in original image coordinates
    image_size: Tuple[int, int]  # (height, width)


def _load_annotation_mapping(annotation_path: Path) -> Dict[str, List[List[float]]]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, List[List[float]]] = {}

    if isinstance(data, dict) and "images" in data and "annotations" in data:
        images = {item.get("id", idx): item for idx, item in enumerate(data["images"])}
        for ann in data.get("annotations", []):
            image_id = ann.get("image_id") or ann.get("id")
            if image_id not in images:
                continue
            image_info = images[image_id]
            file_name = image_info.get("file_name") or image_info.get("path") or image_info.get("name")
            if file_name is None:
                continue

            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                if len(keypoints) % 3 == 0:
                    coords = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)[:, :2]
                else:
                    coords = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
            elif "landmarks" in ann:
                coords = np.asarray(ann["landmarks"], dtype=np.float32).reshape(-1, 2)
            elif "points" in ann:
                coords = np.asarray(ann["points"], dtype=np.float32).reshape(-1, 2)
            else:
                continue
            normalized_name = str(file_name).replace("\\", "/")
            mapping[normalized_name] = coords.tolist()
        return mapping

    if isinstance(data, dict) and "items" in data:
        categories = data.get("categories", {})
        ordered_labels: Sequence[str] = []
        if "label" in categories and isinstance(categories["label"], dict):
            labels = categories["label"].get("labels", [])
            ordered_labels = [label.get("name", str(idx)) for idx, label in enumerate(labels)]

        for item in data["items"]:
            item_id = item.get("id") or item.get("name") or item.get("image_id")
            if item_id is None:
                continue
            annotations = item.get("annotations", [])
            points_by_label: Dict[str, List[float]] = {}
            for ann in annotations:
                label_id = ann.get("label_id")
                label_name = None
                if label_id is not None and label_id < len(ordered_labels):
                    label_name = ordered_labels[label_id]
                label_name = label_name or ann.get("label") or str(label_id)
                points = ann.get("points")
                if not points:
                    continue
                if isinstance(points[0], list):
                    coords = points[0]
                else:
                    coords = points
                points_by_label[label_name] = coords

            if ordered_labels and all(label in points_by_label for label in ordered_labels):
                ordered_points = [points_by_label[label] for label in ordered_labels]
            else:
                ordered_points = [points_by_label[key] for key in sorted(points_by_label.keys())]
            normalized_id = str(item_id).replace("\\", "/")
            mapping[normalized_id] = ordered_points
        return mapping

    raise ValueError(f"Unsupported annotation format in {annotation_path}")


def _resolve_image_paths(data_dir: Path, img_ext: str) -> List[Path]:
    patterns = [f"*{img_ext}"] if img_ext.startswith(".") else [f"*.{img_ext}"]
    image_paths: List[Path] = []
    for pattern in patterns:
        image_paths.extend(sorted(data_dir.rglob(pattern)))
    return image_paths


def _match_landmarks(
    image_path: Path,
    mapping: Dict[str, List[List[float]]],
    data_root: Path,
    image_root: Path,
) -> Optional[np.ndarray]:
    candidates: List[str] = []
    candidates.append(image_path.name)
    candidates.append(image_path.stem)

    try:
        relative_to_image_root = image_path.relative_to(image_root).as_posix()
        candidates.append(relative_to_image_root)
    except ValueError:
        pass

    try:
        relative_to_data_root = image_path.relative_to(data_root).as_posix()
        candidates.append(relative_to_data_root)
    except ValueError:
        pass

    abs_posix = image_path.as_posix()
    candidates.append(abs_posix)

    normalized_candidates = {candidate.replace("\\", "/") for candidate in candidates}

    for candidate in normalized_candidates:
        if candidate in mapping:
            return np.asarray(mapping[candidate], dtype=np.float32)
    return None


def load_image_samples(data_dir: Path, annotation_file: str, img_ext: str) -> List[ImageSample]:
    annotation_path = data_dir / annotation_file
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found at {annotation_path}")

    mapping = _load_annotation_mapping(annotation_path)

    image_root = data_dir
    images_subdir = data_dir / "images"
    if images_subdir.exists():
        image_root = images_subdir

    image_paths = _resolve_image_paths(image_root, img_ext)

    samples: List[ImageSample] = []
    for image_path in image_paths:
        landmarks = _match_landmarks(image_path, mapping, data_dir, image_root)
        if landmarks is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        height, width = image.shape[:2]
        samples.append(ImageSample(image_path=image_path, landmarks=landmarks, image_size=(height, width)))
    if not samples:
        raise RuntimeError("No annotated samples found. Check annotation file and image directory.")
    return samples


def _build_augmentation_pipeline() -> A.Compose:
    return A.Compose(
        [
            A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT_101, p=0.75),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class CervicalSpineDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ImageSample],
        indices: Optional[Sequence[int]] = None,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 128),  # (height, width)
        heatmap_downsample: int = 4,
        sigma: float = 1.5,
        theta_bg: float = 0.4,
        theta_fg: float = 0.9,
        use_flipped: bool = True,
        random_eval_variant: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.samples = list(samples)
        self.indices = list(indices) if indices is not None else list(range(len(self.samples)))
        self.split = split
        self.image_size = image_size
        self.heatmap_downsample = heatmap_downsample
        self.sigma = sigma
        self.theta_bg = theta_bg
        self.theta_fg = theta_fg
        self.use_flipped = use_flipped
        self.random_eval_variant = random_eval_variant
        self.rng = random.Random(seed)

        self.num_landmarks = self.samples[self.indices[0]].landmarks.shape[0]
        self.heatmap_size = (image_size[0] // heatmap_downsample, image_size[1] // heatmap_downsample)

        self.augment = _build_augmentation_pipeline() if split == "train" else None

        self.index_map: List[Tuple[int, int]] = []
        if split == "train" and use_flipped:
            for idx in self.indices:
                self.index_map.append((idx, 0))
                self.index_map.append((idx, 1))
        else:
            for idx in self.indices:
                if split in {"val", "test"} and use_flipped and random_eval_variant:
                    variant = self.rng.randint(0, 1)
                    self.index_map.append((idx, variant))
                else:
                    self.index_map.append((idx, 0))

    def __len__(self) -> int:
        return len(self.index_map)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _crop_and_resize(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        min_x = max(0.0, float(landmarks[:, 0].min()))
        min_y = max(0.0, float(landmarks[:, 1].min()))
        max_x = min(float(w - 1), float(landmarks[:, 0].max()))
        max_y = min(float(h - 1), float(landmarks[:, 1].max()))

        min_x_i = int(math.floor(min_x))
        min_y_i = int(math.floor(min_y))
        max_x_i = int(math.ceil(max_x))
        max_y_i = int(math.ceil(max_y))

        max_x_i = max(min(w - 1, max_x_i), min_x_i + 1)
        max_y_i = max(min(h - 1, max_y_i), min_y_i + 1)

        cropped = image[min_y_i : max_y_i + 1, min_x_i : max_x_i + 1]

        offset = np.array([min_x_i, min_y_i], dtype=np.float32)
        landmarks_cropped = landmarks - offset

        crop_h, crop_w = cropped.shape[:2]
        scale_x = self.image_size[1] / max(crop_w, 1)
        scale_y = self.image_size[0] / max(crop_h, 1)
        resized = cv2.resize(cropped, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        landmarks_resized = landmarks_cropped * np.array([scale_x, scale_y], dtype=np.float32)

        scale = np.array([scale_x, scale_y], dtype=np.float32)
        crop_box = np.array([min_x_i, min_y_i, max_x_i, max_y_i], dtype=np.float32)

        return resized, landmarks_resized, crop_box, scale

    def _preprocess_channels(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32) / 255.0
        equalized = cv2.equalizeHist(gray)
        equalized_float = equalized.astype(np.float32) / 255.0
        inverted_float = 1.0 - gray_float
        return gray_float, equalized_float, inverted_float

    def _build_variant(
        self,
        base_channels: Tuple[np.ndarray, np.ndarray, np.ndarray],
        landmarks: np.ndarray,
        variant: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        gray, equalized, inverted = base_channels
        if variant == 0:
            image = np.stack([gray, equalized, inverted], axis=-1)
            keypoints = landmarks.copy()
        else:
            flipped_gray = np.flip(gray, axis=1)
            flipped_equalized = np.flip(equalized, axis=1)
            flipped_inverted = np.flip(inverted, axis=1)
            image = np.stack([flipped_gray, flipped_inverted, flipped_equalized], axis=-1)
            keypoints = landmarks.copy()
            keypoints[:, 0] = (self.image_size[1] - 1) - keypoints[:, 0]
        return image, keypoints

    def _apply_augmentation(
        self, image: np.ndarray, keypoints: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.augment is None:
            return image, keypoints
        augmented = self.augment(image=image, keypoints=[tuple(kp) for kp in keypoints])
        aug_image = augmented["image"]
        aug_keypoints = np.asarray(augmented["keypoints"], dtype=np.float32)
        aug_keypoints[:, 0] = np.clip(aug_keypoints[:, 0], 0.0, self.image_size[1] - 1)
        aug_keypoints[:, 1] = np.clip(aug_keypoints[:, 1], 0.0, self.image_size[0] - 1)
        return aug_image, aug_keypoints

    def _tensorize(
        self,
        image: np.ndarray,
        heatmaps: np.ndarray,
        attention: np.ndarray,
        landmarks: np.ndarray,
        landmarks_original: np.ndarray,
        crop_box: np.ndarray,
        scale: np.ndarray,
        variant: int,
        image_path: Path,
    ) -> Dict[str, torch.Tensor]:
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        heatmap_tensor = torch.from_numpy(heatmaps.astype(np.float32))
        attention_tensor = torch.from_numpy(attention.astype(np.float32))
        landmark_tensor = torch.from_numpy(landmarks.astype(np.float32))
        landmark_original_tensor = torch.from_numpy(landmarks_original.astype(np.float32))
        crop_tensor = torch.from_numpy(crop_box.astype(np.float32))
        scale_tensor = torch.from_numpy(scale.astype(np.float32))

        return {
            "image": image_tensor,
            "heatmap": heatmap_tensor,
            "attention": attention_tensor,
            "landmarks": landmark_tensor,
            "landmarks_original": landmark_original_tensor,
            "crop_box": crop_tensor,
            "scale": scale_tensor,
            "variant": torch.tensor(float(variant), dtype=torch.float32),
            "image_path": image_path.name,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx, variant = self.index_map[idx]
        sample = self.samples[sample_idx]

        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image at {sample.image_path}")
        image = self._prepare_image(image)
        landmarks_original = sample.landmarks.astype(np.float32)

        resized, landmarks_resized, crop_box, scale = self._crop_and_resize(image, landmarks_original)
        base_channels = self._preprocess_channels(resized)
        variant_image, variant_keypoints = self._build_variant(base_channels, landmarks_resized, variant)
        variant_image, variant_keypoints = self._apply_augmentation(variant_image, variant_keypoints)

        heatmaps = generate_gaussian_heatmaps(
            variant_keypoints,
            self.heatmap_size,
            sigma=self.sigma,
            downsample=self.heatmap_downsample,
        )
        attention_map = build_attention_map(
            torch.from_numpy(heatmaps),
            theta_bg=self.theta_bg,
            theta_fg=self.theta_fg,
        ).numpy()

        return self._tensorize(
            variant_image,
            heatmaps,
            attention_map,
            variant_keypoints,
            landmarks_original,
            crop_box,
            scale,
            variant,
            sample.image_path,
        )


__all__ = ["CervicalSpineDataset", "ImageSample", "load_image_samples"]
