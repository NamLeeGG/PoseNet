from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.dataset import (
    CervicalJSONDataset,
    SampleRecord,
    load_cervical_json_annotations,
)


class CervicalJSONDataModule(LightningDataModule):
    """LightningDataModule orchestrating loading and batching of cervical JSON data."""

    def __init__(
        self,
        data_dir: str,
        json_file: str = "default.json",
        images_subdir: str = "images",
        batch_size: int = 60,
        num_workers: int = 4,
        img_size: Sequence[int] = (256, 128),
        heatmap_size: Sequence[int] = (64, 32),
        sigma: float = 1.5,
        train_val_test_split: Sequence[float] = (0.8, 0.1, 0.1),
        use_flip: bool = True,
        use_hist_eq: bool = True,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = False,
        seed: int = 42,
        num_landmarks: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[CervicalJSONDataset] = None
        self.data_val: Optional[CervicalJSONDataset] = None
        self.data_test: Optional[CervicalJSONDataset] = None

        self._num_landmarks: Optional[int] = num_landmarks
        self._img_size = (int(img_size[0]), int(img_size[1]))
        self._heatmap_size = (int(heatmap_size[0]), int(heatmap_size[1]))

    @property
    def num_landmarks(self) -> int:
        if self._num_landmarks is None:
            raise RuntimeError("DataModule has not been setup yet; num_landmarks is unavailable.")
        return self._num_landmarks

    def prepare_data(self) -> None:  # pragma: no cover - no external downloads required
        data_dir = Path(self.hparams.data_dir).expanduser().resolve()
        json_path = data_dir / self.hparams.json_file
        images_dir = data_dir / self.hparams.images_subdir
        if not json_path.is_file():
            raise FileNotFoundError(f"Annotation file '{json_path}' does not exist.")
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory '{images_dir}' does not exist.")

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None and self.data_val is not None and self.data_test is not None:
            return

        data_dir = Path(self.hparams.data_dir).expanduser().resolve()
        samples = load_cervical_json_annotations(data_dir, self.hparams.json_file, self.hparams.images_subdir)

        if not samples:
            raise RuntimeError(f"No samples discovered in '{data_dir}'.")

        discovered_landmarks = samples[0].landmarks.shape[0]
        if self._num_landmarks is None:
            self._num_landmarks = discovered_landmarks
        elif self._num_landmarks != discovered_landmarks:
            raise ValueError(
                f"Configured num_landmarks={self._num_landmarks} but dataset provides {discovered_landmarks}."
            )

        splits = self._compute_split_lengths(len(samples), list(self.hparams.train_val_test_split))
        generator = torch.Generator().manual_seed(self.hparams.seed)
        indices = torch.randperm(len(samples), generator=generator).tolist()

        subsets: List[List[SampleRecord]] = []
        cursor = 0
        for length in splits:
            subset_indices = indices[cursor : cursor + length]
            subsets.append([samples[i] for i in subset_indices])
            cursor += length

        sigma = float(self.hparams.sigma)
        use_hist_eq = bool(self.hparams.use_hist_eq)

        train_samples, val_samples, test_samples = subsets
        self.data_train = CervicalJSONDataset(
            train_samples,
            split="train",
            img_size=self._img_size,
            heatmap_size=self._heatmap_size,
            sigma=sigma,
            use_flip=bool(self.hparams.use_flip),
            use_hist_eq=use_hist_eq,
        )
        self.data_val = CervicalJSONDataset(
            val_samples,
            split="val",
            img_size=self._img_size,
            heatmap_size=self._heatmap_size,
            sigma=sigma,
            use_flip=False,
            use_hist_eq=use_hist_eq,
        )
        self.data_test = CervicalJSONDataset(
            test_samples,
            split="test",
            img_size=self._img_size,
            heatmap_size=self._heatmap_size,
            sigma=sigma,
            use_flip=False,
            use_hist_eq=use_hist_eq,
        )

    def train_dataloader(self) -> DataLoader:
        if self.data_train is None:
            raise RuntimeError("DataModule.setup() must be called before accessing the train dataloader.")
        return self._build_dataloader(self.data_train, shuffle=True, drop_last=self.hparams.drop_last)

    def val_dataloader(self) -> DataLoader:
        if self.data_val is None:
            raise RuntimeError("DataModule.setup() must be called before accessing the val dataloader.")
        return self._build_dataloader(self.data_val, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        if self.data_test is None:
            raise RuntimeError("DataModule.setup() must be called before accessing the test dataloader.")
        return self._build_dataloader(self.data_test, shuffle=False, drop_last=False)

    def _build_dataloader(
        self,
        dataset: CervicalJSONDataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: List[Dict[str, object]]):
        images = torch.stack([item["image"] for item in batch], dim=0)
        heatmaps = torch.stack([item["heatmaps"] for item in batch], dim=0)
        landmarks = torch.stack([item["landmarks"] for item in batch], dim=0)

        meta = {
            "image_id": [item["meta"]["image_id"] for item in batch],
            "orig_size": [item["meta"]["orig_size"] for item in batch],
            "crop_box": [item["meta"]["crop_box"] for item in batch],
            "variant": [item["meta"]["variant"] for item in batch],
            "img_size": torch.tensor(self._img_size, dtype=torch.float32),
            "heatmap_size": torch.tensor(self._heatmap_size, dtype=torch.float32),
            "landmarks": landmarks.clone(),
        }

        return images, heatmaps, meta

    @staticmethod
    def _compute_split_lengths(total: int, splits: Sequence[float]) -> Tuple[int, int, int]:
        if len(splits) != 3:
            raise ValueError("train_val_test_split must contain three values (train, val, test).")
        if sum(splits) <= 1.0 + 1e-6:
            raw_lengths = [split * total for split in splits]
            lengths = [int(length) for length in raw_lengths]
            remainder = total - sum(lengths)
            for idx in range(remainder):
                lengths[idx % len(lengths)] += 1
        else:
            lengths = [int(s) for s in splits]
            if sum(lengths) != total:
                raise ValueError("Provided split lengths must sum to the dataset size.")
        for idx, length in enumerate(lengths):
            if length <= 0:
                donor = max(range(len(lengths)), key=lambda j: lengths[j])
                if lengths[donor] <= 1:
                    raise ValueError("Unable to assign at least one sample per split.")
                lengths[donor] -= 1
                lengths[idx] += 1
        train, val, test = lengths
        return train, val, test


__all__ = ["CervicalJSONDataModule"]
