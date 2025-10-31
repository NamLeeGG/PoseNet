from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.dataset import CervicalSpineDataset, ImageSample, load_image_samples


class CervicalSpineDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        annotation_file: str = "default.json",
        img_ext: str = ".png",
        train_test_split: Sequence[float] = (0.9, 0.1),
        train_batch_size: int = 60,
        test_batch_size: int = 60,
        num_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        heatmap_downsample: int = 4,
        image_height: int = 256,
        image_width: int = 128,
        sigma: float = 1.5,
        theta_bg: float = 0.4,
        theta_fg: float = 0.9,
        use_flipped: bool = True,
        random_eval_variant: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.annotation_file = annotation_file
        self.img_ext = img_ext
        self.train_split = float(train_test_split[0]) if len(train_test_split) > 0 else 0.9
        self.train_batch_size_per_device = train_batch_size
        self.test_batch_size_per_device = test_batch_size
        self.seed = seed

        self.samples: Optional[List[ImageSample]] = None
        self.train_indices: Optional[List[int]] = None
        self.val_indices: Optional[List[int]] = None

        self.data_train: Optional[CervicalSpineDataset] = None
        self.data_val: Optional[CervicalSpineDataset] = None
        self.data_test: Optional[CervicalSpineDataset] = None

        self.num_landmarks: Optional[int] = None

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.hparams.image_height, self.hparams.image_width)

    @property
    def num_classes(self) -> int:
        return int(self.num_landmarks) if self.num_landmarks is not None else 0

    def prepare_data(self) -> None:
        load_image_samples(self.data_dir, self.annotation_file, self.img_ext)

    def _split_indices(self) -> None:
        if self.samples is None:
            self.samples = load_image_samples(self.data_dir, self.annotation_file, self.img_ext)
        num_samples = len(self.samples)
        train_count = int(round(num_samples * self.train_split))
        train_count = max(1, min(train_count, num_samples - 1)) if num_samples > 1 else num_samples
        val_count = num_samples - train_count

        generator = torch.Generator().manual_seed(self.seed)
        permutation = torch.randperm(num_samples, generator=generator).tolist()
        self.train_indices = permutation[:train_count]
        self.val_indices = permutation[train_count:] if val_count > 0 else permutation[:]

    def setup(self, stage: Optional[str] = None) -> None:
        if self.samples is None or self.train_indices is None or self.val_indices is None:
            self._split_indices()

        if stage in (None, "fit"):
            self.data_train = CervicalSpineDataset(
                samples=self.samples,
                indices=self.train_indices,
                split="train",
                image_size=self.image_size,
                heatmap_downsample=self.hparams.heatmap_downsample,
                sigma=self.hparams.sigma,
                theta_bg=self.hparams.theta_bg,
                theta_fg=self.hparams.theta_fg,
                use_flipped=self.hparams.use_flipped,
                random_eval_variant=False,
                seed=self.seed,
            )
            self.data_val = CervicalSpineDataset(
                samples=self.samples,
                indices=self.val_indices,
                split="val",
                image_size=self.image_size,
                heatmap_downsample=self.hparams.heatmap_downsample,
                sigma=self.hparams.sigma,
                theta_bg=self.hparams.theta_bg,
                theta_fg=self.hparams.theta_fg,
                use_flipped=self.hparams.use_flipped,
                random_eval_variant=self.hparams.random_eval_variant,
                seed=self.seed,
            )

        if stage in (None, "test"):
            if self.data_val is None:
                self.data_val = CervicalSpineDataset(
                    samples=self.samples,
                    indices=self.val_indices,
                    split="val",
                    image_size=self.image_size,
                    heatmap_downsample=self.hparams.heatmap_downsample,
                    sigma=self.hparams.sigma,
                    theta_bg=self.hparams.theta_bg,
                    theta_fg=self.hparams.theta_fg,
                    use_flipped=self.hparams.use_flipped,
                    random_eval_variant=self.hparams.random_eval_variant,
                    seed=self.seed,
                )
            self.data_test = CervicalSpineDataset(
                samples=self.samples,
                indices=self.val_indices,
                split="test",
                image_size=self.image_size,
                heatmap_downsample=self.hparams.heatmap_downsample,
                sigma=self.hparams.sigma,
                theta_bg=self.hparams.theta_bg,
                theta_fg=self.hparams.theta_fg,
                use_flipped=self.hparams.use_flipped,
                random_eval_variant=self.hparams.random_eval_variant,
                seed=self.seed,
            )

        if self.num_landmarks is None and self.data_train is not None:
            self.num_landmarks = self.data_train.num_landmarks

        if self.trainer is not None:
            if self.hparams.train_batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.train_batch_size}) must be divisible by the number of devices ({self.trainer.world_size})."
                )
            self.train_batch_size_per_device = self.hparams.train_batch_size // self.trainer.world_size
            self.test_batch_size_per_device = self.hparams.test_batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.test_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


__all__ = ["CervicalSpineDataModule"]
