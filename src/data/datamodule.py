from typing import Any, Dict, Optional, Tuple

import torch
import hydra
import rootutils
import albumentations as A
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from src.data.components.dataset import BaseDataset, CervicalDataset
import cv2
import numpy as np

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def train_collate_fn(batch):
    images = torch.empty([0, 3, 256, 128])
    labels = torch.empty([0, 24, 64, 32])
    for (x1, y1), (x2, y2) in batch:
        images = torch.cat((images, x1[None, :], x2[None, :]), dim=0)
        labels = torch.cat((labels, y1[None, :], y2[None, :]), dim=0)
    return images, labels

class DataModule(LightningDataModule):
    def __init__(
        self,
        train_test_split: Tuple[float, float] = (0.9, 0.1),
        train_batch_size: int = 32,
        test_batch_size: int = 64,
        num_workers: int = 4,
        train_transforms: Optional[A.Compose] = None,
        test_transforms: Optional[A.Compose] = None,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_batch_size_per_device = train_batch_size
        self.test_batch_size_per_device = test_batch_size

    @property
    def num_classes(self) -> int:
        return 23

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            world_size = self.trainer.world_size
            if self.hparams.train_batch_size % world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.train_batch_size}) not divisible by devices ({world_size})."
                )
            self.train_batch_size_per_device = self.hparams.train_batch_size // world_size
            self.test_batch_size_per_device = self.hparams.test_batch_size // world_size

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = BaseDataset()
            trainset, testset = random_split(
                dataset,
                self.hparams.train_test_split,
                generator=torch.Generator().manual_seed(42)
            )
            self.data_train = CervicalDataset(dataset=trainset, mode='train', transform=self.train_transforms)
            self.data_val = CervicalDataset(dataset=testset, mode='val', transform=self.test_transforms)
            self.data_test = CervicalDataset(dataset=testset, mode='test', transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=train_collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.test_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="data")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule: LightningDataModule = hydra.utils.instantiate(config=cfg)
    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    for sample in val_loader:
        img, heatmap = sample
        img, heatmap = img[0], heatmap[0]
        img = (img[0, :, :] * 255).numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i in range(heatmap.shape[0]):
            idx = np.unravel_index(np.argmax(heatmap[i]), heatmap[i].shape)
            img = cv2.circle(img, (idx[1]*4, idx[0]*4), 2, (255, 255, 0), -1)

        cv2.imwrite("test.png", img)
        break

if __name__ == "__main__":
    main()
