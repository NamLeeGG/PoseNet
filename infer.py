import torch
import hydra
import numpy as np
import cv2
import os
from omegaconf import DictConfig
from typing import Optional
from src.models.posenet_module import PoseNetModule
from src.models.components.PoseNet import PoseNet
from lightning import LightningDataModule


@hydra.main(version_base="1.3", config_path="./configs/data", config_name="data")
def main(cfg: DictConfig) -> Optional[float]:
    ckpt_path = './logs/train/runs/2025-03-07_16-58-13/checkpoints/epoch_116.ckpt'
    model = PoseNetModule.load_from_checkpoint(net = PoseNet(), checkpoint_path = ckpt_path)
    model.eval()
    
    datamodule: LightningDataModule = hydra.utils.instantiate(config=cfg)
    datamodule.setup()
    test_loader = datamodule.data_test
    os.makedirs("inference", exist_ok=True)
    for idx, sample in enumerate(test_loader):
        img, gt = sample
        with torch.no_grad():
            pred = model(img.cuda()).cpu()
        img, gt, pred = img[0], gt[0], pred[0]
        img = (img[0,:,:] * 255).numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i in range(gt.shape[0]):
            id = np.unravel_index(np.argmax(gt[i]), gt[i].shape)
            img = cv2.circle(img, (id[1]*4, id[0]*4), 1, (255, 0, 0), -1) # blue

        for i in range(pred.shape[0]):
            id = np.unravel_index(np.argmax(pred[i]), pred[i].shape)
            img = cv2.circle(img, (id[1]*4, id[0]*4), 1, (0, 0, 255), 1) # red
        
        cv2.imwrite(f"inference/test{idx}.png", img)

if __name__ == "__main__":
    main()