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
    ckpt_path = 'C:/Users/PC/OneDrive/Desktop/WORK/UET-VNU/TestPosenet/PoseNet/logs/train/runs/2025-03-10_22-38-16/checkpoints/epoch_026.ckpt'
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = PoseNetModule.load_from_checkpoint(ckpt_path, net=PoseNet())
    model.to(device)
    model.eval()
    
    # Load dataset
    datamodule: LightningDataModule = hydra.utils.instantiate(config=cfg)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    
    os.makedirs("inference", exist_ok=True)
    
    for idx, sample in enumerate(test_loader):
        img, gt = sample
        img = img.to(device)
        
        with torch.no_grad():
            pred = model(img).cpu()
        
        img, gt, pred = img[0].cpu(), gt[0].cpu(), pred[0].cpu()
        img = (img[0, :, :] * 255).numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i in range(gt.shape[0]):
            id = np.unravel_index(np.argmax(gt[i].numpy()), gt[i].shape)
            img = cv2.circle(img, (id[1] * 4, id[0] * 4), 2, (255, 0, 0), -1)

        for i in range(pred.shape[0]):
            id = np.unravel_index(np.argmax(pred[i].numpy()), pred[i].shape)
            img = cv2.circle(img, (id[1] * 4, id[0] * 4), 2, (0, 0, 255), -1)
        
        cv2.imwrite(f"inference/test{idx}.png", img)

if __name__ == "__main__":
    main()
