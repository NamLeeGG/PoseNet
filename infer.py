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
    ckpt_path = './logs/train/runs/2025-03-14_08-59-52/checkpoints/epoch_110.ckpt'
    model = PoseNetModule.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()
    
    datamodule: LightningDataModule = hydra.utils.instantiate(config=cfg)
    datamodule.setup(stage="test")  # explicitly setup for 'test'
    test_loader = datamodule.test_dataloader()
    os.makedirs("inference", exist_ok=True)
    os.makedirs("inference/heatmaps", exist_ok=True)

    # Configuration
    ORIGINAL_IMAGE_SIZE = (256, 128)  # (height, width)
    HEATMAP_SIZE_PRED = (64, 32)      # (height, width)
    HEATMAP_SIZE_GT = (64, 32)        # (height, width)

    with open("inference/coordinates_log.txt", "w") as log_file:
        for idx, sample in enumerate(test_loader):
            sample = next(iter(test_loader))
            img, gt = sample
            print("Check Input Tensor Shape and Stats:")
            print("Image tensor shape:", img.shape)
            print("Image tensor min/max:", img.min().item(), img.max().item())
            print("Image tensor mean/std:", img.mean().item(), img.std().item())

            if img.dim() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():
                pred = model(img.cuda()).cpu()

            # Verify model output dimensions
            print(f"\n--- Prediction {idx} ---")
            print(f"Model output shape: {pred.shape}")  # Should be [batch, 24, 64, 32]

            img = img[0]
            gt = gt[0]
            pred = pred[0]

            # Convert image for visualization
            img_display = (img[0].numpy() * 255).astype(np.uint8)
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

            # Process Ground Truth (gt)
            for i in range(gt.shape[0]):
                # Reshape to 2D heatmap (8x4)
                heatmap_gt = gt[i].numpy().reshape(HEATMAP_SIZE_GT)
                row, col = np.unravel_index(np.argmax(heatmap_gt), heatmap_gt.shape)

                # Scale coordinates to original image size
                x = int(col * (ORIGINAL_IMAGE_SIZE[1] / HEATMAP_SIZE_GT[1]))
                y = int(row * (ORIGINAL_IMAGE_SIZE[0] / HEATMAP_SIZE_GT[0]))

                # Draw ground truth point (green)
                img_display = cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)

                # Log coordinates
                log_file.write(f"GT {idx}-{i}: ({x}, {y})\n")

            # Process Predictions (pred)
            for i in range(pred.shape[0]):
                heatmap_pred = pred[i].numpy().reshape(HEATMAP_SIZE_PRED)

                # Save heatmap visualization (scaled to 255 for visualization)
                hm_img = (heatmap_pred / heatmap_pred.max() * 255).astype(np.uint8) if heatmap_pred.max() > 0 else np.zeros_like(heatmap_pred)
                cv2.imwrite(f"inference/heatmaps/pred_{idx}_kp{i}.png", hm_img)

                # Find peak coordinates in prediction heatmap
                row, col = np.unravel_index(np.argmax(heatmap_pred), heatmap_pred.shape)

                # Scale coordinates to original image size
                x = int(col * (ORIGINAL_IMAGE_SIZE[1] / HEATMAP_SIZE_PRED[1]))
                y = int(row * (ORIGINAL_IMAGE_SIZE[0] / HEATMAP_SIZE_PRED[0]))

                # Clamp to image dimensions
                x = max(0, min(x, ORIGINAL_IMAGE_SIZE[1] - 1))
                y = max(0, min(y, ORIGINAL_IMAGE_SIZE[0] - 1))

                # Draw prediction point (red)
                overlay = img_display.copy()
                cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
                alpha = 0.5
                img_display = cv2.addWeighted(overlay, 0.5, img_display, 0.5, 0)

                # Log coordinates
                log_file.write(f"Pred {idx}-{i}: ({x}, {y})\n")

            # Save visualization
            cv2.imwrite(f"inference/test_{idx}.png", img_display)

            # Print coordinate summary
            print(f"Processed image {idx}")
            print(f"GT shape: {gt.shape}, Pred shape: {pred.shape}")

    return None

if __name__ == "__main__":
    main()