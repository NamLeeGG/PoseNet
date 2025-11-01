import torch
import hydra
import numpy as np
import cv2
import os
import sys
from pathlib import Path
from omegaconf import DictConfig
from typing import Optional

import rootutils
from hydra.utils import get_original_cwd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.posenet_module import PoseNetModule
from src.data.datamodule import CervicalSpineDataModule


def _resolve_checkpoint(cfg: DictConfig, project_root: Path) -> Path:
    ckpt_cfg = cfg.get("ckpt_path")

    if ckpt_cfg:
        ckpt_path = Path(ckpt_cfg)
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Configured checkpoint path does not exist: {ckpt_path}")
        return ckpt_path

    runs_dir = project_root / "logs" / "train" / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found at {runs_dir}. Provide ckpt_path explicitly.")

    checkpoints = sorted(runs_dir.rglob("epoch_*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoint files found under logs/train/runs. Provide ckpt_path explicitly or run training first."
        )
    return checkpoints[0]


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    project_root = Path(get_original_cwd())
    os.environ.setdefault("PROJECT_ROOT", str(project_root))

    ckpt_path = _resolve_checkpoint(cfg, project_root)
    print(f"Loading checkpoint from: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PoseNetModule.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model.to(device)

    datamodule: CervicalSpineDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    output_dir = project_root / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(test_loader):
        images = batch["image"].to(device)
        heatmaps_gt = batch["heatmap"].to(device)

        with torch.no_grad():
            pred_heatmaps = model(images)

        img = images[0].detach().cpu().numpy()
        gt = heatmaps_gt[0].detach().cpu().numpy()
        pred = pred_heatmaps[0].detach().cpu().numpy()

        img_vis = np.mean(img, axis=0)
        img_vis = np.clip(img_vis * 255.0, 0, 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

        for i in range(gt.shape[0]):
            y_idx, x_idx = np.unravel_index(np.argmax(gt[i]), gt[i].shape)
            img_vis = cv2.circle(img_vis, (x_idx * 4, y_idx * 4), 2, (255, 0, 0), -1)

        for i in range(pred.shape[0]):
            y_idx, x_idx = np.unravel_index(np.argmax(pred[i]), pred[i].shape)
            img_vis = cv2.circle(img_vis, (x_idx * 4, y_idx * 4), 2, (0, 255, 0), -1)

        output_path = output_dir / f"test_{batch_idx:04d}.png"
        cv2.imwrite(str(output_path), img_vis)
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()