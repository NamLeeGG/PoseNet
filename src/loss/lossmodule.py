import torch
import torch.nn as nn
from torchmetrics import Metric
import math

class ICLoss(nn.Module):
    def __init__(self, theta_bg=0.4, theta_fg=0.9, phi_bg=0.5, phi_fg=0.5, phi1_cf=0.5, phi2_cf=0.05, alpha=0.5, beta=4.0):
        super(ICLoss, self).__init__()
        self.theta_bg = theta_bg
        self.theta_fg = theta_fg
        self.phi_bg = phi_bg
        self.phi_fg = phi_fg
        self.phi1_cf = phi1_cf
        self.phi2_cf = phi2_cf
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        return ic_loss(pred, target, self.theta_bg, self.theta_fg, self.phi_bg,
                       self.phi_fg, self.phi1_cf, self.phi2_cf, self.alpha, self.beta)

# create region masks
def get_region_masks(target, theta_bg, theta_fg):
    mask_bg = (target <= theta_bg).float()
    mask_fg = ((target > theta_bg) & (target <= theta_fg)).float()
    mask_cf = (target > theta_fg).float()
    return mask_bg, mask_fg, mask_cf

# compute piecewise losses
def compute_piecewise_losses(diff, phi_bg, phi_fg, phi1_cf, phi2_cf, alpha, beta):
    C1 = 0.5 * (phi_bg ** 2) * (1 - alpha)
    L_BG = torch.where(diff > phi_bg, 0.5 * diff ** 2, alpha * 0.5 * diff ** 2 + C1)

    C2 = 0.5 * (phi_fg ** 2) - phi_fg
    L_FG = torch.where(diff > phi_fg, 0.5 * diff ** 2, diff + C2)

    C3 = phi2_cf - beta * math.log(1 + phi2_cf)
    C4 = beta * math.log(1 + phi1_cf) - beta * math.log(1 + phi2_cf) - 0.5 * phi1_cf ** 2 + phi2_cf
    L_CF = torch.where(
        diff > phi1_cf, 0.5 * diff ** 2,
        torch.where(diff > phi2_cf, beta * torch.log(1 + diff) + C3, diff + C4)
    )

    return L_BG, L_FG, L_CF

# compute misclassification mask
def compute_misclassification_mask(pred, target, theta_bg, mask_bg, mask_fg):
    gt_class = torch.zeros_like(target)
    gt_class[mask_fg.bool()] = 1
    pred_class = (pred > theta_bg).float()
    delta = (pred_class != gt_class).float()
    return delta * (mask_bg + mask_fg)

# compute region-wise weights
def compute_region_weights(mask_bg, mask_fg, mask_cf):
    num_bg = mask_bg.sum(dim=(1, 2, 3), keepdim=True)
    num_fg = mask_fg.sum(dim=(1, 2, 3), keepdim=True)
    num_cf = mask_cf.sum(dim=(1, 2, 3), keepdim=True)
    omega_fg = num_bg / (num_fg + 1e-8)
    omega_cf = num_bg / (num_cf + 1e-8)
    return omega_fg, omega_cf

# main loss computation function
def ic_loss(pred, target, theta_bg, theta_fg, phi_bg, phi_fg, phi1_cf, phi2_cf, alpha, beta):
    diff = (pred - target).abs()

    mask_bg, mask_fg, mask_cf = get_region_masks(target, theta_bg, theta_fg)
    L_BG, L_FG, L_CF = compute_piecewise_losses(diff, phi_bg, phi_fg, phi1_cf, phi2_cf, alpha, beta)
    omega_fg, omega_cf = compute_region_weights(mask_bg, mask_fg, mask_cf)
    delta = compute_misclassification_mask(pred, target, theta_bg, mask_bg, mask_fg)

    CLossBG = (mask_bg * delta * L_BG).sum(dim=(1, 2, 3))
    CLossFG = (mask_fg * delta * L_FG).sum(dim=(1, 2, 3)) * omega_fg
    LossCF = (mask_cf * L_CF).sum(dim=(1, 2, 3)) * omega_cf

    total_pixels = pred.shape[1] * pred.shape[2] * pred.shape[3]

    IC_loss = (CLossBG + CLossFG + LossCF) / total_pixels

    return IC_loss.mean()

if __name__ == "__main__":
    pass