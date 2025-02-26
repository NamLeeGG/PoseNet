import math
import torch
from torch import nn
from torchmetrics import Metric

# classification mask
def L(heatmap, theta_BG=0.4, theta_FG=0.9):
    return torch.where(
        (heatmap >= 0) & (heatmap < theta_BG), 0,
        torch.where((heatmap >= theta_BG) & (heatmap < theta_FG), 1, 0)
    )

# cross-entropy loss for classification
def CE(pred, target, theta_BG=0.4, theta_FG=0.9):
    L_target = L(target, theta_BG, theta_FG)
    L_pred = L(pred, theta_BG, theta_FG)
    return -(L_target * torch.log(target + 1e-6) + L_pred * torch.log(pred + 1e-6))

# weighting func
def omega(pred, target, theta_BG=0.4, theta_FG=0.9):
    cross_entropy = CE(pred, target, theta_BG, theta_FG)
    return torch.where(cross_entropy == 0, 0, 1)

# scaling factors for FG and CF regions
def calc_magnitude(target, theta_BG=0.4, theta_FG=0.9):
    area_BG = torch.sum((target >= 0) & (target <= theta_BG), dim=(2, 3), keepdim=True)
    area_FG = torch.sum((target > theta_BG) & (target <= theta_FG), dim=(2, 3), keepdim=True)
    area_CF = torch.sum(target > theta_FG, dim=(2, 3), keepdim=True)

    omega_FG = area_BG / (area_FG + 1e-6)
    omega_CF = area_BG / (area_CF + 1e-6)
    
    return omega_FG, omega_CF

# absolute difference 
def delta(pred, target):
    return torch.abs(target - pred)

# intensity map func
def intensity_map_function(target, l_b, h_b):
    return ((target >= l_b) & (target < h_b)).float()

### LOSS FUNCTIONS ###

# background loss
def LBG(pred, target, alpha=0.5, C1=-0.125, phi_BG=0.5):
    delta_ = delta(pred, target)
    return torch.where(delta_ > phi_BG, delta_ ** 2, alpha * (delta_ ** 2) + C1)

# foreground loss
def LFG(pred, target, C2=-0.5, phi_FG=0.5):
    delta_ = delta(pred, target)
    return torch.where(delta_ > phi_FG, delta_ ** 2, delta_ + C2)

# core-foreground loss
def LCF(pred, target, beta=4, phi_1_CF=0.5, phi_2_CF=0.05):
    delta_ = delta(pred, target)
    C3 = phi_2_CF - beta * math.log(1 + phi_2_CF)
    C4 = beta * math.log(1 + phi_1_CF) - beta * math.log(1 + phi_2_CF) - phi_1_CF**2 + phi_2_CF

    return torch.where(
        delta_ > phi_1_CF, delta_ ** 2,
        torch.where(delta_ > phi_2_CF, beta * torch.log(1 + delta_) + C3, delta_ + C4)
    )

# total loss

def Loss_BG(pred, target, alpha=0.5, C1=-0.125, phi_BG=0.5, theta_BG=0.4):
    intensity_map = intensity_map_function(target, 0, theta_BG)
    loss = LBG(pred, target, alpha, C1, phi_BG) * intensity_map
    return loss.mean()

def Loss_FG(pred, target, C2=-0.5, phi_FG=0.5, theta_BG=0.4, theta_FG=0.9):
    omega_FG, _ = calc_magnitude(target, theta_BG, theta_FG)
    intensity_map = intensity_map_function(target, theta_BG, theta_FG)
    loss = LFG(pred, target, C2, phi_FG) * omega_FG.unsqueeze(-1).unsqueeze(-1) * intensity_map
    return loss.mean()

def Loss_CF(pred, target, beta=4, phi_1_CF=0.5, phi_2_CF=0.05, theta_BG=0.4, theta_FG=0.9):
    _, omega_CF = calc_magnitude(target, theta_BG, theta_FG)
    intensity_map = intensity_map_function(target, theta_FG, 1)
    loss = LCF(pred, target, beta, phi_1_CF, phi_2_CF) * omega_CF.unsqueeze(-1).unsqueeze(-1) * intensity_map
    
    print(pred.shape, target.shape, omega_CF.shape, intensity_map.shape)

    return loss.mean()

### CATEGORICAL LOSS FUNC ###
def CLoss_BG(pred, target, alpha=0.5, C1=-0.125, phi_BG=0.5, theta_BG=0.4, theta_FG=0.9):
    intensity_map_target = intensity_map_function(target, 0, theta_BG)
    intensity_map_pred = intensity_map_function(pred, 0, theta_BG)
    valid_map = intensity_map_target * intensity_map_pred
    loss = LBG(pred, target, alpha, C1, phi_BG) * omega(pred, target, theta_BG, theta_FG) * valid_map
    return loss.mean()

def CLoss_FG(pred, target, C2=-0.5, phi_FG=0.5, theta_BG=0.4, theta_FG=0.9):
    intensity_map_target = intensity_map_function(target, theta_BG, theta_FG)
    intensity_map_pred = intensity_map_function(pred, theta_BG, theta_FG)
    valid_map = intensity_map_target * intensity_map_pred
    loss = LFG(pred, target, C2, phi_FG) * omega(pred, target, theta_BG, theta_FG) * valid_map
    return loss.mean()

### FINAL IC-LOSS FUNC ###
def IC_Loss(pred, target, alpha=0.5, beta=4, C1=-0.125, C2=-0.5, phi_BG=0.5, phi_FG=0.5, phi_1_CF=0.5, phi_2_CF=0.05, theta_BG=0.4, theta_FG=0.9):
    return (
        CLoss_BG(pred, target, alpha, C1, phi_BG, theta_BG, theta_FG)
        + CLoss_FG(pred, target, C2, phi_FG, theta_BG, theta_FG)
        + Loss_CF(pred, target, beta, phi_1_CF, phi_2_CF, theta_BG, theta_FG)
    )

### WRAPPING IC-LOSS ###
class ICLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ICLoss, self).__init__()
        self.params = kwargs

    def forward(self, pred, target):
        return IC_Loss(pred, target, **self.params)

### NME ###
class NME(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_nme", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        nme = torch.sum((preds - target) ** 2, dim=1).sqrt().sum() / (preds.shape[0] * preds.shape[1])
        self.sum_nme += nme
        self.total += preds.size(0)

    def compute(self):
        return self.sum_nme / self.total

    def reset(self):
        self.sum_nme = torch.tensor(0.0)
        self.total = torch.tensor(0)

if __name__ == "__main__":
    pass
