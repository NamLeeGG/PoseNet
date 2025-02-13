import math
import torch
from torch import nn
from torchmetrics import Metric

def L(heatmap, theta_BG=0.4, theta_FG=0.9):
    return torch.where((heatmap >= 0) & (heatmap < theta_BG), 0, torch.where((heatmap >= theta_BG) & (heatmap < theta_FG), 1, 0))

def CE(pred, target, theta_BG=0.4, theta_FG=0.9):
    L_target = L(target, theta_BG, theta_FG)
    L_pred = L(pred, theta_BG, theta_FG)
    return -(L_target * torch.log(target) + L_pred * torch.log(pred))

def omega(pred, target, theta_BG=0.4, theta_FG=0.9):
    cross_entropy = CE(pred, target, theta_BG, theta_FG)
    return torch.where(cross_entropy == 0, 0, 1)

def calc_magnitude(target, theta_BG=0.4, theta_FG=0.9):
    area_BG = torch.sum((target >= 0) & (target <= theta_BG), dim=(2, 3))
    area_FG = torch.sum((target > theta_BG) & (target <= theta_FG), dim=(2, 3))
    area_CF = torch.sum(target > theta_FG, dim=(2, 3))
    omega_FG = area_BG / area_FG
    omega_CF = area_BG / area_CF
    return omega_FG, omega_CF

def delta(pred, target):
    return torch.abs(target - pred)

def intensity_map_function(target, l_b, h_b):
    return torch.where((target >= l_b) & (target < h_b), 1, 0)

def LBG(pred, target, alpha=0.5, C1=-0.125, phi_BG=0.5):
    delta_ = delta(pred, target)
    return torch.where(delta_ > phi_BG, delta_ ** 2, alpha * (delta_ ** 2) + C1)

def LFG(pred, target, C2=-0.5, phi_FG=0.5):
    delta_ = delta(pred, target)
    return torch.where(delta_ > phi_FG, delta_ ** 2, delta_ + C2)

def LCF(pred, target, beta=4, phi_1_CF=0.5, phi_2_CF=0.05):
    delta_ = delta(pred, target)
    C3 = phi_2_CF - beta * math.log(1 + phi_2_CF)
    C4 = beta * math.log(1 + phi_1_CF) - beta * math.log(1 + phi_2_CF) - phi_1_CF ** 2 + phi_2_CF
    return torch.where(delta_ > phi_1_CF, delta_ ** 2, torch.where(delta_ > phi_2_CF, beta * torch.log(delta_) + C3, delta_ + C4))

def Loss_BG(pred, target, alpha=0.5, C1=-0.125, phi_BG=0.5, theta_BG=0.4):
    intensity_map = intensity_map_function(target, 0, theta_BG)
    loss = LBG(pred, target, alpha, C1, phi_BG) * intensity_map
    return torch.mean(loss)

def Loss_FG(pred, target, C2=-0.5, phi_FG=0.5, theta_BG=0.4, theta_FG=0.9):
    omega_FG, _ = calc_magnitude(target, theta_BG, theta_FG)
    intensity_map = intensity_map_function(target, theta_BG, theta_FG)
    loss = LFG(pred, target, C2, phi_FG) * omega_FG.unsqueeze(-1).unsqueeze(-1) * intensity_map
    return torch.mean(loss)

def Loss_CF(pred, target, beta=4, phi_1_CF=0.5, phi_2_CF=0.05, theta_BG=0.4, theta_FG=0.9):
    _, omega_CF = calc_magnitude(target, theta_BG, theta_FG)
    intensity_map = intensity_map_function(target, theta_FG, 1)
    loss = LCF(pred, target, beta, phi_1_CF, phi_2_CF) * omega_CF.unsqueeze(-1).unsqueeze(-1) * intensity_map
    return torch.mean(loss)

def CLoss_BG(pred, target, alpha=0.5, C1=-0.125, phi_BG=0.5, theta_BG=0.4, theta_FG=0.9):
    intensity_map_target = intensity_map_function(target, 0, theta_BG)
    intensity_map_pred = intensity_map_function(pred, 0, theta_BG)
    valid_map = intensity_map_target * intensity_map_pred
    loss = LBG(pred, target, alpha, C1, phi_BG) * omega(pred, target, theta_BG, theta_FG) * valid_map
    return torch.mean(loss)

def CLoss_FG(pred, target, C2=-0.5, phi_FG=0.5, theta_BG=0.4, theta_FG=0.9):
    intensity_map_target = intensity_map_function(target, theta_BG, theta_FG)
    intensity_map_pred = intensity_map_function(pred, theta_BG, theta_FG)
    valid_map = intensity_map_target * intensity_map_pred
    loss = LFG(pred, target, C2, phi_FG) * omega(pred, target, theta_BG, theta_FG) * valid_map
    return torch.mean(loss)

def IC_Loss(pred, target, alpha=0.5, beta=4, C1=-0.125, C2=-0.5, phi_BG=0.5, phi_FG=0.5, phi_1_CF=0.5, phi_2_CF=0.05, theta_BG=0.4, theta_FG=0.9):
    print(Loss_CF(pred, target, beta, phi_1_CF, phi_2_CF, theta_BG, theta_FG))
    return CLoss_BG(pred, target, alpha, C1, phi_BG, theta_BG, theta_FG) \
         + CLoss_FG(pred, target, C2, phi_FG, theta_BG, theta_FG) \
         + Loss_CF(pred, target, beta, phi_1_CF, phi_2_CF, theta_BG, theta_FG)


class ICLoss(nn.Module):
    def __init__(
            self,
            alpha: float = 0.5,
            beta: float = 4,
            C1: float = -0.125,
            C2: float = -0.5,
            phi_BG: float = 0.5,
            phi_FG: float = 0.5,
            phi_1_CF: float = 0.5,
            phi_2_CF: float = 0.05,
            theta_BG: float = 0.4,
            theta_FG: float = 0.9
        ):
        super(ICLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.C1 = C1
        self.C2 = C2
        self.phi_BG = phi_BG
        self.phi_FG = phi_FG
        self.phi_1_CF = phi_1_CF
        self.phi_2_CF = phi_2_CF
        self.theta_BG = theta_BG
        self.theta_FG = theta_FG
    
    def forward(self, pred, target):
        return IC_Loss(pred, target, self.alpha, self.beta, self.C1, self.C2, self.phi_BG, self.phi_FG, self.phi_1_CF, self.phi_2_CF, self.theta_BG, self.theta_FG)


class NME(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_nme", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate the normalized mean error (NME)
        nme = torch.sum((preds - target) ** 2, dim=1).sqrt().sum() / (preds.shape[0] * preds.shape[1])
        self.sum_nme += nme
        self.total += preds.size(0)

    def compute(self):
        # Compute the average NME
        return self.sum_nme / self.total

    def reset(self):
        self.sum_nme = torch.tensor(0.0)
        self.total = torch.tensor(0)    

if __name__ == "__main__":
    pass