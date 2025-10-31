from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_deconv: bool = True) -> None:
        super().__init__()
        if use_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        self.conv1 = ConvBlock(out_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PoseNet(nn.Module):
    """PoseNet architecture tailored for cervical spine landmark detection."""

    def __init__(
        self,
        in_channels: int = 3,
        num_landmarks: int = 24,
        use_deconv: bool = True,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            ConvBlock(in_channels, 64, stride=2),
            ConvBlock(64, 64, stride=2),
            ConvBlock(64, 64, stride=2),
        )

        self.down1 = DownBlock(64, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)
        self.up1 = UpBlock(1024, 512, 512, use_deconv=use_deconv)
        self.up2 = UpBlock(512, 256, 256, use_deconv=use_deconv)
        self.up3 = UpBlock(256, 128, 128, use_deconv=use_deconv)
        self.up4 = UpBlock(128, 64, 64, use_deconv=use_deconv)

        if use_deconv:
            self.output_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        else:
            self.output_upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(64, 64, kernel_size=1),
            )

        self.output_layer = nn.Conv2d(64, num_landmarks, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)

        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        x = self.output_upsample(x)
        x = self.output_layer(x)
        return x


__all__ = ["PoseNet"]
