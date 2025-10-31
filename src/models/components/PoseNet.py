from typing import Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Head(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(3):
            layers.append(ConvBlock(channels, out_channels, stride=2))
            channels = out_channels
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv1(x)
        features = self.conv2(features)
        pooled = self.pool(features)
        return features, pooled


class KeepBlock(nn.Module):
    def __init__(self, in_channels: int = 512, hidden_channels: int = 1024, out_channels: int = 512):
        super().__init__()
        self.conv = ConvBlock(in_channels, hidden_channels)
        self.deconv = DeconvBlock(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.deconv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, conv_channels: int, deconv_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels + skip_channels, conv_channels)
        self.conv2 = ConvBlock(conv_channels, conv_channels)
        self.deconv = DeconvBlock(conv_channels, deconv_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.deconv(x)
        return x


class PoseNet(nn.Module):
    def __init__(self, num_landmarks: int = 23):
        super().__init__()

        self.head = Head()
        self.down1 = DownBlock(64, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        self.keep = KeepBlock()

        self.up1 = UpBlock(in_channels=512, skip_channels=512, conv_channels=512, deconv_channels=256)
        self.up2 = UpBlock(in_channels=256, skip_channels=256, conv_channels=256, deconv_channels=128)
        self.up3 = UpBlock(in_channels=128, skip_channels=128, conv_channels=128, deconv_channels=64)
        self.up4 = UpBlock(in_channels=64, skip_channels=64, conv_channels=64, deconv_channels=64)

        self.output_layer = nn.Conv2d(64, num_landmarks, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)

        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.keep(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.output_layer(x)


__all__ = ["PoseNet"]
