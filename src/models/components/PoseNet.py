import torch
from torch import nn


class conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class deconv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.deconv2d = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv2d(x)


class Head(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 2,
        num_channels=64,
        in_channels=3,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            conv2d(in_channels, num_channels, kernel_size, stride, padding),
            conv2d(num_channels, num_channels, kernel_size, stride, padding),
            conv2d(num_channels, num_channels, kernel_size, stride, padding),
        )
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DownSampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.downsampling = nn.Sequential(
            conv2d(in_channels, out_channels, kernel_size, stride, padding),
            conv2d(out_channels, out_channels, kernel_size, stride, padding),
        )
        self.conv_maxpool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor, max_pool: bool) -> torch.Tensor:
        x = self.downsampling(x)
        if max_pool:
            x = self.conv_maxpool(x)
        return x


class Keep(nn.Module):
    def __init__(
        self,
        in_channels: int=512,
        out_channels: int=1024,
        deconv_channels: int=512,
        deconv_stride: int=2,
        deconv_kernel_size: int=2,
        kernel_size: int=3,
        stride: int=1,
        padding: int=1,
    ) -> None:
        super().__init__()
        self.keep = nn.Sequential(
            conv2d(in_channels, out_channels, kernel_size, stride, padding),
            deconv2d(out_channels, deconv_channels, deconv_kernel_size, deconv_stride),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.keep(x)


class UpSampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        deconv_kernel_size: int = 2,
        deconv_stride: int = 2,
    ) -> None:
        super().__init__()
        self.upsampling = nn.Sequential(
            conv2d(in_channels, out_channels, kernel_size, stride, padding),
            conv2d(out_channels, out_channels, kernel_size, stride, padding),
            deconv2d(out_channels, out_channels, deconv_kernel_size, deconv_stride),
        )
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampling(x)
    

class PoseNet(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.head = Head()
        self.down1 = DownSampling(64, 64)
        self.down2 = DownSampling(64, 128)
        self.down3 = DownSampling(128, 256)
        self.down4 = DownSampling(256, 512)
        self.keep = Keep()
        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(768, 256)
        self.up3 = UpSampling(384, 128)
        self.up4 = UpSampling(192, 64)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=23, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x) # [64,32,16]
        x1 = self.down1(x, max_pool=True)   # [64,16,8]
        x2 = self.down2(x1, max_pool=True)  # [128,8,4]
        x3 = self.down3(x2, max_pool=True)  # [256,4,2]
        x4 = self.down4(x3, max_pool=True)  # [512,2,1]

        x5 = self.keep(x4) # [512,4,2]

        x5 = self.up1(torch.cat([x5, self.down4(x3, max_pool=False)], dim=1))   # [512,8,4]
        x5 = self.up2(torch.cat([x5, self.down3(x2, max_pool=False)], dim=1))   # [256,16,8]
        x5 = self.up3(torch.cat([x5, self.down2(x1, max_pool=False)], dim=1))   # [128,32,16]
        x5 = self.up4(torch.cat([x5, self.down1(x, max_pool=False)], dim=1))    # [64,64,32]
        x5 = self.output_layer(x5) # [23,64,32]
        
        return x5


if __name__ == "__main__":
    pass
