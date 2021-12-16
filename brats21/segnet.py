import torch
from torch import nn
from torch.nn.modules.conv import Conv3d


class DoubleConv(nn.Module):
    """(Conv3D -> InstanceNorm -> Dropout -> LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout3d(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout3d(inplace=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels * 2
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2, 2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        x, idx = self.pool(x)
        return x, idx


class Up(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        out_channels = in_channels // 3

        self.unpool = nn.MaxUnpool3d(2, 2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.nonlin = nn.LeakyReLU()

    def forward(self, x, idx, skip):
        x = self.unpool(x, idx)
        x = torch.cat((x, skip), dim=1)
        return self.nonlin(self.conv(x))


class SegNet(nn.Module):
    def __init__(self, channels=4, classes=4):
        super().__init__()

        self.nonlin = nn.LeakyReLU()

        self.down1 = Down(channels)
        self.down2 = Down(channels * 2)
        self.down3 = Down(channels * 4)
        self.down4 = Down(channels * 8)
        self.up1 = Up((channels * 16) + (channels * 8))
        self.up2 = Up((channels * 8) + (channels * 4))
        self.up3 = Up((channels * 4) + (channels * 2))
        self.up4 = Up((channels * 2) + (channels))

        self.classify = Conv3d(channels, classes, kernel_size=1, stride=1)

    def forward(self, x):

        # Encoder
        x1, idx1 = self.down1(x)
        x2, idx2 = self.down2(x1)
        x3, idx3 = self.down3(x2)
        x4, idx4 = self.down4(x3)

        # Decoder
        x5 = self.up1(x4, idx4, x3)
        x6 = self.up2(x5, idx3, x2)
        x7 = self.up3(x6, idx2, x1)
        x8 = self.up4(x7, idx1, x)

        return self.classify(x8)


if __name__ == "__main__":

    segnet = SegNet()
    inp = torch.rand(1, 4, 256, 256, 256)
    y = segnet(inp)
    print(y.shape)
