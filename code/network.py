
import torch.nn as nn
import torch.nn.functional as F
import torch



class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, pad='same', pad_mode='reflect', bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=pad, padding_mode=pad_mode, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=pad, padding_mode=pad_mode, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, bilinear=False, bias=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size, bias=bias)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PMRes(nn.Module):
    def __init__(self, device='cpu'):
        super(PMRes, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=60,
                               out_channels=32,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')
        self.conv4 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')
        self.conv5 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')
        self.conv6 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')
        self.conv7 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')
        self.conv8 = nn.Conv2d(in_channels=32,
                               out_channels=1,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')

    def forward(self, inp):
        x1 = F.leaky_relu(self.conv1(inp))
        x2 = F.leaky_relu(self.conv2(x1))
        x3 = F.leaky_relu(self.conv3(x2))
        x4in = x3 + x1
        x4 = F.leaky_relu(self.conv4(x4in))
        x5 = F.leaky_relu(self.conv5(x4))
        x6in = x5 + x4in
        x6 = F.leaky_relu(self.conv6(x6in))
        x7 = F.leaky_relu(self.conv7(x6))
        x8in = x7 + x6in
        x8 = F.relu(self.conv8(x8in))
        return x8



class PMSlim(nn.Module):
    def __init__(self, device='cpu'):
        super(PMSlim, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=60,
                               out_channels=32,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=1,
                               kernel_size=3,
                               padding='same',
                               padding_mode='reflect')

    def forward(self, inp):
        x = F.leaky_relu(self.conv1(inp))
        x = F.leaky_relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class PMLoss(nn.Module):

    def __init__(self, device):
        super(PMLoss, self).__init__()
        self.device = device
        self.loss1 = nn.L1Loss(reduction='mean')

    def forward(self, outputs, labels):
        L = self.loss1(outputs, labels)
        return L


class PMUnet(nn.Module):
    def __init__(self, device='cpu'):
        super(PMUnet, self).__init__()
        self.inc = (DoubleConv(60, 64, kernel_size=3, bias=False))
        factor = 2
        self.down1 = (Down(64, 128, bias=False))
        self.down2 = (Down(128, 256 , bias=False))
        self.down3 = (Down(256, 512 // factor, bias=False))
        self.up1 = (Up(512, 256 // factor, bilinear=True, bias=False))
        self.up2 = (Up(256, 128 // factor, bilinear=True, bias=False))
        self.up3 = (Up(128, 64, bilinear=True, bias=False))
        self.outc = (OutConv(64, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = F.leaky_relu(self.outc(x))
        return logits

