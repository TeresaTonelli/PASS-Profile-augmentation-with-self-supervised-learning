import torch
import torch.nn as nn
import torch.nn.functional as F



########################################
# Simple 1D Conv Block
########################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

########################################
# 1D U-Net-like architecture
########################################
class UNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(64, 128)

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)

        # Final convolution
        self.final = nn.Conv1d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # (B,32,200)
        p1 = self.pool(e1)          # (B,32,100)

        e2 = self.enc2(p1)          # (B,64,100)
        p2 = self.pool(e2)          # (B,64,50)

        # Bottleneck
        b = self.bottleneck(p2)     # (B,128,50)

        # Decoder
        u2 = self.up2(b)            # (B,128,100)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up2(d2)           # (B,64,200)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # Output
        return self.final(d1)