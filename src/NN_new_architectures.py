import torch
import torch.nn as nn
import torch.nn.functional as F


#OPTION 1: TCN Encoder + Temporal Interpolator + TCN Decoder
#1. Encoder = 1D dilated CNN (TCN-like)
#2. Temporal fusion = learned interpolation via MLP on encoded features
#3. Decoder = symmetric 1D CNN stack

########################################
# 1D Dilated CNN block (TCN-like)
########################################
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, padding=dilation, dilation=dilation
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

########################################
# Encoder: maps 2-channel profile → latent features
########################################
class Encoder(nn.Module):
    def __init__(self, in_channels=2, latent_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(in_channels, 32, dilation=1),
            TCNBlock(32, 64, dilation=2),
            TCNBlock(64, latent_channels, dilation=4)
        )

    def forward(self, x):
        return self.net(x)  # output shape: (B, latent_channels, 200)

########################################
# Temporal interpolator
# Takes z(t0) and z(t2), fuses them into z(t1)
########################################
class TemporalFusion(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()
        # Small MLP to compute mixing coefficient
        self.alpha_predictor = nn.Sequential(
            nn.Conv1d(latent_channels * 2, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),     # scalar at each depth location
            nn.Sigmoid()
        )

    def forward(self, z0, z2):
        # Concatenate features over channel dimension
        z_cat = torch.cat([z0, z2], dim=1)
        alpha = self.alpha_predictor(z_cat)  # shape (B,1,200), between 0–1

        # Interpolate latent space
        z1 = alpha * z0 + (1 - alpha) * z2
        return z1

########################################
# Decoder: latent features → 2 output channels
########################################
class Decoder(nn.Module):
    def __init__(self, latent_channels=64, out_channels=2):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(latent_channels, 64, dilation=4),
            TCNBlock(64, 32, dilation=2),
            nn.Conv1d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

########################################
# Full Model
########################################
class TCNInterpolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.fusion = TemporalFusion()
        self.decoder = Decoder()

    def forward(self, x0, x2):
        z0 = self.encoder(x0)   # (B,64,200)
        z2 = self.encoder(x2)   # (B,64,200)
        z1 = self.fusion(z0, z2)
        out = self.decoder(z1)  # (B,2,200)
        return out

########################################
# Example usage
########################################
if __name__ == "__main_prova__":   #aggiunto io prova, così non runna mai
    model = TCNInterpolationModel()
    x0 = torch.randn(8, 2, 200)  # batch of 8: 2 channels, length 200   --> Questa è la struttura che devono avere i dati di input
    x2 = torch.randn(8, 2, 200)
    y1 = model(x0, x2)
    print(y1.shape)  # (8,2,200)


#OPTION 2: Pure CNN (U-Net-style) With Time as Channels

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

########################################
# Example usage
########################################
if __name__ == "__main_prova__":
    model = UNet1D()
    x = torch.randn(8, 2, 200)  # (batch, channels, length)
    y = model(x)
    print(y.shape)  # (8,2,200)



#OPTION 3: Old 3D-UNet, but with time as channel, and only with 3 variables (lat, lon, variable)
#Quite similar to original architecture --> Maybe I only need to change the parameters and the Dataset 
#POTREBBE ESSERE LA PRIMA COSA DA FARE, RICHIEDE MENO MODIFICHE