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
    



class UNet_1D_mode(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
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
    


class MLP_1D_mode(nn.Module):
    def __init__(self, in_channels=7, out_channels=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 32),  
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, out_channels)  
        )
        
    def forward(self, x):
        return self.net(x).unsqueeze(1)

    


class UNet_2D_mode(nn.Module):
    def __init__(self, depth_dim=200):
        super().__init__()

        self.depth_dim = depth_dim

        # =====================================================
        # Spatial encoder
        # Input: [B, 4, 7, 7]
        # =====================================================

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.spatial_fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )

        # =====================================================
        # Profile encoder
        # Input: [B, 2, 200]
        # =====================================================

        self.profile_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1)
        )

        self.profile_fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )

        # =====================================================
        # Fusion
        # =====================================================

        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        # =====================================================
        # Decoder
        # =====================================================

        self.decoder_fc = nn.Linear(
            256,
            depth_dim * 64
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 2, kernel_size=1)
        )

    def build_interpolation_baseline(self, profiles):
        """
        profiles: [B, 2, D]

        profiles[:,0] = point 1
        profiles[:,1] = point 4

        returns:
        baseline: [B, 2, D]
            baseline[:,0] = point 2 estimate
            baseline[:,1] = point 3 estimate
        """

        p1 = profiles[:, 0]
        p4 = profiles[:, 1]

        p2_base = (2.0 / 3.0) * p1 + (1.0 / 3.0) * p4
        p3_base = (1.0 / 3.0) * p1 + (2.0 / 3.0) * p4

        baseline = torch.stack(
            [p2_base, p3_base],
            dim=1
        )

        return baseline

    def forward(self, surface, known_profiles):
        """
        surface:
            [B, 4, 7, 7]

        known_profiles:
            [B, 2, 200]

        returns:
            [B, 2, 200]
        """

        B = surface.shape[0]

        # ==========================================
        # Spatial branch
        # ==========================================

        spatial_feat = self.spatial_encoder(surface)
        spatial_feat = spatial_feat.view(B, -1)
        spatial_feat = self.spatial_fc(spatial_feat)

        # ==========================================
        # Profile branch
        # ==========================================

        profile_feat = self.profile_encoder(known_profiles)
        profile_feat = profile_feat.squeeze(-1)
        profile_feat = self.profile_fc(profile_feat)

        # ==========================================
        # Fusion
        # ==========================================

        latent = torch.cat(
            [spatial_feat, profile_feat],
            dim=1
        )

        latent = self.fusion(latent)

        # ==========================================
        # Decoder
        # ==========================================

        x = self.decoder_fc(latent)

        x = x.view(
            B,
            64,
            self.depth_dim
        )

        residual = self.decoder_conv(x)

        # ==========================================
        # Baseline interpolation
        # ==========================================

        baseline = self.build_interpolation_baseline(
            known_profiles
        )

        prediction = baseline + residual

        return prediction