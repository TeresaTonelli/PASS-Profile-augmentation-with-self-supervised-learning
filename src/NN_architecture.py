import torch
import torch.nn as nn
import torch.nn.functional as F


def crop_to_match(enc, dec):
    """Crop encoder feature map to match the decoder's size."""
    _, _, d_enc, h_enc, w_enc = enc.shape
    _, _, d_dec, h_dec, w_dec = dec.shape

    d_diff = d_enc - d_dec
    h_diff = h_enc - h_dec
    w_diff = w_enc - w_dec

    if d_diff > 0 or h_diff > 0 or w_diff > 0:
        enc = enc[:, :,
                  d_diff // 2 : d_diff // 2 + d_dec if d_diff > 0 else d_enc,
                  h_diff // 2 : h_diff // 2 + h_dec if h_diff > 0 else h_enc,
                  w_diff // 2 : w_diff // 2 + w_dec if w_diff > 0 else w_enc]

    if d_diff < 0 or h_diff < 0 or w_diff < 0:
        pad = [
            max(0, -w_diff // 2), max(0, -w_diff - (-w_diff // 2)),
            max(0, -h_diff // 2), max(0, -h_diff - (-h_diff // 2)),
            max(0, -d_diff // 2), max(0, -d_diff - (-d_diff // 2)),
        ]
        enc = F.pad(enc, pad)

    return enc




    #_, _, d_enc, h_enc, w_enc = enc.shape
    #_, _, d_dec, h_dec, w_dec = dec.shape

    #d_crop = (d_enc - d_dec) // 2
    #h_crop = (h_enc - h_dec) // 2
    #w_crop = (w_enc - w_dec) // 2

    #return enc[:, :, d_crop:d_crop + d_dec,
    #            h_crop:h_crop + h_dec,
    #            w_crop:w_crop + w_dec]


class DoubleConv(nn.Module):
    """(Conv3D → BN → ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        super().__init__()
        self.encoder1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))    #2
        
        self.encoder2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))   #2 

        self.bottleneck = DoubleConv(features[1], features[2])

        self.upconv2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.decoder2 = DoubleConv(features[2], features[1])

        self.upconv1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.decoder1 = DoubleConv(features[1], features[0])

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        #print("Input shape:", x.shape)
        enc1 = self.encoder1(x)
        #print("enc1:", enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
        #print("enc2:", enc2.shape)
        bottleneck = self.bottleneck(self.pool2(enc2))
        #print("bottleneck:", bottleneck.shape)

        dec2 = self.upconv2(bottleneck)
        #print("dec2 after upconv2:", dec2.shape)
        enc2_cropped = crop_to_match(enc2, dec2)
        dec2 = torch.cat((dec2, enc2_cropped), dim=1)
        dec2 = self.decoder2(dec2)
        #print("dec2 after decoder2:", dec2.shape)

        dec1 = self.upconv1(dec2)
        #print("dec1 after upconv1:", dec1.shape)
        enc1_cropped = crop_to_match(enc1, dec1)
        dec1 = torch.cat((dec1, enc1_cropped), dim=1)
        dec1 = self.decoder1(dec1)
        #print("dec1 after decoder1:", dec1.shape)

        #out = self.final_conv(dec1)
        #print("Output shape:", out.shape)

        return self.final_conv(dec1)