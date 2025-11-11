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


class DoubleConv(nn.Module):
    """(Conv3D → BN → ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(5, 3, 3), padding=(2, 1, 1)),   #Ho cambiato qua e nel conv3d linea 41 sia kernel size che padding --> prima erano 3 e 1
            #nn.BatchNorm3d(out_channels),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True), 
            nn.Conv3d(out_channels, out_channels, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            #nn.BatchNorm3d(out_channels),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),    
            #addition of 3 Conv block
            #nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            #nn.InstanceNorm3d(out_channels, affine=True),
            #nn.ReLU(inplace=True)
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

        #self.encoder3 = DoubleConv(features[1], features[2])
        #self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        self.bottleneck = DoubleConv(features[1], features[2])    #Anche qua ho dovuto modificare
        self.dropout = nn.Dropout3d(p=0.1)

        #self.upconv3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=(2, 2, 1), stride=(2, 2, 1))
        #self.decoder3 = DoubleConv(features[3], features[2])

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

        #enc3 = self.encoder3(self.pool2(enc2))    #questo pezzo e` nuovo
        #print("enc3:", enc3.shape)

        bottleneck = self.bottleneck(self.pool2(enc2))    #Il bottleneck e` vecchio ma e` modificato --> ora c'e` 3, prima 2
        bottleneck = self.dropout(bottleneck)
        #print("bottleneck:", bottleneck.shape)
        #bottleneck = self.bottleneck(self.pool3(enc3))  
        #print("bottleneck:", bottleneck.shape)

        #nuovo blocco 
        #dec3 = self.upconv3(bottleneck)
        #print("dec3 after upconv3:", dec3.shape)
        #enc3_cropped = crop_to_match(enc3, dec3)
        #print("enc3 cropped shape", enc3_cropped.shape)
        #dec3 = torch.cat((dec3, enc3_cropped), dim=1)
        #print("dec3 cropping shape", dec3.shape)
        #dec3 = self.decoder3(dec3)
        #print("dec3 after decoder3:", dec3.shape)

        #dec2 = self.upconv2(dec3)    #linea per encoder-decoder 3
        dec2 = self.upconv2(bottleneck)
        #print("dec2 after upconv2:", dec2.shape)
        enc2_cropped = crop_to_match(enc2, dec2)
        #print("enc2 cropped shape", enc2_cropped.shape)
        dec2 = torch.cat((dec2, enc2_cropped), dim=1)
        #print("dec 2 cropping shape", dec2.shape)
        dec2 = self.decoder2(dec2)
        #print("dec2 after decoder2:", dec2.shape)

        dec1 = self.upconv1(dec2)
        #print("dec1 after upconv1:", dec1.shape)
        enc1_cropped = crop_to_match(enc1, dec1)
        dec1 = torch.cat((dec1, enc1_cropped), dim=1)
        dec1 = self.decoder1(dec1)
        #print("dec1 after decoder1:", dec1.shape)

        out = self.final_conv(dec1)
        #print("Output shape:", out.shape)

        return self.final_conv(dec1)