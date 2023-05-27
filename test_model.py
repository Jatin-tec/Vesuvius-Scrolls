import torch
import torch.nn as nn

class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class CNNModel(nn.Module):
    def __init__(self, z_dim):
        super(CNNModel, self).__init__()
        # Encoder
        self.encoder_conv1 = SeparableConv2D(z_dim, 64, 3, padding=1)
        self.encoder_conv2 = SeparableConv2D(64, 128, 3, padding=1)
        self.encoder_pool = nn.MaxPool2d(2, 2)

        # Bridge
        self.bridge_conv1 = SeparableConv2D(128, 256, 3, padding=1)
        self.bridge_conv2 = SeparableConv2D(256, 256, 3, padding=1)

        # Decoder
        self.decoder_upconv = nn.ConvTranspose2d(256, 128, 2, stride=2)
        
        self.decoder_conv1 = SeparableConv2D(256, 128, 3, padding=1)
        self.decoder_conv2 = SeparableConv2D(128, 64, 3, padding=1)
        self.decoder_conv3 = SeparableConv2D(64, 1, 1)
        
        # Sigmoid layer for binary output
        self.sigmoid = nn.Sigmoid()  
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder_conv1(x)
        enc2 = self.encoder_conv2(enc1)
        enc_pool = self.encoder_pool(enc2)

        # Bridge
        bridge = self.bridge_conv1(enc_pool)
        bridge = self.bridge_conv2(bridge)

        # Decoder
        dec_upconv = self.decoder_upconv(bridge)
        dec_concat = torch.cat([dec_upconv, enc2], dim=1)
        
        dec_conv1 = self.decoder_conv1(dec_concat)
        dec_conv2 = self.decoder_conv2(dec_conv1)
        dec_conv3 = self.decoder_conv3(dec_conv2)
        
        # Apply sigmoid activation
        output = self.sigmoid(dec_conv3)  

        return dec_conv3