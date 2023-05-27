import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), activation=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = activation

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()

        self.theta = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(3, 3, 3))
        self.phi = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(3, 3, 3))
        self.g = nn.Conv3d(in_channels, in_channels // 2, kernel_size=(3, 3, 3))
        self.o = nn.Conv3d(in_channels // 2, in_channels, kernel_size=(3, 3, 3))

        self.norm = nn.BatchNorm3d(in_channels)
        self.activation = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        theta = theta.view(theta.size(0), theta.size(1), -1)
        phi = phi.view(phi.size(0), phi.size(1), -1)
        g = g.view(g.size(0), g.size(1), -1)

        theta_phi = torch.matmul(theta.transpose(1, 2), phi)
        attention = self.softmax(theta_phi)
        attention_g = torch.matmul(attention, g.transpose(1, 2))
        attention_g = attention_g.transpose(1, 2).contiguous().view(x.size())
        attention_g = self.o(attention_g)
        attention_g = self.norm(attention_g)
        out = self.activation(x + attention_g)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out