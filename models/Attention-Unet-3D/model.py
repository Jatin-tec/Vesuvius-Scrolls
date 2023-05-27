import torch
import torch.nn as nn
from torchsummary import summary
from buildingBlocks import ConvBlock, AttentionBlock, ResidualBlock
from torchviz import make_dot

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()

        self.down1 = ConvBlock(in_channels, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)
        
        self.bridge = ConvBlock(512, 1024)
        
        self.up1 = nn.ConvTranspose3d(1024, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.up2 = nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.up4 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.up5 = nn.ConvTranspose3d(64, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        
        self.att1 = AttentionBlock(512)
        self.att2 = AttentionBlock(256)
        self.att3 = AttentionBlock(128)
        self.att4 = AttentionBlock(64)
        
        self.residual1 = ResidualBlock(1024, 512)
        self.residual2 = ResidualBlock(512, 256)
        self.residual3 = ResidualBlock(256, 128)
        self.residual4 = ResidualBlock(128, 64)

        # Lazy linear layer
        self.lazy_linear = nn.Linear(65 * 512 * 128, 1)
        
    def forward(self, x):
        # Downsample
        down1 = self.down1(x)
        down2 = self.down2(nn.functional.max_pool3d(down1, kernel_size=(3, 3, 3), stride=(1, 1, 1)))
        down3 = self.down3(nn.functional.max_pool3d(down2, kernel_size=(3, 3, 3), stride=(1, 1, 1)))
        down4 = self.down4(nn.functional.max_pool3d(down3, kernel_size=(3, 3, 3), stride=(1, 1, 1)))
        
        # Bridge
        bridge = self.bridge(nn.functional.max_pool3d(down4, kernel_size=(3, 3, 3), stride=(1, 1, 1)))
        
        # Upsample with attention and residual connections
        up1 = self.up1(bridge)
        att1 = self.att1(down4)
        up1 = self.residual1(torch.cat([up1, att1], dim=1))
        
        up2 = self.up2(up1)
        att2 = self.att2(down3)
        up2 = self.residual2(torch.cat([up2, att2], dim=1))
        
        up3 = self.up3(up2)
        att3 = self.att3(down2)
        up3 = self.residual3(torch.cat([up3, att3], dim=1))
        
        up4 = self.up4(up3)
        att4 = self.att4(down1)
        up4 = self.residual4(torch.cat([up4, att4], dim=1))
        
        # Output
        out = self.up5(up4)

        # Reshape output tensor
        out = out.view(out.size(0), -1)
        
        # Apply lazy linear layer
        out = self.lazy_linear(out)
        
        return out


if __name__ == "__main__":
    # Creating an instance of the UNet3D model
    INPUT_SHAPE = (2, 1, 32, 32, 65)
    OUTPUT_SHAPE = (32, 1)

    model = UNet3D()

    # Checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Moving the model to GPU if available

    # Creating a random input tensor with batch size 2
    batch_size = 2
    input_tensor = torch.randn(INPUT_SHAPE).to(device)  # Moving input tensor to GPU if available

    print(input_tensor.shape)

    # Forward pass
    output = model(input_tensor)

    make_dot(output, params=dict(model.named_parameters())).render("mode_graph", format="png")

    # Print the shape of the output tensor
    print("Output shape:", output.shape)