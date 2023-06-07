import torch
from model import AttentionUNet


input = torch.randn(2, 65, 64, 64)

attention_unet = AttentionUNet(in_channel=65, out_channel=1)

output = attention_unet(input)

print(output.size())    