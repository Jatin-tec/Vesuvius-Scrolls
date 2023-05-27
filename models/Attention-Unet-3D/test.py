import torch
import torch.nn as nn

m = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

input = torch.randn(2, 1, 512, 128, 65)
output = m(input)

print(output.size())    