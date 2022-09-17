import torch.nn as nn
from torch import Tensor

class Convolution(nn.Module):
    def __init__(self, d_model: int, in_channels:int, out_channels:int, kernel_size:int, kernels_per_layer: int, dropout: float = 0.1) -> None:
        super(Convolution, self).__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pointwise_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
        self.depthwise_conv = nn.Conv2d(self.in_channels, self.in_channels * kernels_per_layer, self.kernel_size, padding=1, groups=self.in_channels)
        self.dropout = dropout
    
    def forward(self, x: Tensor): # ?
        X = x
        x = nn.LayerNorm(self.d_model)
        x = self.pointwise_conv(x)
        x = nn.GLU()
        x = self.depthwise_conv(x)
        x = nn.BatchNorm1d()
        x = nn.SiLU()
        x = self.pointwise_conv(x)
        x = nn.Dropout(p=self.dropout)        
        x = X + x
        return x


