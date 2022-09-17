import torch.nn as nn
from torch import Tensor

from feedforward import FeedForward
from attention import RelativeMultiHeadSelfAttention
from convolution import Convolution

class Conformer(nn.Module):
    def __init__(self, d_model, dropout_rate, expansion_rate, num_head, in_channels, out_channels, kernel_size, kernels_per_layer) -> None:
        super(Conformer, Conformer).__init__()

        # feed-forward
        self.d_model = d_model
        self.expansion_rate = expansion_rate
        self.dropout_rate = dropout_rate
        # attention
        self.num_head = num_head
        # convolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernels_per_layer = kernels_per_layer

    def forward(self, x: Tensor) -> Tensor:
        x = x + (0.5 * FeedForward(self.d_model, self.expansion_factor, self.dropout_rate))
        x = x + RelativeMultiHeadSelfAttention(self.d_model, self.num_head, self.dropout_rate)
        x = x + Convolution(self.d_model, self.in_channels, self.out_channels, self.kernel_size, self.kernels_per_layer, self.dropout_rate)
        x = nn.LayerNorm(x + (0.5 * FeedForward(self.d_model, self.expansion_rate, self.dropout_rate)))
        return x