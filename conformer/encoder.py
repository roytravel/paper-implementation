""" 
Reference
- https://github.com/sooftware/conformer/
- https://github.com/msalhab96/Conformer/
- https://github.com/lucidrains/conformer/
- https://github.com/jaketae/conformer/
"""

import torch.nn as nn
from torch import Tensor

from feedforward import FeedForward
from attention import RelativeMultiHeadSelfAttention
from convolution import Convolution

class ConformerBlock(nn.Module):
    def __init__(self, 
        d_model: int = 512,
        num_heads: int = 8,
        dim_heads: int = 64,
        expansion_factor: int = 4,
        dropout_rate: float = 0.1,
        ) -> None:
        
        super(ConformerBlock, self).__init__()
        self.ff1 = FeedForward(d_model = d_model, expansion_factor = 4, dropout_rate = dropout_rate)
        self.attn = RelativeMultiHeadSelfAttention()
        self.conv = Convolution()
        self.ff2 = FeedForward()
        self.lnorm = nn.LayerNorm()
    

    def forward(self, x: Tensor):
        x = x + (0.5 * self.ff1(x))
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + (0.5 * self.ff2(x))
        x = self.lnorm(x)
        return x


class ConformerEncoder(nn.module):
    def __init__(self) -> None:
        super(ConformerEncoder, self).__init__()
        self.spec_aug = False
        self.conv_subsampling = False
        self.conformer_block = ConformerBlock()

    
    def forward(self, x: Tensor):
        x = self.spec_aug(x)
        x = self.conv_subsampling(x)
        x = nn.Linear()
        x = nn.Dropout(p=0.1)
        x = self.conformer_block()
        return x