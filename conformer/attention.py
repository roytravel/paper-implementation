import math
import torch
from torch import Tensor
from typing import Optional
import torch.nn as nn


class RelativeMultiHeadSelfAttention(nn.Moudle):
    """
    Multi head attention with relative positinal encoding is used in Transformer-XL
    """
    def __init__(self, d_model: int = 512, num_heads: int = 16, dropout_rate: float = 0.1) -> None:
        super(RelativeMultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
    
    def _query_key_matmul(self, Q: Tensor, K: Tensor) -> Tensor:
        return torch.matmul(Q, K.T)
    
    def _scaled_dot_attention(self, Q: Tensor, K: Tensor) -> Tensor:
        x = self._query_key_matmul(Q, K)
        x = x / math.sqrt(self.dk)
        x = nn.Softmax(dim=-1)
        return x

    def self_attention(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        (b, m, *_) = Q.shape
        attention = self._query_key_matmul(Q, K)
        x = torch.matmul(attention, V)
        return x.view(b, m, -1)