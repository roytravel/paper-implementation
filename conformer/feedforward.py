import torch.nn as nn
from torch import Tensor
from .activation import Swish

class FeedForward(nn.Module):
    """
    Args:
        d_model (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward. you can multiply it by d_model.
        dropout (float): Ratio of dropout
    """

    def __init__(self, d_model: int = 512, expansion_factor: int = 4, dropout: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.expand_dim = d_model * expansion_factor

    def forward(self, x: Tensor) -> Tensor:
        x = nn.LayerNorm(self.d_model),
        x = nn.Linear(self.d_model, self.expand_dim, bias=True)
        x = Swish(x)
        x = nn.Dropout(p=self.dropout)
        x = nn.Linear(self.expand_dim, self.d_model, bias=True)
        x = nn.Dropout(p=self.dropout)
        return x