import torch.nn as nn
from torch import Tensor

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int, dropout_rate: float) -> None:
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout_rate = dropout_rate

    def forward(self, x: Tensor) -> Tensor:
        x = nn.LayerNorm(self.d_model),
        x = nn.Linear(in_features = self.d_model, out_features= self.d_model * self.expansion_factor, bias = True)
        x = nn.SiLU()
        x = nn.Dropout(p = self.dropout_rate)
        x = nn.Linear(in_features= self.d_model * self.expansion_factor, out_features= self.d_model, bias = True)
        x = nn.Dropout(p = self.dropout_rate)
        return x