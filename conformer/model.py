import torch.nn as nn
from torch import Tensor

class Conformer(nn.Module):
    def __init__(self) -> None:
        super(Conformer, Conformer).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x