import math
import torch.nn as nn
from torch import Tensor


class Sigmoid(nn.Module):
    def __init__(self) -> None:
        super(Sigmoid).__init__()
    
    def forward(self, x):
        return 1 / (1 + math.exp(-x))


class Swish(nn.Module):
    """
    Swish is smooth and a non-monotonic function.
    using Swish is better than ReLU.
    """
    def __init__(self) -> None:
        super(Swish, self).__init__()
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.sigmoid(x)


class GLU(nn.Module):
    """
    GLU: Gated Linear Units
    It helps gradient vanishing problem to relieve.
    """
    def __init__(self, dim) -> None:
        super(GLU, self).__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(chunks=2, dim=self.dim)
        return x * Sigmoid(gate)