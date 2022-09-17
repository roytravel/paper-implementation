
# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import torch.nn as nn
from torch import Tensor


class ReLU(nn.Module):
    def __init__(self) -> None:
        super(ReLU).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return np.maximum(0, x)


class Swish(nn.Module):
    """
    Swish is smooth and a non-monotonic function.
    using Swish is better than ReLU.
    """
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class GLU(nn.Module):
    """
    GLU: Gated Linear Units
    It helps gradient vanishing problem to relieve.
    """
    def __init__(self, dim) -> None:
        super(GLU, self).__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=self.dim)
        return x * gate.sigmoid()