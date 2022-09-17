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

import torch.nn as nn
from torch import Tensor
from typing import Tuple
from modules import Transpose
from activation import GLU, Swish


class DepthWiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False) -> None:
        super(DepthWiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=in_channels, stride=stride, padding=padding, bias=bias)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x

class PointWiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super(PointWiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x

class Conv2dSubsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubsampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
    def forward(self, x: Tensor, x_len: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.sequential(x.unsqueeze(1))
        batch_size, channels, subsampled_len, subsampled_dim = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, subsampled_len, channels*subsampled_dim)

        x_len = x_len >> 2
        x_len -= 1

        return x, x_len

class Convolution(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 31, expansion_factor: int = 2, dropout_p: float = 0.1) -> None:
        super(Convolution, self).__init__()

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointWiseConv1d(in_channels, in_channels*expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthWiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointWiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x).transpose(1, 2)