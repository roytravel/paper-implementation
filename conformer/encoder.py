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
from modules import ResidualConnection, Linear

from feedforward import FeedForward
from attention import MultiHeadSelfAttention
from convolution import Conv2dSubsampling, Convolution

class ConformerBlock(nn.Module):
    def __init__(self, 
        encoder_dim: int = 512, 
        num_attention_head: int = 8, 
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2, 
        dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True) -> None:
        super(ConformerBlock, self).__init__()

        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequencial = nn.Sequential(
            ResidualConnection(
                module=FeedForward(
                    encoder_dim = encoder_dim,
                    expansion_factor = feed_forward_expansion_factor,
                    dropout_p = dropout_p
                ),
                module_factor = self.feed_forward_residual_factor,
            ),
            ResidualConnection(
                module=MultiHeadSelfAttention(
                    d_model = encoder_dim,
                    num_head = num_attention_head,
                    dropout_p = dropout_p,
                ),
            ),
            ResidualConnection(
                module=Convolution(
                    in_channels = encoder_dim,
                    kernel_size = conv_kernel_size,
                    expansion_factor = conv_expansion_factor,
                    dropout_p = dropout_p
                ),
            ),
            ResidualConnection(
                module = FeedForward(
                    encoder_dim = encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p = dropout_p,
                ),
                module_factor = self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequencial(x)


class ConformerEncoder(nn.Module):
    def __init__(self, 
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layer: int = 17,
        num_attention_head: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        super(ConformerEncoder, self).__init__()
        self.conv_subsampling = Conv2dSubsampling(in_channels=1, out_channels=encoder_dim)
        self.input_proj = nn.Sequential(
            Linear(encoder_dim * (((input_dim-1) //2-1) // 2), encoder_dim),
            nn.Dropout(p=dropout_p),
        )

        self.layers = nn.ModuleList([ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_head = num_attention_head,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor = conv_expansion_factor,
                dropout_p = dropout_p,
                conv_kernel_size = conv_kernel_size,
                half_step_residual = half_step_residual,
            ) for _ in range(num_layer)])

    # def count_parameters(self) -> int:
    #     """ Count parameters of encoder """
    #     return sum([p.numel() for p in self.parameters()])

    # def update_dropout(self, dropout_p: float) -> None:
    #     """ Update dropout probability of encoder """
    #     for name, child in self.named_children():
    #         if isinstance(child, nn.Dropout):
    #             child.p = dropout_p

    
    def forward(self, x: Tensor, x_len: Tensor) -> Tuple[Tensor, Tensor]:
        x, x_len = self.conv_subsampling(x, x_len)
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)
        
        return x, x_len