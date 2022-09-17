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
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from modules import Linear


class RelativeMultiHeadSelfAttention(nn.Module):
    """
        Multi head attention with relative positional encoding is used in Transformer-XL
    """
    def __init__(self, d_model: int = 512, num_head: int = 16, dropout_p: float = 0.1) -> None:
        super(RelativeMultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_model // num_head
        self.num_head = num_head

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_head, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_head, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))
    
    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_head, seq_len1, seq_len2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_head, seq_len1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_head, seq_len2+1, seq_len1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, pos_embedding: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = value.size(0)
        
        query = self.query_proj(query).view(batch_size, -1, self.num_head, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_head, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_head, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_head, self.d_head)

        content_score = torch.matmul((query+self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query+self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)
        
        score = (content_score + pos_score) / math.sqrt(self.d_model)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)
        
        attn = self.softmax(score)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int, dropout_p: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.pos_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadSelfAttention(d_model, num_head, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)


    def forward(self, x:Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.size()
        pos_embedding = self.pos_encoding(seq_len)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        
        x = self.layer_norm(x)
        x = self.attention(x, x, x, pos_embedding=pos_embedding, mask=mask)
        x = self.dropout(x)
        return x