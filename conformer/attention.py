import math
import torch
from torch import Tensor
from typing import Optional
import torch.nn as nn


class RelativeMultiHeadSelfAttention(nn.Moudle):
    """
    Multi head attention with relative positional encoding is used in Transformer-XL
    """
    def __init__(self, d_model: int = 512, num_head: int = 8, dropout: float = 0.1) -> None:
        super(RelativeMultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.d_head = d_model // num_head
    
    # def _query_key_matmul(self, Q: Tensor, K: Tensor) -> Tensor:
    #     return torch.matmul(Q, K.T)
    
    # def _scaled_dot_attention(self, Q: Tensor, K: Tensor) -> Tensor:
    #     x = self._query_key_matmul(Q, K)
    #     x = x / math.sqrt(self.dk)
    #     x = nn.Softmax(dim=-1)
    #     return x

    # def self_attention(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    #     (b, m, *_) = Q.shape
    #     attention = self._query_key_matmul(Q, K)
    #     x = torch.matmul(attention, V)
    #     return x.view(b, m, -1)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_head, seq_len1, seq_len2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_head, seq_len1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_head, seq_len2+1, seq_len1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, pos_embedding: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = value.size(0)
        
        query = nn.Linear(self.d_model, self.d_model)(query).view(batch_size, -1, self.num_head, self.d_head)
        key = nn.Linear(self.d_model, self.d_model)(key).view(batch_size, -1, self.num_head, self.d_head).permute(0, 2, 1, 3)
        value = nn.Linear(self.d_model, self.d_model)(value).view(batch_size, -1, self.num_head, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = nn.Linear(self.d_model, self.d_model, bias=False).view(batch_size, -1, self.num_head, self.d_head)

        u_bias = nn.Parameter(torch.Tensor(self.num_head, self.d_head))
        v_bias = nn.Parameter(torch.Tensor(self.num_head, self.d_head))
        torch.nn.init.xavier_uniform_(u_bias)
        torch.nn.init.xavier_uniform_(v_bias)

        content_score = torch.matmul((query+u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query+v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self.relative_shift(pos_score)
        
        score = (content_score + pos_score) / math.sqrt(self.d_model)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)
        
        attn = nn.Softmax(dim=-1)(score)
        attn = nn.Dropout(self.dropout)(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        out = nn.Linear(self.d_model, self.d_model)(context)
        return out


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
    def __init__(self, d_model: int, num_head: int, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.pos_encoding = PositionalEncoding(d_model)
        self.attention = RelativeMultiHeadSelfAttention(d_model, num_head, dropout)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x:Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.size()
        pos_embedding = self.pos_encoding(seq_len)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        x = nn.LayerNorm(self.d_model)(x)
        x = self.attention(x, x, x, pos_embedding=pos_embedding, mask=mask)
        x = self.dropout(x)
        return x