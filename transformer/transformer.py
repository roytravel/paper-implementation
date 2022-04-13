"""
Reference
- https://github.com/hyunwoongko/transformer
- https://github.com/tunz/transformer-pytorch/blob/master/model/transformer.py
- https://github.com/nawnoes/pytorch-transformer/blob/main/model/transformer.py
"""
import math
import torch
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_weight(x):
    nn.init.xavier_uniform(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class PositionalEncoding(nn.Module): # OK
    """ compute sinusoid encoding """
    def __init__(self, max_seq_len=128, d_model=512, device=DEVICE):
        super(PositionalEncoding, self).__init__()

        # same shape with input matrix
        self.pe = torch.zeros(max_seq_len, d_model, requires_grad=False) # 128 * 512

        pos = torch.arange(0, max_seq_len, dtype=torch.float, device=DEVICE)
        pos = pos.unsqueeze(dim=1).float() # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=DEVICE).float()

        self.pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.pe[:seq_len, :] # token_embedding : [batch_size, seq_len, d_model]


class ScaledDotProductAttention(nn.Module): # OK
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        """ self attention 4 steps
            1. x = matmul(query, key^T)
            2. attn_score = x / math.sqrt(d_k) // It is scaled dot product
            3. softmax_attn_score = softmax(attn_score)
            4. x = matmul(softmax_attn_score, value)
        """
        #batch_size, head, length, d_tensor = query.size()
        d_k = query.size()[-1]
        key_t = key.transpose(2, 3)
        attn_score = torch.matmul(query, key_t) / math.sqrt(d_k) # scaled dot product (=math.sqrt(d_k))

        if mask is not None: # mask is optional but, I wonder why I need to consider it.
            attn_score = attn_score.masked_fill(mask == 0, -1e-20)

        attn_score = nn.Softmax(attn_score, dim=-1)
        x = torch.matmul(attn_score, value)
        return x, attn_score


class MultiHeadAttention(nn.Module): # OK
    def __init__(self, num_head=8, d_model=512):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.attention = ScaledDotProductAttention()
        self.w_query = nn.Linear(d_model, d_model) # W_Q = model dimension
        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)
        initialize_weight(self.w_query)
        initialize_weight(self.w_key)
        initialize_weight(self.w_value)
        initialize_weight(self.w_concat)
        self.d_k = self.d_v = d_model // num_head

    def forward(self, query, key, value, mask=None, cache=None):
        # 1. dot product with weight matrices
        query, key, value = self.w_query(query), self.w_key(key), self.w_value(value)

        # 2. split tensor by number of heads
        query, key, value = self.split(query), self.split(key), self.split(value)

        x, attn_score = self.attention(query, key, value, mask=mask)
        x = self.concat(x)
        x = self.w_concat(x)
        return x

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_k = d_model // self.num_head
        tensor = tensor.view(batch_size, length, self.num_head, d_k).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_k = tensor.size()
        d_model = head * d_k
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class LayerNorm(nn.Module): # OK
    def __init__(self, d_model, eps=1e-6): # why eps=1e-6
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) # What is nn.Parameter?
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # -1 means last dimension.
        std = x.std(-1, keepdim=True) # 표준편차

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module): # OK
    def __init__(self, d_model=512, d_ff=2048, dropout_rate=0.1): # Why use the term "filter size" instead of "output size"?
        super(PositionwiseFeedForward, self).__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module): # OK
    def __init__(self, d_model=512, d_ff=2048, dropout_rate=0.1, num_heads=None):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_head=num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.self_attention(query=x, key=x, value=x, mask=src_mask)

        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # 3. position-wise feed forward network
        _x = x
        x = self.ff(x)

        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x


class Encoder(nn.Module): # OK
    def __init__(self, d_model=512, d_ff=2048,
                 dropout_rate=0.1, num_layers=6,
                 num_heads=None, enc_voc_size=None,
                 max_seq_len=None, device=None):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(d_model=d_model,
                                              max_seq_len=max_seq_len,
                                              vocab_size=enc_voc_size,
                                              dropout_rate=dropout_rate,
                                              device=DEVICE)

        encoders = [EncoderLayer(d_model, d_ff, dropout_rate, num_heads) for _ in range(num_layers)]
        self.layers = nn.ModuleList(encoders) # What ModuleList is?

    def forward(self, x, src_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class DecoderLayer(nn.Module): # OK
    def __init__(self, d_model, d_ff, num_heads, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_head=num_heads)
        self.norm1 = LayerNorm(d_model=d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, num_head=num_heads)
        self.norm2 = LayerNorm(d_model=d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)
        self.norm3 = LayerNorm(d_model=d_model, eps=1e-6)
        self.dropout3 = nn.Dropout(p=dropout_rate)

    def forward(self, dec, enc, src_mask, trg_mask):
        _x = dec
        x = self.self_attention(query=dec, key=dec, mask=trg_mask)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attention(query=x, key=enc, value=enc, mask=src_mask)

            x = self.norm2(x + _x)
            x = self.dropout2(x)

        _x = x
        x = self.ff(x)

        x = self.norm3(x + _x)
        x = self.dropout3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model=512, d_ff=2048,
                 dropout_rate=0.1, num_layers=6,
                 dec_vob_size=None, num_heads=None,
                 max_seq_length=None, device=DEVICE):
        super(Decoder, self).__init__()

        self.emb = TransformerEmbedding(d_model=d_model,
                                        dropout_rate=dropout_rate,
                                        vocab_size=dec_vob_size,
                                        max_seq_len=max_seq_length,
                                        device=DEVICE)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, num_heads=num_heads) for _ in range(num_layers)])
        self.linear = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        out = self.linear(trg)
        return out


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout_rate, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_seq_len, device)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx,
                 enc_voc_size, dec_voc_size, d_model, d_ff,
                 max_seq_len, num_layers, num_heads, dropout_rate, device):
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.d_model = d_model
        self.emb_scale = pow(d_model, 0.5)


        self.encoder = Encoder(d_model=d_model, num_heads=num_heads,
                               max_seq_len=max_seq_len, d_ff=d_ff,
                               enc_voc_size=enc_voc_size, dropout_rate=dropout_rate,
                               num_layers=num_layers, device=DEVICE)

        self.decoder = Decoder(d_model=d_model, num_heads=num_heads,
                               max_seq_length=max_seq_len, d_ff=d_ff,
                               dec_vob_size=dec_voc_size, dropout_rate=dropout_rate,
                               num_layers=num_layers, device=DEVICE)

    def forward(self, src, trg):
        src_mask = self._make_pad_mask(src, src)
        src_trg_mask = self._make_pad_mask(trg, src)
        trg_mask = self._make_pad_mask(trg, trg) * self._make_no_peak_mask(trg, trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        return output

    def _make_pad_mask(self, query, key):
        len_query, len_key = query.size(1), key.size(1)


        key = key.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        key = key.repeat(1, 1, len_query, 1)

        query = query.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        queryr = query.repeat(1, 1, 1, len_key)

        mask = query & key
        return mask

    def _make_no_peak_mask(self, query, key):
        len_query, len_key = query.size(1), key.size(1)

        mask = torch.tril(torch.ones(len_query, len_key)).type(torch.BoolTensor).to(DEVICE)
        return mask