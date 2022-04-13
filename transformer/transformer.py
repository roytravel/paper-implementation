"""
Reference
- https://github.com/hyunwoongko/transformer
- https://github.com/tunz/transformer-pytorch/blob/master/model/transformer.py
- https://github.com/nawnoes/pytorch-transformer/blob/main/model/transformer.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

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

        # x = x.transpose(1, 2).contiguous()
        # x = x.view(batch_size, -1, self.num_head * self.d_v)
        # x = self.output_layer(x)
        # attention_result, attention_score = self.self_attention(query, key, value, mask)
        # attention_result = attention_result.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k)
        # return self.w_o(attention_result)


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
        self.self_attention = MultiHeadAttention(num_head=num_heads)
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
                                              max_len=max_seq_len,
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
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, dropout_rate)
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.enc_dec_attention = MultiHeadAttention(d_model, dropout_rate)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.norm3 = LayerNorm(d_model, eps=1e-6)
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
    def __init__(self, d_model=512, d_ff=2048, dropout_rate=0.1, num_layers=6):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(d_model, d_ff, dropout_rate) for _ in range(num_layers)]
        self.layers = nn.ModuleList(num_layers)
        self.last_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, targets, enc_output, i_mask, t_self_mask, cache):
        decoder_output = targets
        for i, dec_layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                layer_cache = cache[i]
            decoder_output = dec_layer(decoder_output, enc_output, t_self_mask, i_mask, layer_cache)
        return self.last_norm(decoder_output)


class TokenEmbedding(nn.Embdding):
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
    def __init__(self,i_vocab_size, t_vocab_size,
                 num_layers=6, d_model=512, d_ff=2048,
                 dropout_rate=0.1, share_target_embedding=True,
                 has_inputs=True, src_pad_idx=None, trg_pad_idx=None):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.emb_scale = pow(d_model, 0.5)
        self.has_inputs = has_inputs
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.t_vocab_embedding = nn.Embedding(t_vocab_size, d_model)
        nn.init.normal_(self.t_vocab_embedding.weeight, mean=0, std=pow(d_model, -0.5))
        self.t_emb_dropout = nn.Dropout(p=dropout_rate)
        self.decoder = Decoder(d_model, d_ff, dropout_rate, num_layers)

        if has_inputs:
            if not share_target_embedding:
                self.i_vocab_embedding = nn.Embedding(i_vocab_size, d_model)
                nn.init.normal_(self.i_vocab_embedding.weight, mean=0, std=pow(d_model, -0.5))
            else:
                self.i_vocab_embedding = self.t_vocab_embedding

            self.i_emb_dropout = nn.Dropout(p=dropout_rate)
            self.encoder = Encoder(d_model, d_ff, dropout_rate, num_layers)


        # For Positional Encoding
        num_timescales = self.d_model // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales -1, 1))
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float32))
        self.register_buffer('inv_timescales', inv_timescales)

    def forward(self, inputs, targets):
        enc_output, i_mask = None, None
        if self.has_inputs:
            i_mask = utils.create_pad_mask(inputs, self.src_pad_idx)
            enc_output = self.encode(inputs, i_mask)

        t_mask = utils.create_pad_mask(targets, self.trg_pad_idx)
        target_size = targets.size()[1]
        t_self_mask = utils.create_trg_self_mask(target_size, device=targets.device)

        return self.decode(targets, enc_output, i_mask, t_self_mask, t_mask)


    def encode(self, inputs, i_mask):
        # Input embedding
        input_embedded = self.i_vocab_embedding(inputs)
        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)
        input_embedded *= self.emb_scale
        input_embedded += self.get_position_encoding(inputs)
        input_embedded = self.i_emb_dropout(input_embedded)

        return self.encoder(input_embedded, i_mask)

    def decode(self, targets, enc_output, i_mask, t_self_mask, t_mask, cache=None):
        # target embedding
        target_embedded = self.t_vocab_embedding(targets)
        target_embedded.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)

        # Shifting
        target_embedded = target_embedded[:, :-1]
        target_embedded = F.pad(target_embedded, (0, 0, 1, 0))

        target_embedded *= self.emb_scale
        target_embedded += self.get_position_encoding(targets)
        target_embedded = self.t_emb_dropout(target_embedded)

        # decoder
        decoder_output = self.decoder(target_embedded, enc_output, i_mask, t_self_mask, cache)
        # linear
        output = torch.matmul(decoder_output, self.t_vocab_embedding.weight.transpose(0, 1))

        return output

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal