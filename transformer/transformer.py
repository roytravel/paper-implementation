"""
Reference
- https://github.com/hyunwoongko/transformer
- https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.py
- https://github.com/tunz/transformer-pytorch/blob/master/model/transformer.py
- https://github.com/nawnoes/pytorch-transformer/blob/main/model/transformer.py
- https://github.com/jayparks/transformer/blob/master/transformer/modules.py
"""
import argparse
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """ compute sinusoid encoding """
    def __init__(self, d_model=512, max_seq_len=128):
        super(PositionalEncoding, self).__init__()

        # same shape with input matrix
        self.pe = torch.zeros(max_seq_len, d_model, requires_grad=False) # 128 * 512

        pos = torch.arange(0, max_seq_len, dtype=torch.float)
        pos = pos.unsqueeze(dim=1).float() # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()

        self.pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.pe[:seq_len, :] # token_embedding : [batch_size, seq_len, d_model]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        """ self attention 4 steps
            1. x = matmul(query, key^T)
            2. attn_score = x / math.sqrt(d_k) // It is scaled dot product
            3. softmax_attn_score = softmax(attn_score)
            4. x = matmul(softmax_attn_score, value)
        """
        """ Shape of query, key, value
            query = [batch_size x num_heads x len_query x d_K]
            key = [batch_size x num_heads x len_key x d_k]
            value = [batch_size x num_heads x len_query x d_v] // len_key == len_value
        """
        #batch_size, num_heads, len_query, d_v = query.size()
        d_k = query.size()[-1]
        key_t = key.transpose(2, 3)
        scale_factor = math.sqrt(d_k)
        attn_score = torch.matmul(query, key_t) / scale_factor # Scaled Dot Product

        if mask is not None: # optional
            attn_score.masked_fill_(mask, -1e9)

        attn_score = self.softmax(attn_score)
        context = torch.matmul(attn_score, value)
        return context, attn_score


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, d_model=512):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.d_k = self.d_v = d_model // num_head
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)


    def forward(self, query, key, value, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(query), self.w_k(key), self.w_v(value)

        # 2. split tensor by number of heads
        q, k, v = self.split(query), self.split(key), self.split(value)

        # 3. do scale-dot product to compute similarity
        x, attn_score = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        x = self.concat(x)
        x = self.w_concat(x)
        return x

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        tensor = tensor.view(batch_size, length, self.num_head, self.d_k).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_k = tensor.size()
        d_model = head * d_k
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # -1 means last dimension.
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super(PositionwiseFeedForward, self).__init__()
        # self.layer1 = nn.Linear(d_model, d_ff)
        # self.layer2 = nn.Linear(d_ff, d_model)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout_rate)
        self.layer1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.layer2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer3 = LayerNorm(d_model=d_model, eps=1e-6)

    def forward(self, x):
        # x = self.layer1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.layer2(x)
        residual = x
        output = nn.ReLU()(self.layer1(x.transpose(1, 2)))
        output = self.layer2(output).transpose(1, 2)
        output = self.layer3(output + residual)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout_rate=0.1, num_heads=None):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_head=num_heads)
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
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


class Encoder(nn.Module):
    def __init__(self, d_model=512, d_ff=2048,
                 dropout_rate=0.1, num_layers=6,
                 num_heads=None, enc_voc_size=None,
                 max_seq_len=None):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(d_model=d_model,
                                              max_seq_len=max_seq_len,
                                              vocab_size=enc_voc_size,
                                              dropout_rate=dropout_rate)

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, dropout_rate, num_heads) for _ in range(num_layers)])

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

        self.ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm3 = LayerNorm(d_model=d_model, eps=1e-6)
        self.dropout3 = nn.Dropout(p=dropout_rate)

    def forward(self, dec, enc, src_mask, trg_mask):
        _x = dec
        x = self.self_attention(query=dec, key=dec, value=dec, mask=trg_mask)

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
                 max_seq_length=None):
        super(Decoder, self).__init__()

        self.emb = TransformerEmbedding(d_model=d_model,
                                        dropout_rate=dropout_rate,
                                        vocab_size=dec_vob_size,
                                        max_seq_len=max_seq_length)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, num_heads=num_heads) for _ in range(num_layers)])
        self.linear = LayerNorm(d_model, eps=1e-6)

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
    def __init__(self, vocab_size, d_model, max_seq_len, dropout_rate):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)


class Transformer(nn.Module):
    def __init__(self, # src_pad_idx, trg_pad_idx, trg_sos_idx,
                 enc_voc_size, dec_voc_size, d_model, d_ff,
                 max_seq_len, num_layers, num_heads, dropout_rate):
        super(Transformer, self).__init__()

        #self.src_pad_idx = src_pad_idx
        #self.trg_pad_idx = trg_pad_idx
        #self.trg_sos_idx = trg_sos_idx
        self.d_model = d_model
        self.emb_scale = pow(d_model, 0.5)

        self.encoder = Encoder(d_model=d_model, num_heads=num_heads,
                               max_seq_len=max_seq_len, d_ff=d_ff,
                               enc_voc_size=enc_voc_size, dropout_rate=dropout_rate,
                               num_layers=num_layers)

        self.decoder = Decoder(d_model=d_model, num_heads=num_heads,
                               max_seq_length=max_seq_len, d_ff=d_ff,
                               dec_vob_size=dec_voc_size, dropout_rate=dropout_rate,
                               num_layers=num_layers)

        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, src, trg):
        src_mask = self._make_pad_mask(src, src)
        src_trg_mask = self._make_pad_mask(trg, src)
        trg_mask = self._make_pad_mask(trg, trg) * self._make_no_peak_mask(trg, trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        dec_logits = self.projection(output)
        return dec_logits.view(-1, dec_logits.size(-1))

    def _make_pad_mask(self, query, key):
        batch_size, len_q = query.size()
        batch_size, len_k = key.size()
        # eq(zero) is PAD token
        pad_attn_mask = key.data.eq(0).unsqueeze(1) # batch_size x 1 len_q, one is masking
        return pad_attn_mask.expand(batch_size, len_q, len_k)

        # len_query, len_key = query.size(1), key.size(1)
        # query = query.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # query = query.repeat(1, 1, 1, len_key)
        # key = key.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # key = key.repeat(1, 1, len_query, 1)
        # mask = query & key
        # return mask

    def _make_no_peak_mask(self, query, key):
        len_query, len_key = query.size(1), key.size(1)

        mask = torch.tril(torch.ones(len_query, len_key)).type(torch.BoolTensor)
        return mask


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


if __name__ == '__main__':
    # sentence example of Machine Translation task
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4} # Padding Should be Zero
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6} # Padding Should be Zero
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    src_len, tgt_len = 5, 5

    # set hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=512, help='size of embedding')
    parser.add_argument('--d_ff', type=int, default=2048, help='feed-forward dimension')
    parser.add_argument('--d_k', type=int, default=64, help='dimension of K(=Q)')
    parser.add_argument('--d_v', type=int, default=64, help='dimension of V')
    parser.add_argument('--num_layers', type=int, default=6, help='number of Encoder/Decoder Layer')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads in Multi-Head Attention')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warmup steps')
    parser.add_argument('--beta1', type=float, default=0.9, help='first beta in Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.98, help='second beta in Adam optimizer')
    args = parser.parse_args()

    # create model
    model = Transformer(d_model=args.d_model, d_ff=args.d_ff, num_heads=args.num_heads, num_layers=args.num_layers,
                        dropout_rate=0.1, max_seq_len=128, enc_voc_size=src_vocab_size, dec_voc_size= tgt_vocab_size)

    # create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

    # create sample from example sentences
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    # Train
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('[*] Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    predict = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '→', *[number_dict[n.item()] for n in predict.squeeze()])