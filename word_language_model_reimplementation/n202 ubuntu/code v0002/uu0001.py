import torch
import torch.nn as nn
import copy
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
def ss(s):
    print(s)
    import sys
    sys.exit(1)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base model for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):

        "Take in and process masked src and target sequences."
        # print(src_mask.shape)
        # print(self.src_embed(src).shape)
        memory = self.encoder(self.src_embed(src), src_mask)
        # print(memory.shape)
        output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        # print(output.shape)
        # print(tgt.shape)
        # ss('---in encoderdecoder forward')
        return output


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # print(mask.shape)
        # print(x.shape)

        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            # print(x)
            x = layer(x, mask)
            # ss('in encoder forward')
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer function that maintains the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of two sublayers, self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # "Follow Figure 1 (left) for connections."
        # print(x)
        # print(x)
        # print(x.shape)
        # print(mask.shape)
        # a = self.self_attn(x, x, x, mask)

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        output = self.sublayer[1](x, self.feed_forward)
        # ss('--encoderlayer forward')
        return output


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            # ss('-in decoder')
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made up of three sublayers, self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, gg=False):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        output = self.sublayer[2](x, self.feed_forward)
        # print(output.shape)
        # ss('in decoder layer')
        return output



def attention(query, key, value, mask=None, dropout=0.0, gg=False):
    # "Compute 'Scaled Dot Product Attention'"
    # ss('--in attention')
    if gg:
        print('--in attention')
        print(query.shape)
        print(query)
        print(key)
        print(value)
        print(mask)
        print(query.size())
        print()
    # print(key.shape)
    # print(key.transpose(-2, -1).shape)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print('o'*80)
    # print(scores.shape)
    # print(mask.shape)
    # print(mask[0,0,0,0])
    # mask[0,0,0,0] = False
    if gg:
        print(mask)
        print(scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    if gg:
        print(scores)
    p_attn = F.softmax(scores, dim = -1)
    # print(scores)
    # print(p_attn)
    # (Dropout described below)
    p_attn = F.dropout(p_attn, p=dropout)
    # print(p_attn.shape)
    # print(torch.matmul(p_attn, value).shape)
    output = torch.matmul(p_attn, value), p_attn
    # print('--end attention')
    return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None, gg=False):
        # print('-in multihead attention')
        # print(mask.shape)
        # "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # print(nbatches)
        # print(mask.shape)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # l = self.linears[0]
        # x = query
        # # x = x.squeeze(0)
        # query = l(x)
        # print()
        # print(x.shape)
        # print(query.shape)
        # query = query.view(nbatches, -1, self.h, self.d_k)
        # print(query.shape)
        # query = query.transpose(1, 2)
        # print(query.shape)
        # ss('--in mh att')
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p, gg=gg)

        # 3) "Concat" using a view and apply a final linear.
        # print(x.shape)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # print(x.shape)
        output = self.linears[-1](x)
        # print(output.shape)
        # print('-out multihead attention')
        return output

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print(x.shape)
        # ss('-----in positionwisefeedforward forward')
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # print('*'*80)
        # print(d_model)
        # print(vocab)
        # print('*'*80)
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # print(x.shape)
        output = self.lut(x) * math.sqrt(self.d_model)
        # print(output.shape)
        # ss('----in embeddings')
        return output


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # print('*'*80)
        pe = torch.zeros(max_len, d_model)
        # print(pe.shape)
        position = torch.arange(0, max_len).unsqueeze(1)
        # print(position.shape)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # print(div_term.shape)
        # print(torch.sin(position * div_term).shape)
        # a = torch.Tensor([2])
        # print('a',a)
        # b = torch.exp(a * - (math.log(10000.0) / d_model))
        # print(b)
        # print(torch.sin(b))
        # p = torch.Tensor([0,1,2])
        # print(torch.sin(p*b))
        # print(torch.arange(0, d_model, 2))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print(pe[0])
        # print(pe[1])
        # print(pe[2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # print('*' * 80)
    def forward(self, x):
        # print(x.shape)
        # print(self.pe[:, :x.size(1)])
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        output = self.dropout(x)
        # ss('in posenco forward')
        return output


class Generator(nn.Module):
    "Standard generation step. (Not described in the paper.)"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # print(vocab)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # print(x.shape)
        # print(x)
        output = F.log_softmax(self.proj(x), dim=-1)
        # print(output.shape)
        # print(output)
        # ss('in generator')
        return output


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Construct a model object based on hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code. Initialize parameters with Glorot or fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)
    return model

if __name__ == '__main__':
    tmp_model = make_model(10, 10, 2)
    # print(tmp_model)
    # y = tmp_model()