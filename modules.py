import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderDecoder(nn.Module):

    def __init__(self):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.src_embed = None
        self.tgt_embed = None
        self.generator = None
        
    def forward(self, src, tgt, src_mask, tgt_mask):

        x = self.encoder(self.src_embed(src), src_mask)
        x = self.decoder(x, src_mask, tgt, tgt_mask)
        
        return x

class LayerNorm(nn.Module):

    def __init__(self, layer_size, eps=1e-8):

        super(LayerNorm, self).__init__()
        self.layer_size = layer_size
        self.alpha = nn.Parameter(torch.ones(self.layer_size))
        self.beta = nn.Parameter(torch.zeros(self.layer_size))
        self.eps = eps

    def forward(self, x):
        # x size: (batch_size, seq_len, d_model)
        x_hat = (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + self.eps)
        x_tilde = self.alpha*x_hat + self.beta
        return x_tilde

def attention(q, k, v, mask=None, dropout=None):

    d_k = q.size(-1)
    assert q.size(-1) == k.size(-1)

    scores = q @ k.transpose(-2,-1)
    scores /= np.sqrt(k.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)

    scores = scores @ v

    return scores

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):

        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_k = d_model // h
        self.h = h
        self.d_model = d_model

        self.v_linear = [nn.Linear(d_model, d_k) for _ in range(h)]
        self.k_linear = [nn.Linear(d_model, d_k) for _ in range(h)]
        self.q_linear = [nn.Linear(d_model, d_k) for _ in range(h)]

        self.attention = attention
        
        self.output_linear = nn.Linear(d_k * h, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, mask=None):

        attention_out_list = []
        for i in range(self.h):
            v2 = self.v_linear[i](v)
            k2 = self.k_linear[i](k)
            q2 = self.q_linear[i](q)

            attention_out = self.attention(v2, k2, q2, mask, self.dropout)
            attention_out_list.append(attention_out)

        attention_out_list = torch.cat(attention_out_list, dim=-1)
        return self.output_linear(attention_out_list)

class ResidualSublayer(nn.Module):

    def __init__(self, sublayer, size, dropout=0.5):

        super(ResidualSublayer, self).__init__()
        self.sublayer = sublayer
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *x):

        return self.norm(x[0] + self.dropout(self.sublayer(*x)))



class TransformerEncoderLayer(nn.Module):

    def __init__(self, layer_size=512, ff_dim=2048, head_num=8):

        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attn = ResidualSublayer(
            MultiHeadAttention(head_num, layer_size),
            layer_size)
        self.ff_network = ResidualSublayer(
            nn.Sequential(
                nn.Linear(layer_size, ff_dim), 
                nn.ReLU(),
                nn.Linear(ff_dim, layer_size)
            ), layer_size
        )

    def forward(self, x):
        
        x = self.multi_head_attn(x,x,x)

        x = self.ff_network(x)

        return x


class Encoder(nn.Module):

    def __init__(self, layer_size=512, N=6):

        super(Encoder, self).__init__()
        self.norm = LayerNorm(layer_size)
        self.layers = nn.ModuleList([TransformerEncoderLayer() for _ in range(N)])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, layer_size=512, ff_dim=2048, head_num=8):

        super(TransformerDecoderLayer, self).__init__()
        self.attention_1 = ResidualSublayer(
            MultiHeadAttention(head_num, layer_size),
            layer_size)
        self.attention_2 = ResidualSublayer(
            MultiHeadAttention(head_num, layer_size),
            layer_size)
        self.ff_network = ResidualSublayer(
            nn.Sequential(
                nn.Linear(layer_size, ff_dim), 
                nn.ReLU(),
                nn.Linear(ff_dim, layer_size)
            ), layer_size
        )

    def forward(self, x, memory):
        
        x = self.attention_1(x, x, x)

        x = self.attention_2(x, memory, memory)

        x = self.ff_network(x)

        return x

class Decoder(nn.Module):

    def __init__(self, layer_size=512, N=6):

        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([TransformerDecoderLayer() for _ in range(N)])

    def forward(self, x, memory):

        for layer in self.layers:
            x = layer(x, memory)
        return x