import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import *

class Transformer(nn.Module):

    def __init__(self, vocab_size, seq_len, layer_dim=512):

        super(Transformer, self).__init__()

        self.src_embed = TransformerEmbedding(vocab_size, seq_len, layer_dim)
        self.tgt_embed = TransformerEmbedding(vocab_size, seq_len, layer_dim)

        self.encoder = nn.Sequential(
            *[TransformerEncoderLayer(layer_dim=layer_dim) for _ in range(6)]
        )
        self.encoder_norm = nn.LayerNorm(layer_dim)
        self.decoder = nn.Sequential(
            *[TransformerDecoderLayer(layer_dim=layer_dim) for _ in range(6)]
        )
        self.decoder_norm = nn.LayerNorm(layer_dim)

        self.output_linear = nn.Linear(layer_dim, layer_dim)

    def forward(self, src, tgt):

        src = self.src_embed(src)
        tgt = self.tgt_embed(tgt)

        x = self.encoder(src)
        x = self.encoder_norm(x)
        x = self.decoder(x, tgt)
        x = self.decoder_norm(x)

        x = output_linear(x)

        return F.softmax(x)


