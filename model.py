import torch
import torch.nn as nn
import math

from reformer_pytorch import Reformer
from charformer_pytorch import GBST

class Detokenizer(nn.Module):
    def __init__(self, embedding_dim=512, vocab_size=257, pool_length=4):
        super().__init__()
        self.conv = nn.ConvTranspose1d(embedding_dim, vocab_size, pool_length, pool_length, 0)
        
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = torch.transpose(x, 1, 2)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        x = x + self.pe[:x.size(0)]
        x = torch.transpose(x, 0, 1)
        return self.dropout(x)

class ByteT2T(nn.Module):
    def __init__(self, d_model=256, num_layers=8, d_byte=128, num_heads=4, max_seq_len=64, bucket_size=16):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.gbst = GBST(num_tokens=257, dim=d_byte, downsample_factor=4)
        self.embedding = nn.Linear(d_byte, d_model)
        self.encoder = Reformer(d_model, num_layers, max_seq_len=max_seq_len, heads=num_heads)
        self.decoder = Reformer(d_model, num_layers, max_seq_len=max_seq_len, heads=num_heads)
        self.detokenizer = Detokenizer(d_model, 257)

    def predict(self, input_sentence, ):
        pass
