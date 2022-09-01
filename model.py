import torch
import torch.nn as nn
import math
import random

from reformer_pytorch import Reformer
from tokenizer import encode, decode

class SeparatedConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, activation=nn.GELU):
        super().__init__()
        self.c1 = nn.Conv1d(input_channels, input_channels, kernel_size, stride, padding, groups=input_channels)
        self.c2 = nn.Conv1d(input_channels, output_channels, 1, 1, 0)
        self.act = activation()

    def forward(self, x):
        return self.act(self.c2(self.c1(x)))

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        return self.module(x) + x

class Tokenizer(nn.Module):
    def __init__(self, embedding_dim=512, vocab_size=257, byte_dim=128, num_layers=4, kernel_size=4):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, byte_dim)
        self.layers = nn.Sequential(*[Residual(SeparatedConv1d(byte_dim, byte_dim, 3, 1, 1, nn.ReLU)) for _ in range(num_layers)])
        self.last = nn.Conv1d(byte_dim, embedding_dim, kernel_size, kernel_size, 0)
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.layers(x)
        x = self.last(x)
        x = torch.transpose(x, 1, 2)
        return x

class Detokenizer(nn.Module):
    def __init__(self, embedding_dim=512, vocab_size=257, kernel_size=4, num_layers=4):
        super().__init__()
        self.conv = nn.ConvTranspose1d(embedding_dim, vocab_size, kernel_size, kernel_size, 0)
        
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = torch.transpose(x, 1, 2)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096, shape=True, trainable=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if trainable:
            self.pe = torch.nn.Parameter(pe)
        else:
            self.register_buffer('pe', pe)
        self.shape = shape

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        if self.shape:
            pe = torch.roll(self.pe, 0, random.randint(0, self.pe.shape[0]))
        else:
            pe = self.pe
        pe = pe[:x.size(0)]

        x = x + pe
        x = torch.transpose(x, 0, 1)
        return self.dropout(x)

class ByteT2T(nn.Module):
    def __init__(self, d_model=256, num_layers=8, d_byte=128, num_heads=4, max_seq_len=64, bucket_size=16):
        super().__init__()
        self.encoder_pe = PositionalEncoding(d_model, max_len=max_seq_len)
        self.decoder_pe = PositionalEncoding(d_model, max_len=max_seq_len, shape=False)
        self.encoder = Reformer(d_model, num_layers, heads=num_heads)
        self.decoder = Reformer(d_model, num_layers, heads=num_heads)
        self.tokenzer = Tokenizer(d_model, 257)
        self.detokenizer = Detokenizer(d_model, 257)

    def forward(self, src, tgt):
        src, tgt = self.tokenzer(src), self.tokenzer(tgt)
        src, tgt = self.encoder_pe(src), self.decoder_pe(tgt)
        mem = self.encoder(src)
        out = self.decoder(src, keys=mem)
        out = self.detokenizer(out)
        return out

    def predict(self, input_sentences, num_iterations = 16):
        pass
