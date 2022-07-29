import torch
import torch.nn as nn
import random

class Shift1d(nn.Module):
    def __init__(self, shift_directions=[1,0]):
        super().__init__()
        self.pad = (min(*shift_directions), max(*shift_directions))
        self.shift_directions = shift_directions

    def forward(self, x):
        # x: [batch, length, d_model]
        b, l, d = x.shape

        # pad
        right_pad = torch.zeros(b, self.pad[0], d, device=x.device)
        left_pad = torch.zeros(b, self.pad[1], d, device=x.device)
        x = torch.cat([left_pad, x, right_pad], dim=1)

        # split
        x = torch.split(x, len(d // self.shift_directions), dim=2)

        # shift and concatenate
        x = torch.cat([torch.roll(t, s, dims=1) for t, s in zip(x, self.shift_directions)], dim=2)

        # remove pads
        x = x[:, self.pad[1]:l]
        return x

class Tokenizer():
    def __init__(self, bos=256, eos=257, mask=258, pad=259, concat_kernel=4):
        self.bos = bos
        self.eos = eos
        self.mask = mask
        self.pad = pad
        self.concat_kernel = concat_kernel

    def tokenize_single(self, sentence : str, mask_prob=0., mask_width=8):
        l = list(sentence.encode(encoding='utf-8'))
        # random masking
        buff = []
        i = 0
        while i < (len(l)):
            a = random.randint(1, mask_width)
            if mask_prob >= random.random():
                buff += [self.mask] * a
            else:
                buff += l[i:i+a]
            i += a
        buff = buff[:len(l)]

        l = [self.bos] + buff + [self.eos]
        return l

    def untokenize_single(self, list_of_integers):
        buff = []
        for i in list_of_integers:
            if i == self.eos:
                break
            if not (i == self.mask or i == self.bos or i == self.pad):
                buff.append(i)
            elif i == self.mask:
                buff += "_".encode("utf-8")
        return bytes(buff).decode(encoding='utf-8', errors="ignore")

    def tokenize_sentences(self, sentences, mask_prob=0., max_length=256):
        s = [ self.tokenize_single(a, mask_prob=mask_prob) for a in sentences ]
        max_len = min(max([len(a) for a in s]), max_length)
        if max_len % self.concat_kernel != 0:
            max_len += (self.concat_kernel - (max_len % self.concat_kernel))
        def _pad(sent, mlen, pad_id):
            b = sent[:mlen]
            while len(b) < mlen:
                b.append(pad_id)
            return b
        s = [ _pad(a, max_len, self.pad) for a in s ]
        return s

    def untokenize_sentences(self, mat):
        return [ self.untokenize_single(l) for l in mat ]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=None, max_length=512, auto_expand=False):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.initialized = False
        self.max_length = max_length
        self.auto_expand = auto_expand

    def __lazy_init__(self, d_model):
        self.embeddings = torch.randn(self.max_length, d_model)
        self.initialized = True
        self.d_model = d_model

    def forward(self, x):
        if not self.initialized:
            self.__lazy_init__(x.shape[2])
            self.embeddings.to(x.device)
        if x.shape[1] > self.embeddings.shape[1] and self.auto_expand:
            # expand embeddings
            self.embeddings = torch.cat([self.embeddings, torch.randn(self.d_model, x.shape[1]-self.embeddings.shape[0]).to(x.device)])
        return x + self.embeddings.to(x.device)

class ByteEmbeddingBlock(nn.module):
    def __init__(self, dim=128, activation=nn.ReLU):
        self.layer = nn.Linear(dim, dim)
        self.shift = Shift1d()
        self.act = activation()

    def forward(self, x):
        res = x
        x = self.shift(x)
        x = self.layer(x)
        x = self.act(x)
        x += res
        return x

class ByteEmbedding(nn.Module):
    def __init__(self, vocab_size=260, dim_per_byte=128, depth=4, activation=nn.ReLU, embedding_dim=512, concatenate_length=4):
        super().__init__()
        self.concatenate_length = concatenate_length
        self.embedding = nn.Embedding(vocab_size, dim_per_byte)
        self.layers = nn.Sequential(*[ByteEmbeddingBlock(dim_per_byte, activation) for _ in range(depth)])
        self.out_linear = nn.Linear(dim_per_byte * concatenate_length, embedding_dim)

    def forward(self, x):
        b, l, d = x.shape
        # pad
        pad = torch.zeros(b, self.concatenate_length - (l % self.concatenate_length), d, device=x.device)
        x = torch.cat([x, papd], dim=1)
        
        # embedding
        x = self.embedding(x)
        x = self.layers(x)

