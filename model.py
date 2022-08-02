import torch
import torch.nn as nn

from reformer_pytorch import ReformerLM
from charformer_pytorch import GBST

class Detokenizer(nn.Module):
    def __init__(self, embedding_dim=512, vocab_size=257, pool_length=4):
        super().__init__()
        self.conv = nn.ConvTranspose1d(embedding_dim, vocab_size, pool_length, pool_length, 0)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class ProgressiveLayerWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.status = True

    def forward(self, *args, **kwargs):
        if self.status:
            return self.layer(*args, **kwargs)
        else:
            return (*args, )

class ReformerLMWrapper(nn.Module):
    def __init__(self, reformerlm):
        super().__init__()
        self.reformerlm = reformerlm
        nl = len(self.reformerlm.reformer.layers.blocks)
        for n in range(nl):
            self.reformerlm.reformer.layers.blocks[n].f = ProgressiveLayerWrapper(self.reformerlm.reformer.layers.blocks[n].f)
            self.reformerlm.reformer.layers.blocks[n].g = ProgressiveLayerWrapper(self.reformerlm.reformer.layers.blocks[n].g)

    def set_num_available_layers(self, num_layers):
        nl = len(self.reformerlm.reformer.layers.blocks)
        for n in range(nl):
            s = nl <= num_layers
            self.reformerlm.reformer.layers.blocks[n].f.status = s
            self.reformerlm.reformer.layers.blocks[n].g.status = s


