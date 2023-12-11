#!pip install causal-conv1d==1.0.2 mamba-ssm

import torch
import torch.nn as nn

class MambaBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.mamba =  Mamba(
        d_model=embed_dim,
        d_state=16,
        d_conv=4,
        expand=2)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm(self.mamba(x) + x)
        return x


class MambaTower(nn.Module):
    def __init__(self, embed_dim, n_layers, seq_len=None, global_pool=False):
        super().__init__()
        self.blocks = nn.Sequential(*[MambaBlock(embed_dim) for _ in range(n_layers)])
        self.global_pool = global_pool

        #self.pos_embed = nn.Embedding(seq_len, embed_dim)
        #self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_len).long())

    def forward(self, x):
        #pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
        #x = x+self.pos_embed(pos_enc)

        out = self.blocks(x) if not self.global_pool else torch.mean(self.blocks(x),1)
        return out
