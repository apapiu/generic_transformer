from einops import rearrange
import torch.nn as nn
from einops.layers.torch import Rearrange


class ImgClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_size = 4
        self.img_size = 32
        self.n_channels = 3
        embed_dim = 256
        dropout = 0.1
        mlp_multiplier = 2
        n_layers = 6
        seq_len = int((self.img_size/self.patch_size)*((self.img_size/self.patch_size)))
        patch_dim = self.n_channels*self.patch_size*self.patch_size

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                                   p1=self.patch_size, p2=self.patch_size)

        self.func = nn.Sequential(self.rearrange,
                                  nn.LayerNorm(patch_dim), #really seems to help
                                  nn.Linear(patch_dim, embed_dim),
                                  nn.LayerNorm(embed_dim),
                                  Tower(embed_dim=embed_dim, 
                                        seq_len=seq_len,
                                        n_layers=n_layers,
                                        use_pos_embeddings=True,
                                        mlp_multiplier = mlp_multiplier, 
                                        mlp_class=MLPSepConv,
                                        dropout=dropout, 
                                        global_pool=True),
                                  nn.Linear(embed_dim, 10))

    def forward(self, x):

        return self.func(x)

