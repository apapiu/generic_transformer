import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, mlp_multiplier, is_causal):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3*embed_dim)
        self.n_heads = n_heads
        self.dropout_level = dropout
        self.is_causal = is_causal
        self.dropout = nn.Dropout(self.dropout_level)
        self.d_k = embed_dim // n_heads
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(self.dropout_level)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)


    def forward(self, x):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)

        d_k = self.d_k

        #MHA - split into heads -> (bs, h, n, d_k)
        #TODO: can we use MultiHeadAttention here?
        #TODO: can we use rearrange? rearrange('b n (h dk) -> b h n d_k', dk = d_k)
        q, k, v = [x.view(x.size(0), x.size(1), self.n_heads, d_k).transpose(1, 2) for x in [q, k, v]]
        attn = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                is_causal=self.is_causal,
                                                                dropout_p=self.dropout_level)
        attn =  attn.transpose(1,2).contiguous().view(attn.size(0), attn.size(2), -1) #END MHA

        #GPT2 moves layer norm on the inside of the function - applied before attn and mlp. 
        # also I use no out proj here
        x = self.norm1(x + attn)
        x = self.norm2(x + self.mlp(x))
        return x

class Tower(nn.Module):
    # input is (bs,n,d) n sequences of dim d (e.g. word embeddings, or flattened image patches) 
    # output is (bs, n,d) OR (bs,d) if global_pool is True
    def __init__(self, embed_dim, seq_len, n_layers, use_pos_embeddings,
                 dropout=0.1, n_heads=4, n_class=1, mlp_multiplier=2, is_causal=False, global_pool=False):
        super().__init__()
        self.use_pos_embeddings = use_pos_embeddings
        self.global_pool = global_pool

        #self.blocks = nn.ModuleList()

        self.tower = nn.Sequential(*[Block(embed_dim=embed_dim, 
                                    n_heads=n_heads, 
                                    dropout=dropout, 
                                    mlp_multiplier=mlp_multiplier, 
                                    is_causal=is_causal) for i in range(n_layers)])

        if use_pos_embeddings:
            #simple fixed learned positional encodings for now:
            self.pos_embed = nn.Embedding(seq_len, embed_dim)
            self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_len).long())

    def forward(self, x):
            
        if self.use_pos_embeddings:
            pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
            out = self.tower(x+self.pos_embed(pos_enc))
        else:
            out = self.tower(x)


        if self.global_pool:
            return torch.mean(out, dim=1)
        else:
            return out


def test_dims():
    b = Block(embed_dim=256, n_heads=2, dropout=0, mlp_multiplier=2, is_causal=False)
    t = Tower(embed_dim=256, n_heads=2, dropout=0, mlp_multiplier=2, n_layers=4, 
            seq_len=64, use_pos_embeddings=True, global_pool=True)

    x = torch.randn(32, 64, 256)

    assert b(x).shape == (32, 64, 256) 
    assert t(x).shape == (32, 256) 
