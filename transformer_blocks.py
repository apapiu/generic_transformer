
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
        # also could add another projection on attn output.
        x = self.norm1(x + attn)
        x = self.norm2(x + self.mlp(x))
        return x
