from transformer_blocks import EncoderBlock, DecoderBlock, MLPSepConv, SinusoidalEmbedding
import torch.nn as nn
import numpy as np
import torch
import torchvision
from einops.layers.torch import Rearrange


class DenoiserTransBlock(nn.Module):
    def __init__(self, patch_size, img_size, n_channels, embed_dim, dropout, n_layers, mlp_multiplier=4):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = int((self.img_size/self.patch_size)*((self.img_size/self.patch_size)))
        patch_dim = self.n_channels*self.patch_size*self.patch_size

        self.patchify_and_embed = nn.Sequential(
                                       nn.Conv2d(self.n_channels, patch_dim, kernel_size=self.patch_size, stride=self.patch_size),
                                       Rearrange('bs d h w -> bs (h w) d'),
                                       nn.LayerNorm(patch_dim),
                                       nn.Linear(patch_dim, self.embed_dim),
                                       nn.LayerNorm(self.embed_dim)
                                       )

        self.rearrange2 = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                                   h=int(self.img_size/self.patch_size),
                                   p1=self.patch_size, p2=self.patch_size)


        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_dim=self.embed_dim,
                                                          mlp_multiplier=self.mlp_multiplier,
                                                          is_causal=False,
                                                          dropout_level=self.dropout,
                                                          mlp_class=MLPSepConv)
                                              for _ in range(self.n_layers)])

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, patch_dim),
                                                self.rearrange2)


    def forward(self, x, cond):
        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
        x = x+self.pos_embed(pos_enc)

        for block in self.decoder_blocks:
            x = block(x, cond)

        return self.out_proj(x)

class Denoiser(nn.Module):
    def __init__(self,
                 image_size,
                 noise_embed_dims):
        super().__init__()

        n = n_channels

        self.scaler = GradScaler()

        self.img_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim


        self.fourier_feats = nn.Sequential(SinusoidalEmbedding(embedding_dims=noise_embed_dims),
                                           nn.Linear(noise_embed_dims, self.embed_dim),
                                           nn.GELU(),
                                           nn.Linear(self.embed_dim, self.embed_dim)
                                           )

        self.denoiser_trans_block = DenoiserTransBlock()
        self.global_step = 0

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.norm = nn.LayerNorm(self.embed_dim)

        self.label_proj = nn.Linear(512, self.embed_dim)


    def forward(self, x, noise_level, label):

        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.label_proj(label).unsqueeze(1)

        noise_label_emb = torch.cat([noise_level, label], dim=1) #bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)

        x = self.denoiser_trans_block(x,noise_label_emb)

        return x

    def train_step(self, x, noise_level, label, y):

        with autocast():
                pred = self.forward(x, noise_level, label)
                loss = self.loss_fn(pred, y)

        self.optimizer.zero_grad()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.global_step += 1

        return loss

        x = self.denoiser_trans_block(x,noise_label_emb)

        return x


###########
#diffusion:
###########

@torch.no_grad()
def diffusion(n_iter=30,
              labels=None,
              num_imgs=64,
              class_guidance=3,
              seed=10,
              scale_factor=8,
              dyn_thresh=False,
              model=model):

    noise_levels = 1 - np.power(np.arange(0.0001, 0.99, 1 / n_iter), 1 / 3)
    noise_levels[-1] = 0.01

    torch.manual_seed(seed)
    seeds = torch.randn(num_imgs,4,img_size,img_size).to(device)
    new_img = seeds

    empty_labels = torch.zeros_like(labels)
    labels = torch.cat([labels, empty_labels])

    model.eval()


    for i in tqdm(range(len(noise_levels) - 1)):

        curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]

        noises = torch.full((num_imgs,1), curr_noise)
        noises = torch.cat([noises, noises])


        x0_pred = model(torch.cat([new_img, new_img]),
                        noises.to(device),
                        labels.to(device)
                        )

        x0_pred_label = x0_pred[:num_imgs]
        x0_pred_no_label = x0_pred[num_imgs:]

        # classifier free guidance:
        x0_pred = class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label

        # new image at next_noise level is a weighted average of old image and predicted x0:

        new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise
        #new_img = (np.sqrt(1 - next_noise**2)) * x0_pred + next_noise * (new_img - np.sqrt(1 - curr_noise**2)* x0_pred)/ curr_noise

        if dyn_thresh:
            s = x0_pred.abs().quantile(0.99)
            x0_pred = x0_pred.clip(-s, s)/(s/2) #rescale to -2,2

    #predict with model one more time to get x0

    x0_pred_img = vae.decode((x0_pred*scale_factor).half())[0].cpu()

    return x0_pred_img, x0_pred

########
###utils
########


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_per_layer(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")

to_pil = torchvision.transforms.ToPILImage()



