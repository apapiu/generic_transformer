from transformer_blocks import EncoderBlock, DecoderBlock, MLPSepConv, SinusoidalEmbedding, MHAttention

import torch.nn as nn
import numpy as np
import random
import torch
import torchvision
from einops.layers.torch import Rearrange
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
import lightning as L
import wandb


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DenoiserTransBlock(nn.Module):
    def __init__(self, patch_size, img_size, embed_dim, dropout, n_layers, mlp_multiplier=4, n_channels=4):
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
                 image_size, noise_embed_dims, patch_size, embed_dim, dropout, n_layers,
                 text_emb_size=768):
        super().__init__()

        self.image_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim

        self.fourier_feats = nn.Sequential(SinusoidalEmbedding(embedding_dims=noise_embed_dims),
                                           nn.Linear(noise_embed_dims, self.embed_dim),
                                           nn.GELU(),
                                           nn.Linear(self.embed_dim, self.embed_dim)
                                           )

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, image_size, embed_dim, dropout, n_layers)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.label_proj = nn.Linear(text_emb_size, self.embed_dim)


    def forward(self, x, noise_level, label):

        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.label_proj(label).unsqueeze(1)

        noise_label_emb = torch.cat([noise_level, label], dim=1) #bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)

        x = self.denoiser_trans_block(x,noise_label_emb)

        return x

class LigtningDenoiser(L.LightningModule):

    def __init__(self, model_params, lr=3e-4):
        super().__init__()
        self.model = Denoiser(**model_params)
        self.ema_model = copy.deepcopy(self.model)

        for param in self.ema_model.parameters():
            param.requires_grad = False
    
        self.lr = lr
        self.loss_fn = torch.nn.MSELoss()
        self.scaling_factor = 8

    def forward(self, x, noise_level, label):
        return self.model(x, noise_level, label)

    def training_step(self, batch, batchidx):

        x, y = batch

        x = x/self.scaling_factor
        noise_level = torch.tensor(np.random.beta(1, 2.7, len(x)), device=self.device).float()
        signal_level = 1 - noise_level
        noise = torch.randn_like(x)

        x_noisy = noise_level.view(-1,1,1,1)*noise + signal_level.view(-1,1,1,1)*x
        x_noisy = x_noisy.float()
        label = y
        
        prob = 0.15
        mask = torch.rand(y.size(0), device=self.device) < prob
        label[mask] = 0 # OR replacement_vector

        preds = self(x_noisy, noise_level.view(-1,1), label)
        loss = self.loss_fn(preds, x)

        wandb.log({"train_loss":loss}, step=self.global_step)

        update_ema(self.ema_model, self.model, alpha=alpha)

        #put it in on batch end:
        if self.global_step % 500 == 0:
            out, out_latent = diffusion(
                                    model=self.ema_model,
                                    vae=vae,
                                    labels=torch.repeat_interleave(emb_val, 8, dim=0),
                                    num_imgs=64, n_iter=35,
                                    class_guidance=3,
                                    scale_factor=self.scaling_factor,
                                    dyn_thresh=True)

            to_pil((vutils.make_grid((out+1)/2, nrow=8)).clip(0, 1)).save('img.jpg')
            wandb.log({f"step: {self.global_step}": wandb.Image("img.jpg")})

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    

#lmodel = LigtningDenoiser(model_params)
#trainer = L.Trainer(precision="16-mixed")
#trainer.fit(lmodel, train_loader)



###########
#diffusion:
###########

@torch.no_grad()
def diffusion(model,
              vae,
              n_iter=30,
              labels=None,
              num_imgs=64,
              class_guidance=3,
              seed=10,
              scale_factor=8,
              dyn_thresh=False,
              img_size=16
              ):

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
            s = x0_pred.abs().float().quantile(0.99)
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

from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_warmup_steps, initial_lr, final_lr, last_epoch=-1):
        self.total_warmup_steps = total_warmup_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.total_warmup_steps:
            # Calculate the learning rate based on the current epoch
            lr = self.initial_lr + (self.final_lr - self.initial_lr) * (self.last_epoch / self.total_warmup_steps)
            return [lr for _ in self.base_lrs]
        else:
            # After warm-up, continue with the base learning rate
            return self.base_lrs
        

class CustomDataset(Dataset):
    def __init__(self, latent_data, label_embeddings1, label_embeddings2):
        self.latent_data = latent_data
        self.label_embeddings1 = label_embeddings1
        self.label_embeddings2 = label_embeddings2

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, idx):
        x = self.latent_data[idx]
        if random.random() < 0.5:
            y = self.label_embeddings1[idx]
        else:
            y = self.label_embeddings2[idx]
        return x, y

@torch.no_grad()
def update_ema(ema_model, model, alpha=0.999):
    """update ema model in place"""
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(model_param.data, alpha=1-alpha)


def set_dropout_to_zero(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
            print(module)

    for module in model.modules():
        if isinstance(module, MHAttention):
            #module.p = 0.0
            module.dropout_level = 0
            print(module.dropout_level)

    
