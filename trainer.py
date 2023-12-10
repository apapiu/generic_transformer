import torch
import torch.nn as nn
import numpy as np

from torch.cuda.amp import autocast

import torchmetrics
from torchview import draw_graph
from torchsummary import summary

from tqdm import tqdm
import wandb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

class Trainer():
    def __init__(self, model, loss_fn, lr=0.0003, T_max=10000, weight_decay=0, wandb_log=False):
        super().__init__()

        self.loss_fn = loss_fn
        self.model = model
        self.wandb_log = wandb_log

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        self.global_step = 0

    def forward(self, x):
        return self.model(x)

    def train_step(self, x, y):
        self.optimizer.zero_grad()

        x, y = to_device(x, device), to_device(y, device)

        # Forward pass with mixed precision
        with autocast():
            preds = self.model(x)
            loss = self.loss_fn(preds, y)

            if self.wandb_log:
                wandb.log({"train_loss":loss}, step=self.global_step)

        # Backward pass
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()

        self.global_step += 1

    @torch.no_grad()
    def validation(self, val_loader, val_metric):
        val_loss = []

        self.model.eval()
        for x, y in val_loader:
            x, y = to_device(x, device), to_device(y, device)
            preds = self.model(x)

            loss = self.loss_fn(preds, y)
            val_loss.append(loss.item())

            val_metric.update(preds, y)

        final_accuracy = val_metric.compute() * 100
        final_val_loss = np.array(val_loss).mean()
        print(f"Iteration: {self.global_step}, Val Metric: {final_accuracy}%, Val Loss: {final_val_loss}, LR: {self.scheduler.get_last_lr()}")

        if self.wandb_log:
            wandb.log({"val_accuracy":final_accuracy}, step=self.global_step)
            wandb.log({"val_loss":final_val_loss}, step=self.global_step)

        val_metric.reset()
        self.model.train()


    def plot_architecture(self, dataloader, depth=4):
        #draw graph and output summary:
        for x, y in dataloader:
            model_graph = draw_graph(self.model, input_data=x,
                         device=device, depth=depth,
                         save_graph=True, filename='model_graph_new',
                         expand_nested=True, )
            try:
                print(summary(self.model, input_size=x.size()[1:], batch_size=64))
            except Exception as e:
                print(e)
                pass
            break

    def train_loop(self, train_loader, val_loader, n_epochs, val_metric, val_every_n_iters=None):
        for epoch in range(n_epochs):
            if val_every_n_iters is None:
                self.validation(val_loader, val_metric)
            for x, y in tqdm(train_loader):
                if val_every_n_iters is not None:
                    if self.global_step % val_every_n_iters == 0:
                         self.validation(val_loader, val_metric)
                self.train_step(x, y)

### example:
#wandb.init(
#    project="project_name",
#    config = config)

# model = ImgClassifier(config_params)
# model.to(device)
# loss = nn.CrossEntropyLoss()
# val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
# trainer = Trainer(model, loss, lr=0.0005, T_max=100000, weight_decay=1e-5)
# trainer.train_loop(train_loader, val_loader, n_epochs, val_metric)
