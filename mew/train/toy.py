import torch
from torch import nn
import pytorch_lightning as pl


class ToyModel(pl.LightningModule):
    def __init__(self, dim_in=16, dim_out=8, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(dim_in, dim_out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def toy_dataloader():
    x = torch.randn(1024, 16)
    y = torch.zeros(1024, 8)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
