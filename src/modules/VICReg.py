from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader

from src.architectures import mlp
from src.optimizers import LARS, adjust_learning_rate, include_bias_and_norm
from src.transforms import AudioSplit
from src.utils import off_diagonal


class VICReg(L.LightningModule):
    def __init__(self, args, dataset, backbone):
        super().__init__()
        self.args = args
        self.num_features = int(args.projector.split("-")[-1])
        self.backbone = backbone
        self.projector = mlp(args.projector)
        self.val_outputs = []
        self.train_outputs = []
        self.dataset = dataset

    def internal_forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))
        return x, y

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return x

    def vicreg_loss(self, batch):
        (x, y), _ = batch
        x, y = self.internal_forward(x, y)

        repr_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss, (repr_loss, std_loss, cov_loss)

    def training_step(self, batch, batch_idx):
        loss, vic = self.vicreg_loss(batch)

        losses = {
            "loss": loss,
            "invariance": vic[0],
            "variance": vic[1],
            "covariance": vic[2],
        }
        self.train_outputs.append(losses)
        return losses

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        lr = adjust_learning_rate(
            self.args,
            self.optimizers(),
            self.trainer.train_dataloader,
            self.global_step,
        )
        self.log("lr", lr, sync_dist=True)

    def _on_epoch_end(self, outputs, name):
        v = np.mean([x["variance"].cpu().detach().numpy() for x in outputs])
        i = np.mean([x["invariance"].cpu().detach().numpy() for x in outputs])
        c = np.mean([x["covariance"].cpu().detach().numpy() for x in outputs])
        reg = None  # 😎
        loss = np.mean([x["loss"].cpu().detach().numpy() for x in outputs])
        self.log(f"{name}_loss", loss, sync_dist=True)
        self.log(f"{name}_variance_loss", v, sync_dist=True)
        self.log(f"{name}_invariance_loss", i, sync_dist=True)
        self.log(f"{name}_covariance_loss", c, sync_dist=True)

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end(self.train_outputs, "train")
        self.train_outputs = []

    def configure_optimizers(self):
        optimizer = LARS(
            self.parameters(),
            lr=0,
            weight_decay=self.args.weight_decay,
            weight_decay_filter=include_bias_and_norm,
            lars_adaptation_filter=include_bias_and_norm,
        )
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.dataset(
                "train", transforms=AudioSplit(self.args), mixing=self.args.mixing
            ),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=self.args.prefetch_factor,
        )
