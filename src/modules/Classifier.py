import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L


class Classifier(L.LightningModule):
    def __init__(self, 
                args,
                multilabel: bool,
                num_features: int,
                backbone: nn.Module,
                ):
        super().__init__()
        self.args = args
        self.multilabel = multilabel
        self.num_features = num_features
        self.embedding = 2048 
        self.backbone = backbone.eval()
        
        if bool(self.args.linear):
            self.mlp = nn.Linear(self.embedding, self.num_features)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.embedding, self.embedding),
                nn.ReLU(),
                nn.Linear(self.embedding, self.num_features),
                )
        
        print(self.mlp)

        if self.multilabel:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = F.cross_entropy

        self.save_hyperparameters()
        self.val_outputs = []
        self.test_outputs = []


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.backbone(x)
        if len(x.shape) == 3:
            x = torch.mean(x, dim=1, keepdim=True)
        x = self.mlp(x)
        return x

    def _interal_forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 5:
            x = x.squeeze(0)
            # repeat y to match x first dimenion
            y = y.repeat(x.shape[0], 1)
        with torch.no_grad():
            x = self.backbone(x)
        return self.mlp(x), y

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = self._interal_forward(x, y)
        loss = self.loss(x, y)
        self.log("train_loss", loss)
        return loss    

    def evaluation_step(self, x, y) -> dict:
        x, y = self._interal_forward(x, y)
        loss = self.loss(x, y).cpu().numpy()
        if self.multilabel:
            predictions = torch.sigmoid(x).cpu().numpy()
        else:
            predictions = torch.argmax(x, dim=1).cpu().numpy()
        labels = y.cpu().numpy()
        return {"loss": loss, "predictions": predictions, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.val_outputs.append(self.evaluation_step(x, y))
        return x
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        self.test_outputs.append(self.evaluation_step(x, y))
        return x
    
    def __on_epoch_end(self, outputs: dict, key: str): 
        predictions = np.concatenate([x["predictions"] for x in outputs])
        labels = np.concatenate([x["labels"] for x in outputs])
        loss = np.mean([x["loss"] for x in outputs])
        self.log(f"{key}_loss", loss)
        if self.multilabel:
            roc_auc = metrics.roc_auc_score(labels, predictions, average='macro')
            self.log(f"{key}_roc_auc", roc_auc)
            average_precision = metrics.average_precision_score(labels, predictions, average='macro')
            self.log(f"{key}_pr_auc", average_precision)
        
        else:
            self.log(f"{key}_acc", metrics.accuracy_score(labels, predictions))


    def on_validation_epoch_end(self) -> None:
        self.__on_epoch_end(self.val_outputs, "val")
        self.val_outputs = []

    def on_test_epoch_end(self) -> None:
        self.__on_epoch_end(self.test_outputs, "test")
        self.test_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.args.lr,
            weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
