import random

import pytorch_lightning as pl

import torch
from torchmetrics.functional import accuracy


class PlModule(pl.LightningModule):
    def __init__(self, model, optimizer, loss_fn, ticnn=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.ticnn = ticnn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.ticnn:
            self.model.dropout_rate = random.uniform(0.1, 0.9)
            if batch_idx % 5 == 0:
                self.model.dropout_rate = 0
        X, y = batch
        logit = self(X)
        preds = torch.argmax(logit, 1)
        correct = preds.eq(y).sum().item()
        total = len(y)
        loss = self.loss_fn(logit, y)
        acc = accuracy(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        train_info = {"correct": correct, "total": total, "loss": loss, "acc": acc}

        return train_info

    def evaluate(self, batch, stage=None):
        X, y = batch
        logit = self(X)
        preds = torch.argmax(logit, 1)
        correct = preds.eq(y).sum().item()
        total = len(y)
        val_acc = accuracy(preds, y)
        val_loss = self.loss_fn(logit, y)

        eval_info = {
            "correct": correct,
            "total": total,
            "loss": val_loss,
            "acc": val_acc,
        }

        if stage:
            self.log(
                f"{stage}_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True
            )
            self.log(
                f"{stage}_acc", val_acc, prog_bar=True, on_step=False, on_epoch=True
            )

        return eval_info

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch=batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch=batch, stage="test")

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()