import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl

import torchmetrics
from torchmetrics import (
    MetricCollection,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    StatScores,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineClassifier(pl.LightningModule):
    def __init__(
        self, 
        num_classes: int, 
        input_size: int, 
        learning_rate: float, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        weights: np.ndarray
        ):
        super(BaselineClassifier, self).__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.linear = nn.Linear(input_size, 1028)
        self.classifier = nn.Linear(1028, num_classes)
        metrics = MetricCollection(
            [
                Precision(average="macro", num_classes=num_classes),
                Recall(average="macro", num_classes=num_classes),
                F1Score(average="macro", num_classes=num_classes),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.weights = weights

    def forward(self, x):
        x = F.leaky_relu(self.linear(x))
        x = self.classifier(x)
        
        return torch.log_softmax(x, dim=1)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        metrics = self.train_metrics(preds, y)
        self.log_dict(metrics)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, val_epoch):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = self.val_metrics(preds, y)

        self.log_dict(metrics)
        self.log("val_loss", loss)
        return {"loss": loss, "labels": y}

    def predict_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        return preds

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels, weight=self.weights.to(device))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader