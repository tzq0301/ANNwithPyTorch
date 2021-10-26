from typing import Any, Optional
import os

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT, EPOCH_OUTPUT
from torchmetrics import Accuracy
from torch import nn, optim
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms


class MnistCNN(pl.LightningModule):
    def __init__(self, lr: float = 1e-2):
        super().__init__()
        self.lr = lr
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # reshape(-1, 12 * 4 * 4) between layer2 and layer3
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=12 * 4 * 4, out_features=120),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=120, out_features=60),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=60, out_features=10)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, *args, **kwargs) -> Any:
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(-1, 12 * 4 * 4)
        x = self.layer3(x)
        x = self.layer4(x)
        o = self.layer5(x)
        return o

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        (x, y) = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        (x, y) = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        return {
            "loss": loss,
            "label": y,
            "pred": pred,
        }

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        metric: Accuracy = Accuracy()
        for output in outputs:
            _ = metric(output['pred'].cpu(), output['label'].cpu())
        print(f"\nAccuracy: {metric.compute().item() : .2f}\n")


class MnistDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, data_folder: str = 'data', download: bool = True):
        super().__init__()
        self.batch_size: int = batch_size
        self.data_path: str = os.path.join(os.getcwd(), data_folder)
        self.download: bool = download
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def prepare_data(self) -> None:
        MNIST(root=self.data_path, train=True, download=self.download)
        MNIST(root=self.data_path, train=False, download=self.download)

    def setup(self, stage: Optional[str] = None) -> None:
        # transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # split dataset
        if stage in (None, "fit"):
            mnist_train = MNIST(self.data_path, train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage in (None, "test"):
            self.mnist_test = MNIST(self.data_path, train=False, transform=transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.mnist_test, batch_size=self.batch_size)
