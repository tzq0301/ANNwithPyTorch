import os
from typing import Any, Optional, Dict, List, Callable, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST


class MnistANN(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 1e-2):
        super().__init__()
        self.automatic_optimization = False  # 不使用 PyTorch 的 optimizer 进行 loss 反向传播
        self.lr = lr
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        self.params: Dict[str, Tensor] = {
            "v": torch.empty(self.input_size, self.hidden_size),
            "gamma": torch.zeros(self.hidden_size),
            "w": torch.empty(self.hidden_size, self.output_size),
            "theta": torch.zeros(self.output_size),
        }
        nn.init.xavier_normal_(self.params["v"])
        nn.init.xavier_normal_(self.params["w"])

    def forward(self, x, *args, **kwargs) -> Any:
        def tzq_pipeline(inputs: List[Tensor], functions: List[Callable]) -> Tensor:
            return functions[2](functions[1](functions[0](inputs[0], inputs[1]), inputs[2]))

        # print(batch)
        b: Tensor = tzq_pipeline(
            inputs=[x, self.params["v"], self.params["gamma"]],
            functions=[torch.matmul, torch.add, torch.sigmoid],
        )
        y_hat: Tensor = tzq_pipeline(
            inputs=[b, self.params["w"], self.params["theta"]],
            functions=[torch.matmul, torch.add, torch.sigmoid],
        )

        return y_hat, b

    def backward_propagation(self, x: Tensor, y: Tensor, y_hat: Tensor, b: Tensor) -> None:
        y = F.one_hot(y, num_classes=10)
        # print(y.size(), y_hat.size())  # torch.Size([16, 10]) torch.Size([16, 10])
        g: Tensor = y_hat * (torch.ones_like(y_hat) - y_hat) * (y - y_hat)  # torch.Size([16, 10])
        # print(b.size())  # b: torch.Size([16, 32])
        tmp1: Tensor = torch.stack([g] * list(b.size())[1], dim=1)  # torch.Size([16, 32, 10])
        tmp2: Tensor = torch.stack([b] * list(g.size())[1], dim=2)  # torch.Size([16, 32, 10])
        # print(g.size(), tmp1.size(), tmp2.size(), self.params["w"].size())  # w: torch.Size([32, 10])
        for idx, x_tmp1 in enumerate(tmp1):
            self.params["w"] += self.lr * x_tmp1 * tmp2[idx]
        for x_g in g:
            self.params["theta"] += - self.lr * x_g

        # e: (16, 32), w: (32, 10), g: (16, 10)
        e: Tensor = b * (torch.ones_like(b) - b) * torch.matmul(g, torch.t(self.params["w"]))
        # print(e.size())  # torch.Size([16, 32])

        tmp3: Tensor = torch.stack([e] * list(x.size())[1], dim=1)
        tmp4: Tensor = torch.stack([x] * list(e.size())[1], dim=2)
        for idx, x_tmp3 in enumerate(tmp3):
            self.params["v"] += self.lr * x_tmp3 * tmp4[idx]
        for x_e in e:
            self.params["gamma"] += - self.lr * x_e

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        (x, y) = batch

        # torch.Size([16, 1, 1, 784]) => torch.Size([16, 784])
        x = torch.squeeze(x)

        # print(x.size())  # torch.Size([16, 784]) == torch.Size([16, 1 * 28 * 28])
        # print(y.size())  # torch.Size([16])

        y_hat, b = self(x)

        loss = 0.5 * torch.sum(torch.pow(torch.argmax(y_hat, dim=1) - y, 2))

        # Backward Propagation
        self.backward_propagation(x, y, y_hat, b)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        (x, y) = batch
        y_hat, b = self(x)
        # print(y.size(), y)
        # print(y_hat.size(), y_hat)
        y_hat = torch.squeeze(y_hat)
        loss = 0.5 * torch.sum(torch.pow(torch.argmax(y_hat, dim=1) - y, 2))
        pred = torch.argmax(y_hat, dim=1)
        return {
            "loss": loss,
            "label": y,
            "pred": pred,
        }

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        metric: Accuracy = Accuracy()
        for output in outputs:
            _ = metric(output['pred'], output['label'])
        print(f"\nAccuracy: {metric.compute().item() : .2f}\n")

    def configure_optimizers(self):
        return None


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
        transform = transforms.Compose([
            transforms.Resize((1, 28 * 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
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
