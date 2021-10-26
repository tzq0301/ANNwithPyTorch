import pytorch_lightning as pl
from cnn_model import MnistCNN, MnistDataLoader

seed: int = 42
download: bool = False

pl.seed_everything(seed)

model = MnistCNN()

dm = MnistDataLoader(download=download)

trainer = pl.Trainer(gpus=1, max_epochs=120)

trainer.fit(model, datamodule=dm)

trainer.test(model, datamodule=dm)
