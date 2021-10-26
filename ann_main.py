import pytorch_lightning as pl
from ann_model import MnistANN, MnistDataLoader

seed: int = 42
download: bool = False

pl.seed_everything(seed)

model = MnistANN(
    input_size=1 * 28 * 28,
    hidden_size=32,
    output_size=10,
)

dm = MnistDataLoader(download=download)

trainer = pl.Trainer(max_epochs=64)

trainer.fit(model, datamodule=dm)

trainer.test(model, datamodule=dm)
