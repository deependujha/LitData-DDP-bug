# train.py
import lightning as L
import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from litdata import StreamingDataset, StreamingDataLoader
import torch.utils.data
import torch.utils.data.distributed


class MyDummyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(),
        )

    def training_step(self, batch, batch_idx):
        print("data idx:", batch["idx"])
        x, y = batch["input"], batch["output"]
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat.squeeze(), y.squeeze())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self):
        # Load data from the directory
        self.data = []
        for _ in range(100):
            fake_input = torch.randn(1)
            fake_output = torch.randn(1)
            data = {"input": fake_input, "output": fake_output}
            self.data.append(data)

    def load_data(self, data_dir):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return data item at the specified index
        return self.data[idx]


def start():
    use_litdata = True
    if use_litdata:
        dataset = StreamingDataset("data")
        dataloader = StreamingDataLoader(
            dataset,
            batch_size=1,
            # num_workers=2,
            drop_last=True,
            # sampler = DistributedSampler(dataset, num_replicas=4, rank=0),
        )
    else:
        # Instantiate the dataset
        dataset = CustomDataset()

        # Define the DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            drop_last=True,  # Drop the last incomplete batch if the dataset size isn't divisible by batch size
        )

    model = MyDummyModel()
    trainer = L.Trainer(max_epochs=100, devices=4, accelerator="cpu", strategy="ddp")
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    start()
