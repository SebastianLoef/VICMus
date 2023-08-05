import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transforms import MelSpectrogram
from typing import Tuple


class TestDataset(Dataset):
    def __init__(self, args, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.n_samples = args.n_samples
        self.melspec = MelSpectrogram(args)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, label = self.dataset[index]
        batch = torch.split(audio, self.n_samples, dim=1)
        batch = torch.cat(batch[:-1], dim=0)
        batch = batch.unsqueeze(dim=1)
        batch = self.melspec(batch)
        return batch, label

    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == "__main__":
    from magnatagatune import MagnaTagATune
    from torch.utils.data import DataLoader

    dataset = MagnaTagATune("test")
    tdataset = TestDataset(dataset)
    loader = DataLoader(tdataset, batch_size=1, shuffle=False)

    for batch, label in loader:
        print(batch.shape)

    print(dataset[0][0].shape)
