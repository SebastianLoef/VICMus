import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Tuple


class TestDataset(Dataset):
    def __init__(self, dataset: Dataset, audio_length: int = 59049) -> None:
        super().__init__()
        self.dataset = dataset
        self.audio_length = audio_length

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, label = self.dataset[index]
        batch = torch.split(audio, self.audio_length, dim=1)
        batch = torch.cat(batch[:-1], dim=0)
        batch = batch.unsqueeze(dim=1)
        return batch, label

if __name__ == "__main__":
    from magnatagatune import MagnaTagATune
    from torch.utils.data import DataLoader

    dataset = MagnaTagATune("test")
    tdataset = TestDataset(dataset)
    loader = DataLoader(tdataset, batch_size=256, shuffle=False)

    for batch, label in loader:
        print(batch.shape)

    print(dataset[0][0].shape)
