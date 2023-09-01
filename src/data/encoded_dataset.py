from typing import Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import Compose

from src.transforms import MelSpectrogram, RandomResizedCrop
from src.utils import generate_encodings


class EncodedDataset(nn.Module):
    def __init__(
        self,
        module: L.LightningModule,
        args,
        subset: str,
        transforms: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encodings, self.labels = self.generate_dataset(args, module, subset)
        self.transforms = None

    def get_integral_dataset(
        self, args, backbone_args, subset: str
    ) -> Tuple[Tensor, Tensor]:
        transforms = Compose(
            [
                RandomResizedCrop(n_samples=args.n_samples),
                MelSpectrogram(backbone_args),
            ]
        )
        dataset = DATASETS[args.dataset](subset=subset, transforms=transforms)
        self.MULTILABEL = dataset.MULTILABEL
        self.NUM_LABELS = dataset.NUM_LABELS
        return dataset

    def generate_dataset(
        self, args, module: L.LightningModule, subset: str
    ) -> Tuple[Tensor, Tensor]:
        dataset = self.get_integral_dataset(args, module.args, subset)
        encodings, labels = generate_encodings(args, module, dataset, subset)
        encodings = torch.from_numpy(encodings)
        labels = torch.from_numpy(labels)
        return encodings, labels

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if index >= len(self):
            raise IndexError
        encoding, label = self.encodings[index], self.labels[index]
        if self.transforms:
            encoding = self.transforms(encoding)
        return encoding, label

    def __len__(self) -> int:
        return len(self.encodings)
