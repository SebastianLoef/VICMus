from typing import Tuple, Union

import torch
import torch.nn as nn
import torchaudio.transforms as transforms
from torchaudio_augmentations import (
    Compose,
    Delay,
    Gain,
    Noise,
    PitchShift,
    Reverb,
    PolarityInversion,
    RandomApply,
    RandomResizedCrop,
    HighLowPass,
)


def get_transforms(
    num_samples: int = 59049,
    sample_rate: int = 22050,
    win_length: int = 250,
    hop_length: int = 128,
):
    transform = Compose(
        [
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise()], p=0.01),
            RandomApply([Gain()], p=0.3),
            RandomApply([HighLowPass(sample_rate=sample_rate)], p=0.8),
            RandomApply([Delay(sample_rate=sample_rate)], p=0.3),
            RandomApply(
                [PitchShift(n_samples=num_samples, sample_rate=sample_rate)],
                p=0.6,
            ),
            RandomApply([Reverb(sample_rate=sample_rate)], p=0.6),
            transforms.MelSpectrogram(
                sample_rate=sample_rate,
            ),
        ]
    )
    return transform


class AudioSplit(nn.Module):
    def __init__(
        self,
        normal: bool = False,
        return_idxs: bool = False,
        transforms: Union[Compose, None] = get_transforms(),
        k: int = 27,
        n_samples: int = 59049,
    ):
        super().__init__()
        self.split = RandomResizedCrop(n_samples=n_samples)
        self.normal = normal
        self.transforms = transforms
        self.return_idxs = return_idxs

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform1 = self.split(waveform)
        waveform2 = self.split(waveform)
        if self.transforms is not None:
            melspec1 = self.transforms(waveform1)
            melspec2 = self.transforms(waveform2)
        return melspec1, melspec2


if __name__ == "__main__":
    import sys

    sys.path.append("src/")
    from torch.utils.data import DataLoader

    from data.magnatagatune import MagnaTagATune

    audio_split = AudioSplit()
    dataset = MagnaTagATune("train", transforms=audio_split)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)
    for batch in dataloader:
        (x, y), label = batch
        print(f"x: max: {x.max()}, min: {x.min()}, mean: {x.mean()}, std: {x.std()}")
        print(f"y: max: {y.max()}, min: {y.min()}, mean: {y.mean()}, std: {y.std()}")
