import os
import random
from typing import Tuple

import pandas as pd
import random

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
import numpy as np 

class FreeMusicArchive(Dataset):
    NUM_LABELS = None
    MULTILABEL = None
    def __init__(self, 
                 subset=None,
                 root: str = "data/processed/FMA/fma_medium/", 
                 sample_rate: int=22050,
                 transforms=None,
                 mixing=False,
                 **kwargs) -> None:
        super().__init__()
        self.root = root
        self.sample_rate = sample_rate
        self.transforms = transforms
        self.mixing = mixing
        self.subset = subset
        # Broken files to be removed :(
        self.broken_FMA = [316, 977, 10675, 13146, 15626, 
                           15627, 15628, 15629, 15630, 15631,
                           15632, 15633, 15634, 15836, 16305, 
                           16959, 20621, 20780, 21988, 23620]
        self.df = self._get_song_list()
        self.df["mp3_path"] = self.df.mp3_path.apply(lambda x: os.path.join(self.root, x))


    def _load_audio(self, index):
        audio_path, _ = self.df.iloc[index, 1:]
        audio, sr = torchaudio.load(audio_path, format='mp3')
        if sr != self.sample_rate:
            transform = Resample(sr, self.sample_rate)
            audio = transform(audio)
        
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
        return audio
    def __getitem__(self, index) -> Tuple[Tensor, int]:
        audio = self._load_audio(index)
        if self.mixing:
            new_idx = random.randint(0, len(self))
            new_audio = self._load_audio(index)
            p = np.random.beta(5, 2)
            audio = audio * p + new_audio * (1 - p)

        if self.transforms:
            audio = self.transforms(audio)
        return audio, Tensor(1)

    def _get_song_list(self):
        print(os.path.join(self.root, "metadata.csv"))
        df = pd.read_csv(os.path.join(self.root, "metadata.csv"), sep=',')
        df.drop(index=self.broken_FMA, inplace=True)
        return df
    
    def __len__(self):
        return len(self.df)
    
if __name__ == "__main__":
    dataset = FreeMusicArchive()
    print(dataset[0][0].shape)