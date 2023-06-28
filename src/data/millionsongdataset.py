import torchaudio
from torch.utils.data import Dataset
from torch import Tensor, FloatTensor
import os
import numpy as np
from typing import Tuple
from torchaudio.transforms import Resample

class MillionSongDataset(Dataset):
    def __init__(self,
                 subset: str,
                 root: str="data/processed/msd/audio/",
                 meta_path: str="data/processed/MSD/", 
                 split: str="lastfm",
                 sample_rate: int=22050,
                 transforms=None,
                 **kwargs) -> None:
        self.root = root
        self.meta_path = meta_path
        self.subset = subset
        self.split = split
        self.sample_rate = sample_rate
        self.transforms = transforms
        self._get_song_list()
        if self.split == "lastfm":
            self._get_binaries()

    def __getitem__(self, index) -> Tuple[Tensor, FloatTensor]:
        audio_path = os.path.join(self.root, self.fl[index])
        audio, sr = torchaudio.load(audio_path, format='mp3')
        if sr != self.sample_rate:
            transform = Resample(sr, self.sample_rate)
            audio = transform(audio)
        
        binaries = FloatTensor(self.binaries[index])
        audio = audio.mean(dim=0, keepdim=True)

        if self.transforms:
            audio = self.transforms(audio)

        return audio, binaries

    def _get_song_list(self):
        fl_path = os.path.join(self.meta_path, self.subset + '_files.npy')
        self.fl = np.load(fl_path, allow_pickle=True)
    
    def _get_binaries(self):
        binaries_path = os.path.join(self.meta_path, self.subset + '_binaries.npy')
        self.binaries = np.load(binaries_path, allow_pickle=True)

    def __len__(self):
        return len(self.fl)