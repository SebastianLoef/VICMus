import torchaudio
from torch.utils.data import Dataset
from torch import Tensor, FloatTensor
import os
import pandas as pd
from typing import Tuple
from torchaudio.transforms import Resample

class MillionSongDataset(Dataset):
    MULTI_LABEL = True
    NUM_LABELS = 50
    def __init__(self,
                 subset: str = 'train',
                 root: str="data/processed/msd/audio/",
                 meta_path: str="data/processed/MSD/", 
                 sample_rate: int=22050,
                 transforms=None,
                 ) -> None:
        self.root = root
        self.meta_path = meta_path
        self.subset = subset
        self.sample_rate = sample_rate
        self.transforms = transforms
        print(f"{self.meta_path} {self.subset}")
        self._get_song_list()

    def __getitem__(self, index) -> Tuple[Tensor, FloatTensor]:
        audio_path = os.path.join(self.root, self.fl[index])
        audio, sr = torchaudio.load(audio_path, format='mp3')

        if sr != self.sample_rate:
            transform = Resample(sr, self.sample_rate)
            audio = transform(audio)

        audio = audio.mean(dim=0, keepdim=True)
        if self.transforms:
            audio = self.transforms(audio)
        return audio, FloatTensor(0)

    def _get_song_list(self):
        fl_path = os.path.join(self.meta_path, self.subset + '_filepaths.csv')
        df = pd.read_csv(fl_path, header=None)
        
        broken_fl_path = os.path.join(self.meta_path, 'broken_filepaths.csv')
        df_broken = pd.read_csv(broken_fl_path, header=None)

        # removes broken filepaths 
        df = (pd.merge(df, df_broken, on=[0,0], how='outer', indicator=True)
                .query("_merge == 'left_only'")
                .drop('_merge', axis=1)
                .reset_index(drop=True))

        self.fl = df[0].values
    
    def __len__(self):
        return len(self.fl)


if __name__ == '__main__':
    dataset = MillionSongDataset()
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][1].shape)
    print(dataset[0][1])
    print(dataset[0][1].dtype)
    print(dataset[0][0].dtype)
    print(dataset[0][0].shape)
    print(dataset[0][0].max())
    print(dataset[0][0].min())
    print(dataset[0][0].mean())
    print(dataset[0][0].std())
    print(dataset[0][0].median())
    print(dataset[0][0].var())
    print(dataset[0][0].sum())
