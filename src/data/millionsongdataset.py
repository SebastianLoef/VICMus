import os
from typing import Tuple

import pandas as pd
import torchaudio
from torch import FloatTensor, Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import Resample


class MillionSongDataset(Dataset):
    MULTILABEL = True
    NUM_LABELS = 50

    def __init__(
        self,
        subset: str = "train",
        root: str = "data/processed/msd/audio/",
        meta_path: str = "data/processed/MSD/",
        sample_rate: int = 22050,
        transforms=None,
        **kwargs
    ) -> None:
        super().__init__()
        self.root = root
        self.meta_path = meta_path
        self.subset = subset
        self.sample_rate = sample_rate
        self.transforms = transforms
        self.fl = self._get_song_list()

    def __getitem__(self, index) -> Tuple[Tensor, FloatTensor]:
        audio_path = os.path.join(self.root, self.fl.iloc[index][0])
        audio, sr = torchaudio.load(audio_path, format="mp3")

        if sr != self.sample_rate:
            transform = Resample(sr, self.sample_rate)
            audio = transform(audio)

        audio = audio.mean(dim=0, keepdim=True)
        if self.transforms:
            audio = self.transforms(audio)
        return audio, FloatTensor(0)

    def _get_song_list(self):
        fl_path = os.path.join(self.meta_path, self.subset + "_filepaths.csv")
        df = pd.read_csv(fl_path, header=None)

        broken_fl_path = os.path.join(self.meta_path, "broken_filepaths.csv")
        df_broken = pd.read_csv(broken_fl_path, header=None)

        # removes broken filepaths
        df = (
            pd.merge(df, df_broken, on=[0, 0], how="outer", indicator=True)
            .query("_merge == 'left_only'")
            .drop("_merge", axis=1)
            .reset_index(drop=True)
        )

        return df

    def __len__(self):
        return len(self.fl)


if __name__ == "__main__":
    train_dataset = MillionSongDataset(subset="train")
    print(len(train_dataset))
    print(train_dataset[55][0].shape)
    print(train_dataset[97][1].shape)
