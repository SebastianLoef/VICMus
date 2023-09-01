import os
from typing import Tuple

import pandas as pd
import torchaudio
from torch import FloatTensor, Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import Resample


class MagnaTagATune(Dataset):
    NUM_LABELS = 50
    MULTILABEL = True
    SIZE = 25_863

    def __init__(
        self,
        subset: str,
        root: str = "data/processed/MagnaTagATune/",
        sample_rate: int = 22050,
        transforms=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.subset = subset
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.binary = self._get_binaries()

        self._get_song_list()
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.fl = self.fl.reset_index(drop=True)

    def __getitem__(self, index) -> Tuple[Tensor, FloatTensor]:
        if index >= len(self):
            raise IndexError
        audio, sr, binary_label = self.get_audio(index)
        binary_label = FloatTensor(binary_label)

        transform = Resample(sr, self.sample_rate)
        audio = transform(audio)

        if self.transforms:
            audio = self.transforms(audio)
        return audio, binary_label

    def _get_song_list(self):
        assert self.subset in [
            "train",
            "test",
            "valid",
        ], "Split should be one of [train, valid, test]"
        pons_path = os.path.join(self.root, "index_mtt.tsv")
        self.fl = pd.read_csv(pons_path, header=None, sep="\t")
        # filter out songs that are not in the binary file
        self.fl = self.fl.loc[self.fl[0].isin(self.binary.keys())]

    def _get_binaries(self):
        subset_map = {
            "train": "train_gt_mtt.tsv",
            "valid": "val_gt_mtt.tsv",
            "test": "test_gt_mtt.tsv",
        }

        tsv_path = os.path.join(self.root, subset_map[self.subset])
        df = pd.read_csv(tsv_path, sep="\t", header=None, index_col=0)
        df[1] = df.apply(lambda x: eval(x[1]), axis=1)
        return df.to_dict()[1]

    def get_audio(self, index):
        ix, fn = self.fl.iloc[index]
        audio_path = os.path.join(self.root, fn)
        audio, sr = torchaudio.load(audio_path, format="mp3")
        label_binary = self.binary[int(ix)]
        return audio, sr, label_binary

    def __len__(self):
        return len(self.fl)
