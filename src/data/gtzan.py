import os
import torchaudio
from torchaudio.datasets.gtzan import gtzan_genres
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from torch import Tensor
from typing import Tuple

class GTZAN(Dataset):
    NUM_LABELS = 50 ### ????
    MULTILABEL = True
    def __init__(self, 
                 subset: str,
                 root: str = "data/processed/gtzan", 
                 sample_rate: int=22050,
                 transforms=None,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.root = root
        self.subset = subset
        self.genres = gtzan_genres
        self.transforms = transforms
        self.sample_rate = sample_rate
        self.broken_GTZAN_train = [252]
        self._get_song_list()

    def _get_song_list(self):
        list_filename = os.path.join(self.root, f'{self.subset}_filtered.txt')
        with open(list_filename) as f:
            lines = f.readlines()

        if self.subset == 'train':
            lines = [ele for idx, ele in enumerate(lines) if idx not in self.broken_GTZAN_train]
        self.song_list = [line.strip() for line in lines]

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        line = self.song_list[index]
        
        # get label
        genre = line.split('/')[0]
        label = self.genres.index(genre)

        # get audio
        audio_file = os.path.join(self.root, 'genres_original', line)
        audio, sr = torchaudio.load(audio_file, format='wav')

        transform = Resample(sr, self.sample_rate)
        audio = transform(audio)
        
        if self.transforms:
            audio = self.transforms(audio)

        return audio, label
    
    def __len__(self):
        return len(self.song_list)