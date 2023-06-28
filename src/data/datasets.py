
import torch
import torchaudio
from torchaudio.datasets.gtzan import gtzan_genres
from torch.utils.data import Dataset
import pandas as pd
import os

class MagnaTagaTune(Dataset):
    def __init__(
            self, 
            transforms=None,
            data_path: str = 'data/processed/MagnaTagATune') -> None:
        super().__init__()
        self.data_path = data_path
        self.meta_data = pd.read_csv(
            data_path + '/clip_info_final.csv',
            sep='\t')
        self.meta_data = self.meta_data[['clip_id','mp3_path']]

        tags = pd.read_csv(
            data_path + '/top_50_tags.csv')
        
        self.meta_data = pd.merge(self.meta_data, tags, on='clip_id', how='right')
        self.meta_data.dropna(subset=['mp3_path'], inplace=True)
        self.meta_data.reset_index(drop=True, inplace=True)
        self.meta_data["mp3_path"] = self.meta_data["mp3_path"].apply( 
            lambda x: os.path.join(data_path, x))
        # remove the row containing the mp3_path f/american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.mp3
        self.meta_data = self.meta_data[self.meta_data["mp3_path"].str.contains("american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.mp3") == False]
        print(self.__len__())
        # remove indices 16249, 24866 and 25545
        self.meta_data = self.meta_data.drop([16248, 16249, 24865, 24866, 25545])
        self.meta_data.reset_index(drop=True , inplace=True)
        self.meta_data = self.meta_data.drop([16247, 24862])
        self.meta_data.reset_index(drop=True, inplace=True)
        self.meta_data = self.meta_data.drop([25538])
        self.meta_data.reset_index(drop=True, inplace=True)



        self.transforms = transforms
        

    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, index: int):
        path = self.meta_data.iloc[index][['mp3_path']].values[0]
        tags = self.meta_data.iloc[index, 2:].values.astype(float)
        waveform, sample_rate =  torchaudio.load(path, format="mp3")
        #print(f"Error loading {path}, for idx {index}")
        #return None, torch.Tensor(tags)
        assert sample_rate == 16000, f"Sample rate is not 16000, it is {sample_rate}"
        if self.transforms:
            waveform = self.transforms(waveform)
        return waveform, torch.Tensor(tags)
    

class GTZAN(Dataset):

    subset_map = {"train": "training", "valid": "validation", "test": "testing"}

    def __init__(self, root, download, subset):
        self.dataset = torchaudio.datasets.GTZAN(
            root=root, download=download, subset=self.subset_map[subset]
        )
        self.labels = gtzan_genres

        self.label2idx = {}
        for idx, label in enumerate(self.labels):
            self.label2idx[label] = idx

        self.n_classes = len(self.label2idx.keys())

    def __getitem__(self, idx):
        audio, sr, label = self.dataset[idx]
        label = self.label2idx[label]
        return audio, label

    def __len__(self):
        return len(self.dataset)
        

if __name__ == '__main__':
    mtt = MagnaTagaTune()
    #gtzan = GTZAN('.', True, 'train')
    for v in mtt:
        v