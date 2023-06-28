
from tqdm import tqdm
from torch.utils.data import Dataset

class PreloadedDataset(Dataset):

    def __init__(self, 
                 dataset: Dataset, 
                 transforms=None,
                 ):
        self.transforms = transforms
        self.MULTILABEL = dataset.MULTILABEL
        self.NUM_LABELS = dataset.NUM_LABELS
        self.data = self.preload_data(dataset)

    def preload_data(self, dataset: Dataset, ) -> list:
        data = []
        print(f"Predloading dataset: {dataset.__class__.__name__}...")
        for _, (audio, label) in enumerate(tqdm(dataset)):
            data.append((audio, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        audio, label = self.data[index]
        if self.transforms:
            audio = self.transforms(audio)

        return audio, label