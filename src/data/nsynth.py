import json
import os
import shutil
from typing import Tuple

import requests
import torch
import torchaudio
from torch import FloatTensor, Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import Resample


class NSynth(Dataset):
    NUM_LABELS = 50
    MULTILABEL = False

    def __init__(
        self,
        subset: str,
        root: str = "data/",
        sample_rate: int = 22050,
        transforms=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.subset = subset
        self.transforms = transforms
        self.sample_rate = sample_rate
        data_folder = os.path.join(root, "processed", "nsynth", f"nsynth-{self.subset}")
        if not os.path.exists(data_folder):
            self.download()
        self.file_paths, self.labels = self.load_data(data_folder)

    @property
    def _label(self) -> str:
        pass

    def __getitem__(self, index) -> Tuple[Tensor, FloatTensor]:
        if index >= len(self):
            raise IndexError
        file_path = self.file_paths[index]
        audio, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.sample_rate:
            resample = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            audio = resample(audio)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if self.transforms:
            audio = self.transforms(audio)
        return audio, torch.tensor(self.labels[index])

    def load_data(self, data_folder: str) -> dict:
        with open(os.path.join(data_folder, "examples.json")) as f:
            data = json.load(f)
        file_paths = []
        labels = []
        for key in data.keys():
            file_paths.append(os.path.join(data_folder, "audio", f"{key}.wav"))
            labels.append(data[key][self._label])
        return file_paths, labels

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        url = f"http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{self.subset}.jsonwav.tar.gz"
        download_and_extract(url, self.root, "nsynth")

    def __len__(self):
        return len(self.file_paths)


class NSynthPitch(NSynth):
    NUM_LABELS = 128

    @property
    def _label(self) -> str:
        return "pitch"


class NSynthInstrument(NSynth):
    NUM_LABELS = 11

    @property
    def _label(self) -> str:
        return "instrument_family"


def extract(file_path, target_folder):
    """Extracts a zip file to a target folder and removes zip file."""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    shutil.unpack_archive(file_path, target_folder)


def download_url(url, target_folder):
    # download into raw folder
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    filename = url.split("/")[-1]
    file_path = os.path.join(target_folder, filename)
    r = requests.get(url, stream=True)
    with open(file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return file_path


def download_and_extract(url, root, name):
    print(f"Downloading from {url}")
    download_path = os.path.join(root, "raw", name)
    file_path = download_url(url, download_path)

    print(f"Extracting {name}")
    processed_path = os.path.join(root, "processed", name)
    extract(file_path, processed_path)

    print(f"Cleaning up {name}")
    os.remove(file_path)
