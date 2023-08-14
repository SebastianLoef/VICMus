import os
import re
import glob
import json
import names
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
from data.gtzan import GTZAN
from data.magnatagatune import MagnaTagATune
from data.millionsongdataset import MillionSongDataset
from data.freemusicarchive import FreeMusicArchive


def generate_encodings(args, module, dataset, subset, normalize=False):
    path = f"data/models/{args.name}/{args.dataset}"
    enc_path = os.path.join(path, f"{subset}_encodings.npy")
    label_path = os.path.join(path, f"{subset}_labels.npy")
    if os.path.exists(enc_path) and os.path.exists(label_path):
        print("Loading encodings from file")
        encodings = np.load(enc_path)
        labels = np.load(label_path)
        return encodings, labels
    print("Generating encodings")
    encodings = []
    labels = []
    module.eval().to("cuda")
    for x, y in tqdm(dataset):
        x = x.unsqueeze(0).to("cuda")
        y = y.unsqueeze(0)
        encoding = module(x)
        encodings.append(encoding.detach().cpu().numpy())
        labels.append(y.numpy())
    encodings = np.concatenate(encodings, axis=0)
    labels = np.concatenate(labels, axis=0)
    if normalize:
        encodings = encodings / np.linalg.norm(encodings, axis=1, keepdims=True)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(enc_path, encodings)
    np.save(label_path, labels)
    return encodings, labels


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_model_name() -> str:
    return names.get_full_name().replace(" ", "_").lower()


def get_model_number() -> int:
    return len(os.listdir("data/models/")) + 1


def get_best_metric_checkpoint_path(name: str, metric: str) -> str:
    path = f"data/models/{name}"

    model_names = glob.glob(f"data/models/{name}/*{metric}*")

    assert len(model_names) > 0, f"No models found with metric: {metric}"
    return path + model_names[0]


def get_epoch_checkpoint_path(name: str, epoch: int = 0) -> str:
    model_names = glob.glob(f"data/models/{name}/*epoch*")
    assert len(model_names) > 0, "No models found"
    d = {int(re.split("=|\.", model_name)[1]): model_name for model_name in model_names}
    if epoch:
        return d[min(d.keys(), key=lambda i: abs(i - epoch))]

    idx = max(d.keys())
    return d[idx]


def get_dataset(name: str):
    if name == "mtat":
        print("Using MagnaTagATune dataset")
        return MagnaTagATune
    elif name == "fma":
        print("Using FreeMusicArchive dataset")
        return FreeMusicArchive
    elif name == "gtzan":
        print("Using GTZAN dataset")
        return GTZAN
    elif name == "msd":
        print("Using MillionSongDataset dataset")
        return MillionSongDataset
    else:
        raise NotImplementedError


def save_parameters(args, name):
    if not os.path.exists(f"data/models/{name}"):
        os.makedirs(f"data/models/{name}")
    with open(f"data/models/{name}/parameters.json", "w") as f:
        json.dump(vars(args), f)


def load_parameters(name):
    with open(f"data/models/{name}/parameters.json", "r") as f:
        return SimpleNamespace(**json.load(f))


if __name__ == "__main__":
    path = get_epoch_checkpoint_path("irma_orourke-17")
    print(path)
