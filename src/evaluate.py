import argparse

import yaml
import lightning as L
from transforms import MelSpectrogram
from architectures import resnet
from modules.VICReg import VICReg

from utils import (
    get_best_metric_checkpoint_path,
    get_dataset,
    get_epoch_checkpoint_path,
    load_parameters,
)

from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from tqdm import tqdm

def generate_encodings(args, module, dataset, normalize=True):
    encodings = []
    labels = []
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
    return encodings, labels

def fit_model(args, module, train_dataset):
    # fit linear model using scikit
    print("Generate training encodings")
    encodings, labels = generate_encodings(args, module, train_dataset)
    print(encodings.shape, labels.shape)
    model = LinearRegression()
    print("Fit model")
    model.fit(encodings, labels)
    return model

def evaluate_model(args, module, model, dataset):
    encodings, labels = generate_encodings(args, module, dataset)
    preds = model.predict(encodings)
    print(preds.shape, labels.shape)
    return preds, labels

def print_results(preds, labels):
    mean_ap = metrics.average_precision_score(labels, preds, average="macro")
    roc_auc = metrics.roc_auc_score(labels, preds, average="macro")
    print(f"mAP: {mean_ap}")
    print(f"ROC AUC: {roc_auc}")

def run(args, module, train_dataset, val_dataset, test_dataset):
    print("Fit model")
    model = fit_model(args, module, train_dataset)
    print("Evaluate model on validation set")
    preds, labels = evaluate_model(args, module, model, val_dataset)
    print_results(preds, labels)
    print("Evaluate model on test set")
    preds, labels = evaluate_model(args, module, model, test_dataset)
    print_results(preds, labels)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--best_metric", type=str, default=None)
    parser.add_argument("--multilabel", type=bool, default=True)
    args, _ = parser.parse_known_args()
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config["evaluation"].items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    for k, v in config["general"].items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    return args


def main(args):
    if args.best_metric is None:
        metric = "last_epoch"
        backbone_path = get_epoch_checkpoint_path(args.name)
    elif "epoch_" in args.best_metric:
        metric = args.best_metric
        epoch = int(args.best_metric.split("_")[-1])
        backbone_path = get_epoch_checkpoint_path(args.name, epoch)
    else:
        metric = args.best_metric
        backbone_path = get_best_metric_checkpoint_path(args.name, args.best_metric)

    backbone_args = load_parameters(args.name)
    ############################
    # datasets
    ############################
    transforms = MelSpectrogram(backbone_args)
    train_dataset = get_dataset(args.train_dataset)(subset="train", transforms=transforms)
    val_dataset = get_dataset(args.val_dataset)(subset="valid", transforms=transforms)
    test_dataset = get_dataset(args.test_dataset)(subset="test", transforms=transforms)

    ############################
    # model
    ############################
    module = VICReg.load_from_checkpoint(
        backbone_path, args=backbone_args, backbone=resnet()
    )
    module.freeze()
    module.eval()
    module.to("cuda")
    run(args, module, train_dataset, val_dataset, test_dataset)

if __name__ == "__main__":
    args = get_arguments()
    main(args)
