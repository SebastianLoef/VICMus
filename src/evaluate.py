import argparse

import lightning as L
import numpy as np
import yaml
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeRegressor

from architectures import resnet
from modules.VICReg import VICReg
from transforms import MelSpectrogram
from utils import (
    generate_encodings,
    get_best_metric_checkpoint_path,
    get_dataset,
    get_epoch_checkpoint_path,
    load_parameters,
)


def fit_model(args, module, train_dataset):
    # fit linear model using scikit
    encodings, labels = generate_encodings(args, module, train_dataset, "train")
    print(encodings.shape, labels.shape)
    model = OneVsRestClassifier(SGDRegressor())
    print("Fit model")
    model.fit(encodings, labels)
    return model


def evaluate_model(args, module, model, dataset, subset):
    encodings, labels = generate_encodings(args, module, dataset, subset)
    preds = model.predict(encodings)
    print(
        preds.shape,
        labels.shape,
    )
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
    preds, labels = evaluate_model(args, module, model, val_dataset, "val")
    print_results(preds, labels)
    print("Evaluate model on test set")
    preds, labels = evaluate_model(args, module, model, test_dataset, "test")
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
        backbone_path = get_epoch_checkpoint_path(args.name)
    elif "epoch_" in args.best_metric:
        args.best_metric
        epoch = int(args.best_metric.split("_")[-1])
        backbone_path = get_epoch_checkpoint_path(args.name, epoch)
    else:
        args.best_metric
        backbone_path = get_best_metric_checkpoint_path(args.name, args.best_metric)

    backbone_args = load_parameters(args.name)
    ############################
    # datasets
    ############################
    transforms = MelSpectrogram(backbone_args)
    train_dataset = get_dataset(args.train_dataset)(
        subset="train", transforms=transforms
    )
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
