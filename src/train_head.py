import argparse

import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchaudio_augmentations import RandomResizedCrop
from torchvision.transforms import Compose
from transforms import MelSpectrogram
from architectures import resnet
from modules.Classifier import Classifier
from modules.VICReg import VICReg

from utils import (
    get_best_metric_checkpoint_path,
    get_dataset,
    get_epoch_checkpoint_path,
    get_model_number,
    load_parameters,
    class_balanced_sampler
)

from data.test_dataset import TestDataset


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

    name_linear = args.name + "-linear" + f"-{get_model_number()}-{metric}"
    backbone_args = load_parameters(args.name)
    ###########################
    # transforms
    ############################
    transforms = Compose(
        [
            RandomResizedCrop(n_samples=backbone_args.n_samples),
            MelSpectrogram(backbone_args),
        ]
    )
    ############################
    # dataset
    ############################
    dataset = get_dataset(args.dataset)
    train_dataset = dataset(subset="train", transforms=transforms)
    val_dataset = dataset(subset="valid", transforms=transforms)
    test_dataset = dataset(subset="test", transforms=None)
    test_dataset = TestDataset(backbone_args, test_dataset)
    if args.class_balanced:
        sampler = class_balanced_sampler(train_dataset)
        shuffle=False
    else:
        sampler = None
        shuffle=True
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    MULTILABELS = train_dataset.MULTILABEL
    NUM_LABELS = train_dataset.NUM_LABELS
    ############################
    # model
    ############################
    print(backbone_path)
    backbone_module = VICReg.load_from_checkpoint(
        backbone_path, args=backbone_args, backbone=resnet()
    )
    backbone = backbone_module.backbone.cpu()
    model = Classifier(args, MULTILABELS, NUM_LABELS, backbone)
    #
    ############################
    # logger
    ############################
    wandb_logger = WandbLogger(
        project="ICASSP2024",
        name=name_linear,
        entity="sebastianl",
        save_dir="data/logs",
        tags=["downstream"],
    )
    for k, v in args.__dict__.items():
        wandb_logger.experiment.config[k] = v
    for k, v in backbone_module.args.__dict__.items():
        wandb_logger.experiment.config[f"backbone_{k}"] = v
    wandb_logger.experiment.config["backbone_file"] = backbone_path.split("/")[-1]

    ############################
    # Checkpointing
    ############################
    checkpoint_callbacks = []
    for metric in ["val_loss"]:
        checkpoint_callbacks.append(
            ModelCheckpoint(
                monitor=metric,
                dirpath=f"data/models/{name_linear}",
                filename=f"classifier-best-{metric}",
                save_top_k=1,
                mode="min",
            )
        )
    checkpoint_callbacks.append(
        ModelCheckpoint(
            dirpath=f"data/models/{name_linear}",
            filename="classifier" + "-{epoch:02d}",
            save_top_k=-1,
        )
    )

    ############################
    # Training
    ############################
    trainer = L.Trainer(
        callbacks=checkpoint_callbacks,
        logger=wandb_logger,
        max_epochs=args.epochs,
        accelerator="gpu",
        precision=32,
        num_sanity_val_steps=0,
        devices=args.devices,
        check_val_every_n_epoch=args.check_val_every_n_epoch
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
