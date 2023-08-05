import argparse

import yaml
import lightning as L

from utils import save_parameters
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from modules.VICReg import VICReg
from utils import get_model_name, get_model_number
from architectures import resnet


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")

    args, _ = parser.parse_known_args()
    # add arguments from yaml file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config["vicreg"].items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    for k, v in config["general"].items():
        parser.add_argument(f"--{k}", type=type(v), default=v)

    args = parser.parse_args()
    return args


def main(args):
    name = get_model_name() + f"-{get_model_number()}"
    save_parameters(args, name)

    ############################
    # Logging
    ############################
    wandb_logger = WandbLogger(
        project="ICASSP2024", entity="sebastianl", name=name, save_dir="data/logs"
    )
    wandb_logger.experiment.config.update(args.__dict__, allow_val_change=True)
    if args.devices > 1:
        args.batch_size = int(args.batch_size / args.devices)
    ############################
    # model
    ############################
    backbone = resnet(args.pretrained)
    model = VICReg(args, backbone)
    checkpoint = ModelCheckpoint(
            dirpath=f"data/models/{name}",
            filename="vicreg-{epoch:02d}",
            save_top_k=-1,
            every_n_epochs=50,
            save_weights_only=True,
        )
    ############################
    # Training
    ############################
    trainer = L.Trainer(
        callbacks=[checkpoint],
        logger=wandb_logger,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        num_sanity_val_steps=0,
        log_every_n_steps=3,
        strategy=args.strategy,
    )
    trainer.fit(
        model,
    )


if __name__ == "__main__":
    args = get_arguments()
    main(args)
