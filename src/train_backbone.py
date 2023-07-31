import argparse

import yaml
import lightning as L

from utils import save_parameters
from lightning.pytorch.callbacks import ModelCheckpoint
#from lightning.pytorch.loggers import WandbLogger
from modules.VICReg import VICReg
from utils import get_model_name, get_model_number
from architectures import resnet


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--loss", type=str, default="vicreg")
    parser.add_argument("--encoder_train_dataset", type=str, default="freemusicarchive")
    parser.add_argument("--encoder_val_dataset", type=str, default="magnatagatune")

    args, _ = parser.parse_known_args()
    # add arguments from yaml file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config[args.loss].items():
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
    #wandb_logger = WandbLogger(project=args.loss, name=name, save_dir="data/logs")
    #wandb_logger.experiment.config.update(args.__dict__, allow_val_change=True)
    if args.devices > 1:
        args.batch_size = int(args.batch_size / args.devices)
    ############################
    # model
    ############################
    backbone = resnet()
    model = VICReg(backbone, args)
    ############################
    # Checkpointing
    ############################
    checkpoint_callbacks = []
    if args.loss == "vicreg":
        checkpoint_metrics = [
            "val_loss",
            "train_loss",
            "train_invariance_loss",
            "train_variance_loss",
            "train_covariance_loss",
            "val_invariance_loss",
            "val_variance_loss",
            "val_covariance_loss",
        ]
    else:
        checkpoint_metrics = ["val_loss", "train_loss"]

    for metric in checkpoint_metrics:
        checkpoint_callbacks.append(
            ModelCheckpoint(
                monitor=metric,
                dirpath=f"data/models/{name}",
                filename=f"{args.loss}-best-{metric}",
                save_top_k=1,
                mode="min",
                save_weights_only=True,
            )
        )
    checkpoint_callbacks.append(
        ModelCheckpoint(
            dirpath=f"data/models/{name}",
            filename=f"{args.loss}" + "-{epoch:02d}",
            save_top_k=-1,
            every_n_epochs=100,
            save_weights_only=True,
        )
    )

    ############################
    # Training
    ############################
    compiled_model = model #torch.compile(model, backend='torchxla_trace_once')
    trainer = L.Trainer(
        #callbacks=checkpoint_callbacks,
        #logger=wandb_logger,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        strategy=args.strategy,
    )
    trainer.fit(
        compiled_model,
    )


if __name__ == "__main__":
    args = get_arguments()
    main(args)
