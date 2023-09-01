import argparse

import lightning
import torch
from torch.utils.data import Dataset

from src.architectures import resnet
from src.modules.VICReg import VICReg


class RandomDataset(Dataset):
    def __init__(self, subset, mixing, transforms) -> None:
        self.transforms = transforms

    def __getitem__(self, index: int):
        x = torch.rand(3, 128, 128, requires_grad=True)
        y = torch.rand(3, 128, 128, requires_grad=True)
        z = torch.rand(1, requires_grad=True)
        return (x, y), z

    def __len__(self) -> int:
        return 10


def test_train():
    args = argparse.Namespace(
        batch_size=2,
        cov_coeff=1.0,
        devices=1,
        epochs=1,
        f_max=1.0,
        f_min=1.0,
        hop_length=1,
        mixing=False,
        n_fft=1,
        n_samples=2,
        normalize=False,
        num_workers=0,
        prefetch_factor=None,
        projector="2048-2-2048",
        sample_rate=3,
        sim_coeff=1.0,
        std_coeff=1.0,
        strategy="auto",
        weight_decay=1e-9,
        win_length=1,
        base_lr=1e-3,
    )

    backbone = resnet(pretrained=False)
    dataset = RandomDataset
    model = VICReg(args=args, dataset=dataset, backbone=backbone)

    trainer = lightning.Trainer(max_epochs=1)
    trainer.fit(model)

    assert trainer.state.finished
