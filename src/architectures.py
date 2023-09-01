import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def resnet(pretrained: bool = False, **kwargs):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    print(f"Loading ResNet50 with weights: {weights}")
    model = resnet50(weights=weights, **kwargs)
    model.fc = nn.Identity()
    return model


def mlp(mlp_spec, norm: bool = True, bias: bool = True):
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1], bias=bias))
        if norm:
            layers.append(nn.BatchNorm1d(f[i + 1]))  # Have check this out later
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)
