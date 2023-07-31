from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def resnet(**kwargs):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, **kwargs)
    #model.fc = Identity() 
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
