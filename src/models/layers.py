import math
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ConvBlock(
    in_channels,
    out_channels,
    kernel_size,
    padding='same',
    stride=1,
    dilation=1,
    activation='relu'
):
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU()
    }
    if activation not in activations:
        raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activations.keys())}")
    activation = activations[activation]
    
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        ),
        activation,
        nn.BatchNorm1d(out_channels),
    )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x