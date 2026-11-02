"""Dependency shims; preserve fvcore's Caffe2 Xavier initialization semantics."""
import math
import torch
from torch import nn


def c2_xavier_fill(module):
    # fvcore.nn.weight_init.c2_xavier_fill uses this exact initialization.
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def autocast(enabled=False):
    # Upstream uses torch.cuda.amp.autocast(enabled=False) for float32 attention.
    return torch.autocast(device_type="cuda", enabled=enabled)
