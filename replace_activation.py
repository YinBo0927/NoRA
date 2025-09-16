from nora import NoRA
from rational_group import Rational_Group1d, Rational_Group2d
import torch.nn as nn


def replace_activation(module: nn.Module, mode="gelu", dim=1):
    mapping = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU,
    }
    target_type = mapping.get(mode.lower())
    if target_type is None:
        raise ValueError(f"Unsupported mode: {mode}")

    RG = Rational_Group1d

    for name, child in module.named_children():
        if isinstance(child, target_type):
            setattr(module, name, RG(mode=mode))
        else:
            replace_activation(child, mode=mode, dim=dim)
    return module

def replace_activation_with_nora(module: nn.Module, target_types=(nn.GELU,), factory=lambda _: NoRA(groups=8, rank=3, lora_alpha=6, mode="gelu")):
    for name, child in module.named_children():
        if isinstance(child, target_types):
            setattr(module, name, factory(child))
        else:
            replace_activation_with_nora(child, target_types=target_types, factory=factory)
    return module