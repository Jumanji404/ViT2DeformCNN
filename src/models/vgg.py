"""VGG model implementation in PyTorch.

This module defines a base VGG architecture and provides functions to create
variants like VGG11, VGG13, VGG16, and VGG19. Supports configurable number of
output classes and custom convolution layers.
"""

from __future__ import annotations

from typing import Any

import torch
from timm.models import register_model
from torch import Tensor, nn

from models.layers import DeformableConv2d
from models.utils import initialize_weights

cfgs: dict[str, list[int | str]] = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

supported_conv_layers = {
    "conv_layer_types": {
        "nn.Conv2d": nn.Conv2d,
        "DeformableConv2d": DeformableConv2d,
    },
}


class VGGBase(nn.Module):
    """Base class for VGG models."""

    features: nn.Sequential
    avgpool: nn.AdaptiveAvgPool2d

    def __init__(
        self,
        features: nn.Sequential,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        initialize_weights(self.modules)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def make_layers(cfg: list[int | str], conv_layer: type[nn.Module] = nn.Conv2d) -> nn.Sequential:
    """Builds a sequential layer stack based on configuration."""
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if isinstance(v, int):
            layers.append(conv_layer(in_channels, v, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)


@register_model
def custom_vgg11(conv_layer_type: str, **kwargs: dict[str, Any]) -> VGGBase:
    """Constructs a VGG11 model."""
    return VGGBase(make_layers(cfgs["VGG11"], supported_conv_layers["conv_layer_types"][conv_layer_type]), **kwargs)


@register_model
def custom_vgg13(conv_layer_type: str, **kwargs: dict[str, Any]) -> VGGBase:
    """Constructs a VGG13 model."""
    return VGGBase(make_layers(cfgs["VGG13"], supported_conv_layers["conv_layer_types"][conv_layer_type]), **kwargs)


@register_model
def custom_vgg16(conv_layer_type: str, **kwargs: dict[str, Any]) -> VGGBase:
    """Constructs a VGG16 model."""
    return VGGBase(make_layers(cfgs["VGG16"], supported_conv_layers["conv_layer_types"][conv_layer_type]), **kwargs)


@register_model
def custom_vgg19(conv_layer_type: str, **kwargs: dict[str, Any]) -> VGGBase:
    """Constructs a VGG19 model."""
    return VGGBase(make_layers(cfgs["VGG19"], supported_conv_layers["conv_layer_types"][conv_layer_type]), **kwargs)
