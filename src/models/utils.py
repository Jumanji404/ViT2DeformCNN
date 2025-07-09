"""Utility functions for model initialization."""

from typing import Callable, Iterator

from torch import nn


def initialize_weights(modules: Callable[[], Iterator[nn.Module]]) -> None:
    """Initialize weights for convolutional and linear layers using Kaiming Normal initialization."""
    for m in modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
