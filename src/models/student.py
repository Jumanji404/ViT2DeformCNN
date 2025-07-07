"""Definition of the student model using deformable convolutional layers."""

from __future__ import annotations

from typing import Any, cast

import torch
from timm.models import register_model
from torch import nn

from models.layers import DeformableConv2d


class DummyModel(nn.Module):
    """A simple model composed of deformable convolutional layers."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any):
        """
        Initialize the DummyModel.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__()
        self.layers = nn.Sequential(
            DeformableConv2d(in_channels=in_channels, out_channels=20),
            nn.ReLU(),
            DeformableConv2d(in_channels=20, out_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DummyModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the layers.
        """
        return cast(torch.Tensor, self.layers(x))


@register_model
def dummy_model(**kwargs: Any) -> DummyModel:
    """
    Factory function to register and return an instance of DummyModel.

    Args:
        **kwargs: Keyword arguments passed to DummyModel.

    Returns:
        DummyModel: An instance of the dummy model.
    """
    return DummyModel(**kwargs)
