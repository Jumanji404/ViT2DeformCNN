"""Implementation of a Deformable Convolutional Layer using PyTorch and torchvision."""

from __future__ import annotations

from typing import cast

import torch
import torchvision.ops
from torch import nn

from models.layers.utils import to_tuple


class DeformableConv2d(nn.Module):
    """
    A PyTorch implementation of a Deformable Convolutional layer, which enhances spatial modeling
    by learning offsets and modulation masks.

    Attributes:
        stride (tuple[int, int]): Stride of the convolution.
        padding (tuple[int, int]): Padding of the convolution.
        dilation (tuple[int, int]): Dilation of the convolution.
        offset_conv (nn.Conv2d): Learns spatial offsets.
        modulator_conv (nn.Conv2d): Learns modulation masks.
        regular_conv (nn.Conv2d): Standard convolutional layer.
    """

    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    offset_conv: nn.Conv2d
    modulator_conv: nn.Conv2d
    regular_conv: nn.Conv2d

    def __init__(  # pylint: disable=[R0913,R0917]
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        bias: bool = False,
    ) -> None:
        """
        Initializes the DeformableConv2d layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolutional kernel size.
            stride: Stride of the convolution.
            padding: Padding for input.
            dilation: Dilation rate.
            bias: If True, includes bias in the convolution.
        """
        super().__init__()

        kernel_size = to_tuple(kernel_size)
        self.stride = to_tuple(stride)
        self.padding = to_tuple(padding)
        self.dilation = to_tuple(dilation)

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)  # type: ignore[arg-type]

        self.modulator_conv = nn.Conv2d(
            in_channels,
            kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)  # type: ignore[arg-type]

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs deformable convolution on input tensor.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Output after deformable convolution.
        """
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

        out = cast(
            torch.Tensor,
            torchvision.ops.deform_conv2d(
                input=x,
                offset=offset,
                weight=self.regular_conv.weight,
                bias=self.regular_conv.bias,
                padding=self.padding,
                mask=modulator,
                stride=self.stride,
                dilation=self.dilation,
            ),
        )

        return out
