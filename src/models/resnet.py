"""ResNet model implementation in PyTorch.

Defines the base ResNet architecture and utilities to instantiate variants
(ResNet-18, ResNet-50, etc.). Supports configurable number of classes and custom
convolution layers.
"""

from __future__ import annotations

from typing import Any, Optional, cast

import torch
from timm.models import register_model
from torch import nn

from models.layers import DeformableConv2d
from models.utils import initialize_weights

cfgs: dict[str, list[int]] = {
    "ResNet18": [2, 2, 2, 2],
    "ResNet34": [3, 4, 6, 3],
    "ResNet50": [3, 4, 6, 3],
    "ResNet101": [3, 4, 23, 3],
    "ResNet152": [3, 8, 36, 3],
}

supported_conv_layers = {
    "conv_layer_types": {
        "nn.Conv2d": nn.Conv2d,
        "DeformableConv2d": DeformableConv2d,
    },
}


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    conv_layer: type[nn.Module] = nn.Conv2d,
) -> nn.Module:
    """3x3 convolution with padding."""
    return conv_layer(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    conv_layer: type[nn.Module] = nn.Conv2d,
) -> nn.Module:
    """1x1 convolution."""
    return conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic ResNet block

    Attributes:
        expansion: Expansion factor for the block. Always 1.
        conv1: First 3x3 convolution layer.
        bn1: Batch normalization after the first conv layer.
        relu: ReLU activation function.
        conv2: Second 3x3 convolution layer.
        bn2: Batch normalization after the second conv layer.
        downsample: Optional downsampling layer for residual path.
        stride: Stride used in the first convolution layer.
    """

    expansion: int = 1

    conv1: nn.Module
    bn1: nn.BatchNorm2d
    relu: nn.ReLU
    conv2: nn.Module
    bn2: nn.BatchNorm2d
    downsample: Optional[nn.Module]
    stride: int

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        conv_layer: type[nn.Module] = nn.Conv2d,
    ):
        """Initialize BasicBlock."""
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, conv_layer)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv_layer=conv_layer)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for BasicBlock."""
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return cast(torch.Tensor, self.relu(out))


class ResNetBase(nn.Module):
    """
    Base class for ResNet architectures.

    Attributes:
        inplanes: Number of channels before entering a residual block.
        conv1: Initial 7x7 convolutional layer.
        bn1: Batch normalization after conv1.
        relu: ReLU activation function.
        maxpool: Max pooling after initial convolution and BN.
        layer1: First stack of residual blocks.
        layer2: Second stack of residual blocks.
        layer3: Third stack of residual blocks.
        layer4: Fourth stack of residual blocks.
        avgpool: Global average pooling layer.
        fc: Final fully connected layer for classification.
    """

    inplanes: int
    conv1: nn.Module
    bn1: nn.BatchNorm2d
    relu: nn.ReLU
    maxpool: nn.MaxPool2d
    layer1: nn.Sequential
    layer2: nn.Sequential
    layer3: nn.Sequential
    layer4: nn.Sequential
    avgpool: nn.AdaptiveAvgPool2d
    fc: nn.Linear

    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        num_classes: int = 1000,
        conv_layer: type[nn.Module] = nn.Conv2d,
        **kwargs: Any,
    ):
        """Initialize ResNetBase."""
        super().__init__()
        self.inplanes = 64
        self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], conv_layer=conv_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, conv_layer=conv_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, conv_layer=conv_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, conv_layer=conv_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        initialize_weights(self.modules)

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int = 1,
        conv_layer: type[nn.Module] = nn.Conv2d,
    ) -> nn.Sequential:
        """Create one ResNet layer composed of blocks."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, conv_layer),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, conv_layer)]
        self.inplanes = planes * block.expansion

        for _ in range(1, num_blocks):  # pylint: disable=W0612
            layers.append(block(self.inplanes, planes, conv_layer=conv_layer))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ResNetBase."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return cast(torch.Tensor, self.fc(x))


@register_model
def custom_resnet18(conv_layer_type: str, **kwargs: Any) -> ResNetBase:
    """Constructs a ResNet-18 model."""
    return ResNetBase(
        BasicBlock, cfgs["ResNet18"], conv_layer=supported_conv_layers["conv_layer_types"][conv_layer_type], **kwargs
    )


@register_model
def custom_resnet50(conv_layer_type: str, **kwargs: Any) -> ResNetBase:
    """Constructs a ResNet-50 model."""
    return ResNetBase(
        BasicBlock, cfgs["ResNet50"], conv_layer=supported_conv_layers["conv_layer_types"][conv_layer_type], **kwargs
    )


@register_model
def custom_resnet101(conv_layer_type: str, **kwargs: Any) -> ResNetBase:
    """Constructs a ResNet-101 model."""
    return ResNetBase(
        BasicBlock, cfgs["ResNet101"], conv_layer=supported_conv_layers["conv_layer_types"][conv_layer_type], **kwargs
    )


@register_model
def custom_resnet152(conv_layer_type: str, **kwargs: Any) -> ResNetBase:
    """Constructs a ResNet-152 model."""
    return ResNetBase(
        BasicBlock, cfgs["ResNet152"], conv_layer=supported_conv_layers["conv_layer_types"][conv_layer_type], **kwargs
    )
