import torch
import torch.nn as nn
from lazytorch import (
    LazyConv2dInChannelModule,
    create_lazy_signature,
    NamedSequential,
)
from .depth_sep_conv import DepthwiseConv2d, PointwiseConv2d
from .squeeze_excitation import SqueezeExcitation
from typing import Optional


class InvertedBottleneck(nn.Module):
    """An inverted bottleneck block with optional squeeze-and-excitiation
    layer. References:
    - MobileNetV2 (https://arxiv.org/abs/1801.04381)
    - MnasNet (https://arxiv.org/abs/1807.11626)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion_ratio: int = 1,
        use_se: bool = False,
        se_reduction_ratio: Optional[int] = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        mid_channels = in_channels * expansion_ratio

        self.layers = NamedSequential(
            pw=PointwiseConv2d(
                in_channels,
                mid_channels,
                norm_layer=norm_layer,
                activation=activation,
            ),
            dw=DepthwiseConv2d(
                mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation=activation,
            ),
            se=nn.Identity(),
            bottleneck=nn.Conv2d(mid_channels, out_channels, 1),
        )

        if use_se:
            self.layers.se = SqueezeExcitation(
                mid_channels, reduction_ratio=se_reduction_ratio
            )

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        if x.shape == out.shape:
            out += x

        return out


@create_lazy_signature(exclude=["in_channels"])
class LazyInvertedBottleneck(LazyConv2dInChannelModule, InvertedBottleneck):
    """Lazily-initialized InvertedBottleneck module"""
