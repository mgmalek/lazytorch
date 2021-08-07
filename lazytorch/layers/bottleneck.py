import torch
import torch.nn as nn
from lazytorch import LazyConv2dInChannelModule, NamedSequential, create_lazy_signature
from .conv_norm_activ import ConvNormActivation
from typing import Optional


class BottleneckBlock(nn.Module):
    """A bottleneck block. References:
    - Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)"""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        activation: Optional[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.stride = stride
        bottleneck_padding = (kernel_size + 2 * (dilation - 1)) // 2
        self.layers = NamedSequential(
            conv1=ConvNormActivation(
                in_channels, mid_channels, kernel_size=1, activation=activation
            ),
            conv2=ConvNormActivation(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=bottleneck_padding,
                groups=groups,
                dilation=dilation,
                activation=activation,
            ),
            conv3=ConvNormActivation(
                mid_channels, out_channels, kernel_size=1, activation=nn.Identity
            ),
        )

        self.proj = nn.Identity()
        if self.stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.activ = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.proj(x) + self.layers(x))


@create_lazy_signature(exclude=["in_channels"])
class LazyBottleneckBlock(LazyConv2dInChannelModule, BottleneckBlock):
    """Lazily-initialized BottleneckBlock module"""

    pass
