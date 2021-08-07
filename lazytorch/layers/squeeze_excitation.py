import torch
import torch.nn as nn
from lazytorch import LazyConv2dInChannelModule, create_lazy_signature


class SqueezeExcitation(nn.Module):
    """A Squeeze and Excitation module. References:
    - Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)"""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        mid_channels = self.in_channels // self.reduction_ratio
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, self.in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return x * self.layers(x)


@create_lazy_signature(exclude=["in_channels"])
class LazySqueezeExcitation(LazyConv2dInChannelModule, SqueezeExcitation):
    """Lazily-initialized SqueezeExcitation Module"""

    pass
