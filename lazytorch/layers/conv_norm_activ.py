import torch.nn as nn
from lazytorch import LazyConv2dInChannelModule, NamedSequential, create_lazy_signature
from typing import Type


class ConvNormActivation(NamedSequential):
    """2d Convolution followed by normalization and an activation function"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        bias: bool = False,
        **kwargs
    ):
        if "padding" not in kwargs:
            kwargs["padding"] = kernel_size // 2

        super().__init__(
            conv=nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, bias=bias, **kwargs
            ),
            norm=norm_layer(out_channels),
            act=activation(),
        )


@create_lazy_signature(exclude=["in_channels"])
class LazyConvNormActivation(LazyConv2dInChannelModule, ConvNormActivation):
    """Lazily-initialized ConvNormActivation module"""

    pass
