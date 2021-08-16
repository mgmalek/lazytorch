import torch.nn as nn
from lazytorch import (
    LazyConv2dInChannelModule,
    LazyConv2dExpansionModule,
    LazyConv2dReductionModule,
    NamedSequential,
    create_lazy_signature,
)
from .conv_norm_activ import ConvNormActivation


class DepthwiseConv2d(ConvNormActivation):
    """2D Depthwise Convolution"""

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, **kwargs):
        if 'padding' not in kwargs:
            kwargs['padding'] = kernel_size // 2

        super().__init__(
            channels,
            channels,
            kernel_size,
            stride,
            groups=channels,
            **kwargs
        )


class PointwiseConv2d(ConvNormActivation):
    """2D Pointwise convolution"""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=1, **kwargs)


class DepthSepConv2d(NamedSequential):
    """2D Depthwise Separable Convolution. References:
    - MobileNetV1 (https://arxiv.org/abs/1704.04861)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__(
            dw=DepthwiseConv2d(
                in_channels,
                kernel_size,
                stride,
                norm_layer=norm_layer,
                activation=activation,
                **kwargs,
            ),
            pw=PointwiseConv2d(
                in_channels, out_channels, norm_layer=norm_layer, activation=activation
            ),
        )


@create_lazy_signature(exclude=["channels"])
class LazyDepthwiseConv2d(LazyConv2dInChannelModule, DepthwiseConv2d):
    """Lazily-initialized DepthwiseConv2d module"""

    pass


@create_lazy_signature(exclude=["in_channels"])
class LazyPointwiseConv2d(LazyConv2dInChannelModule, PointwiseConv2d):
    """Lazily-initialized PointwiseConv2d module"""

    pass


@create_lazy_signature(exclude=["in_channels", "out_channels"])
class LazyExpansionPointwiseConv2d(LazyConv2dExpansionModule, PointwiseConv2d):
    """Lazily-initialized PointwiseConv2d module that increases the number of
    input channels by a factor of expansion_ratio"""

    pass


@create_lazy_signature(exclude=["in_channels", "out_channels"])
class LazyReductionPointwiseConv2d(LazyConv2dReductionModule, PointwiseConv2d):
    """Lazily-initialized PointwiseConv2d module that reduces the number of
    input channels by a factor of reduction_ratio"""

    pass


@create_lazy_signature(exclude=["in_channels"])
class LazyDepthSepConv2d(LazyConv2dInChannelModule, DepthSepConv2d):
    """Lazily-initialized DepthSepConv2d module"""

    pass


@create_lazy_signature(exclude=["in_channels", "out_channels"])
class LazyExpansionDepthSepConv2d(LazyConv2dExpansionModule, DepthSepConv2d):
    """Lazily-initialized DepthSepConv2d module that increases the number of
    input channels by a factor of expansion_ratio"""

    pass


@create_lazy_signature(exclude=["in_channels", "out_channels"])
class LazyReductionDepthSepConv2d(LazyConv2dReductionModule, DepthSepConv2d):
    """Lazily-initialized DepthSepConv2d module that reduces the number of
    input channels by a factor of reduction_ratio"""

    pass
