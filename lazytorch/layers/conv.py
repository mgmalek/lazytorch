import torch.nn as nn
from lazytorch import (
    LazyConv2dExpansionModule,
    LazyConv2dReductionModule,
    create_lazy_signature,
)


@create_lazy_signature(exclude=("in_channels", "out_channels"))
class LazyExpansionConv2d(LazyConv2dExpansionModule, nn.Conv2d):
    """Lazily-initialized Conv2d that expands the number of channels by a
    factor of expansion_ratio"""

    pass


@create_lazy_signature(exclude=("in_channels", "out_channels"))
class LazyReductionConv2d(LazyConv2dReductionModule, nn.Conv2d):
    """Lazily-initialized Conv2d that reduces the number of channels by a
    factor of reduction_ratio"""

    pass
