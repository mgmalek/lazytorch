# LazyTorch


## Introduction

LazyTorch lets you convert your PyTorch modules into versions that lazily infer the dimensions of their parameters when the module is first used. For example, based on an existing `ResidualBlock` class, you could create a `LazyResidualBlock` class that lazily initializes its parameters based on the shape of the first input it receives. The `lazytorch.layers` module also contains lazily-initialised versions of popular layers. See the section entitled *List of Layers Available in `lazytorch.layers`* below for more details.


## Installation

```shell
$ git clone https://github.com/mgmalek/lazytorch
$ cd lazytorch
$ pip install .
```


## Examples

### Lazify-ing an Existing Module

Converting an existing PyTorch module into a lazily-initialised module involves subclassing `lazytorch.LazyModule` (which is itself a subclass of `nn.Module`) and overriding its `get_initialized_module` method.

The `get_initialized_module` method takes three arguments and it should return an initialized version of the module. Its arguments are:
- `inputs`: the first input received by the module
- `module_args`: the **positional** arguments passed when initializing the lazy module (e.g. `64` in the example above, which represents the number of output channels)
- `module_kwargs`: the **keyword** arguments passed when initializing the lazy module (e.g. `kernel_size=3` and `padding=1` in the example below)

In addition to subclassing `LazyModule`, you must also subclass the type of module you want to lazily-initialize (e.g. `nn.Conv2d` in the example below). This limitation arises from the way that hooks interact with the call to `forward` in PyTorch.

```Python
import torch
import torch.nn as nn
from lazytorch import LazyModule
from typing import Tuple, Dict


class LazyConv2d(LazyModule, nn.Conv2d):
    """Lazily-initialized 2d Convolution Layer"""

    def get_initialized_module(
        self,
        inputs: Tuple[torch.Tensor],
        module_args: Tuple,
        module_kwargs: Dict,
    ) -> nn.Module:
        # Validate the input shape
        assert len(inputs) == 1 and inputs[0].dim() == 4

        # Infer the number of input channels
        in_channels = inputs[0].size(1)

        # Initialize the module
        module = nn.Conv2d(in_channels, *module_args, **module_kwargs)

        return module


# Create a lazily-initialized 2d convolution layer with 64 output channels
lazy_conv = LazyConv2d(64, kernel_size=3, padding=1)
x = torch.randn(1, 3, 225, 225)
y = lazy_conv(x)
```

### Setting the Init Signature of a Lazy Module

By default, the init signature of a `LazyModule` subclass will be `LazyConv2d(*module_args, **module_kwargs)` (when viewed in a Jupyter Notebook).
This isn't particularly user friendly, so LazyTorch offers a `create_lazy_signature` decorator to override this init signature with the init signature of the module it will transform into.
For example, you could make the init signature of `LazyConv2d` match the init signature of `nn.Conv2d` as follows (excluding the `in_channels` parameter since that will be lazily inferred):

```Python
...

from lazytorch import create_lazy_signature

@create_lazy_signature(exclude=["in_channels"])
class LazyConv2d(LazyModule, nn.Conv2d):
    ...

...
```

Now, the init signature of `LazyConv2d` will be the following:
```Python
LazyConv2d(
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros',
    device=None,
    dtype=None,
)
```


## List of Layers Available in `lazytorch.layers`

The `lazytorch.layers` module also contains lazily-initialised versions of popular layers. These layers can be placed into one of three categories:
- **Standard Lazy Modules** which are direct conversions of regular PyTorch modules into lazily-initialised modules.
- **Lazy Expansion Modules** which are modules where the number of output channels is a constant factor (`expansion_ratio`) larger than the number of input channels
- **Lazy Reduction Modules** which are modules where the number of output channels is a constant factor (`reduction_ratio`) smaller than the number of input channels


### List of Standard Lazy Modules
- `LazyConvNormActivation`
- `LazyBottleneckBlock`
- `LazyInvertedBottleneck`
- `LazyDepthwiseConv2d`
- `LazyPointwiseConv2d`
- `LazyDepthSepConv2d`
- `LazySqueezeExcitation`


### List of Lazy Expansion Modules

- `LazyReductionConv2d`
- `LazyReductionDepthSepConv2d`
- `LazyReductionPointwiseConv2d`


### List of Lazy Reduction Modules

- `LazyExpansionConv2d`
- `LazyExpansionPointwiseConv2d`
- `LazyExpansionDepthSepConv2d`


## Why not use `nn.Lazy.LazyModuleMixin`?

PyTorch's `nn.Lazy.LazyModuleMixin` class is great for smaller modules (e.g. `nn.LazyConv2d` and `nn.LazyLinear`). However, it requires that all lazily-initialized parameters are declared as `nn.UninitializedParameter` objects, which limits composability and doesn't let users easily create lazy versions of existing modules without completely rewriting them.


## Known Limitations of LazyTorch
- It must be possible to infer parameter dimensions using only the positional arguments passed to a module's `.forward()` method. Keyword arguments cannot be used since hooks (which are used under the hood by `LazyModule`) don't have access to them.
- Hooks must be registered *after* parameters have been initialized.
