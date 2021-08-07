from __future__ import annotations
from collections import namedtuple
import inspect
import torch
import torch.nn as nn
from typing import Type, Tuple, Dict, List


FuncToApply = namedtuple("FuncToApply", ["fn", "hidden_apply"])


class LazyModule(nn.Module):
    """LazyModule offers a simple mechanism to create PyTorch modules with
    lazily-initialized parameters.

    Every lazily-initialised module must be paired with a regular PyTorch
    module. For example, to implement `LazyDepthwiseConv2d`, you must first
    implement a vanilla `DepthwiseConv2d` module.

    Modules that subclass `LazyModule` must also subclass the module they want
    to transform into, or else the first forward pass won't work. This is
    because the `_call_impl` method of PyTorch's `nn.Module` class saves the
    forward method in a variable before invoking forward pre hooks (which are
    used by LazyTorch to initialize the module's parameters).

    Example Usage:
    >>> class LazyConv2d(LazyModule, nn.Conv2d):
    ...     '''Lazily-initialized 2d Convolution Layer'''
    ...
    ...     def get_initialized_module(
    ...         self,
    ...         inputs: Tuple[torch.Tensor],
    ...         module_args: Tuple,
    ...         module_kwargs: Dict,
    ...     ) -> nn.Module:
    ...         # Validate the input shape
    ...         assert len(inputs) == 1 and inputs[0].dim() == 4
    ...
    ...         # Infer the number of input channels
    ...         in_channels = inputs[0].size(1)
    ...
    ...         # Initialize the module
    ...         module = nn.Conv2d(in_channels, *module_args, **module_kwargs)
    ...
    ...         return module
    >>> # Create a lazily-initialized 2d convolution layer with 64 output channels
    >>> lazy_conv = LazyConv2d(64, kernel_size=3, padding=1)
    >>> x = torch.randn(1, 3, 225, 225)
    >>> # Initialize the parameters of `lazy_conv` by running a forward pass
    >>> y = lazy_conv(x)
    """

    def __init__(self, *module_args, **module_kwargs):
        """
        Args:
            *args: Positional arguments used to initialize module.
            **kwargs: Keyword arguments used to initialize module.
        """
        # Since LazyModule will only be used as a parent class in a multiple
        # inheritance pattern, we need to explicitly initialize nn.Module
        # since just calling super().__init__() would initialize the other
        # parent class of LazyModule's child. We don't want this to happen
        # since we don't have all the necessary parameters to initialize that
        # other parent class (hence why we're using LazyModule in the first
        # place)
        nn.Module.__init__(self)
        self.module_args = module_args
        self.module_kwargs = module_kwargs
        self.funcs_to_apply: List[FuncToApply] = []
        self._initialize_hook = self.register_forward_pre_hook(self._update_module)

    def _update_module(self, module: LazyModule, inputs):
        self._initialize_hook.remove()

        initialized_module = module.get_initialized_module(
            inputs=inputs,
            module_args=module.module_args,
            module_kwargs=module.module_kwargs,
        )

        if not isinstance(initialized_module, nn.Module):
            raise TypeError(
                f"get_initialized_module must return an object of type nn.Module"
            )

        # Add attributes (e.g. _parameters, _buffers, etc.) from initialized
        # module to module. The only relevant attributes are those of type
        # dict or set (see implementation of nn.Module for a list of those
        # attributes: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py)
        for attr_name, attr_val in initialized_module.__dict__.items():
            if isinstance(attr_val, dict):
                new_dict = dict()
                if hasattr(module, attr_name):
                    new_dict = getattr(module, attr_name)
                # Merge dictionaries, giving preference to values in
                # initialized_module if there is a key conflict
                new_dict = {**new_dict, **attr_val}
                setattr(module, attr_name, new_dict)
            elif isinstance(attr_val, set):
                if hasattr(module, attr_name):
                    attr_val |= getattr(module, attr_name)
                setattr(module, attr_name, attr_val)
            elif attr_name not in ("training",):
                setattr(module, attr_name, attr_val)

        # Update special attributes of module
        special_attrs_to_update = (
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
            "__class__",
        )
        for attr_name in special_attrs_to_update:
            if hasattr(module, attr_name):
                setattr(module, attr_name, getattr(initialized_module, attr_name))

        # Apply functions. Note that this will call the _apply and apply
        # methods of nn.Module since the __class__ attribute was updated above
        for fn_to_apply in self.funcs_to_apply:
            if fn_to_apply.hidden_apply:
                module._apply(fn_to_apply.fn)
            else:
                module.apply(fn_to_apply.fn)

        # Run a forward pass to initialize any `LazyModule`s that are children
        # of initialized_module
        if isinstance(module, LazyModule) or has_lazy_children(module):
            module(*inputs)

        # Cleanup all traces of LazyModule
        del self._initialize_hook
        del self.module_args
        del self.module_kwargs
        del self.funcs_to_apply

    def get_initialized_module(
        self, inputs, module_args: Tuple, module_kwargs: Dict
    ) -> nn.Module:
        raise NotImplementedError(
            f"Method get_initialized_module is not implemented for module {self._get_name()}"
        )

    def _apply(self, fn):
        self.funcs_to_apply.append(FuncToApply(fn, hidden_apply=True))
        return self

    def apply(self, fn):
        self.funcs_to_apply.append(FuncToApply(fn, hidden_apply=False))
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def has_lazy_children(module: nn.Module) -> bool:
    for child in module.children():
        if isinstance(child, LazyModule) or has_lazy_children(child):
            return True

    return False


def infer_cls_to_become(lazy_module: Type[LazyModule]) -> Type[nn.Module]:
    """Infer the class that an instance of LazyModule should be transformed
    into"""
    base_classes = lazy_module.__bases__
    base_classes = [x for x in base_classes if not issubclass(x, LazyModule)]
    base_classes = [x for x in base_classes if issubclass(x, nn.Module)]
    assert len(base_classes) == 1, (
        "A child class of LazyModule must only "
        "inherit from one nn.Module class (excluing LazyModule itself)"
    )
    return base_classes[0]


def create_lazy_signature(exclude: Tuple[str] = tuple()):
    """A decorator to create the signature of a LazyModule child class
    by merging in the signature of its other base classes."""

    def get_params(o):
        sig = inspect.signature(o)
        return list(sig.parameters.values())

    def _create_lazy_signature(tgt_cls: Type):
        tgt_params = get_params(tgt_cls)
        cls_to_become = infer_cls_to_become(tgt_cls)
        src_params = get_params(cls_to_become)

        # Remove *args (VAR_POSITIONAL) and **kwargs (VAR_KEYWORD) from
        # tgt_cls signature
        tgt_params = [
            p
            for p in tgt_params
            if p.kind.name not in ["VAR_POSITIONAL", "VAR_KEYWORD"]
        ]

        for val in src_params:
            if val.name not in exclude:
                tgt_params.append(val)

        tgt_cls.__signature__ = inspect.Signature(parameters=tgt_params)
        return tgt_cls

    return _create_lazy_signature


class LazyConv2dInChannelModule(LazyModule):
    """A class to help with a common pattern when initializing lazy
    convolutional modules, where the only lazily-determined input is the
    number of input channels"""

    def get_initialized_module(
        self, inputs, module_args: Tuple, module_kwargs: Dict
    ) -> nn.Module:
        assert len(inputs) == 1 and inputs[0].dim() == 4
        in_channels = inputs[0].size(1)
        cls_to_become = infer_cls_to_become(self.__class__)
        return cls_to_become(in_channels, *module_args, **module_kwargs)


class LazyConv2dExpansionModule(LazyModule):
    """A class to help with a common pattern when initializing lazy
    convolutional modules, where the conv layer should increase the number
    of input channels by a constant factor (expansion_ratio)"""

    def __init__(self, expansion_ratio: int, *args, **kwargs):
        kwargs["expansion_ratio"] = expansion_ratio
        super().__init__(*args, **kwargs)

    def get_initialized_module(
        self, inputs, module_args: Tuple, module_kwargs: Dict
    ) -> nn.Module:
        assert len(inputs) == 1 and inputs[0].dim() == 4
        expansion_ratio = module_kwargs.pop("expansion_ratio")
        in_channels = inputs[0].size(1)
        out_channels = in_channels * expansion_ratio
        cls_to_become = infer_cls_to_become(self.__class__)
        return cls_to_become(in_channels, out_channels, *module_args, **module_kwargs)


class LazyConv2dReductionModule(LazyModule):
    """A class to help with a common pattern when initializing lazy
    convolutional modules, where the conv layer should decrease the number
    of input channels by a constant factor (reduction_factor)"""

    def __init__(self, reduction_ratio: int, *args, **kwargs):
        kwargs["reduction_ratio"] = reduction_ratio
        super().__init__(*args, **kwargs)

    def get_initialized_module(
        self, inputs, module_args: Tuple, module_kwargs: Dict
    ) -> nn.Module:
        assert len(inputs) == 1 and inputs[0].dim() == 4
        reduction_ratio = module_kwargs.pop("reduction_ratio")
        in_channels = inputs[0].size(1)
        out_channels = in_channels // reduction_ratio
        assert in_channels > reduction_ratio
        assert in_channels % reduction_ratio == 0, (
            f"The number of input channels ({in_channels}) must be divisible "
            f"by the reduction ratio ({reduction_ratio})"
        )
        cls_to_become = infer_cls_to_become(self.__class__)
        return cls_to_become(in_channels, out_channels, *module_args, **module_kwargs)
