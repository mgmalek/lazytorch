#!/usr/bin/env python

from __future__ import annotations
import sys
import unittest
import torch
import torch.nn as nn

sys.path.append("../lazytorch")

from lazytorch import LazyModule
from lazytorch.layers import *


def get_lazy_module_with_input() -> Tuple[LazyModule, torch.Tensor]:
    """Return an uninitialized lazy module and an input tensor for
    initializing it"""
    layer = LazyConvNormActivation(64, kernel_size=3, stride=1)
    x = torch.randn(1, 3, 32, 32)
    return layer, x


class BasicTests(unittest.TestCase):
    def test_initialisation(self):
        """Test basic lazy initialiation"""
        layer, x = get_lazy_module_with_input()
        y = layer(x)
        self.assertEqual(layer.conv.weight.data.shape, (64, 3, 3, 3))
        self.assertEqual(y.shape, (1, 64, 32, 32))
        self.assertEqual(layer.__class__.__name__, "ConvNormActivation")

    def test_multiple_forward_passes(self):
        """Test multiple forward passes. This is here for completeness since
        test_initialization only tests a single forward pass"""
        layer, x = get_lazy_module_with_input()
        for _ in range(10):
            y = layer(x)
            self.assertEqual(y.shape, (1, 64, 32, 32))

    def test_training_status_change(self):
        """Test changing the status fo the module from training to eval before
        inittialization"""
        layer, x = get_lazy_module_with_input()
        layer.eval()
        _ = layer(x)
        self.assertEqual(layer.training, False)

    def test_device_move_before_init(self):
        """Test moving a module to cuda before initialisation"""
        layer, x = get_lazy_module_with_input()
        layer.cuda()
        x = x.cuda()
        y = layer(x)
        self.assertEqual(y.device.type, "cuda")

    def test_jit_script_after_init(self):
        """Test scripting a model after initialisation"""
        layer, x = get_lazy_module_with_input()
        _ = layer(x)
        scripted_layer = torch.jit.script(layer)
        y = scripted_layer(x)
        self.assertEqual(y.shape, (1, 64, 32, 32))

    def test_jit_trace_after_init(self):
        """Test tracing a model after initialisation"""
        layer, x = get_lazy_module_with_input()
        _ = layer(x)
        traced_layer = torch.jit.trace(layer, x)
        y = traced_layer(x)
        self.assertEqual(y.shape, (1, 64, 32, 32))

    def test_arbitrary_data_retention(self):
        """Test whether arbitrary properties set on the unititialisd module
        are retained post-initialization"""
        layer, x = get_lazy_module_with_input()
        layer.arbitrary_attr = "arbitrary_attr_content"
        _ = layer(x)
        self.assertEqual(layer.arbitrary_attr, "arbitrary_attr_content")

    def test_sequential_layers(self):
        """Test multiple lazily-initialized layers in series using
        nn.Sequential"""
        layer1, x = get_lazy_module_with_input()
        layer2, _ = get_lazy_module_with_input()
        module = nn.Sequential(layer1, layer2)
        y = module(x)
        self.assertEqual(y.shape, (1, 64, 32, 32))

    def test_module(self):
        """Test a lazily-initialised layer running in a standard nn.Module"""
        layer, x = get_lazy_module_with_input()

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = layer

            def forward(self, x):
                return self.layer(x)

        module = TestModule()
        y = module(x)
        self.assertEqual(y.shape, (1, 64, 32, 32))

    def test_lazy_composition(self):
        """Test creating a LazyModule out of another LazyModule"""

        class BaseLazyModule(LazyModule, nn.Conv2d):
            def get_initialized_module(self, inputs, module_args, module_kwargs):
                in_channels = inputs[0].size(1)
                return nn.Conv2d(in_channels, *module_args, **module_kwargs)

        class LazyModuleComposed(LazyModule, nn.Conv2d):
            def get_initialized_module(self, inputs, module_args, module_kwargs):
                return BaseLazyModule(*module_args, **module_kwargs)

        x = torch.randn(1, 3, 32, 32)
        module = LazyModuleComposed(out_channels=64, kernel_size=3, stride=1, padding=1)
        y = module(x)
        self.assertEqual(y.shape, (1, 64, 32, 32))

    def test_lazy_submodule(self):
        """Test creating a LazyModule which has a LazyModule as its child"""

        class BaseLazyModule(LazyModule, nn.Conv2d):
            def get_initialized_module(self, inputs, module_args, module_kwargs):
                in_channels = inputs[0].size(1)
                return nn.Conv2d(in_channels, *module_args, **module_kwargs)

        class LazyModuleWithLazySubmodule(LazyModule, nn.Sequential):
            def get_initialized_module(self, inputs, module_args, module_kwargs):
                return nn.Sequential(BaseLazyModule(*module_args, **module_kwargs))

        module = LazyModuleWithLazySubmodule(
            out_channels=64, kernel_size=3, stride=1, padding=1
        )

        x = torch.randn(1, 3, 32, 32)
        y = module(x)
        self.assertEqual(y.shape, (1, 64, 32, 32))

    def test_apply(self):
        """Test calling .apply() on an uninitialized LazyModule"""
        layer, x = get_lazy_module_with_input()

        def add_arbitrary_attr(module):
            module.arbitrary_attr = "arbitrary_attr_content"

        layer.apply(add_arbitrary_attr)
        _ = layer(x)
        self.assertEqual(layer.arbitrary_attr, "arbitrary_attr_content")

    def test_lazy_expansion_conv_2d(self):
        """Test the LazyExpansionConv2d module"""
        layer = LazyExpansionConv2d(expansion_ratio=4, kernel_size=3, padding=1)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 64, 32, 32))

    def test_lazy_reduction_conv_2d(self):
        """Test the LazyReductionConv2d module"""
        layer = LazyReductionConv2d(reduction_ratio=4, kernel_size=3, padding=1)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 4, 32, 32))

    def test_lazy_conv_norm_activation(self):
        """Test the LazyConvNormActivation module"""
        layer = LazyConvNormActivation(out_channels=32, kernel_size=1)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 32, 32, 32))

    def test_lazy_depthwise_conv_2d(self):
        """Test the LazyDepthwiseConv2d module"""
        layer = LazyDepthwiseConv2d(kernel_size=1)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 16, 32, 32))

    def test_lazy_pointwise_conv_2d(self):
        """Test the LazyPointwiseConv2d module"""
        layer = LazyPointwiseConv2d(out_channels=32)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 32, 32, 32))

    def test_lazy_expansion_pointwise_conv_2d(self):
        """Test the LazyExpansionPointwiseConv2d module"""
        layer = LazyExpansionPointwiseConv2d(expansion_ratio=4)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 64, 32, 32))

    def test_lazy_reduction_pointwise_conv_2d(self):
        """Test the LazyReductionPointwiseConv2d module"""
        layer = LazyReductionPointwiseConv2d(reduction_ratio=4)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 4, 32, 32))

    def test_lazy_depth_sep_conv_2d(self):
        """Test the LazyDepthSepConv2d module"""
        layer = LazyDepthSepConv2d(out_channels=32, kernel_size=3)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 32, 32, 32))

    def test_lazy_expansion_depth_sep_conv_2d(self):
        """Test the LazyExpansionDepthSepConv2d module"""
        layer = LazyExpansionDepthSepConv2d(expansion_ratio=4, kernel_size=3)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 64, 32, 32))

    def test_lazy_reduction_depth_sep_conv_2d(self):
        """Test the LazyReductionDepthSepConv2d module"""
        layer = LazyReductionDepthSepConv2d(reduction_ratio=4, kernel_size=3)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 4, 32, 32))

    def test_lazy_squeeze_excitation(self):
        """Test the LazySqueezeExcitation module"""
        layer = LazySqueezeExcitation(reduction_ratio=16)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 16, 32, 32))

    def test_lazy_bottleneck_block(self):
        """Test the LazyBottleneckBlock module"""
        layer = LazyBottleneckBlock(mid_channels=32, out_channels=16)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 16, 32, 32))

    def test_lazy_inverted_bottleneck(self):
        """Test the LazyInvertedBottleneck module"""
        layer = LazyInvertedBottleneck(out_channels=32, kernel_size=3)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 32, 32, 32))
    
    def test_lazy_dilated_depth_sep_conv_2d(self):
        """Test the LazyDepthSepConv2d module with dilated convolutions"""
        layer = LazyDepthSepConv2d(out_channels=32, kernel_size=3, padding=2, dilation=2)
        x = torch.randn(1, 16, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 32, 32, 32))


if __name__ == "__main__":
    unittest.main()
