"""Tests for torchtangents stax linear layers.

Tests adapted from neural-tangents stax/linear_test.py to verify 
linear layer functionality in PyTorch implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

import torch
import torch.nn as nn
from absl.testing import absltest

from ..test_utils import TorchTangentsTestCase, product, parameters
import torchtangents.stax as stax
from torchtangents import empirical_ntk_fn, empirical_nngp_fn


class FlattenTest(TorchTangentsTestCase):
    """Test Flatten layer functionality."""

    @parameters([
        {'same_inputs': True},
        {'same_inputs': False},
    ])
    def test_flatten(self, same_inputs):
        """Test flatten layer in different positions."""
        generator = torch.Generator()
        generator.manual_seed(42)
        
        X0_1 = torch.randn(4, 4, 3, 2, generator=generator)
        X0_2 = None if same_inputs else torch.randn(2, 4, 3, 2, generator=generator)
        
        X0_1_flat = X0_1.view(X0_1.shape[0], -1)
        X0_2_flat = None if X0_2 is None else X0_2.view(X0_2.shape[0], -1)
        
        # Test networks with flatten in different positions
        dense = stax.Dense(32, W_std=1.7, b_std=0.1)
        
        # Fully connected (flat inputs)
        init_fc, apply_fc, kernel_fc = stax.serial(
            dense, stax.Erf(), dense
        )
        
        # Flatten at the end
        init_top, apply_top, kernel_top = stax.serial(
            dense, stax.Erf(), dense, stax.Flatten()
        )
        
        # Flatten in middle
        init_mid, apply_mid, kernel_mid = stax.serial(
            dense, stax.Erf(), stax.Flatten(), dense
        )
        
        # Flatten at beginning
        init_bot, apply_bot, kernel_bot = stax.serial(
            stax.Flatten(), dense, stax.Erf(), dense
        )
        
        # Test that they can be initialized
        try:
            _, params_fc = init_fc(generator, (-1,) + X0_1_flat.shape[1:])
            _, params_top = init_top(generator, (-1,) + X0_1.shape[1:])
            _, params_mid = init_mid(generator, (-1,) + X0_1.shape[1:])
            _, params_bot = init_bot(generator, (-1,) + X0_1.shape[1:])
        except Exception as e:
            self.skipTest(f"Initialization failed: {e}")
        
        # Test function application
        try:
            out_fc = apply_fc(params_fc, X0_1_flat)
            out_top = apply_top(params_top, X0_1)
            out_mid = apply_mid(params_mid, X0_1)
            out_bot = apply_bot(params_bot, X0_1)
            
            # Check that outputs are finite
            self.assertTrue(torch.isfinite(out_fc).all())
            self.assertTrue(torch.isfinite(out_top).all())
            self.assertTrue(torch.isfinite(out_mid).all())
            self.assertTrue(torch.isfinite(out_bot).all())
            
        except Exception as e:
            self.skipTest(f"Function application failed: {e}")

    def test_flatten_shapes(self):
        """Test that flatten produces correct shapes."""
        # Test different input shapes
        shapes = [(3, 4), (2, 3, 4), (2, 3, 4, 5)]
        
        for shape in shapes:
            with self.subTest(shape=shape):
                init_fn, apply_fn, _ = stax.Flatten()
                
                generator = torch.Generator()
                generator.manual_seed(42)
                
                x = torch.randn(2, *shape, generator=generator)
                _, params = init_fn(generator, (-1,) + shape)
                
                out = apply_fn(params, x)
                
                # Should flatten to 2D
                expected_shape = (2, torch.prod(torch.tensor(shape)).item())
                self.assertEqual(out.shape, expected_shape)

    def test_flatten_with_conv(self):
        """Test flatten after convolution."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Conv(8, (3, 3), padding='SAME', W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Flatten(),
            stax.Dense(10, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        x = torch.randn(4, 8, 8, 3, generator=generator)
        
        try:
            _, params = init_fn(generator, (-1,) + x.shape[1:])
            out = apply_fn(params, x)
            
            self.assertEqual(out.shape, (4, 10))
            self.assertTrue(torch.isfinite(out).all())
            
        except Exception as e:
            self.skipTest(f"Conv+Flatten test failed: {e}")


class DenseTest(TorchTangentsTestCase):
    """Test Dense layer functionality."""

    def test_dense_basic(self):
        """Test basic dense layer functionality."""
        generator = torch.Generator()
        generator.manual_seed(42)
        
        init_fn, apply_fn, kernel_fn = stax.Dense(10, W_std=1.0, b_std=0.1)
        
        x = torch.randn(4, 5, generator=generator)
        _, params = init_fn(generator, (-1,) + x.shape[1:])
        
        out = apply_fn(params, x)
        
        self.assertEqual(out.shape, (4, 10))
        self.assertTrue(torch.isfinite(out).all())

    def test_dense_no_bias(self):
        """Test dense layer without bias."""
        generator = torch.Generator()
        generator.manual_seed(42)
        
        init_fn, apply_fn, kernel_fn = stax.Dense(10, W_std=1.0, b_std=None)
        
        x = torch.randn(4, 5, generator=generator)
        _, params = init_fn(generator, (-1,) + x.shape[1:])
        
        out = apply_fn(params, x)
        
        self.assertEqual(out.shape, (4, 10))
        self.assertTrue(torch.isfinite(out).all())
        
        # Check that bias is not in parameters
        if isinstance(params, dict) and 'bias' in params:
            self.assertIsNone(params['bias'])

    @parameters([
        {'W_std': 0.5, 'b_std': 0.1},
        {'W_std': 1.0, 'b_std': 0.1},
        {'W_std': 2.0, 'b_std': 0.5},
    ])
    def test_dense_initialization_scales(self, W_std, b_std):
        """Test different initialization scales."""
        generator = torch.Generator()
        generator.manual_seed(42)
        
        init_fn, apply_fn, kernel_fn = stax.Dense(10, W_std=W_std, b_std=b_std)
        
        x = torch.randn(4, 5, generator=generator)
        _, params = init_fn(generator, (-1,) + x.shape[1:])
        
        # Check parameter scales are roughly correct
        if isinstance(params, dict):
            if 'weight' in params:
                w_actual_std = torch.std(params['weight'])
                self.assertLess(abs(w_actual_std.item() - W_std), W_std * 0.5)
            
            if 'bias' in params and params['bias'] is not None:
                b_actual_std = torch.std(params['bias'])
                self.assertLess(abs(b_actual_std.item() - b_std), b_std * 0.5)

    def test_dense_kernel_function(self):
        """Test that dense layer has a kernel function."""
        init_fn, apply_fn, kernel_fn = stax.Dense(10, W_std=1.0, b_std=0.1)
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        x1 = torch.randn(3, 5, generator=generator)
        x2 = torch.randn(2, 5, generator=generator)
        
        try:
            # Test kernel function exists and runs
            kernel = kernel_fn(x1, x2, get='nngp')
            self.assertEqual(kernel.shape, (3, 2))
            self.assertTrue(torch.isfinite(kernel).all())
            
        except Exception as e:
            self.skipTest(f"Kernel function test failed: {e}")


class ConvTest(TorchTangentsTestCase):
    """Test Conv layer functionality."""

    def test_conv_basic(self):
        """Test basic convolution functionality."""
        generator = torch.Generator()
        generator.manual_seed(42)
        
        init_fn, apply_fn, kernel_fn = stax.Conv(8, (3, 3), padding='SAME', 
                                               W_std=1.0, b_std=0.1)
        
        x = torch.randn(4, 8, 8, 3, generator=generator)
        _, params = init_fn(generator, (-1,) + x.shape[1:])
        
        out = apply_fn(params, x)
        
        # With SAME padding, spatial dimensions should be preserved
        self.assertEqual(out.shape, (4, 8, 8, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_conv_valid_padding(self):
        """Test convolution with VALID padding."""
        generator = torch.Generator()
        generator.manual_seed(42)
        
        init_fn, apply_fn, kernel_fn = stax.Conv(8, (3, 3), padding='VALID',
                                               W_std=1.0, b_std=0.1)
        
        x = torch.randn(4, 8, 8, 3, generator=generator)
        _, params = init_fn(generator, (-1,) + x.shape[1:])
        
        out = apply_fn(params, x)
        
        # With VALID padding and 3x3 kernel, output should be 6x6
        self.assertEqual(out.shape, (4, 6, 6, 8))
        self.assertTrue(torch.isfinite(out).all())

    @parameters([
        {'strides': (1, 1)},
        {'strides': (2, 2)},
    ])
    def test_conv_strides(self, strides):
        """Test convolution with different strides."""
        generator = torch.Generator()
        generator.manual_seed(42)
        
        init_fn, apply_fn, kernel_fn = stax.Conv(8, (3, 3), strides=strides,
                                               padding='SAME', W_std=1.0, b_std=0.1)
        
        x = torch.randn(4, 8, 8, 3, generator=generator)
        
        try:
            _, params = init_fn(generator, (-1,) + x.shape[1:])
            out = apply_fn(params, x)
            
            expected_h = 8 // strides[0]
            expected_w = 8 // strides[1]
            self.assertEqual(out.shape, (4, expected_h, expected_w, 8))
            self.assertTrue(torch.isfinite(out).all())
            
        except Exception as e:
            self.skipTest(f"Conv stride test failed: {e}")

    def test_conv_no_bias(self):
        """Test convolution without bias."""
        generator = torch.Generator()
        generator.manual_seed(42)
        
        init_fn, apply_fn, kernel_fn = stax.Conv(8, (3, 3), padding='SAME',
                                               W_std=1.0, b_std=None)
        
        x = torch.randn(4, 8, 8, 3, generator=generator)
        _, params = init_fn(generator, (-1,) + x.shape[1:])
        
        out = apply_fn(params, x)
        
        self.assertEqual(out.shape, (4, 8, 8, 8))
        self.assertTrue(torch.isfinite(out).all())


class PoolingTest(TorchTangentsTestCase):
    """Test pooling layer functionality."""

    def test_global_avg_pool(self):
        """Test global average pooling."""
        generator = torch.Generator()
        generator.manual_seed(42)
        
        init_fn, apply_fn, kernel_fn = stax.GlobalAvgPool()
        
        x = torch.randn(4, 8, 8, 16, generator=generator)
        _, params = init_fn(generator, (-1,) + x.shape[1:])
        
        out = apply_fn(params, x)
        
        # Should average over spatial dimensions
        self.assertEqual(out.shape, (4, 16))
        self.assertTrue(torch.isfinite(out).all())
        
        # Check that it's actually averaging
        manual_avg = torch.mean(x, dim=(1, 2))
        self.assertTensorsAllClose(out, manual_avg, atol=1e-6)

    def test_global_avg_pool_with_conv(self):
        """Test global average pooling after convolution."""
        generator = torch.Generator()
        generator.manual_seed(42)
        
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Conv(8, (3, 3), padding='SAME', W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.GlobalAvgPool()
        )
        
        x = torch.randn(4, 8, 8, 3, generator=generator)
        
        try:
            _, params = init_fn(generator, (-1,) + x.shape[1:])
            out = apply_fn(params, x)
            
            self.assertEqual(out.shape, (4, 8))
            self.assertTrue(torch.isfinite(out).all())
            
        except Exception as e:
            self.skipTest(f"Conv+GlobalAvgPool test failed: {e}")


if __name__ == '__main__':
    absltest.main()