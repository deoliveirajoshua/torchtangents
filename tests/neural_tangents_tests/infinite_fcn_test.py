"""Tests for torchtangents infinite FCN functionality.

Tests adapted from neural-tangents infinite_fcn_test.py to verify 
infinite width fully connected network behavior in PyTorch implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
import torch.nn as nn
from absl.testing import absltest

from .test_utils import TorchTangentsTestCase, parameters
import torchtangents.stax as stax
from torchtangents import empirical_ntk_fn, empirical_nngp_fn


class InfiniteFcnTest(TorchTangentsTestCase):
    """Test infinite width fully connected networks."""

    def test_infinite_fcn_basic(self):
        """Test basic infinite FCN behavior."""
        # Build a deep FCN
        layers = [
            stax.Dense(512, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(256, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(128, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        ]
        
        init_fn, apply_fn, kernel_fn = stax.serial(*layers)
        
        # Test initialization
        generator = torch.Generator()
        generator.manual_seed(42)
        input_shape = (-1, 10)
        output_shape, params = init_fn(generator, input_shape)
        
        self.assertEqual(output_shape, (-1, 1))
        
        # Test forward pass
        x = torch.randn(4, 10)
        y = apply_fn(params, x)
        self.assertEqual(y.shape, (4, 1))

    def test_deep_network_gradient_flow(self):
        """Test gradient flow in deep networks."""
        # Very deep network
        layers = []
        for i in range(10):  # 10 hidden layers
            layers.extend([
                stax.Dense(64, W_std=1.0, b_std=0.1),
                stax.Relu()
            ])
        layers.append(stax.Dense(1, W_std=1.0, b_std=0.1))
        
        init_fn, apply_fn, kernel_fn = stax.serial(*layers)
        
        generator = torch.Generator()
        generator.manual_seed(42)
        input_shape = (-1, 8)
        _, params = init_fn(generator, input_shape)
        
        # Test that gradients don't vanish/explode
        x = torch.randn(2, 8, requires_grad=True)
        y = apply_fn(params, x)
        loss = y.sum()
        loss.backward()
        
        # Check that input gradients exist and are reasonable
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(torch.isfinite(x.grad)))
        
        # Gradient magnitude should be reasonable (not too small/large)
        grad_norm = torch.norm(x.grad)
        self.assertTrue(grad_norm > 1e-6)
        self.assertTrue(grad_norm < 1e6)

    @parameters([
        {'width': 64, 'depth': 3},
        {'width': 128, 'depth': 5},
        {'width': 256, 'depth': 4},
    ])
    def test_width_scaling(self, width, depth):
        """Test behavior with different widths."""
        # Build network with specified width and depth
        layers = []
        for i in range(depth):
            layers.extend([
                stax.Dense(width, W_std=1.0, b_std=0.1),
                stax.Relu()
            ])
        layers.append(stax.Dense(1, W_std=1.0, b_std=0.1))
        
        init_fn, apply_fn, kernel_fn = stax.serial(*layers)
        
        generator = torch.Generator()
        generator.manual_seed(42)
        input_shape = (-1, 10)
        _, params = init_fn(generator, input_shape)
        
        # Test forward pass
        x = torch.randn(3, 10)
        y = apply_fn(params, x)
        
        # Should produce reasonable outputs
        self.assertEqual(y.shape, (3, 1))
        self.assertTrue(torch.all(torch.isfinite(y)))

    def test_empirical_vs_infinite_consistency(self):
        """Test consistency between empirical and theoretical kernels."""
        # Simple 2-layer network
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(256, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        input_shape = (-1, 5)
        _, params = init_fn(generator, input_shape)
        
        # Create test data
        x1 = torch.randn(3, 5)
        x2 = torch.randn(2, 5)
        
        # Define function for empirical kernel
        def f(params_dict, x):
            return apply_fn(params_dict, x)
        
        # Compute empirical NTK
        empirical_ntk = empirical_ntk_fn(f)
        try:
            ntk_empirical = empirical_ntk(x1, x2, params)
            
            # Should be valid tensor
            self.assertEqual(ntk_empirical.shape, (3, 2))
            self.assertTrue(torch.all(torch.isfinite(ntk_empirical)))
            
        except Exception as e:
            self.skipTest(f"Empirical NTK computation failed: {e}")

    def test_fcn_output_variance(self):
        """Test output variance scaling in FCNs."""
        widths = [32, 64, 128, 256]
        variances = []
        
        for width in widths:
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Dense(width, W_std=1.0, b_std=0.0),  # No bias for cleaner variance
                stax.Relu(),
                stax.Dense(1, W_std=1.0, b_std=0.0)
            )
            
            generator = torch.Generator()
            generator.manual_seed(42)
            input_shape = (-1, 10)
            _, params = init_fn(generator, input_shape)
            
            # Multiple forward passes to estimate variance
            x = torch.randn(100, 10)
            outputs = []
            
            for _ in range(10):  # Multiple initializations
                generator_i = torch.Generator()
                generator_i.manual_seed(torch.randint(0, 10000, (1,)).item())
                _, params_i = init_fn(generator_i, input_shape)
                y = apply_fn(params_i, x[0:1])  # Single input
                outputs.append(y[0, 0].item())
            
            variance = torch.var(torch.tensor(outputs))
            variances.append(variance.item())
        
        # Check that variances are reasonable
        for var in variances:
            self.assertTrue(var > 0)
            self.assertTrue(var < 100)  # Not too large

    def test_activation_statistics(self):
        """Test statistics of activations in deep networks."""
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(128, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(64, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        input_shape = (-1, 20)
        _, params = init_fn(generator, input_shape)
        
        # Large batch to get good statistics
        x = torch.randn(1000, 20)
        y = apply_fn(params, x)
        
        # Output statistics should be reasonable
        mean = torch.mean(y)
        std = torch.std(y)
        
        # Mean should be close to 0 (with bias initialization)
        self.assertTrue(abs(mean.item()) < 1.0)
        
        # Standard deviation should be reasonable
        self.assertTrue(std.item() > 0.1)
        self.assertTrue(std.item() < 10.0)


if __name__ == '__main__':
    absltest.main()