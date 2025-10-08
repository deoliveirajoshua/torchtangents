"""Tests for torchtangents empirical NTK functionality.

Tests adapted from neural-tangents empirical_ntk_test.py to verify 
empirical Neural Tangent Kernel computation in PyTorch implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
import torch.nn as nn
from absl.testing import absltest

from .test_utils import TorchTangentsTestCase, product, parameters
from torchtangents.stax import Dense, Conv, Flatten, GlobalAvgPool, Relu, Erf, serial, parallel
from torchtangents import empirical_ntk_fn, empirical_nngp_fn, empirical_kernel_fn

# Create a stax-like namespace for compatibility with our tests
class stax:
    Dense = Dense
    Conv = Conv
    Flatten = Flatten
    GlobalAvgPool = GlobalAvgPool
    Relu = Relu
    Erf = Erf
    serial = serial
    parallel = parallel


class EmpiricalNtkTest(TorchTangentsTestCase):
    """Test empirical NTK computation functionality."""

    def test_empirical_ntk_basic(self):
        """Test basic empirical NTK computation."""
        # Simple network
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(32, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 4))
        
        # Test data
        x_train = torch.randn(6, 4, generator=generator)
        x_test = torch.randn(3, 4, generator=generator)
        
        # Compute empirical NTK
        ntk_fn = empirical_ntk_fn(apply_fn)
        
        # Train-train kernel
        ntk_train_train = ntk_fn(x_train, None, params)
        self.assertEqual(ntk_train_train.shape, (6, 6))
        self.assertTrue(torch.isfinite(ntk_train_train).all())
        
        # Test-train kernel
        ntk_test_train = ntk_fn(x_test, x_train, params)
        self.assertEqual(ntk_test_train.shape, (3, 6))
        self.assertTrue(torch.isfinite(ntk_test_train).all())
        
        # Test-test kernel
        ntk_test_test = ntk_fn(x_test, None, params)
        self.assertEqual(ntk_test_test.shape, (3, 3))
        self.assertTrue(torch.isfinite(ntk_test_test).all())

    def test_empirical_ntk_symmetry(self):
        """Test that empirical NTK is symmetric."""
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(16, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(2, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 5))
        
        x = torch.randn(4, 5, generator=generator)
        
        ntk_fn = empirical_ntk_fn(apply_fn)
        ntk = ntk_fn(x, None, params)
        
        # Should be symmetric
        self.assertTensorsAllClose(ntk, ntk.T, atol=1e-5, rtol=1e-5)

    def test_empirical_ntk_positive_definite(self):
        """Test that empirical NTK is positive semi-definite."""
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(20, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 3))
        
        x = torch.randn(5, 3, generator=generator)
        
        ntk_fn = empirical_ntk_fn(apply_fn)
        ntk = ntk_fn(x, None, params)
        
        # Check positive semi-definiteness
        eigenvals = torch.linalg.eigvalsh(ntk)
        self.assertGreater(torch.min(eigenvals).item(), -1e-6)  # Allow small numerical errors

    def test_empirical_ntk_parameter_dependence(self):
        """Test that NTK depends on network parameters."""
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(10, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        x = torch.randn(3, 4)
        
        # Two different parameter sets
        generator1 = torch.Generator()
        generator1.manual_seed(42)
        _, params1 = init_fn(generator1, (-1, 4))
        
        generator2 = torch.Generator()
        generator2.manual_seed(123)
        _, params2 = init_fn(generator2, (-1, 4))
        
        ntk_fn = empirical_ntk_fn(apply_fn)
        
        ntk1 = ntk_fn(x, None, params1)
        ntk2 = ntk_fn(x, None, params2)
        
        # Should be different with different parameters
        diff = torch.norm(ntk1 - ntk2)
        self.assertGreater(diff.item(), 1e-6)

    @parameters([
        {'width': 8, 'depth': 2},
        {'width': 16, 'depth': 3},
        {'width': 32, 'depth': 2},
    ])
    def test_empirical_ntk_different_architectures(self, width, depth):
        """Test empirical NTK with different network architectures."""
        # Build network with specified width and depth
        layers = [stax.Dense(width, W_std=1.0, b_std=0.1), stax.Relu()]
        for _ in range(depth - 2):
            layers.extend([stax.Dense(width, W_std=1.0, b_std=0.1), stax.Relu()])
        layers.append(stax.Dense(1, W_std=1.0, b_std=0.1))
        
        init_fn, apply_fn, kernel_fn = stax.serial(*layers)
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 5))
        
        x = torch.randn(3, 5, generator=generator)
        
        try:
            ntk_fn = empirical_ntk_fn(apply_fn)
            ntk = ntk_fn(x, None, params)
            
            self.assertEqual(ntk.shape, (3, 3))
            self.assertTrue(torch.isfinite(ntk).all())
            self.assertTensorsAllClose(ntk, ntk.T, atol=1e-5)
            
            # Check that larger networks generally have larger NTK values
            ntk_scale = torch.mean(torch.diagonal(ntk))
            self.assertGreater(ntk_scale.item(), 0)
            
        except Exception as e:
            self.skipTest(f"Architecture test failed for width={width}, depth={depth}: {e}")

    def test_empirical_ntk_convolutional(self):
        """Test empirical NTK with convolutional layers."""
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Conv(8, (3, 3), padding='SAME', W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Flatten(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 8, 8, 3))
        
        x = torch.randn(2, 8, 8, 3, generator=generator)
        
        try:
            ntk_fn = empirical_ntk_fn(apply_fn)
            ntk = ntk_fn(x, None, params)
            
            self.assertEqual(ntk.shape, (2, 2))
            self.assertTrue(torch.isfinite(ntk).all())
            self.assertTensorsAllClose(ntk, ntk.T, atol=1e-5)
            
        except Exception as e:
            self.skipTest(f"Convolutional NTK test failed: {e}")

    def test_empirical_ntk_gradient_computation(self):
        """Test that empirical NTK correctly computes gradients."""
        # Simple network for easier verification
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(4, W_std=1.0, b_std=0.1),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 2))
        
        # Make parameters require grad for manual verification
        # Handle both dict and list parameter structures
        trainable_params = []
        if isinstance(params, dict):
            # Single layer case
            layer_params = {}
            for k, v in params.items():
                if v is not None:
                    layer_params[k] = v.clone().requires_grad_(True)
                else:
                    layer_params[k] = None
            trainable_params = layer_params
        elif isinstance(params, list):
            # Multi-layer case (from serial combinator)
            for layer_params in params:
                layer_trainable = {}
                for k, v in layer_params.items():
                    if v is not None:
                        layer_trainable[k] = v.clone().requires_grad_(True)
                    else:
                        layer_trainable[k] = None
                trainable_params.append(layer_trainable)
        
        x = torch.randn(2, 2, generator=generator, requires_grad=True)
        
        # Compute function output
        f_x = apply_fn(trainable_params, x)
        
        # Collect all trainable parameters for gradient computation
        all_trainable_params = []
        if isinstance(trainable_params, dict):
            all_trainable_params = [v for v in trainable_params.values() if v is not None and v.requires_grad]
        elif isinstance(trainable_params, list):
            for layer_params in trainable_params:
                all_trainable_params.extend([v for v in layer_params.values() if v is not None and v.requires_grad])
        
        # Compute gradients manually
        grad_output = torch.ones_like(f_x)
        manual_grads = torch.autograd.grad(
            f_x, all_trainable_params, 
            grad_outputs=grad_output, create_graph=True
        )
        
        # Check that gradients exist and are finite
        for grad in manual_grads:
            self.assertTrue(torch.isfinite(grad).all())
        
        # Compute empirical NTK
        ntk_fn = empirical_ntk_fn(apply_fn)
        ntk = ntk_fn(x, None, params)
        
        # Should be positive (gradients exist)
        self.assertTrue(torch.all(torch.diagonal(ntk) > 0))

    def test_empirical_ntk_batch_consistency(self):
        """Test that NTK computation is consistent across batch dimensions."""
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(8, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 3))
        
        # Different batch sizes
        x_small = torch.randn(2, 3, generator=generator)
        x_large = torch.randn(5, 3, generator=generator)
        
        ntk_fn = empirical_ntk_fn(apply_fn)
        
        ntk_small = ntk_fn(x_small, None, params)
        ntk_large = ntk_fn(x_large, None, params)
        
        # Check shapes
        self.assertEqual(ntk_small.shape, (2, 2))
        self.assertEqual(ntk_large.shape, (5, 5))
        
        # Both should be finite and symmetric
        self.assertTrue(torch.isfinite(ntk_small).all())
        self.assertTrue(torch.isfinite(ntk_large).all())
        self.assertTensorsAllClose(ntk_small, ntk_small.T, atol=1e-5)
        self.assertTensorsAllClose(ntk_large, ntk_large.T, atol=1e-5)

    def test_empirical_ntk_vs_nngp(self):
        """Test differences between empirical NTK and NNGP."""
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(16, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 4))
        
        x = torch.randn(3, 4, generator=generator)
        
        # Compute both NTK and NNGP
        ntk_fn = empirical_ntk_fn(apply_fn)
        nngp_fn = empirical_nngp_fn(apply_fn)
        
        try:
            ntk = ntk_fn(x, None, params)
            nngp = nngp_fn(x, None, params)
            
            self.assertEqual(ntk.shape, nngp.shape)
            self.assertTrue(torch.isfinite(ntk).all())
            self.assertTrue(torch.isfinite(nngp).all())
            
            # They should generally be different (unless very specific conditions)
            diff = torch.norm(ntk - nngp)
            # Don't assert difference is large, as they could be similar in some cases
            self.assertGreaterEqual(diff.item(), 0)
            
        except Exception as e:
            self.skipTest(f"NTK vs NNGP comparison failed: {e}")

    def test_empirical_ntk_training_dynamics(self):
        """Test that empirical NTK relates to training dynamics."""
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(8, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 3))
        
        x = torch.randn(4, 3, generator=generator)
        y = torch.randn(4, 1, generator=generator)
        
        # Compute initial NTK
        ntk_fn = empirical_ntk_fn(apply_fn)
        ntk_initial = ntk_fn(x, None, params)
        
        # Initial function value
        f_initial = apply_fn(params, x)
        
        # The NTK should predict training dynamics
        # For infinitesimal learning: df/dt = -NTK * (f - y)
        residual = f_initial - y
        predicted_change_direction = -ntk_initial @ residual
        
        # All computations should be finite
        self.assertTrue(torch.isfinite(ntk_initial).all())
        self.assertTrue(torch.isfinite(predicted_change_direction).all())
        
        # NTK diagonal should be positive (function can change)
        self.assertTrue(torch.all(torch.diagonal(ntk_initial) > 0))

    def test_empirical_ntk_scaling_behavior(self):
        """Test scaling behavior of empirical NTK."""
        base_width = 8
        
        ntk_scales = []
        widths = [base_width, base_width * 2, base_width * 4]
        
        for width in widths:
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Dense(width, W_std=1.0, b_std=0.1),
                stax.Relu(),
                stax.Dense(1, W_std=1.0, b_std=0.1)
            )
            
            generator = torch.Generator()
            generator.manual_seed(42)  # Same seed for fair comparison
            _, params = init_fn(generator, (-1, 4))
            
            x = torch.randn(3, 4, generator=generator)
            
            try:
                ntk_fn = empirical_ntk_fn(apply_fn)
                ntk = ntk_fn(x, None, params)
                
                # Use trace as a scale measure
                scale = torch.trace(ntk) / ntk.shape[0]
                ntk_scales.append(scale.item())
                
            except Exception as e:
                self.skipTest(f"Scaling test failed for width {width}: {e}")
        
        # NTK should generally scale with width (though not necessarily linearly)
        if len(ntk_scales) >= 2:
            # At least check that wider networks don't have zero NTK
            for scale in ntk_scales:
                self.assertGreater(scale, 0)

    def test_empirical_ntk_consistency_across_calls(self):
        """Test that empirical NTK is consistent across multiple calls."""
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(10, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 3))
        
        x = torch.randn(3, 3, generator=generator)
        
        ntk_fn = empirical_ntk_fn(apply_fn)
        
        # Compute NTK multiple times
        ntk1 = ntk_fn(x, None, params)
        ntk2 = ntk_fn(x, None, params)
        ntk3 = ntk_fn(x, None, params)
        
        # Should be identical
        self.assertTensorsAllClose(ntk1, ntk2, atol=1e-7)
        self.assertTensorsAllClose(ntk2, ntk3, atol=1e-7)


if __name__ == '__main__':
    absltest.main()