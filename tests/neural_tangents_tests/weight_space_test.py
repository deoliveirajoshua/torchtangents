"""Tests for torchtangents weight space functionality.

Tests adapted from neural-tangents weight_space_test.py to verify 
weight space dynamics in PyTorch implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
import torch.nn as nn
from absl.testing import absltest

from .test_utils import TorchTangentsTestCase, product, parameters
import torchtangents.stax as stax
from torchtangents import empirical_ntk_fn, empirical_nngp_fn


class WeightSpaceTest(TorchTangentsTestCase):
    """Test weight space dynamics."""

    def test_weight_space_basic(self):
        """Test basic weight space functionality."""
        # Create a simple network
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(32, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        # Initialize parameters
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 4))
        
        # Test data
        x = torch.randn(8, 4, generator=generator)
        y = torch.randn(8, 1, generator=generator)
        
        # Test that we can compute function values
        fx = apply_fn(params, x)
        self.assertEqual(fx.shape, y.shape)
        self.assertTrue(torch.isfinite(fx).all())
        
        # Test that we can compute NTK
        ntk_fn = empirical_ntk_fn(apply_fn)
        ntk = ntk_fn(x, None, params)
        
        self.assertEqual(ntk.shape, (8, 8))
        self.assertTrue(torch.isfinite(ntk).all())
        
        # Test symmetry
        self.assertTensorsAllClose(ntk, ntk.T, atol=1e-5)

    def test_weight_space_gradient_descent(self):
        """Test weight space gradient descent."""
        # Simple network for testing
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(16, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 3))
        
        # Make parameters require grad
        def make_trainable(param_structure):
            """Helper to make parameters trainable regardless of structure."""
            if isinstance(param_structure, dict):
                trainable = {}
                for k, v in param_structure.items():
                    if v is not None:
                        trainable[k] = v.clone().requires_grad_(True)
                    else:
                        trainable[k] = v
                return trainable
            elif isinstance(param_structure, list):
                return [make_trainable(layer_params) for layer_params in param_structure]
            else:
                return param_structure
        
        trainable_params = make_trainable(params)
        
        # Test data
        x = torch.randn(4, 3, generator=generator)
        y = torch.randn(4, 1, generator=generator)
        
        # Define loss
        def loss_fn(params, x, y):
            pred = apply_fn(params, x)
            return 0.5 * torch.mean((pred - y) ** 2)
        
        # Initial loss
        initial_loss = loss_fn(trainable_params, x, y)
        self.assertTrue(torch.isfinite(initial_loss))
        
        # Compute gradients
        initial_loss.backward()
        
        # Check that gradients exist
        def check_gradients(param_structure):
            """Helper to check gradients regardless of structure."""
            if isinstance(param_structure, dict):
                for k, v in param_structure.items():
                    if v is not None and v.requires_grad:
                        self.assertIsNotNone(v.grad)
                        self.assertTrue(torch.isfinite(v.grad).all())
            elif isinstance(param_structure, list):
                for layer_params in param_structure:
                    check_gradients(layer_params)
        
        check_gradients(trainable_params)

    def test_parameter_space_properties(self):
        """Test properties of parameter space."""
        # Create network
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(8, W_std=1.0, b_std=0.1),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        # Test multiple parameter initializations
        params_list = []
        for seed in [42, 123, 456]:
            gen = torch.Generator()
            gen.manual_seed(seed)
            _, params = init_fn(gen, (-1, 4))
            params_list.append(params)
        
        x = torch.randn(3, 4, generator=generator)
        
        # Test that different parameters give different outputs
        outputs = []
        for params in params_list:
            output = apply_fn(params, x)
            outputs.append(output)
        
        # Check that outputs are different
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                diff = torch.norm(outputs[i] - outputs[j])
                self.assertGreater(diff.item(), 1e-6)

    def test_weight_space_ntk_properties(self):
        """Test NTK properties in weight space."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(16, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(2, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 4))
        
        x1 = torch.randn(3, 4, generator=generator)
        x2 = torch.randn(2, 4, generator=generator)
        
        ntk_fn = empirical_ntk_fn(apply_fn)
        
        # Test NTK computation
        ntk_11 = ntk_fn(x1, None, params)
        ntk_12 = ntk_fn(x1, x2, params)
        ntk_22 = ntk_fn(x2, None, params)
        
        # Check shapes
        self.assertEqual(ntk_11.shape, (3, 3))
        self.assertEqual(ntk_12.shape, (3, 2))
        self.assertEqual(ntk_22.shape, (2, 2))
        
        # Check finite values
        self.assertTrue(torch.isfinite(ntk_11).all())
        self.assertTrue(torch.isfinite(ntk_12).all())
        self.assertTrue(torch.isfinite(ntk_22).all())
        
        # Check symmetry for diagonal blocks
        self.assertTensorsAllClose(ntk_11, ntk_11.T, atol=1e-5)
        self.assertTensorsAllClose(ntk_22, ntk_22.T, atol=1e-5)

    def test_weight_initialization_effects(self):
        """Test effects of different weight initializations."""
        x = torch.randn(4, 3)
        
        results = {}
        
        for w_std, b_std in [(0.5, 0.1), (1.0, 0.1), (2.0, 0.1)]:
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(8, W_std=w_std, b_std=b_std),
                stax.Relu(),
                stax.Dense(1, W_std=w_std, b_std=b_std)
            )
            
            generator = torch.Generator()
            generator.manual_seed(42)
            _, params = init_fn(generator, (-1, 3))
            
            output = apply_fn(params, x)
            ntk_fn = empirical_ntk_fn(apply_fn)
            ntk = ntk_fn(x, None, params)
            
            results[(w_std, b_std)] = {
                'output': output,
                'ntk': ntk,
                'output_scale': torch.std(output),
                'ntk_scale': torch.mean(torch.diagonal(ntk))
            }
        
        # Check that different initializations give different results
        scales = [results[k]['output_scale'] for k in results]
        for i in range(len(scales)):
            for j in range(i + 1, len(scales)):
                # Different weight scales should give different output scales
                self.assertNotEqual(scales[i].item(), scales[j].item())


if __name__ == '__main__':
    absltest.main()