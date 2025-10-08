"""Tests for torchtangents function space functionality.

Tests adapted from neural-tangents function_space_test.py to verify 
function space dynamics in PyTorch implementation.
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


class FunctionSpaceTest(TorchTangentsTestCase):
    """Test function space dynamics."""

    def test_function_space_basic(self):
        """Test basic function space functionality."""
        # Create a simple network
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(16, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        # Initialize parameters
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 4))
        
        # Test data
        x = torch.randn(6, 4, generator=generator)
        
        # Test function evaluation
        fx = apply_fn(params, x)
        self.assertEqual(fx.shape, (6, 1))
        self.assertTrue(torch.isfinite(fx).all())
        
        # Test that function varies with input
        x_different = torch.randn(6, 4, generator=generator) + 1.0
        fx_different = apply_fn(params, x_different)
        
        diff = torch.norm(fx - fx_different)
        self.assertGreater(diff.item(), 1e-6)

    def test_function_space_linearization(self):
        """Test function space linearization properties."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(8, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 3))
        
        x0 = torch.randn(1, 3, generator=generator)
        
        # Compute function at x0
        fx0 = apply_fn(params, x0)
        
        # Test linearization around x0
        def compute_linearization(x_new):
            # Compute Jacobian at x0
            x0_copy = x0.clone().requires_grad_(True)
            fx0_copy = apply_fn(params, x0_copy)
            
            # Compute gradient w.r.t. input
            grad_x = torch.autograd.grad(fx0_copy.sum(), x0_copy, 
                                       create_graph=True)[0]
            
            # Linear approximation
            dx = x_new - x0
            linear_pred = fx0 + torch.sum(grad_x * dx, dim=-1, keepdim=True)
            return linear_pred
        
        # Test small perturbation
        eps = 0.01
        x_new = x0 + eps * torch.randn_like(x0)
        fx_new = apply_fn(params, x_new)
        fx_linear = compute_linearization(x_new)
        
        # For small perturbations, linearization should be close
        error = torch.abs(fx_new - fx_linear)
        self.assertLess(error.item(), eps * 10)  # Allow some nonlinearity

    def test_function_space_ntk_relationship(self):
        """Test relationship between function space and NTK."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(16, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 4))
        
        x = torch.randn(4, 4, generator=generator)
        
        # Compute NTK
        ntk_fn = empirical_ntk_fn(apply_fn)
        ntk = ntk_fn(x, None, params)
        
        # Test that NTK captures function correlations
        self.assertEqual(ntk.shape, (4, 4))
        self.assertTrue(torch.isfinite(ntk).all())
        
        # NTK should be symmetric
        self.assertTensorsAllClose(ntk, ntk.T, atol=1e-5)
        
        # Diagonal elements should be positive
        diag_elements = torch.diagonal(ntk)
        self.assertTrue(torch.all(diag_elements > 0))

    def test_function_composition(self):
        """Test function composition properties."""
        # Create two small networks to compose
        init_fn1, apply_fn1, _ = stax.Dense(8, W_std=1.0, b_std=0.1)
        init_fn2, apply_fn2, _ = stax.serial(
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        # Initialize both networks
        _, params1 = init_fn1(generator, (-1, 4))
        _, params2 = init_fn2(generator, (-1, 8))
        
        x = torch.randn(3, 4, generator=generator)
        
        # Compute composition
        h1 = apply_fn1(params1, x)
        output = apply_fn2(params2, h1)
        
        self.assertEqual(output.shape, (3, 1))
        self.assertTrue(torch.isfinite(output).all())
        
        # Test that intermediate representation is reasonable
        self.assertEqual(h1.shape, (3, 8))
        self.assertTrue(torch.isfinite(h1).all())

    def test_function_approximation_capacity(self):
        """Test function approximation capacity."""
        # Test different network widths
        x = torch.randn(5, 3)
        target = torch.sin(torch.sum(x, dim=1, keepdim=True))  # Simple target
        
        for width in [4, 8, 16]:
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(width, W_std=1.0, b_std=0.1),
                stax.Relu(),
                stax.Dense(width // 2, W_std=1.0, b_std=0.1),
                stax.Relu(),
                stax.Dense(1, W_std=1.0, b_std=0.1)
            )
            
            generator = torch.Generator()
            generator.manual_seed(42)
            _, params = init_fn(generator, (-1, 3))
            
            output = apply_fn(params, x)
            
            # Should be able to produce some output
            self.assertEqual(output.shape, target.shape)
            self.assertTrue(torch.isfinite(output).all())
            
            # Different widths should give different outputs
            # (unless by coincidence)
            self.assertGreater(torch.std(output).item(), 1e-6)

    def test_function_smoothness(self):
        """Test function smoothness properties."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(8, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 2))
        
        # Test function on a grid
        x_values = torch.linspace(-2, 2, 10)
        y_values = torch.linspace(-2, 2, 10)
        
        outputs = []
        for x_val in x_values:
            for y_val in y_values:
                x_point = torch.tensor([[x_val, y_val]])
                output = apply_fn(params, x_point)
                outputs.append(output.item())
        
        outputs = torch.tensor(outputs)
        
        # Function should be relatively smooth (no extreme values)
        self.assertTrue(torch.all(torch.abs(outputs) < 100))  # Reasonable bounds
        self.assertTrue(torch.all(torch.isfinite(outputs)))

    def test_function_parameter_sensitivity(self):
        """Test sensitivity to parameter changes."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(4, W_std=1.0, b_std=0.1),
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        _, params = init_fn(generator, (-1, 2))
        
        x = torch.randn(3, 2, generator=generator)
        
        # Original output
        fx_original = apply_fn(params, x)
        
        # Perturb parameters slightly
        def perturb_params(param_structure, eps=0.01):
            """Helper to perturb parameters regardless of structure (dict or list)."""
            if isinstance(param_structure, dict):
                perturbed = {}
                for k, v in param_structure.items():
                    if v is not None:
                        noise = torch.randn_like(v) * eps
                        perturbed[k] = v + noise
                    else:
                        perturbed[k] = v
                return perturbed
            elif isinstance(param_structure, list):
                return [perturb_params(layer_params, eps) for layer_params in param_structure]
            else:
                return param_structure
        
        params_perturbed = perturb_params(params)
        fx_perturbed = apply_fn(params_perturbed, x)
        
        # Function should change with parameter perturbation
        diff = torch.norm(fx_original - fx_perturbed)
        self.assertGreater(diff.item(), 1e-8)
        self.assertLess(diff.item(), 10.0)  # But not too much


if __name__ == '__main__':
    absltest.main()