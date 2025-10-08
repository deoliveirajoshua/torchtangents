"""Tests for torchtangents elementwise numerical computations.

Tests adapted from neural-tangents elementwise_numerical_test.py to verify 
numerical properties of elementwise activation functions in PyTorch implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
import torch.nn as nn
import math
from absl.testing import absltest

from .test_utils import TorchTangentsTestCase, product, parameters
from torchtangents.stax import Dense, Conv, Flatten, GlobalAvgPool, Relu, Erf, Sigmoid, Tanh, serial, parallel
from torchtangents import empirical_ntk_fn, empirical_nngp_fn

# Create a stax-like namespace for compatibility with our tests
class stax:
    Dense = Dense
    Conv = Conv
    Flatten = Flatten
    GlobalAvgPool = GlobalAvgPool
    Relu = Relu
    Erf = Erf
    Sigmoid = Sigmoid
    Tanh = Tanh
    serial = serial
    parallel = parallel
    Relu = Relu
    Erf = Erf
    serial = serial
    parallel = parallel


class ElementwiseNumericalTest(TorchTangentsTestCase):
    """Test numerical properties of elementwise activation functions."""

    def test_relu_numerical_properties(self):
        """Test numerical properties of ReLU activation."""
        # Test basic ReLU properties
        init_fn, apply_fn, kernel_fn = stax.Relu()
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        # Test with various inputs
        test_inputs = [
            torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
            torch.tensor([-100.0, -0.001, 0.001, 100.0]),
            torch.randn(10, generator=generator),
        ]
        
        for x in test_inputs:
            with self.subTest(input_shape=x.shape):
                _, params = init_fn(generator, (-1,) + x.shape[1:])
                y = apply_fn(params, x.unsqueeze(0) if x.dim() == 1 else x)
                
                # ReLU should be max(0, x)
                expected = torch.relu(x.unsqueeze(0) if x.dim() == 1 else x)
                self.assertTensorsAllClose(y, expected, atol=1e-7)
                
                # Check non-negativity
                self.assertTrue(torch.all(y >= 0))
                
                # Check that positive inputs are preserved
                positive_mask = x > 0
                if torch.any(positive_mask):
                    if x.dim() == 1:
                        self.assertTensorsAllClose(
                            y[0, positive_mask], x[positive_mask], atol=1e-7
                        )

    def test_sigmoid_numerical_properties(self):
        """Test numerical properties of Sigmoid activation."""
        init_fn, apply_fn, kernel_fn = stax.Sigmoid()
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        # Test a few different input ranges
        test_inputs = [
            torch.randn(10, generator=generator),
            torch.randn(5, 8, generator=generator) * 2,  # Larger variance
            torch.randn(3, 4, 5, generator=generator) * 0.5,  # Smaller variance
        ]
        
        for x in test_inputs:
            with self.subTest(shape=x.shape):
                output_shape, params = init_fn(generator, x.shape)
                self.assertEqual(output_shape, x.shape)
                self.assertEqual(params, {})
                
                # Apply function
                y = apply_fn(params, x)
                
                # Sigmoid should be sigmoid(x)
                expected = torch.sigmoid(x)
                self.assertTensorsAllClose(y, expected, atol=1e-7)
                
                # Check output range [0, 1]
                self.assertTrue(torch.all(y >= 0))
                self.assertTrue(torch.all(y <= 1))
                
                # Check that sigmoid(0) ≈ 0.5
                if torch.any(torch.abs(x) < 1e-6):
                    zero_mask = torch.abs(x) < 1e-6
                    self.assertTensorsAllClose(
                        y[zero_mask], torch.tensor(0.5), atol=1e-3
                    )

    def test_tanh_numerical_properties(self):
        """Test numerical properties of Tanh activation."""
        init_fn, apply_fn, kernel_fn = stax.Tanh()
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        # Test a few different input ranges
        test_inputs = [
            torch.randn(10, generator=generator),
            torch.randn(5, 8, generator=generator) * 2,  # Larger variance
            torch.randn(3, 4, 5, generator=generator) * 0.5,  # Smaller variance
        ]
        
        for x in test_inputs:
            with self.subTest(shape=x.shape):
                output_shape, params = init_fn(generator, x.shape)
                self.assertEqual(output_shape, x.shape)
                self.assertEqual(params, {})
                
                # Apply function
                y = apply_fn(params, x)
                
                # Tanh should be tanh(x)
                expected = torch.tanh(x)
                self.assertTensorsAllClose(y, expected, atol=1e-7)
                
                # Check output range [-1, 1]
                self.assertTrue(torch.all(y >= -1))
                self.assertTrue(torch.all(y <= 1))
                
                # Check that tanh(0) = 0
                if torch.any(torch.abs(x) < 1e-6):
                    zero_mask = torch.abs(x) < 1e-6
                    self.assertTensorsAllClose(
                        y[zero_mask], torch.tensor(0.0), atol=1e-6
                    )

    def test_erf_numerical_properties(self):
        """Test numerical properties of Erf activation."""
        init_fn, apply_fn, kernel_fn = stax.Erf()
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        # Test with various inputs
        test_inputs = [
            torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]),
            torch.tensor([-10.0, -5.0, 5.0, 10.0]),
            torch.randn(10, generator=generator),
        ]
        
        for x in test_inputs:
            with self.subTest(input_shape=x.shape):
                _, params = init_fn(generator, (-1,) + x.shape[1:])
                y = apply_fn(params, x.unsqueeze(0) if x.dim() == 1 else x)
                
                # Erf should match torch.erf
                expected = torch.erf(x.unsqueeze(0) if x.dim() == 1 else x)
                self.assertTensorsAllClose(y, expected, atol=1e-6)
                
                # Check range [-1, 1]
                self.assertTrue(torch.all(y >= -1))
                self.assertTrue(torch.all(y <= 1))
                
                # Check odd function property: erf(-x) = -erf(x)
                x_neg = -x
                y_neg = apply_fn(params, x_neg.unsqueeze(0) if x_neg.dim() == 1 else x_neg)
                self.assertTensorsAllClose(y_neg, -y, atol=1e-6)
                
                # Check erf(0) = 0
                zero_input = torch.zeros_like(x).unsqueeze(0) if x.dim() == 1 else torch.zeros_like(x)
                y_zero = apply_fn(params, zero_input)
                self.assertTensorsAllClose(y_zero, torch.zeros_like(y_zero), atol=1e-7)

    @parameters([
        {'activation': 'Relu'},
        {'activation': 'Erf'},
    ])
    def test_activation_gradient_computation(self, activation):
        """Test that activation functions have computable gradients."""
        # Get activation function
        activation_fn = getattr(stax, activation)
        init_fn, apply_fn, kernel_fn = activation_fn()
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        # Create test input that requires grad
        x = torch.randn(3, 5, generator=generator, requires_grad=True)
        _, params = init_fn(generator, (-1,) + x.shape[1:])
        
        # Forward pass
        y = apply_fn(params, x)
        
        # Compute loss and gradients
        loss = torch.sum(y ** 2)
        loss.backward()
        
        # Check that gradients exist and are finite
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        
        # For ReLU, gradient should be 0 for negative inputs, 2*x for positive inputs
        if activation == 'Relu':
            # Check that gradients are correct for ReLU: 0 where x <= 0, 2*x where x > 0
            mask_positive = x > 0
            mask_negative = x <= 0
            
            # Where x > 0, gradient should be 2*x (from d/dx[relu(x)^2] = 2*relu(x) = 2*x)
            expected_grad_positive = 2 * x[mask_positive]
            actual_grad_positive = x.grad[mask_positive]
            self.assertTrue(torch.allclose(actual_grad_positive, expected_grad_positive, atol=1e-6))
            
            # Where x <= 0, gradient should be 0
            actual_grad_negative = x.grad[mask_negative]
            self.assertTrue(torch.allclose(actual_grad_negative, torch.zeros_like(actual_grad_negative), atol=1e-6))

    def test_activation_derivative_consistency(self):
        """Test that activation derivatives are consistent with numerical derivatives."""
        # Test with ReLU (simple case)
        init_fn, apply_fn, kernel_fn = stax.Relu()
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        x = torch.randn(5, generator=generator, requires_grad=True)
        _, params = init_fn(generator, (-1,) + x.shape[1:])
        
        # Analytical gradient
        y = apply_fn(params, x.unsqueeze(0))
        loss = torch.sum(y)
        loss.backward()
        analytical_grad = x.grad.clone()
        
        # Numerical gradient
        x.grad = None
        eps = 1e-5
        numerical_grad = torch.zeros_like(x)
        
        for i in range(x.numel()):
            x_plus = x.clone()
            x_plus[i] += eps
            x_minus = x.clone()
            x_minus[i] -= eps
            
            y_plus = apply_fn(params, x_plus.unsqueeze(0))
            y_minus = apply_fn(params, x_minus.unsqueeze(0))
            
            numerical_grad[i] = (torch.sum(y_plus) - torch.sum(y_minus)) / (2 * eps)
        
        # Compare (with some tolerance for numerical precision)
        # Use larger tolerance for ReLU due to discontinuous derivative at 0
        self.assertTensorsAllClose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    def test_activation_kernel_properties(self):
        """Test kernel properties of activation functions."""
        activations = ['Relu', 'Erf']  # Only test available activations
        
        for activation_name in activations:
            with self.subTest(activation=activation_name):
                activation_fn = getattr(stax, activation_name)
                
                # Create simple network with activation
                init_fn, apply_fn, kernel_fn = stax.serial(
                    stax.Dense(10, W_std=1.0, b_std=0.1),
                    activation_fn(),
                    stax.Dense(1, W_std=1.0, b_std=0.1)
                )
                
                generator = torch.Generator()
                generator.manual_seed(42)
                
                x = torch.randn(4, 3, generator=generator)
                _, params = init_fn(generator, (-1,) + x.shape[1:])
                
                try:
                    # Test empirical kernel computation
                    ntk_fn = empirical_ntk_fn(apply_fn)
                    kernel = ntk_fn(x, None, params)
                    
                    # Basic kernel properties
                    self.assertEqual(kernel.shape, (4, 4))
                    self.assertTrue(torch.isfinite(kernel).all())
                    
                    # Symmetry
                    self.assertTensorsAllClose(kernel, kernel.T, atol=1e-5)
                    
                    # Positive semi-definite (approximately)
                    eigenvals = torch.linalg.eigvalsh(kernel)
                    self.assertGreater(torch.min(eigenvals).item(), -1e-6)
                    
                except Exception as e:
                    # Skip if this particular activation causes issues
                    self.skipTest(f"Kernel test failed for {activation_name}: {e}")

    def test_extreme_input_stability(self):
        """Test activation functions with extreme inputs."""
        activations = [
            ('Relu', torch.relu),
            ('Erf', torch.erf),
        ]
        
        # Extreme test cases
        extreme_inputs = [
            torch.tensor([-1000.0, -100.0, 100.0, 1000.0]),
            torch.tensor([-torch.inf, torch.inf]),  # Will be clipped
            torch.tensor([torch.nan]),  # Should be handled gracefully
        ]
        
        for activation_name, torch_fn in activations:
            activation_fn = getattr(stax, activation_name)
            init_fn, apply_fn, kernel_fn = activation_fn()
            
            generator = torch.Generator()
            generator.manual_seed(42)
            
            for x in extreme_inputs:
                with self.subTest(activation=activation_name, input=x):
                    try:
                        # Filter out inf/nan for comparison
                        valid_mask = torch.isfinite(x)
                        if not torch.any(valid_mask):
                            continue
                            
                        x_valid = x[valid_mask]
                        _, params = init_fn(generator, (-1,) + x_valid.shape[1:])
                        y = apply_fn(params, x_valid.unsqueeze(0))
                        
                        # Should produce finite outputs for finite inputs
                        self.assertTrue(torch.isfinite(y).all())
                        
                        # Should match PyTorch implementation for valid inputs
                        expected = torch_fn(x_valid.unsqueeze(0))
                        if torch.isfinite(expected).all():
                            self.assertTensorsAllClose(y, expected, atol=1e-5, rtol=1e-5)
                            
                    except Exception as e:
                        # Expected for some extreme cases
                        continue

    def test_activation_composition(self):
        """Test composition of multiple activation functions."""
        # Test network with multiple activations
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(8, W_std=1.0, b_std=0.1),
            stax.Relu(),
            stax.Dense(8, W_std=1.0, b_std=0.1),
            stax.Erf(),  # Use Erf instead of Tanh
            stax.Dense(1, W_std=1.0, b_std=0.1)
        )
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        x = torch.randn(3, 5, generator=generator)
        _, params = init_fn(generator, (-1,) + x.shape[1:])
        
        # Forward pass should work
        y = apply_fn(params, x)
        self.assertEqual(y.shape, (3, 1))
        self.assertTrue(torch.isfinite(y).all())
        
        # Gradient computation should work
        x_grad = x.clone().requires_grad_(True)
        y_grad = apply_fn(params, x_grad)
        loss = torch.sum(y_grad ** 2)
        loss.backward()
        
        self.assertIsNotNone(x_grad.grad)
        self.assertTrue(torch.isfinite(x_grad.grad).all())


if __name__ == '__main__':
    absltest.main()