"""Tests for torchtangents elementwise functionality.

Tests adapted from neural-tangents elementwise_test.py to verify 
elementwise operations in PyTorch implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
import torch.nn as nn
from absl.testing import absltest

from .test_utils import TorchTangentsTestCase, parameters
import torchtangents.stax as stax


class ElementwiseTest(TorchTangentsTestCase):
    """Test elementwise operations."""

    def test_elementwise_activations(self):
        """Test various elementwise activation functions."""
        activations = [
            ('Relu', stax.Relu()),
            ('Sigmoid', stax.Sigmoid()),
            ('Tanh', stax.Tanh()),
            ('Identity', stax.Identity()),
        ]
        
        for name, activation in activations:
            with self.subTest(activation=name):
                init_fn, apply_fn, kernel_fn = activation
                
                # Test initialization
                generator = torch.Generator()
                generator.manual_seed(42)
                input_shape = (-1, 10)
                output_shape, params = init_fn(generator, input_shape)
                
                # Should preserve shape
                self.assertEqual(output_shape, input_shape)
                
                # Test application
                x = torch.randn(5, 10)
                y = apply_fn(params, x)
                self.assertEqual(y.shape, x.shape)

    def test_relu_properties(self):
        """Test specific properties of ReLU activation."""
        init_fn, apply_fn, kernel_fn = stax.Relu()
        
        generator = torch.Generator()
        input_shape = (-1, 5)
        _, params = init_fn(generator, input_shape)
        
        # Test ReLU behavior
        x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        y = apply_fn(params, x)
        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0]])
        
        self.assertTensorsAllClose(y, expected)

    def test_sigmoid_range(self):
        """Test that sigmoid output is in valid range."""
        init_fn, apply_fn, kernel_fn = stax.Sigmoid()
        
        generator = torch.Generator()
        input_shape = (-1, 10)
        _, params = init_fn(generator, input_shape)
        
        # Test with extreme values
        x = torch.tensor([[-100.0, -10.0, 0.0, 10.0, 100.0]])
        y = apply_fn(params, x)
        
        # Sigmoid should be in [0, 1]
        self.assertTrue(torch.all(y >= 0.0))
        self.assertTrue(torch.all(y <= 1.0))

    def test_tanh_range(self):
        """Test that tanh output is in valid range."""
        init_fn, apply_fn, kernel_fn = stax.Tanh()
        
        generator = torch.Generator()
        input_shape = (-1, 10)
        _, params = init_fn(generator, input_shape)
        
        # Test with extreme values
        x = torch.tensor([[-100.0, -10.0, 0.0, 10.0, 100.0]])
        y = apply_fn(params, x)
        
        # Tanh should be in [-1, 1]
        self.assertTrue(torch.all(y >= -1.0))
        self.assertTrue(torch.all(y <= 1.0))

    def test_identity_preservation(self):
        """Test that identity preserves input exactly."""
        init_fn, apply_fn, kernel_fn = stax.Identity()
        
        generator = torch.Generator()
        input_shape = (-1, 8)
        _, params = init_fn(generator, input_shape)
        
        x = torch.randn(3, 8)
        y = apply_fn(params, x)
        
        self.assertTensorsAllClose(x, y)

    @parameters([
        {'batch_size': 1, 'features': 5},
        {'batch_size': 4, 'features': 10},
        {'batch_size': 8, 'features': 3},
    ])
    def test_elementwise_batch_handling(self, batch_size, features):
        """Test elementwise operations with different batch sizes."""
        activations = [stax.Relu(), stax.Sigmoid(), stax.Tanh()]
        
        for activation in activations:
            with self.subTest(activation=str(activation)):
                init_fn, apply_fn, kernel_fn = activation
                
                generator = torch.Generator()
                generator.manual_seed(42)
                input_shape = (-1, features)
                _, params = init_fn(generator, input_shape)
                
                x = torch.randn(batch_size, features)
                y = apply_fn(params, x)
                
                # Should preserve batch and feature dimensions
                self.assertEqual(y.shape, (batch_size, features))

    def test_elementwise_gradient_flow(self):
        """Test that gradients flow through elementwise operations."""
        init_fn, apply_fn, kernel_fn = stax.Relu()
        
        generator = torch.Generator()
        input_shape = (-1, 3)
        _, params = init_fn(generator, input_shape)
        
        # Input that requires gradients
        x = torch.tensor([[1.0, -1.0, 0.5]], requires_grad=True)
        y = apply_fn(params, x)
        
        # Compute loss and backward
        loss = y.sum()
        loss.backward()
        
        # Check gradients
        expected_grad = torch.tensor([[1.0, 0.0, 1.0]])  # ReLU derivative
        self.assertTensorsAllClose(x.grad, expected_grad)

    def test_elementwise_dtype_preservation(self):
        """Test that elementwise operations preserve dtype."""
        init_fn, apply_fn, kernel_fn = stax.Relu()
        
        generator = torch.Generator()
        input_shape = (-1, 4)
        _, params = init_fn(generator, input_shape)
        
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 4, dtype=dtype)
                y = apply_fn(params, x)
                
                self.assertEqual(y.dtype, dtype)

    def test_elementwise_device_consistency(self):
        """Test elementwise operations work on different devices."""
        init_fn, apply_fn, kernel_fn = stax.Relu()
        
        generator = torch.Generator()
        input_shape = (-1, 4)
        _, params = init_fn(generator, input_shape)
        
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        for device in devices:
            with self.subTest(device=device):
                x = torch.randn(2, 4, device=device)
                
                # Move params to device (if they have any)
                if params:
                    params_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in params.items()}
                else:
                    params_device = params
                
                y = apply_fn(params_device, x)
                
                self.assertEqual(y.device.type, device)


if __name__ == '__main__':
    absltest.main()