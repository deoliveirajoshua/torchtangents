"""Tests for torchtangents stax layers.

Basic tests to verify stax layer functionality adapted from neural-tangents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
from absl.testing import absltest

from ..test_utils import TorchTangentsTestCase, parameters
import torchtangents.stax as stax


class StaxTest(TorchTangentsTestCase):
    """Test stax layers."""

    def test_dense_layer(self):
        """Test Dense layer creation and usage."""
        init_fn, apply_fn, kernel_fn = stax.Dense(10, W_std=1.0, b_std=0.1)
        
        # Test initialization
        key = torch.Generator()
        key.manual_seed(42)
        input_shape = (-1, 5)
        output_shape, params = init_fn(key, input_shape)
        
        self.assertEqual(output_shape, (-1, 10))
        self.assertIn('weight', params)
        self.assertIn('bias', params)
        self.assertEqual(params['weight'].shape, (5, 10))
        self.assertEqual(params['bias'].shape, (10,))
        
        # Test forward pass
        x = torch.randn(3, 5)
        y = apply_fn(params, x)
        self.assertEqual(y.shape, (3, 10))

    def test_conv_layer(self):
        """Test Conv layer creation and usage."""
        init_fn, apply_fn, kernel_fn = stax.Conv(8, (3, 3), padding='SAME', W_std=1.0, b_std=0.1)
        
        # Test initialization
        key = torch.Generator()
        key.manual_seed(42)
        input_shape = (-1, 3, 32, 32)  # NCHW format
        output_shape, params = init_fn(key, input_shape)
        
        self.assertEqual(output_shape, (-1, 8, 32, 32))
        self.assertIn('weight', params)
        self.assertIn('bias', params)
        self.assertEqual(params['weight'].shape, (8, 3, 3, 3))  # out_ch, in_ch, h, w
        self.assertEqual(params['bias'].shape, (8,))
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        y = apply_fn(params, x)
        self.assertEqual(y.shape, (2, 8, 32, 32))

    def test_relu_layer(self):
        """Test ReLU layer."""
        init_fn, apply_fn, kernel_fn = stax.Relu()
        
        # Test initialization (ReLU has no parameters)
        key = torch.Generator()
        input_shape = (-1, 10)
        output_shape, params = init_fn(key, input_shape)
        
        self.assertEqual(output_shape, input_shape)
        self.assertEqual(params, {})
        
        # Test forward pass
        x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
        y = apply_fn(params, x)
        expected = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
        self.assertTensorsAllClose(y, expected)

    @parameters([
        {'layers': [stax.Dense(5), stax.Relu(), stax.Dense(2)]},
        {'layers': [stax.Conv(4, (3, 3)), stax.Relu(), stax.Flatten(), stax.Dense(3)]},
    ])
    def test_serial_composition(self, layers):
        """Test serial composition of layers."""
        init_fn, apply_fn, kernel_fn = stax.serial(*layers)
        
        # Test that we can initialize
        key = torch.Generator()
        key.manual_seed(42)
        
        if any('Conv' in str(layer) for layer in layers):
            input_shape = (-1, 3, 8, 8)
            x = torch.randn(2, 3, 8, 8)
        else:
            input_shape = (-1, 10)
            x = torch.randn(2, 10)
        
        output_shape, params = init_fn(key, input_shape)
        
        # Test that we can apply
        y = apply_fn(params, x)
        
        # Output should have correct batch dimension
        self.assertEqual(y.shape[0], x.shape[0])

    def test_parallel_composition(self):
        """Test parallel composition of layers."""
        init_fn, apply_fn, kernel_fn = stax.parallel(stax.Dense(3), stax.Dense(5))
        
        key = torch.Generator()
        key.manual_seed(42)
        input_shape = (-1, 4)
        output_shape, params = init_fn(key, input_shape)
        
        # Should return tuple of shapes
        self.assertIsInstance(output_shape, tuple)
        self.assertEqual(len(output_shape), 2)
        self.assertEqual(output_shape[0], (-1, 3))
        self.assertEqual(output_shape[1], (-1, 5))
        
        # Test forward pass
        x = torch.randn(2, 4)
        y = apply_fn(params, x)
        
        self.assertIsInstance(y, tuple)
        self.assertEqual(len(y), 2)
        self.assertEqual(y[0].shape, (2, 3))
        self.assertEqual(y[1].shape, (2, 5))

    def test_flatten_layer(self):
        """Test Flatten layer."""
        init_fn, apply_fn, kernel_fn = stax.Flatten()
        
        # Test initialization
        key = torch.Generator()
        input_shape = (-1, 3, 4, 5)
        output_shape, params = init_fn(key, input_shape)
        
        self.assertEqual(output_shape, (-1, 60))  # 3*4*5 = 60
        self.assertEqual(params, {})
        
        # Test forward pass
        x = torch.randn(2, 3, 4, 5)
        y = apply_fn(params, x)
        self.assertEqual(y.shape, (2, 60))


if __name__ == '__main__':
    absltest.main()