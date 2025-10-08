"""Tests for torchtangents empirical kernels.

This module contains tests adapted from neural-tangents to verify 
empirical kernel computation in PyTorch.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from absl.testing import absltest

from .test_utils import TorchTangentsTestCase, product, parameters
from torchtangents import empirical_ntk_fn, empirical_nngp_fn, NtkImplementation


TRAIN_SHAPES = [(4, 4), (4, 8), (6, 4, 4, 3)]
TEST_SHAPES = [(2, 4), (6, 8), (2, 4, 4, 3)]
OUTPUT_LOGITS = [1, 2, 3]


def _build_network(input_shape, out_logits):
    """Build a simple network for testing."""
    if len(input_shape) == 1:
        # Flat network
        return nn.Sequential(
            nn.Linear(input_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, out_logits)
        )
    elif len(input_shape) == 3:
        # CNN network
        return nn.Sequential(
            nn.Conv2d(input_shape[0], 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(8 * 2 * 2, out_logits)
        )
    else:
        raise ValueError(f'Unexpected input shape: {input_shape}')


class EmpiricalTest(TorchTangentsTestCase):
    """Test empirical kernel computation."""

    @parameters([
        {
            'train_shape': train_shape,
            'test_shape': test_shape,
            'out_logits': out_logits
        }
        for train_shape, test_shape, out_logits in zip(TRAIN_SHAPES, TEST_SHAPES, OUTPUT_LOGITS)
    ])
    def test_empirical_ntk_nngp_shapes(self, train_shape, test_shape, out_logits):
        """Test that empirical kernels have correct shapes."""
        # Build network
        if len(train_shape) == 2:
            input_shape = (train_shape[1],)
            x_train = torch.randn(*train_shape)
            x_test = torch.randn(*test_shape)
        else:
            input_shape = train_shape[1:]
            x_train = torch.randn(*train_shape)
            x_test = torch.randn(*test_shape)
        
        net = _build_network(input_shape, out_logits)
        
        def f(params, x):
            """Function that uses parameters."""
            try:
                return torch.func.functional_call(net, params, (x,))
            except AttributeError:
                # Fallback if functional_call not available
                original_params = {}
                for name, param in net.named_parameters():
                    original_params[name] = param.data.clone()
                    param.data = params[name]
                
                result = net(x)
                
                # Restore original parameters
                for name, param in net.named_parameters():
                    param.data = original_params[name]
                
                return result
        
        params = dict(net.named_parameters())
        
        # Test NNGP
        nngp_fn = empirical_nngp_fn(f)
        nngp = nngp_fn(x_train, x_test, params)
        
        expected_nngp_shape = (train_shape[0], test_shape[0])
        self.assertEqual(nngp.shape, expected_nngp_shape)
        
        # Test NTK with different implementations
        for implementation in [NtkImplementation.JACOBIAN_CONTRACTION, 
                             NtkImplementation.STRUCTURED_DERIVATIVES]:
            with self.subTest(implementation=implementation):
                ntk_fn = empirical_ntk_fn(f, implementation=implementation)
                ntk = ntk_fn(x_train, x_test, params)
                
                expected_ntk_shape = (train_shape[0], test_shape[0])
                self.assertEqual(ntk.shape, expected_ntk_shape)

    def test_empirical_ntk_diagonal_vs_full(self):
        """Test diagonal vs full NTK computation."""
        x = torch.randn(4, 3)
        net = nn.Linear(3, 2)
        
        def f(params, x):
            try:
                return torch.func.functional_call(net, params, (x,))
            except AttributeError:
                original_params = {}
                for name, param in net.named_parameters():
                    original_params[name] = param.data.clone()
                    param.data = params[name]
                
                result = net(x)
                
                for name, param in net.named_parameters():
                    param.data = original_params[name]
                
                return result
        
        params = dict(net.named_parameters())
        
        # Compute full NTK
        ntk_fn = empirical_ntk_fn(f)
        ntk_full = ntk_fn(x, None, params)
        
        # Extract diagonal
        ntk_diagonal = torch.diag(ntk_full)
        
        # Verify shapes
        self.assertEqual(ntk_full.shape, (4, 4))
        self.assertEqual(ntk_diagonal.shape, (4,))
        
        # Verify symmetry for same inputs
        self.assertTensorsAllClose(ntk_full, ntk_full.T, atol=1e-5)

    def test_empirical_nngp_properties(self):
        """Test basic properties of empirical NNGP."""
        x = torch.randn(3, 5)
        net = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
        
        def f(params, x):
            try:
                return torch.func.functional_call(net, params, (x,))
            except AttributeError:
                original_params = {}
                for name, param in net.named_parameters():
                    original_params[name] = param.data.clone()
                    param.data = params[name]
                
                result = net(x)
                
                for name, param in net.named_parameters():
                    param.data = original_params[name]
                
                return result
        
        params = dict(net.named_parameters())
        
        nngp_fn = empirical_nngp_fn(f)
        nngp = nngp_fn(x, None, params)
        
        # Test symmetry
        self.assertTensorsAllClose(nngp, nngp.T, atol=1e-6)
        
        # Test shape
        self.assertEqual(nngp.shape, (3, 3))

    def test_different_trace_axes(self):
        """Test NTK with different trace axes."""
        x = torch.randn(2, 4)
        net = nn.Linear(4, 3)
        
        def f(params, x):
            try:
                return torch.func.functional_call(net, params, (x,))
            except AttributeError:
                original_params = {}
                for name, param in net.named_parameters():
                    original_params[name] = param.data.clone()
                    param.data = params[name]
                
                result = net(x)
                
                for name, param in net.named_parameters():
                    param.data = original_params[name]
                
                return result
        
        params = dict(net.named_parameters())
        
        # Test different trace axes
        for trace_axes in [(-1,), (), (0,)]:
            with self.subTest(trace_axes=trace_axes):
                ntk_fn = empirical_ntk_fn(f, trace_axes=trace_axes)
                ntk = ntk_fn(x, None, params)
                
                # Should always return a valid tensor
                self.assertIsInstance(ntk, torch.Tensor)
                self.assertEqual(len(ntk.shape), 2)  # Should be 2D kernel


if __name__ == '__main__':
    absltest.main()