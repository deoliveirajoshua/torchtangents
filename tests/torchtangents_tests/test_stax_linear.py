# Copyright 2025 torchtangents contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehensive tests for torchtangents stax linear layers."""

import torch
import sys
import os
import math

# Add the torchtangents module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torchtangents as tt
from torchtangents import stax
from torchtangents._src.utils.kernel import Kernel


class TestDenseLayer:
    """Comprehensive tests for Dense layer."""
    
    def test_dense_basic_functionality(self):
        """Test basic Dense layer functionality."""
        out_dim = 10
        init_fn, apply_fn, kernel_fn = stax.Dense(out_dim, b_std=0.1)
        
        # Test initialization
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (5, 3)
        output_shape, params = init_fn(rng, input_shape)
        
        assert 'weight' in params
        assert 'bias' in params
        assert params['weight'].shape == (3, 10)  # (in_dim, out_dim)
        assert params['bias'].shape == (10,)  # Bias is 1D
        
        # Test application
        x = torch.randn(5, 3)
        output = apply_fn(params, x)
        assert output.shape == (5, 10)
        
        # Test that output follows the NTK parameterization
        # For NTK parameterization: (W_std / sqrt(in_dim)) * x @ W + b_std * b
        in_dim = 3
        expected = (1.0 / math.sqrt(in_dim)) * torch.matmul(x, params['weight']) + 0.1 * params['bias']
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_dense_without_bias(self):
        """Test Dense layer without bias."""
        init_fn, apply_fn, kernel_fn = stax.Dense(5)  # b_std=None by default
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (3, 4)
        output_shape, params = init_fn(rng, input_shape)
        
        assert 'weight' in params
        assert 'bias' in params
        assert params['weight'].shape == (4, 5)
        assert params['bias'] is None
        
        # Test application
        x = torch.randn(3, 4)
        output = apply_fn(params, x)
        assert output.shape == (3, 5)
        
        # Should be matrix multiplication with NTK scaling (no bias)
        in_dim = 4
        expected = (1.0 / math.sqrt(in_dim)) * torch.matmul(x, params['weight'])
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_dense_different_parameterizations(self):
        """Test Dense layer with different parameterizations."""
        # Test NTK parameterization
        init_fn_ntk, apply_fn_ntk, _ = stax.Dense(8, b_std=0.1, parameterization='ntk')
        
        # Test standard parameterization
        init_fn_std, apply_fn_std, _ = stax.Dense(8, b_std=0.1, parameterization='standard')
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (4, 6)
        
        _, params_ntk = init_fn_ntk(rng, input_shape)
        rng.manual_seed(42)  # Reset for fair comparison
        _, params_std = init_fn_std(rng, input_shape)
        
        # Both should have same structure
        assert params_ntk['weight'].shape == params_std['weight'].shape
        assert params_ntk['bias'].shape == params_std['bias'].shape
        
        # NTK parameterization should have different scaling
        # (implementation detail - the scaling factors should be different)
        x = torch.randn(4, 6)
        output_ntk = apply_fn_ntk(params_ntk, x)
        output_std = apply_fn_std(params_std, x)
        
        assert output_ntk.shape == output_std.shape == (4, 8)
    
    def test_dense_weight_and_bias_initialization(self):
        """Test Dense layer weight and bias initialization."""
        init_fn, _, _ = stax.Dense(10, W_std=2.0, b_std=0.5)
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (100, 20)  # Large input for statistical testing
        output_shape, params = init_fn(rng, input_shape)
        
        # Check that weights have approximately the right standard deviation
        W_std_actual = params['weight'].std().item()
        b_std_actual = params['bias'].std().item()
        
        # Allow some tolerance due to finite sample size
        assert abs(W_std_actual - 2.0) < 0.3, f"Weight std: {W_std_actual}, expected ~2.0"
        assert abs(b_std_actual - 0.5) < 0.2, f"Bias std: {b_std_actual}, expected ~0.5"
    
    def test_dense_different_input_shapes(self):
        """Test Dense layer with different input shapes."""
        init_fn, apply_fn, _ = stax.Dense(7)
        
        # 1D input (after batch)
        rng = torch.Generator()
        rng.manual_seed(42)
        _, params_1d = init_fn(rng, (10, 5))
        x_1d = torch.randn(10, 5)
        output_1d = apply_fn(params_1d, x_1d)
        assert output_1d.shape == (10, 7)
        
        # 2D input (Dense operates on last axis)
        rng.manual_seed(42)
        _, params_2d = init_fn(rng, (8, 3, 4))
        x_2d = torch.randn(8, 3, 4)
        output_2d = apply_fn(params_2d, x_2d)
        assert output_2d.shape == (8, 3, 7)  # Dense: (8,3,4) -> (8,3,7)
        
        # Check that parameters are correctly sized for the channel axis
        assert params_2d['weight'].shape == (4, 7)  # Last axis size: 4 -> 7
        assert params_2d['bias'] is None  # No bias by default
    
    def test_dense_kernel_function(self):
        """Test Dense layer kernel function for infinite width."""
        _, _, kernel_fn = stax.Dense(1000)  # Large width for infinite limit
        
        # Create test data
        x1 = torch.randn(5, 3)
        x2 = torch.randn(4, 3)
        
        # Create base kernels from input tensors
        # For NNGP: use simple dot product kernel as base
        x1_flat = x1.view(x1.shape[0], -1)
        x2_flat = x2.view(x2.shape[0], -1)
        
        # Test NNGP computation
        base_nngp_same = torch.matmul(x1_flat, x1_flat.T) / x1_flat.shape[1]
        x1_is_x2 = torch.tensor(True)
        base_kernel_same = Kernel(
            nngp=base_nngp_same, 
            cov1=base_nngp_same,
            cov2=base_nngp_same, 
            ntk=base_nngp_same,
            x1_is_x2=x1_is_x2,
            is_gaussian=True,
            is_reversed=False,
            is_input=True,
            diagonal_batch=True,
            diagonal_spatial=False,
            shape1=(5, 3),
            shape2=(5, 3),
            batch_axis=0,
            channel_axis=1
        )
        
        kernel_same = kernel_fn(base_kernel_same)
        assert isinstance(kernel_same, Kernel)
        assert kernel_same.nngp.shape == (5, 5)
        
        # Test cross-kernel
        base_nngp_cross = torch.matmul(x1_flat, x2_flat.T) / x1_flat.shape[1]
        x1_is_x2_cross = torch.tensor(False)
        base_kernel_cross = Kernel(
            nngp=base_nngp_cross, 
            cov1=base_nngp_same,
            cov2=torch.matmul(x2_flat, x2_flat.T) / x2_flat.shape[1],
            ntk=base_nngp_cross,
            x1_is_x2=x1_is_x2_cross,
            is_gaussian=True,
            is_reversed=False,
            is_input=True,
            diagonal_batch=True,
            diagonal_spatial=False,
            shape1=(5, 3),
            shape2=(4, 3),
            batch_axis=0,
            channel_axis=1
        )
        
        kernel_cross = kernel_fn(base_kernel_cross)
        assert kernel_cross.nngp.shape == (5, 4)
        
        # Test NTK computation - create kernel with NTK component
        base_kernel_ntk = Kernel(
            nngp=base_nngp_same, 
            cov1=base_nngp_same,
            cov2=base_nngp_same, 
            ntk=base_nngp_same,
            x1_is_x2=x1_is_x2,
            is_gaussian=True,
            is_reversed=False,
            is_input=True,
            diagonal_batch=True,
            diagonal_spatial=False,
            shape1=(5, 3),
            shape2=(5, 3),
            batch_axis=0,
            channel_axis=1
        )
        kernel_ntk = kernel_fn(base_kernel_ntk)
        assert kernel_ntk.ntk is not None
        assert kernel_ntk.ntk.shape == (5, 5)
        
        # Test symmetry for same inputs
        torch.testing.assert_close(kernel_same.nngp, kernel_same.nngp.T, rtol=1e-5, atol=1e-8)
    
    def test_dense_batch_processing(self):
        """Test Dense layer with different batch sizes."""
        init_fn, apply_fn, _ = stax.Dense(6)
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (1, 4)  # Minimal batch for initialization
        output_shape, params = init_fn(rng, input_shape)
        
        # Test different batch sizes
        for batch_size in [1, 5, 10, 100]:
            x = torch.randn(batch_size, 4)
            output = apply_fn(params, x)
            assert output.shape == (batch_size, 6)
    
    def test_dense_gradient_flow(self):
        """Test that gradients flow through Dense layer."""
        init_fn, apply_fn, _ = stax.Dense(3, b_std=0.1)  # Explicitly include bias
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (2, 5)
        output_shape, params = init_fn(rng, input_shape)
        
        # Make parameters require gradients
        params['weight'].requires_grad_(True)
        params['bias'].requires_grad_(True)
        
        x = torch.randn(2, 5, requires_grad=True)
        output = apply_fn(params, x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert params['weight'].grad is not None
        assert params['bias'].grad is not None
        assert x.grad is not None
        
        # Check gradient shapes
        assert params['weight'].grad.shape == params['weight'].shape
        assert params['bias'].grad.shape == params['bias'].shape


class TestIdentityLayer:
    """Tests for Identity layer."""
    
    def test_identity_basic(self):
        """Test basic Identity layer functionality."""
        init_fn, apply_fn, kernel_fn = stax.Identity()
        
        # Test initialization (should be empty)
        rng = torch.Generator()
        input_shape = (5, 3)
        output_shape, params = init_fn(rng, input_shape)
        assert params == {}
        assert output_shape == input_shape
        
        # Test application (should return input unchanged)
        x = torch.randn(5, 3)
        output = apply_fn(params, x)
        torch.testing.assert_close(output, x, rtol=1e-5, atol=1e-8)
    
    def test_identity_kernel(self):
        """Test Identity layer kernel function."""
        _, _, kernel_fn = stax.Identity()
        
        x1 = torch.randn(4, 6)
        x2 = torch.randn(3, 6)
        
        # Create base kernel - for Identity, this should be preserved exactly
        base_nngp = torch.matmul(x1, x2.T)
        x1_is_x2 = torch.tensor(False)
        base_kernel = Kernel(
            nngp=base_nngp, 
            cov1=torch.matmul(x1, x1.T),
            cov2=torch.matmul(x2, x2.T), 
            ntk=base_nngp,
            x1_is_x2=x1_is_x2,
            is_gaussian=True,
            is_reversed=False,
            is_input=True,
            diagonal_batch=True,
            diagonal_spatial=False,
            shape1=(4, 6),
            shape2=(3, 6),
            batch_axis=0,
            channel_axis=1
        )
        
        # Identity should preserve kernel structure exactly
        kernel = kernel_fn(base_kernel)
        assert isinstance(kernel, Kernel)
        
        # For identity, everything should be preserved
        torch.testing.assert_close(kernel.nngp, base_kernel.nngp, rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(kernel.cov1, base_kernel.cov1, rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(kernel.cov2, base_kernel.cov2, rtol=1e-5, atol=1e-8)
    
    def test_identity_different_shapes(self):
        """Test Identity with different tensor shapes."""
        init_fn, apply_fn, _ = stax.Identity()
        
        # Test various shapes
        shapes = [(1, 5), (10, 3), (7, 2, 4), (5, 3, 2, 2)]
        
        for shape in shapes:
            rng = torch.Generator()
            output_shape, params = init_fn(rng, shape)
            assert params == {}
            assert output_shape == shape
            
            x = torch.randn(*shape)
            output = apply_fn(params, x)
            torch.testing.assert_close(output, x, rtol=1e-5, atol=1e-8)


class TestFlattenLayer:
    """Tests for Flatten layer."""
    
    def test_flatten_basic(self):
        """Test basic Flatten layer functionality."""
        init_fn, apply_fn, kernel_fn = stax.Flatten()
        
        # Test initialization (should be empty)
        rng = torch.Generator()
        input_shape = (5, 3, 4)
        output_shape, params = init_fn(rng, input_shape)
        assert params == {}
        # Flatten should report the actual flattened output shape
        assert output_shape == (5, 12)  # 3 * 4 = 12
        
        # Test application
        x = torch.randn(5, 3, 4)
        output = apply_fn(params, x)
        assert output.shape == (5, 12)  # 3 * 4 = 12
        
        # Should be equivalent to manual flattening
        expected = x.view(5, -1)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_flatten_different_shapes(self):
        """Test Flatten with different input shapes."""
        init_fn, apply_fn, _ = stax.Flatten()
        
        # 2D input (should be unchanged)
        x_2d = torch.randn(8, 5)
        output_2d = apply_fn({}, x_2d)
        torch.testing.assert_close(output_2d, x_2d, rtol=1e-5, atol=1e-8)
        
        # 3D input
        x_3d = torch.randn(6, 4, 7)
        output_3d = apply_fn({}, x_3d)
        assert output_3d.shape == (6, 28)
        
        # 4D input (like images)
        x_4d = torch.randn(4, 3, 8, 8)
        output_4d = apply_fn({}, x_4d)
        assert output_4d.shape == (4, 192)  # 3 * 8 * 8 = 192
    
    def test_flatten_kernel(self):
        """Test Flatten layer kernel function."""
        _, _, kernel_fn = stax.Flatten()
        
        x1 = torch.randn(3, 2, 4)
        x2 = torch.randn(5, 2, 4)
        
        # Create proper cross-correlation matrices
        x1_flat = x1.view(3, -1)
        x2_flat = x2.view(5, -1)
        nngp_input = torch.matmul(x1_flat, x2_flat.T)  # (3, 5)
        ntk_input = torch.matmul(x1_flat, x2_flat.T)   # (3, 5)

        kernel_obj = Kernel(
            nngp=nngp_input, 
            ntk=ntk_input,
            cov1=torch.matmul(x1_flat, x1_flat.T),
            cov2=torch.matmul(x2_flat, x2_flat.T),
            x1_is_x2=torch.tensor(False),
            is_gaussian=True,
            is_reversed=False,
            is_input=True,
            diagonal_batch=True,
            diagonal_spatial=False,
            shape1=(3, 2, 4),
            shape2=(5, 2, 4),
            batch_axis=0,
            channel_axis=1
        )
        kernel = kernel_fn(kernel_obj)
        assert isinstance(kernel, Kernel)
        
        # Flatten should reshape but preserve dot product structure
        x1_flat = x1.view(3, -1)
        x2_flat = x2.view(5, -1)
        expected_nngp = torch.matmul(x1_flat, x2_flat.T)
        torch.testing.assert_close(kernel.nngp, expected_nngp, rtol=1e-5, atol=1e-8)


class TestDropoutLayer:
    """Tests for Dropout layer."""
    
    def test_dropout_training_mode(self):
        """Test Dropout layer in training mode."""
        rate = 0.5
        init_fn, apply_fn, kernel_fn = stax.Dropout(rate, mode='train')
        
        # Test initialization (should be empty)
        rng = torch.Generator()
        input_shape = (100, 10)
        output_shape, params = init_fn(rng, input_shape)
        assert params == {}
        assert output_shape == input_shape
        
        # Test application - some values should be zeroed
        x = torch.ones(100, 10)  # All ones for easy testing
        output = apply_fn(params, x)
        
        # Should have approximately (1-rate) fraction of non-zero elements
        non_zero_fraction = (output != 0).float().mean().item()
        assert abs(non_zero_fraction - (1 - rate)) < 0.1  # Allow some variance
        
        # Non-zero elements should be scaled up by 1/(1-rate)
        non_zero_values = output[output != 0]
        expected_scale = 1.0 / (1 - rate)
        torch.testing.assert_close(non_zero_values, torch.full_like(non_zero_values, expected_scale), rtol=1e-5, atol=1e-8)
    
    def test_dropout_inference_mode(self):
        """Test Dropout layer in inference mode."""
        rate = 0.3
        init_fn, apply_fn, kernel_fn = stax.Dropout(rate, mode='predict')
        
        # Test initialization
        rng = torch.Generator()
        input_shape = (10, 5)
        output_shape, params = init_fn(rng, input_shape)
        assert params == {}
        assert output_shape == input_shape
        
        # Test application - should return input unchanged
        x = torch.randn(10, 5)
        output = apply_fn(params, x)
        torch.testing.assert_close(output, x, rtol=1e-5, atol=1e-8)
    
    def test_dropout_different_rates(self):
        """Test Dropout with different dropout rates."""
        for rate in [0.0, 0.2, 0.5, 0.8, 0.9]:
            init_fn, apply_fn, _ = stax.Dropout(rate, mode='train')
            
            x = torch.ones(1000, 1)  # Large sample for statistics
            output = apply_fn({}, x)
            
            if rate == 0.0:
                # No dropout
                torch.testing.assert_close(output, x, rtol=1e-5, atol=1e-8)
            else:
                # Check dropout statistics
                non_zero_fraction = (output != 0).float().mean().item()
                assert abs(non_zero_fraction - (1 - rate)) < 0.05
    
    def test_dropout_kernel(self):
        """Test Dropout layer kernel function."""
        rate = 0.5
        _, _, kernel_fn = stax.Dropout(rate)
        
        x1 = torch.randn(4, 6)
        x2 = torch.randn(3, 6)
        
        # Create proper cross-correlation matrices
        nngp_input = torch.matmul(x1, x2.T)  # (4, 3)
        ntk_input = torch.matmul(x1, x2.T)   # (4, 3)

        kernel_obj = Kernel(
            nngp=nngp_input, 
            ntk=ntk_input,
            cov1=torch.matmul(x1, x1.T),
            cov2=torch.matmul(x2, x2.T),
            x1_is_x2=torch.tensor(False),
            is_gaussian=True,
            is_reversed=False,
            is_input=False,
            diagonal_batch=True,
            diagonal_spatial=False,
            shape1=(4, 6),
            shape2=(3, 6),
            batch_axis=0,
            channel_axis=1
        )
        kernel = kernel_fn(kernel_obj)
        assert isinstance(kernel, Kernel)
        
        # In infinite width, dropout becomes deterministic scaling
        expected_nngp = torch.matmul(x1, x2.T) * (1 - rate)
        torch.testing.assert_close(kernel.nngp, expected_nngp, rtol=1e-5, atol=1e-8)


class TestLinearLayerIntegration:
    """Integration tests for linear layers."""
    
    def test_linear_layer_composition(self):
        """Test composing multiple linear layers."""
        # Create a simple MLP
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(8),
            stax.Dense(4),
            stax.Dense(2)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (5, 6)
        output_shape, params = init_fn(rng, input_shape)
        
        # Should have 3 layers
        assert len(params) == 3
        for i, layer_params in enumerate(params):
            assert 'weight' in layer_params
            assert 'bias' in layer_params
        
        # Test forward pass
        x = torch.randn(5, 6)
        output = apply_fn(params, x)
        assert output.shape == (5, 2)
    
    def test_linear_with_flatten(self):
        """Test Dense layer with Flatten preprocessing."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Flatten(),
            stax.Dense(10, b_std=0.1)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (3, 4, 5)  # 3D input
        output_shape, params = init_fn(rng, input_shape)
        
        # Should have empty params for Flatten, Dense params for layer 1
        assert len(params) == 2
        assert params[0] == {}  # Flatten
        assert 'weight' in params[1]  # Dense
        assert params[1]['weight'].shape == (20, 10)  # 4*5 = 20 inputs
        
        # Test application
        x = torch.randn(7, 4, 5)
        output = apply_fn(params, x)
        assert output.shape == (7, 10)
    
    def test_linear_layers_preserve_gradients(self):
        """Test that linear layer compositions preserve gradients."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(5),
            stax.Identity(),
            stax.Dense(3),
            stax.Flatten()  # Should not affect gradients
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (2, 4)
        output_shape, params = init_fn(rng, input_shape)
        
        # Enable gradients
        for layer_params in params:
            for key, param in layer_params.items():
                if param is not None:
                    param.requires_grad_(True)
        
        x = torch.randn(2, 4, requires_grad=True)
        output = apply_fn(params, x)
        loss = output.sum()
        loss.backward()
        
        # Check that all parameters have gradients
        for layer_params in params:
            for key, param in layer_params.items():
                if param is not None and param.requires_grad:
                    assert param.grad is not None
    
    def test_linear_kernel_composition(self):
        """Test kernel functions for composed linear layers."""
        _, _, kernel_fn = stax.serial(
            stax.Dense(100),
            stax.Dense(50)
        )
        
        x1 = torch.randn(3, 4)
        x2 = torch.randn(5, 4)
        
        # Create proper cross-correlation matrices
        nngp_input = torch.matmul(x1, x2.T)  # (3, 5)
        ntk_input = torch.matmul(x1, x2.T)   # (3, 5)

        # Test NNGP computation
        kernel_obj = Kernel(
            nngp=nngp_input, 
            ntk=ntk_input,
            cov1=torch.matmul(x1, x1.T),
            cov2=torch.matmul(x2, x2.T),
            x1_is_x2=torch.tensor(False),
            is_gaussian=True,
            is_reversed=False,
            is_input=True,
            diagonal_batch=True,
            diagonal_spatial=False,
            shape1=(3, 4),
            shape2=(5, 4),
            batch_axis=0,
            channel_axis=1
        )
        kernel = kernel_fn(kernel_obj)
        assert isinstance(kernel, Kernel)
        assert kernel.nngp.shape == (3, 5)
        
        # Test NTK computation with same input
        nngp_self = torch.matmul(x1, x1.T)  # (3, 3)
        ntk_self = torch.matmul(x1, x1.T)   # (3, 3)

        kernel_obj_ntk = Kernel(
            nngp=nngp_self, 
            ntk=ntk_self,
            cov1=nngp_self,
            cov2=None,
            x1_is_x2=torch.tensor(True),
            is_gaussian=True,
            is_reversed=False,
            is_input=True,
            diagonal_batch=True,
            diagonal_spatial=False,
            shape1=(3, 4),
            shape2=(3, 4),
            batch_axis=0,
            channel_axis=1
        )
        kernel_ntk = kernel_fn(kernel_obj_ntk)
        assert kernel_ntk.ntk is not None
        assert kernel_ntk.ntk.shape == (3, 3)


if __name__ == "__main__":
    # Run tests directly
    test_dense = TestDenseLayer()
    test_dense.test_dense_basic_functionality()
    test_dense.test_dense_without_bias()
    test_dense.test_dense_different_parameterizations()
    test_dense.test_dense_weight_and_bias_initialization()
    test_dense.test_dense_different_input_shapes()
    test_dense.test_dense_kernel_function()
    test_dense.test_dense_batch_processing()
    test_dense.test_dense_gradient_flow()
    print("✓ Dense layer tests passed")
    
    test_identity = TestIdentityLayer()
    test_identity.test_identity_basic()
    test_identity.test_identity_kernel()
    test_identity.test_identity_different_shapes()
    print("✓ Identity layer tests passed")
    
    test_flatten = TestFlattenLayer()
    test_flatten.test_flatten_basic()
    test_flatten.test_flatten_different_shapes()
    test_flatten.test_flatten_kernel()
    print("✓ Flatten layer tests passed")
    
    test_dropout = TestDropoutLayer()
    test_dropout.test_dropout_training_mode()
    test_dropout.test_dropout_inference_mode()
    test_dropout.test_dropout_different_rates()
    test_dropout.test_dropout_kernel()
    print("✓ Dropout layer tests passed")
    
    test_integration = TestLinearLayerIntegration()
    test_integration.test_linear_layer_composition()
    test_integration.test_linear_with_flatten()
    test_integration.test_linear_layers_preserve_gradients()
    test_integration.test_linear_kernel_composition()
    print("✓ Linear layer integration tests passed")
    
    print("\n🎉 All stax linear tests completed!")