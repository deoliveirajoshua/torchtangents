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

"""Comprehensive tests for torchtangents stax elementwise operations."""

import torch
import sys
import os
import math

# Add the torchtangents module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torchtangents as tt
from torchtangents import stax
from torchtangents._src.utils.kernel import Kernel


class TestReluActivation:
    """Comprehensive tests for Relu activation."""
    
    def test_relu_basic_functionality(self):
        """Test basic Relu activation functionality."""
        init_fn, apply_fn, kernel_fn = stax.Relu()
        
        # Test initialization (should be empty)
        rng = torch.Generator()
        input_shape = (5, 3)
        output_shape, params = init_fn(rng, input_shape)
        assert params == {}
        assert output_shape == input_shape
        
        # Test application
        x = torch.tensor([[-2., -1., 0., 1., 2.], [3., -4., 5., -6., 7.]])
        output = apply_fn(params, x)
        expected = torch.tensor([[0., 0., 0., 1., 2.], [3., 0., 5., 0., 7.]])
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_relu_zero_preservation(self):
        """Test that Relu preserves zeros correctly."""
        _, apply_fn, _ = stax.Relu()
        
        # Test with zeros
        x = torch.zeros(3, 4)
        output = apply_fn({}, x)
        torch.testing.assert_close(output, x, rtol=1e-5, atol=1e-8)
    
    def test_relu_positive_values(self):
        """Test Relu with all positive values."""
        _, apply_fn, _ = stax.Relu()
        
        x = torch.rand(5, 6) + 0.1  # All positive
        output = apply_fn({}, x)
        torch.testing.assert_close(output, x, rtol=1e-5, atol=1e-8)  # Should be unchanged
    
    def test_relu_negative_values(self):
        """Test Relu with all negative values."""
        _, apply_fn, _ = stax.Relu()
        
        x = -torch.rand(4, 3) - 0.1  # All negative
        output = apply_fn({}, x)
        expected = torch.zeros_like(x)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_relu_kernel_function(self):
        """Test Relu kernel function for infinite width."""
        _, _, kernel_fn = stax.Relu()
        
        # Create test data
        x1 = torch.randn(4, 3)
        x2 = torch.randn(5, 3)
        
        # Create input kernel (like from previous layer)
        input_kernel = Kernel(
            nngp=torch.matmul(x1, x2.T),
            ntk=None,
            cov1=torch.matmul(x1, x1.T),
            cov2=torch.matmul(x2, x2.T),
            x1_is_x2=torch.tensor(False),
            is_gaussian=True,
            is_reversed=False,
            is_input=True,
            diagonal_batch=True,
            diagonal_spatial=False,
            shape1=(4, 3),
            shape2=(5, 3),
            batch_axis=0,
            channel_axis=1
        )
        
        # Apply Relu kernel transformation
        kernel = kernel_fn(input_kernel)
        assert isinstance(kernel, Kernel)
        assert kernel.nngp.shape == (4, 5)
        
        # Relu should preserve non-negative correlations but modify negative ones
        # The exact transformation depends on the implementation but should be valid
        assert not torch.isnan(kernel.nngp).any()
        assert not torch.isinf(kernel.nngp).any()
    
    def test_relu_gradient_flow(self):
        """Test gradient flow through Relu."""
        _, apply_fn, _ = stax.Relu()
        
        x = torch.tensor([[-1., 0., 1., 2.]], requires_grad=True)
        output = apply_fn({}, x)
        loss = output.sum()
        loss.backward()
        
        # Gradients should be 0 for negative inputs, 1 for positive
        expected_grad = torch.tensor([[0., 0., 1., 1.]])
        torch.testing.assert_close(x.grad, expected_grad, rtol=1e-5, atol=1e-8)


class TestAbReluActivation:
    """Tests for ABRelu (absolute value + Relu) activation."""
    
    def test_abrelu_basic_functionality(self):
        """Test basic ABRelu functionality."""
        a = 0.5
        b = 1.0
        init_fn, apply_fn, kernel_fn = stax.ABRelu(a, b)
        
        # Test initialization
        rng = torch.Generator()
        input_shape = (3, 4)
        output_shape, params = init_fn(rng, input_shape)
        assert params == {}
        assert output_shape == input_shape
        
        # Test application: ABRelu(x) = a * |x| + b * max(x, 0)
        x = torch.tensor([[-2., -1., 0., 1., 2.]])
        output = apply_fn(params, x)
        
        # Manual computation
        abs_x = torch.abs(x)
        relu_x = torch.max(x, torch.zeros_like(x))
        expected = a * abs_x + b * relu_x
        
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_abrelu_special_cases(self):
        """Test ABRelu with special parameter values."""
        # a=0, b=1 should be regular ReLU
        _, apply_fn_relu, _ = stax.ABRelu(0.0, 1.0)
        x = torch.tensor([[-1., 0., 1.]])
        output_relu = apply_fn_relu({}, x)
        expected_relu = torch.tensor([[0., 0., 1.]])
        torch.testing.assert_close(output_relu, expected_relu, rtol=1e-5, atol=1e-8)
        
        # a=1, b=0 should be absolute value
        _, apply_fn_abs, _ = stax.ABRelu(1.0, 0.0)
        output_abs = apply_fn_abs({}, x)
        expected_abs = torch.tensor([[1., 0., 1.]])
        torch.testing.assert_close(output_abs, expected_abs, rtol=1e-5, atol=1e-8)
        
        # a=0.5, b=0.5
        _, apply_fn_mixed, _ = stax.ABRelu(0.5, 0.5)
        output_mixed = apply_fn_mixed({}, x)
        expected_mixed = 0.5 * torch.abs(x) + 0.5 * torch.relu(x)
        torch.testing.assert_close(output_mixed, expected_mixed, rtol=1e-5, atol=1e-8)
    
    def test_abrelu_different_parameters(self):
        """Test ABRelu with different parameter combinations."""
        x = torch.randn(5, 10)
        
        for a in [0.0, 0.1, 0.5, 1.0, 2.0]:
            for b in [0.0, 0.1, 0.5, 1.0, 2.0]:
                _, apply_fn, _ = stax.ABRelu(a, b)
                output = apply_fn({}, x)
                
                # Check output shape
                assert output.shape == x.shape
                
                # Check manual computation
                expected = a * torch.abs(x) + b * torch.relu(x)
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)


class TestErfActivation:
    """Tests for Erf (error function) activation."""
    
    def test_erf_basic_functionality(self):
        """Test basic Erf activation functionality."""
        init_fn, apply_fn, kernel_fn = stax.Erf()
        
        # Test initialization
        rng = torch.Generator()
        input_shape = (4, 5)
        output_shape, params = init_fn(rng, input_shape)
        assert params == {}
        assert output_shape == input_shape
        
        # Test application
        x = torch.tensor([[-2., -1., 0., 1., 2.]])
        output = apply_fn(params, x)
        expected = torch.erf(x)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_erf_properties(self):
        """Test mathematical properties of Erf."""
        _, apply_fn, _ = stax.Erf()
        
        # Test that erf(0) = 0
        x_zero = torch.zeros(3, 3)
        output_zero = apply_fn({}, x_zero)
        torch.testing.assert_close(output_zero, x_zero, rtol=1e-5, atol=1e-8)
        
        # Test antisymmetry: erf(-x) = -erf(x)
        x = torch.randn(4, 6)
        output_pos = apply_fn({}, x)
        output_neg = apply_fn({}, -x)
        torch.testing.assert_close(output_neg, -output_pos, rtol=1e-5, atol=1e-8)
        
        # Test that |erf(x)| < 1 for finite x
        x_large = torch.randn(100) * 10
        output_large = apply_fn({}, x_large)
        assert (torch.abs(output_large) <= 1.0).all()
    
    def test_erf_scaling_parameter(self):
        """Test Erf with different scaling parameters."""
        for a in [0.5, 1.0, 1.5, 2.0]:
            _, apply_fn, _ = stax.Erf(a=a)
            
            x = torch.randn(3, 4)
            output = apply_fn({}, x)
            expected = torch.erf(a * x)
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)


class TestGeluActivation:
    """Tests for Gelu (Gaussian Error Linear Unit) activation."""
    
    def test_gelu_basic_functionality(self):
        """Test basic Gelu activation functionality."""
        init_fn, apply_fn, kernel_fn = stax.Gelu()
        
        # Test initialization
        rng = torch.Generator()
        input_shape = (3, 5)
        output_shape, params = init_fn(rng, input_shape)
        assert params == {}
        assert output_shape == input_shape
        
        # Test application
        x = torch.tensor([[-2., -1., 0., 1., 2.]])
        output = apply_fn(params, x)
        
        # GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
        # Approximation: GELU(x) ≈ 0.5 * x * (1 + erf(x / sqrt(2)))
        expected = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_gelu_properties(self):
        """Test mathematical properties of Gelu."""
        _, apply_fn, _ = stax.Gelu()
        
        # Test that GELU(0) = 0
        x_zero = torch.zeros(2, 3)
        output_zero = apply_fn({}, x_zero)
        torch.testing.assert_close(output_zero, x_zero, rtol=1e-5, atol=1e-8)
        
        # Test that GELU(x) ≈ x for large positive x
        x_large = torch.tensor([[10., 20., 100.]])
        output_large = apply_fn({}, x_large)
        torch.testing.assert_close(output_large, x_large, rtol=1e-5, atol=1e-8)
        
        # Test that GELU(x) ≈ 0 for large negative x
        x_neg_large = torch.tensor([[-10., -20., -100.]])
        output_neg_large = apply_fn({}, x_neg_large)
        assert (torch.abs(output_neg_large) < 1e-10).all()
    
    def test_gelu_vs_relu_comparison(self):
        """Compare Gelu with Relu on same inputs."""
        _, apply_gelu, _ = stax.Gelu()
        _, apply_relu, _ = stax.Relu()
        
        x = torch.linspace(-3, 3, 100).unsqueeze(0)
        
        gelu_output = apply_gelu({}, x)
        relu_output = apply_relu({}, x)
        
        # GELU should be smoother than ReLU
        # For positive x, GELU should be close to ReLU
        positive_mask = x > 1
        gelu_pos = gelu_output[positive_mask]
        relu_pos = relu_output[positive_mask]
        
        # Should be approximately equal for large positive values
        # GELU approaches x but is always slightly less, so we need reasonable tolerance
        torch.testing.assert_close(gelu_pos, relu_pos, rtol=0.2, atol=0.2)


class TestSinActivation:
    """Tests for Sin activation."""
    
    def test_sin_basic_functionality(self):
        """Test basic Sin activation functionality."""
        init_fn, apply_fn, kernel_fn = stax.Sin()
        
        # Test initialization
        rng = torch.Generator()
        input_shape = (4, 3)
        output_shape, params = init_fn(rng, input_shape)
        assert params == {}
        assert output_shape == input_shape
        
        # Test application
        x = torch.tensor([[0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]])
        output = apply_fn(params, x)
        expected = torch.sin(x)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_sin_properties(self):
        """Test mathematical properties of Sin."""
        _, apply_fn, _ = stax.Sin()
        
        # Test that sin(0) = 0
        x_zero = torch.zeros(3, 4)
        output_zero = apply_fn({}, x_zero)
        torch.testing.assert_close(output_zero, x_zero, rtol=1e-5, atol=1e-8)
        
        # Test periodicity: sin(x + 2π) = sin(x)
        x = torch.randn(5, 6)
        output1 = apply_fn({}, x)
        output2 = apply_fn({}, x + 2 * math.pi)
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-8)
        
        # Test that |sin(x)| <= 1
        x_large = torch.randn(100) * 100
        output_large = apply_fn({}, x_large)
        assert (torch.abs(output_large) <= 1.0 + 1e-6).all()
    
    def test_sin_scaling_parameter(self):
        """Test Sin with different scaling parameters."""
        for a in [0.5, 1.0, 2.0, math.pi]:
            for b in [0.5, 1.0, 2.0]:
                for c in [0.0, 0.5, 1.0]:
                    _, apply_fn, _ = stax.Sin(a=a, b=b, c=c)
                    
                    x = torch.randn(3, 4)
                    output = apply_fn({}, x)
                    expected = a * torch.sin(b * x + c)
                    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_sin_kernel_function(self):
        """Test Sin kernel function."""
        _, _, kernel_fn = stax.Sin()
        
        x1 = torch.randn(3, 4)
        x2 = torch.randn(5, 4)
        
        # Create input kernel
        input_kernel = Kernel(
            nngp=torch.matmul(x1, x2.T),
            ntk=None,
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
        
        # Apply Sin kernel transformation
        kernel = kernel_fn(input_kernel)
        assert isinstance(kernel, Kernel)
        assert kernel.nngp.shape == (3, 5)
        
        # Sin kernel should produce valid correlations
        assert not torch.isnan(kernel.nngp).any()
        assert not torch.isinf(kernel.nngp).any()


class TestCosActivation:
    """Test Cos activation function."""
    
    def test_cos_basic_functionality(self):
        """Test basic Cos activation."""
        _, apply_fn, _ = stax.Cos()
        
        x = torch.tensor([0., math.pi/4, math.pi/2, math.pi])
        output = apply_fn({}, x)
        expected = torch.cos(x)
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-7)
    
    def test_cos_scaling_parameter(self):
        """Test Cos with different scaling parameters."""
        for a in [0.5, 1.0, 2.0]:
            for b in [0.5, 1.0, 2.0]:
                for c in [0.0, 0.5, 1.0]:
                    _, apply_fn, _ = stax.Cos(a=a, b=b, c=c)
                    
                    x = torch.randn(3, 4)
                    output = apply_fn({}, x)
                    expected = a * torch.cos(b * x + c)
                    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-7)
    
    def test_cos_properties(self):
        """Test mathematical properties of Cos."""
        _, apply_fn, _ = stax.Cos()
        
        # Test that Cos is even function: cos(-x) = cos(x)
        x = torch.randn(5, 3)
        pos_result = apply_fn({}, x)
        neg_result = apply_fn({}, -x)
        torch.testing.assert_close(pos_result, neg_result, rtol=1e-4, atol=1e-7)
        
        # Test range is [-1, 1] for default parameters
        x = torch.randn(100, 10) * 10  # Large range
        output = apply_fn({}, x)
        assert output.min() >= -1.0 - 1e-6
        assert output.max() <= 1.0 + 1e-6


class TestActivationComposition:
    """Test composition of activation functions."""
    
    def test_multiple_activations(self):
        """Test composing multiple activation functions."""
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Relu(),
            stax.Erf(),
            stax.Sin()
        )
        
        # Test initialization
        rng = torch.Generator()
        output_shape, params = init_fn(rng, (5, 4))
        assert len(params) == 3
        assert all(p == {} for p in params)
        
        # Test application
        x = torch.randn(5, 4)
        output = apply_fn(params, x)
        assert output.shape == x.shape
        
        # Manual composition
        temp1 = torch.relu(x)
        temp2 = torch.erf(temp1)
        expected = torch.sin(temp2)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
    
    def test_activation_with_linear_layers(self):
        """Test activations combined with linear layers."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(8),
            stax.Relu(),
            stax.Dense(4),
            stax.Gelu(),
            stax.Dense(2)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (6, 3)
        output_shape, params = init_fn(rng, input_shape)
        
        # Should have 5 layers total
        assert len(params) == 5
        
        # Dense layers should have parameters, activations should not
        assert 'weight' in params[0] and 'bias' in params[0]  # Dense
        assert params[1] == {}  # Relu
        assert 'weight' in params[2] and 'bias' in params[2]  # Dense
        assert params[3] == {}  # Gelu
        assert 'weight' in params[4] and 'bias' in params[4]  # Dense
        
        # Test forward pass
        x = torch.randn(6, 3)
        output = apply_fn(params, x)
        assert output.shape == (6, 2)
    
    def test_same_activation_multiple_times(self):
        """Test using the same activation multiple times."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(5),
            stax.Relu(),
            stax.Dense(5),
            stax.Relu(),
            stax.Dense(3)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        output_shape, params = init_fn(rng, (4, 6))
        
        x = torch.randn(4, 6)
        output = apply_fn(params, x)
        assert output.shape == (4, 3)


class TestActivationKernels:
    """Test kernel functions for activations."""
    
    def test_activation_kernel_shapes(self):
        """Test that activation kernels preserve shapes."""
        activations = [
            stax.Relu(),
            stax.ABRelu(0.5, 1.0),
            stax.Erf(),
            stax.Gelu(),
            stax.Sin()
        ]
        
        x1 = torch.randn(4, 5)
        x2 = torch.randn(3, 5)
        input_kernel = Kernel(
            nngp=torch.matmul(x1, x2.T),
            ntk=None,
            cov1=torch.matmul(x1, x1.T),
            cov2=torch.matmul(x2, x2.T),
            x1_is_x2=torch.tensor(False),
            is_gaussian=True,
            is_reversed=False,
            is_input=True,
            diagonal_batch=True,
            diagonal_spatial=False,
            shape1=(4, 5),
            shape2=(3, 5),
            batch_axis=0,
            channel_axis=1
        )
        
        for init_fn, apply_fn, kernel_fn in activations:
            kernel = kernel_fn(input_kernel)
            assert isinstance(kernel, Kernel)
            assert kernel.nngp.shape == (4, 3)
    
    def test_activation_kernel_symmetry(self):
        """Test that activation kernels preserve symmetry."""
        activations = [
            stax.Relu(),
            stax.Erf(),
            stax.Gelu()
        ]
        
        x = torch.randn(5, 4)
        
        # Create proper kernel with all required fields
        nngp = torch.matmul(x, x.T)
        cov1 = torch.diagonal(nngp).clone()  # Extract diagonal for cov1
        input_kernel = Kernel(
            nngp=nngp,
            ntk=nngp.clone(),
            cov1=cov1,
            cov2=cov1.clone(),
            x1_is_x2=torch.tensor(True),
            is_gaussian=False,
            is_reversed=False,
            is_input=True,
            diagonal_batch=True,
            diagonal_spatial=True,
            shape1=(5, 4),
            shape2=(5, 4),
            batch_axis=0,
            channel_axis=1
        )
        
        for init_fn, apply_fn, kernel_fn in activations:
            kernel = kernel_fn(input_kernel)
            # Should preserve symmetry
            torch.testing.assert_close(kernel.nngp, kernel.nngp.T, rtol=1e-5, atol=1e-8)
    
    def test_activation_kernel_finite_values(self):
        """Test that activation kernels produce finite values."""
        activations = [
            stax.Relu(),
            stax.ABRelu(1.0, 1.0),
            stax.Erf(2.0),
            stax.Gelu(),
            stax.Sin(a=1.0, b=1.0, c=0.0)
        ]
        
        # Test with various input magnitudes
        for scale in [0.1, 1.0, 10.0, 100.0]:
            x1 = torch.randn(3, 4) * scale
            x2 = torch.randn(5, 4) * scale
            input_kernel = Kernel(
                nngp=torch.matmul(x1, x2.T),
                ntk=None,
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
            
            for init_fn, apply_fn, kernel_fn in activations:
                kernel = kernel_fn(input_kernel)
                
                # Should produce finite values
                assert torch.isfinite(kernel.nngp).all()
                assert not torch.isnan(kernel.nngp).any()


if __name__ == "__main__":
    # Run tests directly
    test_relu = TestReluActivation()
    test_relu.test_relu_basic_functionality()
    test_relu.test_relu_zero_preservation()
    test_relu.test_relu_positive_values()
    test_relu.test_relu_negative_values()
    test_relu.test_relu_kernel_function()
    test_relu.test_relu_gradient_flow()
    print("✓ Relu activation tests passed")
    
    test_abrelu = TestAbReluActivation()
    test_abrelu.test_abrelu_basic_functionality()
    test_abrelu.test_abrelu_special_cases()
    test_abrelu.test_abrelu_different_parameters()
    print("✓ ABRelu activation tests passed")
    
    test_erf = TestErfActivation()
    test_erf.test_erf_basic_functionality()
    test_erf.test_erf_properties()
    test_erf.test_erf_scaling_parameter()
    print("✓ Erf activation tests passed")
    
    test_gelu = TestGeluActivation()
    test_gelu.test_gelu_basic_functionality()
    test_gelu.test_gelu_properties()
    test_gelu.test_gelu_vs_relu_comparison()
    print("✓ Gelu activation tests passed")
    
    test_sin = TestSinActivation()
    test_sin.test_sin_basic_functionality()
    test_sin.test_sin_properties()
    test_sin.test_sin_scaling_parameter()
    test_sin.test_sin_kernel_function()
    print("✓ Sin activation tests passed")
    
    test_composition = TestActivationComposition()
    test_composition.test_multiple_activations()
    test_composition.test_activation_with_linear_layers()
    test_composition.test_same_activation_multiple_times()
    print("✓ Activation composition tests passed")
    
    test_kernels = TestActivationKernels()
    test_kernels.test_activation_kernel_shapes()
    test_kernels.test_activation_kernel_symmetry()
    test_kernels.test_activation_kernel_finite_values()
    print("✓ Activation kernel tests passed")
    
    print("\n🎉 All stax elementwise tests completed!")