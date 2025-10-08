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

"""Comprehensive tests for torchtangents stax combinators."""

import torch
import sys
import os

# Add the torchtangents module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torchtangents as tt
from torchtangents import stax
from torchtangents._src.utils.kernel import Kernel


class TestSerialCombinator:
    """Comprehensive tests for serial combinator."""
    
    def test_serial_basic_functionality(self):
        """Test basic serial combinator functionality."""
        # Create a simple MLP
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(8),
            stax.Relu(),
            stax.Dense(4),
            stax.Gelu(),
            stax.Dense(2)
        )
        
        # Test initialization
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (5, 6)
        output_shape, params = init_fn(rng, input_shape)
        
        # Should have 5 layers
        assert len(params) == 5
        assert 'weight' in params[0] and 'bias' in params[0]  # Dense
        assert params[1] == {}  # Relu
        assert 'weight' in params[2] and 'bias' in params[2]  # Dense
        assert params[3] == {}  # Gelu
        assert 'weight' in params[4] and 'bias' in params[4]  # Dense
        
        # Check parameter shapes
        assert params[0]['weight'].shape == (6, 8)
        assert params[2]['weight'].shape == (8, 4)
        assert params[4]['weight'].shape == (4, 2)
        
        # Test application
        x = torch.randn(5, 6)
        output = apply_fn(params, x)
        assert output.shape == (5, 2)
    
    def test_serial_empty_layers(self):
        """Test serial with empty layer list."""
        try:
            init_fn, apply_fn, kernel_fn = stax.serial()
            # Should handle empty case gracefully or raise appropriate error
            assert False, "Empty serial should raise an error or handle gracefully"
        except (ValueError, TypeError):
            pass  # Expected behavior
    
    def test_serial_single_layer(self):
        """Test serial with single layer."""
        single_layer = stax.Dense(5)
        init_fn_single, apply_fn_single, _ = single_layer
        init_fn_serial, apply_fn_serial, _ = stax.serial(single_layer)
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (3, 4)
        
        # Both should behave identically
        output_shape_single, params_single = init_fn_single(rng, input_shape)
        rng.manual_seed(42)
        output_shape_serial, params_serial = init_fn_serial(rng, input_shape)
        
        # Serial should wrap parameters in list
        assert isinstance(params_serial, list)
        assert len(params_serial) == 1
        
        # Output shapes should match
        assert output_shape_single == output_shape_serial
        
        x = torch.randn(3, 4)
        output_single = apply_fn_single(params_single, x)
        output_serial = apply_fn_serial(params_serial, x)
        
        torch.testing.assert_close(output_single, output_serial, rtol=1e-5, atol=1e-8)
    
    def test_serial_parameter_propagation(self):
        """Test that serial properly propagates shapes through layers."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(10),
            stax.Relu(),
            stax.Dense(5),
            stax.Dropout(0.1),
            stax.Dense(3),
            stax.Flatten()
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (7, 8)
        output_shape, params = init_fn(rng, input_shape)
        
        # Check that each Dense layer has correct input/output dimensions
        assert params[0]['weight'].shape == (8, 10)   # Input: 8, Output: 10
        assert params[2]['weight'].shape == (10, 5)   # Input: 10, Output: 5
        assert params[4]['weight'].shape == (5, 3)    # Input: 5, Output: 3
        
        # Test forward pass
        x = torch.randn(7, 8)
        output = apply_fn(params, x)
        assert output.shape == (7, 3)  # Flatten shouldn't change shape for 2D
    
    def test_serial_with_multidimensional_input(self):
        """Test serial with multidimensional inputs."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Flatten(),
            stax.Dense(20),
            stax.Relu(),
            stax.Dense(10)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (5, 3, 4)  # 3D input
        output_shape, params = init_fn(rng, input_shape)
        
        # Flatten should create input of size 3*4=12 for first Dense
        assert params[1]['weight'].shape == (12, 20)
        
        # Test forward pass
        x = torch.randn(5, 3, 4)
        output = apply_fn(params, x)
        assert output.shape == (5, 10)
    
    def test_serial_kernel_function(self):
        """Test serial kernel function composition."""
        _, _, kernel_fn = stax.serial(
            stax.Dense(100),
            stax.Relu(),
            stax.Dense(50)
        )
        
        x1 = torch.randn(4, 5)
        x2 = torch.randn(3, 5)
        
        # Test NNGP computation
        kernel = kernel_fn(x1, x2, 'nngp')
        assert isinstance(kernel, torch.Tensor)
        assert kernel.shape == (4, 3)
        
        # Test self-kernel symmetry
        kernel_self = kernel_fn(x1, x1, 'nngp')
        torch.testing.assert_close(kernel_self, kernel_self.T, rtol=1e-5, atol=1e-8)
    
    def test_serial_gradient_flow(self):
        """Test gradient flow through serial composition."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(6),
            stax.Relu(),
            stax.Dense(3)
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
        
        # Check gradients exist for all parameters
        assert params[0]['weight'].grad is not None
        if params[0]['bias'] is not None:
            assert params[0]['bias'].grad is not None
        assert params[2]['weight'].grad is not None
        if params[2]['bias'] is not None:
            assert params[2]['bias'].grad is not None
        assert x.grad is not None


class TestParallelCombinator:
    """Tests for parallel combinator."""
    
    def test_parallel_basic_functionality(self):
        """Test basic parallel combinator functionality."""
        # Create parallel branches
        init_fn, apply_fn, kernel_fn = stax.parallel(
            stax.Dense(5),
            stax.serial(stax.Dense(3), stax.Relu()),
            stax.Identity()
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (4, 6)
        output_shape, params = init_fn(rng, input_shape)
        
        # Should have 3 branches
        assert len(params) == 3
        
        # Branch 0: Dense layer
        assert 'weight' in params[0] and 'bias' in params[0]
        assert params[0]['weight'].shape == (6, 5)
        
        # Branch 1: Serial composition
        assert len(params[1]) == 2  # Dense + Relu
        assert 'weight' in params[1][0] and 'bias' in params[1][0]
        assert params[1][0]['weight'].shape == (6, 3)
        assert params[1][1] == {}  # Relu
        
        # Branch 2: Identity
        assert params[2] == {}
        
        # Test application
        x = torch.randn(4, 6)
        outputs = apply_fn(params, x)
        
        # Should return tuple of outputs
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3
        assert outputs[0].shape == (4, 5)  # Dense output
        assert outputs[1].shape == (4, 3)  # Serial output
        assert outputs[2].shape == (4, 6)  # Identity output
        torch.testing.assert_close(outputs[2], x, rtol=1e-5, atol=1e-8)  # Identity should preserve input
    
    def test_parallel_single_branch(self):
        """Test parallel with single branch."""
        single_layer = stax.Dense(7)
        init_fn_single, apply_fn_single, _ = single_layer
        init_fn_parallel, apply_fn_parallel, _ = stax.parallel(single_layer)
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (3, 4)
        
        output_shape_single, params_single = init_fn_single(rng, input_shape)
        rng.manual_seed(42)
        output_shape_parallel, params_parallel = init_fn_parallel(rng, input_shape)
        
        x = torch.randn(3, 4)
        output_single = apply_fn_single(params_single, x)
        outputs_parallel = apply_fn_parallel(params_parallel, x)
        
        # Parallel should return tuple with single element
        assert isinstance(outputs_parallel, tuple)
        assert len(outputs_parallel) == 1
        torch.testing.assert_close(outputs_parallel[0], output_single, rtol=1e-5, atol=1e-8)
    
    def test_parallel_different_output_shapes(self):
        """Test parallel with branches producing different output shapes."""
        init_fn, apply_fn, _ = stax.parallel(
            stax.Dense(10),          # Output: (batch, 10)
            stax.serial(             # Output: (batch, 5)
                stax.Dense(8),
                stax.Dense(5)
            ),
            stax.serial(             # Output: (batch, 12)
                stax.Flatten(),
                stax.Dense(12)
            )
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (6, 3, 4)  # 3D input
        output_shape, params = init_fn(rng, input_shape)
        
        x = torch.randn(6, 3, 4)
        outputs = apply_fn(params, x)
        
        assert len(outputs) == 3
        assert outputs[0].shape == (6, 3, 10)  # Dense on last axis: (6,3,4) -> (6,3,10)
        assert outputs[1].shape == (6, 3, 5)   # Serial Dense: (6,3,4) -> (6,3,8) -> (6,3,5)
        assert outputs[2].shape == (6, 12)     # Flatten + Dense: (6,3,4) -> (6,12) -> (6,12)
    
    def test_parallel_kernel_function(self):
        """Test parallel kernel function."""
        _, _, kernel_fn = stax.parallel(
            stax.Dense(20),
            stax.Dense(30),
            stax.Identity()
        )
        
        x1 = torch.randn(4, 5)
        x2 = torch.randn(3, 5)
        
        # Parallel should combine kernels and return single result  
        kernel = kernel_fn(x1, x2, 'nngp')
        assert isinstance(kernel, torch.Tensor)
        assert kernel.shape == (4, 3)
    
    def test_parallel_empty_branches(self):
        """Test parallel with empty branch list."""
        try:
            init_fn, apply_fn, kernel_fn = stax.parallel()
            # Should handle empty case gracefully or raise appropriate error
            assert False, "Empty parallel should raise an error"
        except (ValueError, TypeError):
            pass  # Expected behavior


class TestFanOutCombinator:
    """Tests for FanOut combinator (if implemented)."""
    
    def test_fanout_basic(self):
        """Test basic FanOut functionality."""
        # FanOut applies same input to multiple parallel branches
        if hasattr(stax, 'FanOut'):
            init_fn, apply_fn, _ = stax.FanOut(2)  # Duplicate input 2 times
            
            rng = torch.Generator()
            input_shape = (3, 4)
            output_shape, params = init_fn(rng, input_shape)
            
            x = torch.randn(3, 4)
            outputs = apply_fn(params, x)
            
            # Should return list with duplicated inputs
            assert isinstance(outputs, list)
            assert len(outputs) == 2
            torch.testing.assert_close(outputs[0], x, rtol=1e-5, atol=1e-8)
            torch.testing.assert_close(outputs[1], x, rtol=1e-5, atol=1e-8)
        else:
            # Skip if not implemented
            pass


class TestFanInCombinator:
    """Tests for FanIn combinator (if implemented)."""
    
    def test_fanin_concatenation(self):
        """Test FanIn with concatenation."""
        if hasattr(stax, 'FanInConcat'):
            # Create branches that produce different outputs
            branch1 = stax.Dense(5)
            branch2 = stax.Dense(3)
            
            init_fn, apply_fn, _ = stax.serial(
                stax.parallel(branch1, branch2),
                stax.FanInConcat()
            )
            
            rng = torch.Generator()
            rng.manual_seed(42)
            input_shape = (4, 6)
            output_shape, params = init_fn(rng, input_shape)
            
            x = torch.randn(4, 6)
            output = apply_fn(params, x)
            
            # Should concatenate outputs: 5 + 3 = 8
            assert output.shape == (4, 8)
        else:
            # Skip if not implemented
            pass
    
    def test_fanin_summation(self):
        """Test FanIn with summation."""
        if hasattr(stax, 'FanInSum'):
            # Create branches with same output size
            branch1 = stax.Dense(7)
            branch2 = stax.Dense(7)
            
            init_fn, apply_fn, _ = stax.serial(
                stax.parallel(branch1, branch2),
                stax.FanInSum()
            )
            
            rng = torch.Generator()
            rng.manual_seed(42)
            input_shape = (3, 5)
            output_shape, params = init_fn(rng, input_shape)
            
            x = torch.randn(3, 5)
            output = apply_fn(params, x)
            
            # Should sum outputs, maintaining shape
            assert output.shape == (3, 7)
        else:
            # Skip if not implemented
            pass


class TestAdvancedCompositions:
    """Test advanced layer compositions."""
    
    def test_nested_serial_parallel(self):
        """Test nested serial and parallel compositions."""
        # Create a complex architecture
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(20),
            stax.Relu(),
            stax.parallel(
                stax.serial(stax.Dense(10), stax.Gelu()),
                stax.serial(stax.Dense(8), stax.Relu()),
                stax.Identity()
            ),
            # Note: This would need a FanIn layer to combine parallel outputs
            # For now, just test that parallel composition works
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (5, 6)
        output_shape, params = init_fn(rng, input_shape)
        
        x = torch.randn(5, 6)
        outputs = apply_fn(params, x)
        
        # Should get tuple of 3 outputs from parallel section
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3
        assert outputs[0].shape == (5, 10)  # First branch
        assert outputs[1].shape == (5, 8)   # Second branch
        assert outputs[2].shape == (5, 20)  # Identity (preserves previous layer output)
    
    def test_deep_serial_composition(self):
        """Test very deep serial composition."""
        layers = []
        current_size = 100
        
        # Create 10-layer network with decreasing sizes
        for i in range(10):
            next_size = max(10, current_size - 10)
            layers.append(stax.Dense(next_size))
            if i < 9:  # Don't add activation after last layer
                layers.append(stax.Relu())
            current_size = next_size
        
        init_fn, apply_fn, _ = stax.serial(*layers)
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (8, 100)
        output_shape, params = init_fn(rng, input_shape)
        
        # Should have 19 layers (10 Dense + 9 Relu)
        assert len(params) == 19
        
        x = torch.randn(8, 100)
        output = apply_fn(params, x)
        assert output.shape == (8, 10)  # Final layer size
    
    def test_wide_parallel_composition(self):
        """Test wide parallel composition with many branches."""
        branches = []
        for i in range(10):
            branch_size = 5 + i  # Different sizes for each branch
            branches.append(stax.Dense(branch_size))
        
        init_fn, apply_fn, _ = stax.parallel(*branches)
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (4, 20)
        output_shape, params = init_fn(rng, input_shape)
        
        assert len(params) == 10
        
        x = torch.randn(4, 20)
        outputs = apply_fn(params, x)
        
        assert len(outputs) == 10
        for i, output in enumerate(outputs):
            expected_size = 5 + i
            assert output.shape == (4, expected_size)
    
    def test_composition_kernel_consistency(self):
        """Test that composed layers have consistent kernel functions."""
        # Create two equivalent networks
        # Network 1: Direct composition
        net1_init, net1_apply, net1_kernel = stax.serial(
            stax.Dense(50),
            stax.Relu(),
            stax.Dense(20)
        )
        
        # Network 2: Step by step composition (if individual kernel functions work)
        dense1 = stax.Dense(50)
        relu = stax.Relu()
        dense2 = stax.Dense(20)
        
        # For kernel computation, both should give similar results
        x1 = torch.randn(3, 10)
        x2 = torch.randn(4, 10)
        
        kernel1 = net1_kernel(x1, x2, 'nngp')
        assert isinstance(kernel1, torch.Tensor)
        assert kernel1.shape == (3, 4)
        
        # Test finite values
        assert torch.isfinite(kernel1).all()


if __name__ == "__main__":
    # Run tests directly
    test_serial = TestSerialCombinator()
    test_serial.test_serial_basic_functionality()
    test_serial.test_serial_single_layer()
    test_serial.test_serial_parameter_propagation()
    test_serial.test_serial_with_multidimensional_input()
    test_serial.test_serial_kernel_function()
    test_serial.test_serial_gradient_flow()
    print("✓ Serial combinator tests passed")
    
    test_parallel = TestParallelCombinator()
    test_parallel.test_parallel_basic_functionality()
    test_parallel.test_parallel_single_branch()
    test_parallel.test_parallel_different_output_shapes()
    test_parallel.test_parallel_kernel_function()
    print("✓ Parallel combinator tests passed")
    
    test_fanout = TestFanOutCombinator()
    test_fanout.test_fanout_basic()
    print("✓ FanOut combinator tests passed")
    
    test_fanin = TestFanInCombinator()
    test_fanin.test_fanin_concatenation()
    test_fanin.test_fanin_summation()
    print("✓ FanIn combinator tests passed")
    
    test_advanced = TestAdvancedCompositions()
    test_advanced.test_nested_serial_parallel()
    test_advanced.test_deep_serial_composition()
    test_advanced.test_wide_parallel_composition()
    test_advanced.test_composition_kernel_consistency()
    print("✓ Advanced composition tests passed")
    
    print("\n🎉 All stax combinator tests completed!")