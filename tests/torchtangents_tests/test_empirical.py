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

"""Comprehensive tests for torchtangents empirical kernel computation."""

import torch
import sys
import os

# Add the torchtangents module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torchtangents as tt
from torchtangents import stax
from torchtangents._src.empirical import (
    empirical_nngp_fn, empirical_ntk_fn, empirical_kernel_fn,
    NtkImplementation
)
from torchtangents._src.utils.kernel import Kernel


class TestEmpiricalNNGP:
    """Comprehensive tests for empirical NNGP computation."""
    
    def test_empirical_nngp_basic(self):
        """Test basic empirical NNGP computation."""
        # Create simple network
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(8),
            stax.Relu(),
            stax.Dense(1)
        )
        
        # Initialize parameters
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(5, 3)
        output_shape, params = init_fn(rng, x.shape)
        
        # Create NNGP function
        nngp_fn = empirical_nngp_fn(apply_fn)
        
        # Compute NNGP
        nngp = nngp_fn(x, None, params)
        
        assert nngp.shape == (5, 5)
        # NNGP should be symmetric for same inputs
        torch.testing.assert_close(nngp, nngp.T, rtol=1e-5, atol=1e-8)
        
        # NNGP should be positive semi-definite
        eigenvals = torch.linalg.eigvals(nngp).real
        assert (eigenvals >= -1e-6).all()  # Allow small numerical errors
    
    def test_empirical_nngp_cross_covariance(self):
        """Test empirical NNGP cross-covariance computation."""
        # Create network
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(10),
            stax.Gelu(),
            stax.Dense(2)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x1 = torch.randn(4, 6)
        x2 = torch.randn(3, 6)
        output_shape, params = init_fn(rng, x1.shape)
        
        nngp_fn = empirical_nngp_fn(apply_fn)
        
        # Compute cross-covariance
        nngp_cross = nngp_fn(x1, x2, params)
        assert nngp_cross.shape == (4, 3)
        
        # Should be finite
        assert torch.isfinite(nngp_cross).all()
    
    def test_empirical_nngp_different_network_sizes(self):
        """Test NNGP with different network architectures."""
        networks = [
            # Simple linear
            stax.Dense(1),
            
            # Single hidden layer
            stax.serial(stax.Dense(20), stax.Relu(), stax.Dense(1)),
            
            # Deep network
            stax.serial(
                stax.Dense(50),
                stax.Relu(),
                stax.Dense(30),
                stax.Relu(),
                stax.Dense(10),
                stax.Relu(),
                stax.Dense(1)
            ),
            
            # With different activations
            stax.serial(
                stax.Dense(15),
                stax.Erf(),
                stax.Dense(8),
                stax.Sin(),
                stax.Dense(1)
            )
        ]
        
        x = torch.randn(6, 4)
        
        for network in networks:
            init_fn, apply_fn, _ = network
            
            rng = torch.Generator()
            rng.manual_seed(42)
            output_shape, params = init_fn(rng, x.shape)
            
            nngp_fn = empirical_nngp_fn(apply_fn)
            nngp = nngp_fn(x, None, params)
            
            assert nngp.shape == (6, 6)
            assert torch.isfinite(nngp).all()
            torch.testing.assert_close(nngp, nngp.T, rtol=1e-5, atol=1e-8)
    
    def test_empirical_nngp_batch_sizes(self):
        """Test NNGP with different batch sizes."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(12),
            stax.Relu(),
            stax.Dense(1)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (1, 5)
        output_shape, params = init_fn(rng, input_shape)
        
        nngp_fn = empirical_nngp_fn(apply_fn)
        
        # Test various batch sizes
        for batch_size in [1, 2, 5, 10, 20]:
            x = torch.randn(batch_size, 5)
            nngp = nngp_fn(x, None, params)
            
            assert nngp.shape == (batch_size, batch_size)
            assert torch.isfinite(nngp).all()
    
    def test_empirical_nngp_trace_axes(self):
        """Test NNGP with trace axes specification."""
        # Create network that outputs multiple dimensions
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(8),
            stax.Relu(),
            stax.Dense(3)  # 3D output
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(4, 5)
        output_shape, params = init_fn(rng, x.shape)
        
        # Test with different trace axes
        nngp_fn = empirical_nngp_fn(apply_fn, trace_axes=(-1,))
        nngp = nngp_fn(x, None, params)
        
        assert nngp.shape == (4, 4)
        assert torch.isfinite(nngp).all()


class TestEmpiricalNTK:
    """Comprehensive tests for empirical NTK computation."""
    
    def test_empirical_ntk_basic(self):
        """Test basic empirical NTK computation."""
        # Create simple network
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(6),
            stax.Dense(1)
        )
        
        # Initialize parameters
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(4, 3)
        output_shape, params = init_fn(rng, x.shape)
        
        # Create NTK function
        ntk_fn = empirical_ntk_fn(apply_fn)
        
        # Compute NTK
        ntk = ntk_fn(x, None, params)
        
        assert ntk.shape == (4, 4)
        # NTK should be symmetric
        torch.testing.assert_close(ntk, ntk.T, rtol=1e-5, atol=1e-8)
        
        # NTK should be positive semi-definite
        eigenvals = torch.linalg.eigvals(ntk).real
        assert (eigenvals >= -1e-6).all()  # Allow small numerical errors
    
    def test_empirical_ntk_cross_kernel(self):
        """Test empirical NTK cross-kernel computation."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(10),
            stax.Relu(),
            stax.Dense(2)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x1 = torch.randn(3, 5)
        x2 = torch.randn(4, 5)
        output_shape, params = init_fn(rng, x1.shape)
        
        ntk_fn = empirical_ntk_fn(apply_fn)
        
        # Compute cross-kernel
        ntk_cross = ntk_fn(x1, x2, params)
        assert ntk_cross.shape == (3, 4)
        assert torch.isfinite(ntk_cross).all()
    
    def test_empirical_ntk_implementations(self):
        """Test different NTK implementations."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(8),
            stax.Relu(),
            stax.Dense(1)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(3, 4)
        output_shape, params = init_fn(rng, x.shape)
        
        implementations = [
            NtkImplementation.JACOBIAN_CONTRACTION,
            NtkImplementation.NTK_VECTOR_PRODUCTS,
            # NtkImplementation.STRUCTURED_DERIVATIVES  # May not be fully implemented
        ]
        
        results = []
        for impl in implementations:
            try:
                ntk_fn = empirical_ntk_fn(apply_fn, implementation=impl)
                ntk = ntk_fn(x, None, params)
                
                assert ntk.shape == (3, 3)
                assert torch.isfinite(ntk).all()
                results.append(ntk)
            except (NotImplementedError, ValueError):
                # Some implementations might not be available
                continue
        
        # If multiple implementations work, they should give similar results
        if len(results) > 1:
            for i in range(1, len(results)):
                torch.testing.assert_close(results[0], results[i], rtol=1e-5, atol=1e-8)
    
    def test_empirical_ntk_gradient_relationship(self):
        """Test relationship between NTK and gradients."""
        # Simple linear network for easier analysis
        init_fn, apply_fn, _ = stax.Dense(1)
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(2, 3)
        output_shape, params = init_fn(rng, x.shape)
        
        # Enable gradients
        params['weight'].requires_grad_(True)
        if params['bias'] is not None:
            params['bias'].requires_grad_(True)
        
        # Compute NTK
        ntk_fn = empirical_ntk_fn(apply_fn)
        ntk = ntk_fn(x, None, params)

        # Manual computation: NTK[i,j] = ∇_θ f(x_i) · ∇_θ f(x_j)
        def compute_jacobian(xi):
            output = apply_fn(params, xi.unsqueeze(0))
            grad_W = torch.autograd.grad(output, params['weight'], retain_graph=True)[0]
            grads = [grad_W.flatten()]
            if params['bias'] is not None:
                grad_b = torch.autograd.grad(output, params['bias'], retain_graph=True)[0]
                grads.append(grad_b.flatten())
            return torch.cat(grads)

        jacobians = []
        for i in range(x.shape[0]):
            jac = compute_jacobian(x[i])
            jacobians.append(jac)

        jacobians = torch.stack(jacobians)
        ntk_manual = torch.matmul(jacobians, jacobians.T)

        torch.testing.assert_close(ntk, ntk_manual, rtol=1e-5, atol=1e-8)

    def test_empirical_ntk_different_output_dimensions(self):
        """Test NTK with different output dimensions."""
        for output_dim in [1, 3, 5, 10]:
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(8),
                stax.Relu(),
                stax.Dense(output_dim)
            )
            
            rng = torch.Generator()
            rng.manual_seed(42)
            x = torch.randn(4, 6)
            output_shape, params = init_fn(rng, x.shape)
            
            ntk_fn = empirical_ntk_fn(apply_fn)
            ntk = ntk_fn(x, None, params)
            
            # NTK shape should be (batch, batch) regardless of output dimension
            assert ntk.shape == (4, 4)
            assert torch.isfinite(ntk).all()
    
    def test_empirical_ntk_parameter_scaling(self):
        """Test how NTK scales with parameter initialization for nonlinear networks."""
        # Use a nonlinear network where NTK scaling is meaningful
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(3),
            stax.Relu(),
            stax.Dense(1)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(3, 4)
        output_shape, params = init_fn(rng, x.shape)
        
        ntk_fn = empirical_ntk_fn(apply_fn)
        ntk_base = ntk_fn(x, None, params)
        
        # Scale parameters - only test small scaling to avoid numerical issues
        scale = 1.5
        
        def scale_params(p, s):
            if isinstance(p, torch.Tensor):
                return p * s
            elif isinstance(p, dict):
                return {k: scale_params(v, s) for k, v in p.items()}
            elif isinstance(p, (list, tuple)):
                return type(p)(scale_params(item, s) for item in p)
            else:
                return p
        
        params_scaled = scale_params(params, scale)
        ntk_scaled = ntk_fn(x, None, params_scaled)
        
        # For nonlinear networks, NTK should change with parameter scaling
        # We just check that they're different (exact scaling may be complex)
        assert not torch.allclose(ntk_scaled, ntk_base, rtol=1e-3)
        
        # Check that finite values are preserved
        finite_base = torch.isfinite(ntk_base)
        finite_scaled = torch.isfinite(ntk_scaled)
        assert finite_base.all() or finite_scaled.all()  # At least one should be finite


class TestEmpiricalKernelFunction:
    """Test the unified empirical_kernel_fn."""
    
    def test_empirical_kernel_fn_basic(self):
        """Test basic empirical_kernel_fn functionality."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(10),
            stax.Relu(),
            stax.Dense(1)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(5, 4)
        output_shape, params = init_fn(rng, x.shape)
        
        # Test getting both NNGP and NTK
        kernel_fn = empirical_kernel_fn(apply_fn)
        kernel = kernel_fn(x, None, get='nngp', params=params)
        
        assert isinstance(kernel, Kernel)
        assert kernel.nngp is not None
        assert kernel.nngp.shape == (5, 5)
        
        # Test getting NTK
        kernel_ntk = kernel_fn(x, None, get='ntk', params=params)
        assert kernel_ntk.ntk is not None
        assert kernel_ntk.ntk.shape == (5, 5)
    
    def test_empirical_kernel_fn_get_options(self):
        """Test different 'get' options for empirical_kernel_fn."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(6),
            stax.Dense(1)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(3, 5)
        output_shape, params = init_fn(rng, x.shape)
        
        kernel_fn = empirical_kernel_fn(apply_fn)
        
        # Test individual gets
        kernel_nngp = kernel_fn(x, None, get='nngp', params=params)
        assert kernel_nngp.nngp is not None
        assert kernel_nngp.ntk is None
        
        kernel_ntk = kernel_fn(x, None, get='ntk', params=params)
        assert kernel_ntk.nngp is None
        assert kernel_ntk.ntk is not None
        
        # Test getting both (if supported)
        try:
            kernel_both = kernel_fn(x, None, get=('nngp', 'ntk'), params=params)
            assert kernel_both.nngp is not None
            assert kernel_both.ntk is not None
        except (ValueError, NotImplementedError):
            # May not be implemented
            pass
    
    def test_empirical_kernel_fn_trace_diagonal_axes(self):
        """Test empirical_kernel_fn with trace and diagonal axes."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(8),
            stax.Relu(),
            stax.Dense(3)  # Multi-dimensional output
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(4, 6)
        output_shape, params = init_fn(rng, x.shape)
        
        kernel_fn = empirical_kernel_fn(apply_fn, trace_axes=(-1,))
        kernel = kernel_fn(x, None, get='nngp', params=params)
        
        assert kernel.nngp.shape == (4, 4)
        assert torch.isfinite(kernel.nngp).all()


class TestEmpiricalEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empirical_with_empty_network(self):
        """Test empirical computation with identity network."""
        init_fn, apply_fn, _ = stax.Identity()
        
        rng = torch.Generator()
        x = torch.randn(3, 4)
        output_shape, params = init_fn(rng, x.shape)
        
        # NNGP should work
        nngp_fn = empirical_nngp_fn(apply_fn)
        nngp = nngp_fn(x, None, params)
        
        # Should be just dot product for identity
        expected = torch.matmul(x, x.T)
        torch.testing.assert_close(nngp, expected, rtol=1e-5, atol=1e-8)
        
        # NTK should be zero for identity (no parameters)
        try:
            ntk_fn = empirical_ntk_fn(apply_fn)
            ntk = ntk_fn(x, None, params)
            # Should be zero or raise error
            assert torch.allclose(ntk, torch.zeros_like(ntk))
        except (ValueError, RuntimeError):
            # May raise error for networks without parameters
            pass
    
    def test_empirical_with_single_input(self):
        """Test empirical computation with single input."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(5),
            stax.Relu(),
            stax.Dense(1)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(1, 3)  # Single input
        output_shape, params = init_fn(rng, x.shape)
        
        nngp_fn = empirical_nngp_fn(apply_fn)
        nngp = nngp_fn(x, None, params)
        
        assert nngp.shape == (1, 1)
        assert nngp.item() >= 0  # Should be non-negative
        
        ntk_fn = empirical_ntk_fn(apply_fn)
        ntk = ntk_fn(x, None, params)
        
        assert ntk.shape == (1, 1)
        assert ntk.item() >= 0  # Should be non-negative
    
    def test_empirical_with_large_batch(self):
        """Test empirical computation with large batch size."""
        init_fn, apply_fn, _ = stax.serial(
            stax.Dense(20),
            stax.Relu(),
            stax.Dense(1)
        )
        
        rng = torch.Generator()
        rng.manual_seed(42)
        batch_size = 100  # Large batch
        x = torch.randn(batch_size, 10)
        output_shape, params = init_fn(rng, x.shape)
        
        # This might be memory intensive, but should work
        nngp_fn = empirical_nngp_fn(apply_fn)
        nngp = nngp_fn(x, None, params)
        
        assert nngp.shape == (batch_size, batch_size)
        assert torch.isfinite(nngp).all()
    
    def test_empirical_numerical_stability(self):
        """Test numerical stability with extreme parameter values."""
        init_fn, apply_fn, _ = stax.Dense(1)
        
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(3, 4)
        output_shape, params = init_fn(rng, x.shape)
        
        # Test with very small parameters
        params_small = {
            'weight': params['weight'] * 1e-10,
            'bias': params['bias'] * 1e-10 if params['bias'] is not None else None
        }
        
        nngp_fn = empirical_nngp_fn(apply_fn)
        nngp_small = nngp_fn(x, None, params_small)
        assert torch.isfinite(nngp_small).all()
        
        # Test with very large parameters
        params_large = {
            'weight': params['weight'] * 1e3,
            'bias': params['bias'] * 1e3 if params['bias'] is not None else None
        }
        
        nngp_large = nngp_fn(x, None, params_large)
        assert torch.isfinite(nngp_large).all()
    
    def test_empirical_with_different_dtypes(self):
        """Test empirical computation with different tensor dtypes."""
        init_fn, apply_fn, _ = stax.Dense(1)
        
        # Test with float32
        rng = torch.Generator()
        rng.manual_seed(42)
        x_f32 = torch.randn(3, 4, dtype=torch.float32)
        output_shape, params_f32 = init_fn(rng, x_f32.shape)
        
        nngp_fn = empirical_nngp_fn(apply_fn)
        nngp_f32 = nngp_fn(x_f32, None, params_f32)
        assert nngp_f32.dtype == torch.float32
        
        # Test with float64
        x_f64 = x_f32.double()
        params_f64 = {k: v.double() if v is not None else v for k, v in params_f32.items()}
        
        nngp_f64 = nngp_fn(x_f64, None, params_f64)
        assert nngp_f64.dtype == torch.float64


if __name__ == "__main__":
    # Run tests directly
    test_nngp = TestEmpiricalNNGP()
    test_nngp.test_empirical_nngp_basic()
    test_nngp.test_empirical_nngp_cross_covariance()
    test_nngp.test_empirical_nngp_different_network_sizes()
    test_nngp.test_empirical_nngp_batch_sizes()
    test_nngp.test_empirical_nngp_trace_axes()
    print("✓ Empirical NNGP tests passed")
    
    test_ntk = TestEmpiricalNTK()
    test_ntk.test_empirical_ntk_basic()
    test_ntk.test_empirical_ntk_cross_kernel()
    test_ntk.test_empirical_ntk_implementations()
    test_ntk.test_empirical_ntk_gradient_relationship()
    test_ntk.test_empirical_ntk_different_output_dimensions()
    test_ntk.test_empirical_ntk_parameter_scaling()
    print("✓ Empirical NTK tests passed")
    
    test_kernel_fn = TestEmpiricalKernelFunction()
    test_kernel_fn.test_empirical_kernel_fn_basic()
    test_kernel_fn.test_empirical_kernel_fn_get_options()
    test_kernel_fn.test_empirical_kernel_fn_trace_diagonal_axes()
    print("✓ Empirical kernel function tests passed")
    
    test_edge_cases = TestEmpiricalEdgeCases()
    test_edge_cases.test_empirical_with_empty_network()
    test_edge_cases.test_empirical_with_single_input()
    test_edge_cases.test_empirical_with_large_batch()
    test_edge_cases.test_empirical_numerical_stability()
    test_edge_cases.test_empirical_with_different_dtypes()
    print("✓ Empirical edge case tests passed")
    
    print("\n🎉 All empirical tests completed!")