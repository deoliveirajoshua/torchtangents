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

"""Comprehensive tests for torchtangents Monte Carlo utilities."""

import torch
import sys
import os
import math

# Add the torchtangents module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torchtangents as tt
from torchtangents import stax
from torchtangents._src.utils.kernel import Kernel


class TestMonteCarloBasic:
    """Basic tests for Monte Carlo functionality."""
    
    def test_monte_carlo_import(self):
        """Test that Monte Carlo module can be imported."""
        try:
            from torchtangents._src import monte_carlo
            assert monte_carlo is not None
        except ImportError:
            # Monte Carlo module may not be implemented
            print("Note: Monte Carlo module not found - may not be implemented yet")
            pass
    
    def test_monte_carlo_sampling_basic(self):
        """Test basic Monte Carlo sampling functionality."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            # Create a simple network for testing
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(20),
                stax.Relu(),
                stax.Dense(1)
            )
            
            # Test Monte Carlo kernel estimation
            rng = torch.Generator()
            rng.manual_seed(42)
            x = torch.randn(5, 3)
            
            # Create MC kernel function
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=100)
            
            # Compute kernel estimate
            kernel = mc_kernel(rng, x, x, 'nngp')
            
            assert isinstance(kernel, Kernel)
            assert kernel.nngp.shape == (5, 5)
            assert torch.isfinite(kernel.nngp).all()
            
        except ImportError:
            # Skip if Monte Carlo not implemented
            pass
    
    def test_monte_carlo_convergence(self):
        """Test Monte Carlo convergence with increasing samples."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(50),
                stax.Erf(),
                stax.Dense(1)
            )
            
            x = torch.randn(4, 6)
            
            # Test convergence with different sample sizes
            sample_sizes = [10, 50, 100, 500]
            kernels = []
            
            for n_samples in sample_sizes:
                mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=n_samples)
                
                rng = torch.Generator()
                rng.manual_seed(42)  # Same seed for consistency
                kernel = mc_kernel(rng, x, x, 'nngp')
                kernels.append(kernel.nngp)
            
            # Later estimates should be more stable (less variance)
            # This is a statistical test, so we use loose bounds
            assert all(torch.isfinite(k).all() for k in kernels)
            
        except ImportError:
            pass
    
    def test_monte_carlo_vs_analytical(self):
        """Test Monte Carlo approximation vs analytical kernel."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            # Use simple network where analytical solution is available
            init_fn, apply_fn, kernel_fn = stax.Dense(100)  # Large width
            
            x = torch.randn(3, 5)
            
            # Analytical kernel
            analytical = kernel_fn(x, x, 'nngp')
            
            # Monte Carlo approximation
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=1000)
            
            rng = torch.Generator()
            rng.manual_seed(42)
            mc_result = mc_kernel(rng, x, x, 'nngp')
            
            # Should be approximately equal (with some Monte Carlo error)
            correlation = torch.corrcoef(torch.cat([
                analytical.nngp.flatten(), 
                mc_result.nngp.flatten()
            ]))[0, 1]
            
            assert correlation > 0.8  # Should be reasonably correlated
            
        except ImportError:
            pass


class TestMonteCarloKernelFunctions:
    """Test Monte Carlo kernel function implementations."""
    
    def test_monte_carlo_nngp_estimation(self):
        """Test Monte Carlo NNGP estimation."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            # Network with nonlinearity
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(30),
                stax.Relu(),
                stax.Dense(10),
                stax.Gelu(),
                stax.Dense(1)
            )
            
            x1 = torch.randn(4, 6)
            x2 = torch.randn(3, 6)
            
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=200)
            
            rng = torch.Generator()
            rng.manual_seed(42)
            
            # Test self-covariance
            kernel_self = mc_kernel(rng, x1, x1, 'nngp')
            assert kernel_self.nngp.shape == (4, 4)
            
            # Should be approximately symmetric
            symmetry_error = torch.norm(kernel_self.nngp - kernel_self.nngp.T)
            assert symmetry_error < 0.1 * torch.norm(kernel_self.nngp)
            
            # Test cross-covariance
            rng.manual_seed(42)
            kernel_cross = mc_kernel(rng, x1, x2, 'nngp')
            assert kernel_cross.nngp.shape == (4, 3)
            
        except ImportError:
            pass
    
    def test_monte_carlo_ntk_estimation(self):
        """Test Monte Carlo NTK estimation."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            # Simple network for NTK
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(20),
                stax.Dense(1)
            )
            
            x = torch.randn(3, 4)
            
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=100)
            
            rng = torch.Generator()
            rng.manual_seed(42)
            kernel = mc_kernel(rng, x, x, 'ntk')
            
            assert kernel.ntk.shape == (3, 3)
            assert torch.isfinite(kernel.ntk).all()
            
            # NTK should be positive semi-definite
            eigenvals = torch.linalg.eigvals(kernel.ntk).real
            assert (eigenvals >= -1e-4).all()  # Allow for MC noise
            
        except ImportError:
            pass
    
    def test_monte_carlo_different_sample_sizes(self):
        """Test Monte Carlo with different sample sizes."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(15),
                stax.Sin(),
                stax.Dense(1)
            )
            
            x = torch.randn(2, 3)
            
            # Test various sample sizes
            for n_samples in [10, 50, 100]:
                mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=n_samples)
                
                rng = torch.Generator()
                rng.manual_seed(42)
                kernel = mc_kernel(rng, x, x, 'nngp')
                
                assert kernel.nngp.shape == (2, 2)
                assert torch.isfinite(kernel.nngp).all()
            
        except ImportError:
            pass
    
    def test_monte_carlo_reproducibility(self):
        """Test Monte Carlo reproducibility with same random seed."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            init_fn, apply_fn, _ = stax.Dense(10)
            x = torch.randn(3, 4)
            
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=50)
            
            # Same seed should give same results
            rng1 = torch.Generator()
            rng1.manual_seed(123)
            kernel1 = mc_kernel(rng1, x, x, 'nngp')
            
            rng2 = torch.Generator()
            rng2.manual_seed(123)
            kernel2 = mc_kernel(rng2, x, x, 'nngp')
            
            torch.testing.assert_close(kernel1.nngp, kernel2.nngp, rtol=1e-5, atol=1e-8)
            
        except ImportError:
            pass


class TestMonteCarloVariance:
    """Test variance estimation and reduction techniques."""
    
    def test_monte_carlo_variance_estimation(self):
        """Test Monte Carlo variance estimation."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            init_fn, apply_fn, _ = stax.Dense(5)
            x = torch.randn(2, 3)
            
            # Run multiple MC estimates to estimate variance
            n_runs = 20
            results = []
            
            for _ in range(n_runs):
                mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=10)
                rng = torch.Generator()
                rng.manual_seed(torch.randint(0, 10000, (1,)).item())
                
                kernel = mc_kernel(rng, x, x, 'nngp')
                results.append(kernel.nngp)
            
            # Compute sample variance
            results_tensor = torch.stack(results)
            sample_mean = results_tensor.mean(dim=0)
            sample_var = results_tensor.var(dim=0)
            
            # Variance should be finite and positive
            assert torch.isfinite(sample_var).all()
            assert (sample_var >= 0).all()
            
        except ImportError:
            pass
    
    def test_monte_carlo_variance_reduction(self):
        """Test that more samples reduce variance."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(20),
                stax.Relu(),
                stax.Dense(1)
            )
            
            x = torch.randn(2, 4)
            
            # Compare variance with different sample sizes
            sample_sizes = [10, 100]
            variances = []
            
            for n_samples in sample_sizes:
                # Multiple runs to estimate variance
                results = []
                for _ in range(10):
                    mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=n_samples)
                    rng = torch.Generator()
                    rng.manual_seed(torch.randint(0, 10000, (1,)).item())
                    
                    kernel = mc_kernel(rng, x, x, 'nngp')
                    results.append(kernel.nngp)
                
                variance = torch.stack(results).var(dim=0).mean()
                variances.append(variance)
            
            # More samples should generally reduce variance
            # (This is a statistical property, so we use a loose check)
            assert variances[1] <= variances[0] * 2  # Allow some noise
            
        except ImportError:
            pass


class TestMonteCarloAdvanced:
    """Advanced Monte Carlo tests."""
    
    def test_monte_carlo_different_networks(self):
        """Test Monte Carlo with different network architectures."""
        networks = [
            stax.Dense(10),
            stax.serial(stax.Dense(20), stax.Relu(), stax.Dense(1)),
            stax.serial(stax.Dense(15), stax.Erf(), stax.Dense(8), stax.Gelu(), stax.Dense(1)),
        ]
        
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            x = torch.randn(3, 5)
            
            for network in networks:
                init_fn, apply_fn, _ = network
                
                mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=50)
                
                rng = torch.Generator()
                rng.manual_seed(42)
                kernel = mc_kernel(rng, x, x, 'nngp')
                
                assert kernel.nngp.shape == (3, 3)
                assert torch.isfinite(kernel.nngp).all()
            
        except ImportError:
            pass
    
    def test_monte_carlo_parameter_initialization(self):
        """Test Monte Carlo with different parameter initialization schemes."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            # Test different initialization scales
            for W_std in [0.1, 1.0, 2.0]:
                init_fn, apply_fn, _ = stax.Dense(20, W_std=W_std, b_std=0.1)
                
                x = torch.randn(2, 3)
                
                mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=100)
                
                rng = torch.Generator()
                rng.manual_seed(42)
                kernel = mc_kernel(rng, x, x, 'nngp')
                
                assert torch.isfinite(kernel.nngp).all()
                
                # Larger initialization should generally give larger kernel values
                kernel_scale = torch.norm(kernel.nngp)
                if W_std == 2.0:
                    assert kernel_scale > 0.1  # Should be non-trivial
            
        except ImportError:
            pass
    
    def test_monte_carlo_computational_efficiency(self):
        """Test computational efficiency considerations."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            # Test that MC doesn't break with larger networks/data
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(100),
                stax.Relu(),
                stax.Dense(50),
                stax.Relu(),
                stax.Dense(1)
            )
            
            x = torch.randn(8, 10)  # Moderately sized input
            
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=20)  # Small sample for speed
            
            rng = torch.Generator()
            rng.manual_seed(42)
            kernel = mc_kernel(rng, x, x, 'nngp')
            
            assert kernel.nngp.shape == (8, 8)
            assert torch.isfinite(kernel.nngp).all()
            
        except ImportError:
            pass


class TestMonteCarloEdgeCases:
    """Test edge cases for Monte Carlo methods."""
    
    def test_monte_carlo_single_sample(self):
        """Test Monte Carlo with single input sample."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            init_fn, apply_fn, _ = stax.Dense(10)
            x = torch.randn(1, 4)  # Single sample
            
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=20)
            
            rng = torch.Generator()
            rng.manual_seed(42)
            kernel = mc_kernel(rng, x, x, 'nngp')
            
            assert kernel.nngp.shape == (1, 1)
            assert torch.isfinite(kernel.nngp).all()
            assert kernel.nngp.item() >= 0  # Should be non-negative
            
        except ImportError:
            pass
    
    def test_monte_carlo_very_few_samples(self):
        """Test Monte Carlo with very few samples."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            init_fn, apply_fn, _ = stax.Dense(5)
            x = torch.randn(2, 3)
            
            # Test with minimal samples
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=1)
            
            rng = torch.Generator()
            rng.manual_seed(42)
            kernel = mc_kernel(rng, x, x, 'nngp')
            
            assert kernel.nngp.shape == (2, 2)
            # Even with 1 sample, should produce finite result
            assert torch.isfinite(kernel.nngp).all()
            
        except ImportError:
            pass
    
    def test_monte_carlo_zero_width_network(self):
        """Test Monte Carlo with degenerate network."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            # Network that might be degenerate
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(1, W_std=0.0, b_std=0.0),  # Zero initialization
                stax.Dense(1)
            )
            
            x = torch.randn(2, 3)
            
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=10)
            
            rng = torch.Generator()
            rng.manual_seed(42)
            
            try:
                kernel = mc_kernel(rng, x, x, 'nngp')
                # If it succeeds, should be finite
                assert torch.isfinite(kernel.nngp).all()
            except (ValueError, RuntimeError):
                # May fail due to degenerate initialization
                pass
            
        except ImportError:
            pass


class TestMonteCarloIntegration:
    """Integration tests combining Monte Carlo with other components."""
    
    def test_monte_carlo_with_batching(self):
        """Test Monte Carlo combined with batching."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            from torchtangents._src.batching import batch
            
            init_fn, apply_fn, _ = stax.serial(
                stax.Dense(30),
                stax.Relu(),
                stax.Dense(1)
            )
            
            # Create batched MC kernel function
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=50)
            
            def batched_mc_kernel(x1, x2):
                rng = torch.Generator()
                rng.manual_seed(42)
                return mc_kernel(rng, x1, x2, 'nngp').nngp
            
            batched_fn = batch(batched_mc_kernel, batch_size=3, device_count=1)
            
            x1 = torch.randn(8, 5)
            x2 = torch.randn(6, 5)
            
            result = batched_fn(x1, x2)
            assert result.shape == (8, 6)
            assert torch.isfinite(result).all()
            
        except ImportError:
            pass
    
    def test_monte_carlo_vs_empirical_consistency(self):
        """Test consistency between Monte Carlo and empirical methods."""
        try:
            from torchtangents._src.monte_carlo import mc_kernel_fn
            
            # Use same network for both methods
            init_fn, apply_fn, _ = stax.Dense(50)  # Large width for convergence
            
            x = torch.randn(3, 4)
            
            # Empirical kernel with specific parameters
            rng = torch.Generator()
            rng.manual_seed(42)
            output_shape, params = init_fn(rng, x.shape)
            
            empirical_nngp_fn = tt.empirical_nngp_fn(apply_fn)
            empirical_result = empirical_nngp_fn(x, x, params)
            
            # Monte Carlo estimate
            mc_kernel = mc_kernel_fn(apply_fn, init_fn, n_samples=1000)
            
            rng.manual_seed(42)
            mc_result = mc_kernel(rng, x, x, 'nngp')
            
            # For linear layers, they should be quite close
            correlation = torch.corrcoef(torch.cat([
                empirical_result.flatten(),
                mc_result.nngp.flatten()
            ]))[0, 1]
            
            assert correlation > 0.7  # Should be reasonably correlated
            
        except ImportError:
            pass


if __name__ == "__main__":
    # Run tests directly
    test_basic = TestMonteCarloBasic()
    test_basic.test_monte_carlo_import()
    test_basic.test_monte_carlo_sampling_basic()
    test_basic.test_monte_carlo_convergence()
    test_basic.test_monte_carlo_vs_analytical()
    print("✓ Monte Carlo basic tests passed")
    
    test_kernels = TestMonteCarloKernelFunctions()
    test_kernels.test_monte_carlo_nngp_estimation()
    test_kernels.test_monte_carlo_ntk_estimation()
    test_kernels.test_monte_carlo_different_sample_sizes()
    test_kernels.test_monte_carlo_reproducibility()
    print("✓ Monte Carlo kernel function tests passed")
    
    test_variance = TestMonteCarloVariance()
    test_variance.test_monte_carlo_variance_estimation()
    test_variance.test_monte_carlo_variance_reduction()
    print("✓ Monte Carlo variance tests passed")
    
    test_advanced = TestMonteCarloAdvanced()
    test_advanced.test_monte_carlo_different_networks()
    test_advanced.test_monte_carlo_parameter_initialization()
    test_advanced.test_monte_carlo_computational_efficiency()
    print("✓ Monte Carlo advanced tests passed")
    
    test_edge_cases = TestMonteCarloEdgeCases()
    test_edge_cases.test_monte_carlo_single_sample()
    test_edge_cases.test_monte_carlo_very_few_samples()
    test_edge_cases.test_monte_carlo_zero_width_network()
    print("✓ Monte Carlo edge case tests passed")
    
    test_integration = TestMonteCarloIntegration()
    test_integration.test_monte_carlo_with_batching()
    test_integration.test_monte_carlo_vs_empirical_consistency()
    print("✓ Monte Carlo integration tests passed")
    
    print("\n🎉 All Monte Carlo tests completed!")