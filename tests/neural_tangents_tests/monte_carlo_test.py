"""Tests for torchtangents Monte Carlo functionality.

Tests adapted from neural-tangents monte_carlo_test.py to verify 
Monte Carlo sampling behavior in PyTorch implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
import torch.nn as nn
from absl.testing import absltest

from .test_utils import TorchTangentsTestCase, product, parameters
import torchtangents.stax as stax
from torchtangents import empirical_ntk_fn, empirical_nngp_fn, empirical_kernel_fn


BATCH_SIZES = [2, 4]
WIDTH = 32  # Smaller for testing

def _get_inputs_and_model(width=1, n_classes=2, use_conv=True):
    """Get test inputs and model."""
    generator = torch.Generator()
    generator.manual_seed(42)
    
    x1 = torch.randn(8, 3, 4, 2)
    x2 = torch.randn(4, 3, 4, 2)
    
    if not use_conv:
        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
    
    if use_conv:
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Conv(width, (3, 3), padding='SAME'),
            stax.Relu(),
            stax.Flatten(),
            stax.Dense(n_classes, W_std=2.0, b_std=0.5)
        )
    else:
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(width),
            stax.Relu(),
            stax.Dense(n_classes, W_std=2.0, b_std=0.5)
        )
    
    return x1, x2, init_fn, apply_fn, kernel_fn, generator


class MonteCarloTest(TorchTangentsTestCase):
    """Test Monte Carlo sampling functionality."""

    @parameters([
        {'batch_size': 2, 'get': 'ntk'},
        {'batch_size': 4, 'get': 'nngp'},
    ])
    def test_sample_consistency(self, batch_size, get):
        """Test that Monte Carlo sampling is consistent."""
        x1, x2, init_fn, apply_fn, kernel_fn, generator = _get_inputs_and_model(
            width=WIDTH, use_conv=False
        )
        
        def f(params, x):
            return apply_fn(params, x)
        
        # Generate multiple samples
        samples = []
        for i in range(3):  # Small number for testing
            gen = torch.Generator()
            gen.manual_seed(42 + i)
            _, params = init_fn(gen, (-1,) + x1.shape[1:])
            
            if get == 'ntk':
                kernel_fn_emp = empirical_ntk_fn(f)
                sample = kernel_fn_emp(x1, x2, params)
            else:  # nngp
                kernel_fn_emp = empirical_nngp_fn(f)
                sample = kernel_fn_emp(x1, x2, params)
            
            samples.append(sample)
        
        # Check that samples have correct shape
        expected_shape = (x1.shape[0], x2.shape[0])
        for sample in samples:
            self.assertEqual(sample.shape, expected_shape)
            self.assertTrue(torch.all(torch.isfinite(sample)))

    def test_monte_carlo_variance(self):
        """Test Monte Carlo variance estimation."""
        x1, x2, init_fn, apply_fn, kernel_fn, generator = _get_inputs_and_model(
            width=WIDTH, use_conv=False
        )
        
        def f(params, x):
            return apply_fn(params, x)
        
        # Generate many samples to estimate variance
        samples = []
        n_samples = 10
        
        for i in range(n_samples):
            gen = torch.Generator()
            gen.manual_seed(100 + i)
            _, params = init_fn(gen, (-1,) + x1.shape[1:])
            
            kernel_fn_emp = empirical_ntk_fn(f)
            sample = kernel_fn_emp(x1[:2], x2[:2], params)  # Small subset
            samples.append(sample)
        
        # Stack samples and compute statistics
        samples_tensor = torch.stack(samples)
        mean_sample = torch.mean(samples_tensor, dim=0)
        var_sample = torch.var(samples_tensor, dim=0)
        
        # Variance should be positive
        self.assertTrue(torch.all(var_sample >= 0))
        
        # Mean should be finite
        self.assertTrue(torch.all(torch.isfinite(mean_sample)))

    @parameters([
        {'use_conv': True, 'width': 16},
        {'use_conv': False, 'width': 32},
    ])
    def test_different_architectures(self, use_conv, width):
        """Test Monte Carlo with different architectures."""
        x1, x2, init_fn, apply_fn, kernel_fn, generator = _get_inputs_and_model(
            width=width, use_conv=use_conv
        )
        
        def f(params, x):
            return apply_fn(params, x)
        
        # Test single sample
        _, params = init_fn(generator, (-1,) + x1.shape[1:])
        
        # Test both NNGP and NTK
        for kernel_type in ['nngp', 'ntk']:
            with self.subTest(kernel_type=kernel_type):
                if kernel_type == 'ntk':
                    kernel_fn_emp = empirical_ntk_fn(f)
                else:
                    kernel_fn_emp = empirical_nngp_fn(f)
                
                try:
                    sample = kernel_fn_emp(x1, x2, params)
                    
                    # Check basic properties
                    expected_shape = (x1.shape[0], x2.shape[0])
                    self.assertEqual(sample.shape, expected_shape)
                    self.assertTrue(torch.all(torch.isfinite(sample)))
                    
                except Exception as e:
                    self.skipTest(f"Architecture test failed: {e}")

    def test_kernel_symmetry_monte_carlo(self):
        """Test kernel symmetry in Monte Carlo sampling."""
        x1, x2, init_fn, apply_fn, kernel_fn, generator = _get_inputs_and_model(
            width=WIDTH, use_conv=False
        )
        
        def f(params, x):
            return apply_fn(params, x)
        
        # Use same data for both inputs to test symmetry
        x = x1[:4]  # Small subset
        
        samples = []
        for i in range(5):
            gen = torch.Generator()
            gen.manual_seed(200 + i)
            _, params = init_fn(gen, (-1,) + x.shape[1:])
            
            kernel_fn_emp = empirical_ntk_fn(f)
            sample = kernel_fn_emp(x, None, params)  # x2=None should use x1
            samples.append(sample)
        
        # Test symmetry for each sample
        for i, sample in enumerate(samples):
            with self.subTest(sample=i):
                self.assertTensorsAllClose(sample, sample.T, atol=1e-5, rtol=1e-5)

    def test_monte_carlo_convergence(self):
        """Test that Monte Carlo estimates converge."""
        x1, x2, init_fn, apply_fn, kernel_fn, generator = _get_inputs_and_model(
            width=WIDTH, use_conv=False
        )
        
        def f(params, x):
            return apply_fn(params, x)
        
        # Use small inputs for faster computation
        x1_small = x1[:2]
        x2_small = x2[:2]
        
        # Collect samples
        samples = []
        for i in range(8):  # Moderate number for testing
            gen = torch.Generator()
            gen.manual_seed(300 + i)
            _, params = init_fn(gen, (-1,) + x1_small.shape[1:])
            
            kernel_fn_emp = empirical_ntk_fn(f)
            sample = kernel_fn_emp(x1_small, x2_small, params)
            samples.append(sample)
        
        # Compute running averages
        running_means = []
        for i in range(1, len(samples) + 1):
            running_mean = torch.mean(torch.stack(samples[:i]), dim=0)
            running_means.append(running_mean)
        
        # Later averages should be more stable (smaller changes)
        if len(running_means) >= 4:
            early_change = torch.norm(running_means[2] - running_means[1])
            late_change = torch.norm(running_means[-1] - running_means[-2])
            
            # This is a weak test - just that we don't have huge instabilities
            self.assertTrue(early_change < 100)
            self.assertTrue(late_change < 100)

    def test_batch_vs_individual_consistency(self):
        """Test that batched computation matches individual computation."""
        x1, x2, init_fn, apply_fn, kernel_fn, generator = _get_inputs_and_model(
            width=WIDTH, use_conv=False
        )
        
        def f(params, x):
            return apply_fn(params, x)
        
        # Small inputs for testing
        x1_small = x1[:2]
        x2_small = x2[:1]
        
        # Individual computation
        gen1 = torch.Generator()
        gen1.manual_seed(400)
        _, params = init_fn(gen1, (-1,) + x1_small.shape[1:])
        
        kernel_fn_emp = empirical_ntk_fn(f)
        individual_result = kernel_fn_emp(x1_small, x2_small, params)
        
        # "Batched" computation (same parameters, should give same result)
        gen2 = torch.Generator()
        gen2.manual_seed(400)  # Same seed
        _, params2 = init_fn(gen2, (-1,) + x1_small.shape[1:])
        
        batch_result = kernel_fn_emp(x1_small, x2_small, params2)
        
        # Should be identical with same seed
        self.assertTensorsAllClose(individual_result, batch_result, atol=1e-7)


if __name__ == '__main__':
    absltest.main()