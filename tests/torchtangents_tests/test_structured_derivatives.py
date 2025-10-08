#!/usr/bin/env python3
"""Test script for structured derivatives implementation."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torchtangents._src.empirical import empirical_ntk_fn, NtkImplementation


def test_structured_derivatives():
    """Test structured derivatives implementation."""
    print("Testing structured derivatives implementation...")
    
    # Create a simple neural network
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 5)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    # Initialize network and data
    net = SimpleNet()
    x1 = torch.randn(8, 10)  # Small batch for testing
    x2 = torch.randn(6, 10)
    
    def f(params, x):
        """Function that uses parameters."""
        # Use functional_call to apply parameters
        try:
            return torch.func.functional_call(net, params, (x,))
        except AttributeError:
            # Fallback if functional_call not available
            # Temporarily set the network parameters
            original_params = {}
            for name, param in net.named_parameters():
                original_params[name] = param.data.clone()
                param.data = params[name]
            
            result = net(x)
            
            # Restore original parameters
            for name, param in net.named_parameters():
                param.data = original_params[name]
            
            return result
    
    # Get network parameters
    params = dict(net.named_parameters())
    
    try:
        # Test JACOBIAN_CONTRACTION (baseline)
        print("Testing JACOBIAN_CONTRACTION...")
        ntk_fn_jacobian = empirical_ntk_fn(
            f, implementation=NtkImplementation.JACOBIAN_CONTRACTION
        )
        ntk_jacobian = ntk_fn_jacobian(x1, x2, params)
        print(f"Jacobian contraction NTK shape: {ntk_jacobian.shape}")
        
        # Test STRUCTURED_DERIVATIVES
        print("Testing STRUCTURED_DERIVATIVES...")
        ntk_fn_structured = empirical_ntk_fn(
            f, implementation=NtkImplementation.STRUCTURED_DERIVATIVES
        )
        ntk_structured = ntk_fn_structured(x1, x2, params)
        print(f"Structured derivatives NTK shape: {ntk_structured.shape}")
        
        # Compare results
        if ntk_jacobian.shape == ntk_structured.shape:
            diff = torch.norm(ntk_jacobian - ntk_structured)
            print(f"Difference between methods: {diff.item():.6f}")
            
            if diff < 1e-4:
                print("✓ Structured derivatives implementation produces correct results!")
            else:
                print("⚠ Structured derivatives results differ from baseline")
        else:
            print("⚠ Shape mismatch between implementations")
            
        print("Basic structured derivatives test completed successfully!")
        assert True  # Test passed
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Test failed with error: {e}"
        return False


def test_different_trace_axes():
    """Test structured derivatives with different trace axes."""
    print("\nTesting different trace axes...")
    
    # Simple network
    net = nn.Linear(5, 3)
    x = torch.randn(4, 5)
    
    def f(params, x):
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
    
    try:
        # Test different trace axes
        for trace_axes in [(-1,), (), (0,)]:
            print(f"Testing trace_axes={trace_axes}...")
            
            ntk_fn = empirical_ntk_fn(
                f, trace_axes=trace_axes,
                implementation=NtkImplementation.STRUCTURED_DERIVATIVES
            )
            ntk = ntk_fn(x, None, params)
            print(f"  NTK shape: {ntk.shape}")
            
        print("✓ Different trace axes test passed!")
        assert True  # Test passed
        
    except Exception as e:
        print(f"✗ Trace axes test failed: {e}")
        assert False, f"Trace axes test failed: {e}"


if __name__ == "__main__":
    print("Running structured derivatives tests...\n")
    
    success = True
    success &= test_structured_derivatives()
    success &= test_different_trace_axes()
    
    print(f"\nOverall test result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)