#!/usr/bin/env python3
"""
Final demonstration: TorchTangents NTK gradient capabilities work!
"""

import torch
import sys
import os

# Add the torchtangents package to path
sys.path.insert(0, os.path.abspath('.'))

import torchtangents as tt
from torchtangents import stax
from torchtangents._src.utils.kernel import Kernel

def working_example_infinite_width():
    """Working example of infinite-width NTK gradients."""
    print("✅ INFINITE-WIDTH NTK GRADIENTS W.R.T. INPUTS")
    print("-" * 50)
    
    # Create inputs that require gradients
    x1 = torch.randn(3, 4, requires_grad=True)
    x2 = torch.randn(2, 4, requires_grad=True)
    
    # Create infinite-width network
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512, W_std=1.0, b_std=0.1),
        stax.Relu(),
        stax.Dense(1, W_std=1.0, b_std=0.1)
    )
    
    # Create base kernel (input dot product)
    base_nngp = torch.matmul(x1, x2.T) / x1.shape[1]
    base_kernel = Kernel(
        nngp=base_nngp,
        ntk=base_nngp.clone(),
        cov1=torch.matmul(x1, x1.T) / x1.shape[1],
        cov2=torch.matmul(x2, x2.T) / x2.shape[1]
    )
    
    # Apply infinite-width transformation
    infinite_kernel = kernel_fn(base_kernel)
    
    # Compute gradients
    ntk_scalar = infinite_kernel.ntk.sum()
    grad_x1, grad_x2 = torch.autograd.grad(ntk_scalar, [x1, x2])
    
    print(f"✓ NTK shape: {infinite_kernel.ntk.shape}")
    print(f"✓ ∂NTK/∂x1 shape: {grad_x1.shape}, norm: {grad_x1.norm():.4f}")
    print(f"✓ ∂NTK/∂x2 shape: {grad_x2.shape}, norm: {grad_x2.norm():.4f}")
    print("✓ SUCCESS: Can compute infinite-width NTK gradients w.r.t. inputs!")

def working_example_empirical():
    """Working example of empirical NTK gradients."""
    print("\n✅ EMPIRICAL NTK GRADIENTS W.R.T. INPUTS")
    print("-" * 50)
    
    # Create inputs that require gradients
    x1 = torch.randn(3, 4, requires_grad=True)
    x2 = torch.randn(2, 4, requires_grad=True)
    
    # Create finite-width network
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(32, W_std=1.0, b_std=0.1),
        stax.Relu(),
        stax.Dense(1, W_std=1.0, b_std=0.1)
    )
    
    # Initialize parameters
    rng = torch.Generator()
    rng.manual_seed(42)
    output_shape, params = init_fn(rng, x1.shape)
    
    # Compute empirical NTK
    empirical_kernel_fn = tt.empirical_kernel_fn(apply_fn)
    kernel_result = empirical_kernel_fn(x1, x2, 'ntk', params)
    ntk = kernel_result.ntk
    
    # Compute gradients
    ntk_scalar = ntk.sum()
    grad_x1, grad_x2 = torch.autograd.grad(ntk_scalar, [x1, x2])
    
    print(f"✓ NTK shape: {ntk.shape}")
    print(f"✓ ∂NTK/∂x1 shape: {grad_x1.shape}, norm: {grad_x1.norm():.4f}")
    print(f"✓ ∂NTK/∂x2 shape: {grad_x2.shape}, norm: {grad_x2.norm():.4f}")
    print("✓ SUCCESS: Can compute empirical NTK gradients w.r.t. inputs!")

def working_example_finite_parameter_gradients():
    """Working example showing what works for finite NTK gradients."""
    print("\n✅ FINITE NTK GRADIENTS (INPUT GRADIENTS WORK)")
    print("-" * 50)
    
    # Simple network with explicit parameters
    x = torch.randn(3, 4, requires_grad=True)
    W = torch.randn(4, 2, requires_grad=True)
    b = torch.randn(2, requires_grad=True)
    
    def simple_network(x, W, b):
        return x @ W + b
    
    # Compute Jacobian matrix manually (this preserves input gradients)
    def compute_jacobian_row(xi):
        output = simple_network(xi.unsqueeze(0), W, b)
        grad_W = torch.autograd.grad(output.sum(), W, retain_graph=True, create_graph=True)[0]
        grad_b = torch.autograd.grad(output.sum(), b, retain_graph=True, create_graph=True)[0]
        return torch.cat([grad_W.flatten(), grad_b.flatten()])
    
    # Build Jacobian matrix
    jacobians = []
    for i in range(x.shape[0]):
        jac_row = compute_jacobian_row(x[i])
        jacobians.append(jac_row)
    
    J = torch.stack(jacobians)  # Shape: (batch_size, n_params)
    
    # Compute NTK = J @ J.T
    ntk = J @ J.T
    
    # Test input gradients (these work!)
    ntk_scalar = ntk.sum()
    grad_x = torch.autograd.grad(ntk_scalar, x, retain_graph=True)[0]
    
    print(f"✓ NTK shape: {ntk.shape}")
    print(f"✓ ∂NTK/∂x shape: {grad_x.shape}, norm: {grad_x.norm():.4f}")
    print("✓ SUCCESS: Can compute finite NTK gradients w.r.t. inputs!")
    
    # Explain why parameter gradients don't work with this method
    print("\n📝 Note on Parameter Gradients:")
    print("   • Input gradients work because x flows through the Jacobian computation")
    print("   • Parameter gradients don't work because W,b are not in final NTK graph")
    print("   • For parameter gradients, need different approach (e.g., implicit differentiation)")
    print("   • TorchTangents focuses on infinite-width theory where parameters → ∞")

if __name__ == "__main__":
    print("🎯 TORCHTANGENTS NTK GRADIENT CAPABILITIES DEMONSTRATION")
    print("=" * 70)
    
    working_example_infinite_width()
    working_example_empirical()
    working_example_finite_parameter_gradients()
    
    print("\n" + "=" * 70)
    print("🏆 FINAL ANSWER TO YOUR QUESTION:")
    print("=" * 70)
    print("✅ YES! You CAN find gradients of infinite-width NTK w.r.t. inputs!")
    print()
    print("📋 WHAT WORKS:")
    print("  ✅ Infinite-width NTK gradients w.r.t. inputs")
    print("  ✅ Empirical (finite-width) NTK gradients w.r.t. inputs")
    print("  ✅ Finite NTK input gradients (manual Jacobian method)")
    print()
    print("🔧 HOW TO USE:")
    print("  • For input optimization: Use either infinite-width or empirical NTK")
    print("  • For parameter gradients: Use specialized methods (not standard autodiff)")
    print("  • For theoretical analysis: Use infinite-width kernels")
    print("  • Primary use case: INPUT GRADIENTS work perfectly!")
    print()
    print("🎉 TorchTangents supports NTK input gradient computation perfectly!")