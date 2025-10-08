#!/usr/bin/env python3
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

"""
NTK Gradient Computation Examples

This example demonstrates how to compute gradients of Neural Tangent Kernels (NTK)
with respect to inputs using TorchTangents. This is useful for:

- Input optimization and adversarial analysis
- Input sensitivity analysis  
- Gradient-based input design
- Understanding infinite-width network behavior

The example shows three methods:
1. Infinite-width NTK gradients (theoretical limit)
2. Empirical NTK gradients (finite-width approximation)
3. Manual Jacobian computation (educational)
"""

import torch
import torchtangents as tt
from torchtangents import stax
from torchtangents._src.utils.kernel import Kernel


def infinite_width_ntk_gradients():
    """
    Example 1: Compute gradients of infinite-width NTK w.r.t. inputs.
    
    This method uses the theoretical infinite-width limit implemented in stax.
    The gradients represent how the infinite-width kernel changes with input variations.
    """
    print("Example 1: Infinite-Width NTK Gradients")
    print("=" * 50)
    
    # Create inputs that require gradients
    torch.manual_seed(42)
    x1 = torch.randn(4, 3, requires_grad=True)  # First set of inputs
    x2 = torch.randn(3, 3, requires_grad=True)  # Second set of inputs
    
    print(f"Input shapes: x1={x1.shape}, x2={x2.shape}")
    
    # Build infinite-width network architecture
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1024, W_std=1.0, b_std=0.1),  # Large width for infinite limit
        stax.Relu(),
        stax.Dense(512, W_std=1.0, b_std=0.1),
        stax.Relu(),
        stax.Dense(1, W_std=1.0, b_std=0.1)      # Output layer
    )
    
    # Create base kernel from input correlations
    # This simulates the "input layer" kernel (normalized dot products)
    base_nngp = torch.matmul(x1, x2.T) / x1.shape[1]
    base_kernel = Kernel(
        nngp=base_nngp,
        ntk=base_nngp.clone(),  # Start with same values
        cov1=torch.matmul(x1, x1.T) / x1.shape[1],  # Self-correlation x1
        cov2=torch.matmul(x2, x2.T) / x2.shape[1]   # Self-correlation x2
    )
    
    print(f"Base kernel NTK shape: {base_kernel.ntk.shape}")
    
    # Apply infinite-width kernel transformation through the network
    infinite_kernel = kernel_fn(base_kernel)
    
    print(f"Infinite-width NTK shape: {infinite_kernel.ntk.shape}")
    print(f"NTK requires gradients: {infinite_kernel.ntk.requires_grad}")
    
    # Compute gradients of NTK w.r.t. inputs
    # We take a scalar function of the NTK (sum) to get gradients
    ntk_scalar = infinite_kernel.ntk.sum()
    
    grad_x1, grad_x2 = torch.autograd.grad(
        ntk_scalar, [x1, x2], 
        retain_graph=True, 
        create_graph=True  # Enable higher-order gradients if needed
    )
    
    print(f"✓ ∂NTK/∂x1 shape: {grad_x1.shape}")
    print(f"✓ ∂NTK/∂x2 shape: {grad_x2.shape}")
    print(f"✓ ∂NTK/∂x1 norm: {grad_x1.norm():.6f}")
    print(f"✓ ∂NTK/∂x2 norm: {grad_x2.norm():.6f}")
    
    # Example: Use gradients for input optimization
    print("\nExample: Input optimization using NTK gradients")
    
    # Optimize x1 to maximize NTK trace (diagonal sum)
    x1_opt = x1.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x1_opt], lr=0.01)
    
    for step in range(10):
        optimizer.zero_grad()
        
        # Recompute kernel with updated x1
        base_nngp_opt = torch.matmul(x1_opt, x2.T) / x1_opt.shape[1]
        base_kernel_opt = Kernel(
            nngp=base_nngp_opt,
            ntk=base_nngp_opt.clone(),
            cov1=torch.matmul(x1_opt, x1_opt.T) / x1_opt.shape[1],
            cov2=torch.matmul(x2, x2.T) / x2.shape[1]
        )
        
        infinite_kernel_opt = kernel_fn(base_kernel_opt)
        
        # Maximize trace (diagonal sum) of NTK
        objective = torch.trace(infinite_kernel_opt.ntk)
        loss = -objective  # Negative for maximization
        
        loss.backward()
        optimizer.step()
        
        if step % 3 == 0:
            print(f"  Step {step}: NTK trace = {objective.item():.6f}")
    
    print("✓ Successfully optimized inputs using infinite-width NTK gradients!\n")
    
    return {
        'method': 'infinite_width',
        'x1': x1, 'x2': x2,
        'ntk': infinite_kernel.ntk,
        'grad_x1': grad_x1, 'grad_x2': grad_x2
    }


def empirical_ntk_gradients():
    """
    Example 2: Compute gradients of empirical (finite-width) NTK w.r.t. inputs.
    
    This method uses a finite-width network and computes the empirical NTK,
    which approximates the infinite-width limit as width increases.
    """
    print("Example 2: Empirical NTK Gradients")
    print("=" * 50)
    
    # Create inputs that require gradients
    torch.manual_seed(42)
    x1 = torch.randn(4, 3, requires_grad=True)
    x2 = torch.randn(3, 3, requires_grad=True)
    
    print(f"Input shapes: x1={x1.shape}, x2={x2.shape}")
    
    # Build finite-width network (smaller for computational efficiency)
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(64, W_std=1.0, b_std=0.1),
        stax.Relu(),
        stax.Dense(32, W_std=1.0, b_std=0.1),
        stax.Relu(),
        stax.Dense(1, W_std=1.0, b_std=0.1)
    )
    
    # Initialize network parameters
    rng = torch.Generator()
    rng.manual_seed(123)
    output_shape, params = init_fn(rng, x1.shape)
    
    print(f"Network output shape: {output_shape}")
    print(f"Number of parameter groups: {len(params) if isinstance(params, list) else 1}")
    
    # Create empirical kernel function
    empirical_kernel_fn = tt.empirical_kernel_fn(apply_fn)
    
    # Compute empirical NTK
    kernel_result = empirical_kernel_fn(x1, x2, 'ntk', params)
    
    # Extract NTK tensor from Kernel object
    if isinstance(kernel_result, Kernel):
        ntk = kernel_result.ntk
    else:
        ntk = kernel_result
    
    print(f"Empirical NTK shape: {ntk.shape}")
    print(f"NTK requires gradients: {ntk.requires_grad}")
    
    # Compute gradients w.r.t. inputs
    ntk_scalar = ntk.sum()
    
    grad_x1, grad_x2 = torch.autograd.grad(
        ntk_scalar, [x1, x2],
        retain_graph=True,
        create_graph=True
    )
    
    print(f"✓ ∂NTK/∂x1 shape: {grad_x1.shape}")
    print(f"✓ ∂NTK/∂x2 shape: {grad_x2.shape}")
    print(f"✓ ∂NTK/∂x1 norm: {grad_x1.norm():.6f}")
    print(f"✓ ∂NTK/∂x2 norm: {grad_x2.norm():.6f}")
    
    # Example: Adversarial perturbation using gradients
    print("\nExample: Adversarial input perturbation")
    
    # Find perturbation that maximally changes NTK
    epsilon = 0.1  # Perturbation magnitude
    
    # Normalize gradients to get direction of steepest increase
    grad_direction = grad_x1 / grad_x1.norm()
    
    # Create adversarial input
    x1_adv = x1 + epsilon * grad_direction
    
    # Compute NTK for adversarial input
    kernel_result_adv = empirical_kernel_fn(x1_adv, x2, 'ntk', params)
    ntk_adv = kernel_result_adv.ntk if isinstance(kernel_result_adv, Kernel) else kernel_result_adv
    
    ntk_change = (ntk_adv - ntk).norm()
    print(f"  Original NTK norm: {ntk.norm():.6f}")
    print(f"  Adversarial NTK norm: {ntk_adv.norm():.6f}")
    print(f"  NTK change magnitude: {ntk_change:.6f}")
    print("✓ Successfully computed adversarial perturbation!\n")
    
    return {
        'method': 'empirical',
        'x1': x1, 'x2': x2,
        'ntk': ntk,
        'grad_x1': grad_x1, 'grad_x2': grad_x2,
        'params': params
    }


def manual_jacobian_gradients():
    """
    Example 3: Manual Jacobian computation for educational purposes.
    
    This shows how to manually compute NTK gradients using explicit Jacobian
    computation. This is mainly for understanding the underlying mathematics.
    """
    print("Example 3: Manual Jacobian NTK Gradients")
    print("=" * 50)
    
    # Simple network for demonstration
    torch.manual_seed(42)
    x = torch.randn(3, 4, requires_grad=True)
    W1 = torch.randn(4, 6, requires_grad=True)
    b1 = torch.randn(6, requires_grad=True)
    W2 = torch.randn(6, 1, requires_grad=True)
    b2 = torch.randn(1, requires_grad=True)
    
    def simple_network(x, W1, b1, W2, b2):
        """Simple 2-layer network."""
        h1 = torch.relu(x @ W1 + b1)
        return h1 @ W2 + b2
    
    print(f"Input shape: {x.shape}")
    print(f"Network: {x.shape[1]} → {W1.shape[1]} → {W2.shape[1]}")
    
    # Compute Jacobian matrix manually
    def compute_jacobian_row(xi):
        """Compute Jacobian row for single input xi."""
        output = simple_network(xi.unsqueeze(0), W1, b1, W2, b2)
        
        # Compute gradients w.r.t. all parameters
        grad_W1 = torch.autograd.grad(output.sum(), W1, retain_graph=True, create_graph=True)[0]
        grad_b1 = torch.autograd.grad(output.sum(), b1, retain_graph=True, create_graph=True)[0]
        grad_W2 = torch.autograd.grad(output.sum(), W2, retain_graph=True, create_graph=True)[0]
        grad_b2 = torch.autograd.grad(output.sum(), b2, retain_graph=True, create_graph=True)[0]
        
        # Flatten and concatenate all parameter gradients
        return torch.cat([
            grad_W1.flatten(),
            grad_b1.flatten(),
            grad_W2.flatten(),
            grad_b2.flatten()
        ])
    
    # Build Jacobian matrix: each row is ∇_θ f(x_i)
    jacobians = []
    for i in range(x.shape[0]):
        jac_row = compute_jacobian_row(x[i])
        jacobians.append(jac_row)
    
    J = torch.stack(jacobians)  # Shape: (batch_size, n_parameters)
    
    print(f"Jacobian matrix shape: {J.shape}")
    print(f"Number of parameters: {J.shape[1]}")
    
    # Compute NTK = J @ J.T
    ntk = J @ J.T
    
    print(f"Manual NTK shape: {ntk.shape}")
    print(f"NTK requires gradients: {ntk.requires_grad}")
    
    # Compute gradients w.r.t. input
    ntk_scalar = ntk.sum()
    grad_x = torch.autograd.grad(ntk_scalar, x, retain_graph=True)[0]
    
    print(f"✓ ∂NTK/∂x shape: {grad_x.shape}")
    print(f"✓ ∂NTK/∂x norm: {grad_x.norm():.6f}")
    
    # Show relationship between NTK and network Jacobian
    print("\nEducational: NTK structure analysis")
    print(f"  NTK diagonal (self-interactions): {torch.diag(ntk)}")
    print(f"  NTK off-diagonal (cross-interactions): {ntk[0, 1]:.6f}")
    print(f"  Jacobian norms per input: {J.norm(dim=1)}")
    print("✓ Manual computation demonstrates NTK = J @ J.T structure!\n")
    
    return {
        'method': 'manual_jacobian',
        'x': x,
        'ntk': ntk,
        'grad_x': grad_x,
        'jacobian': J
    }


def compare_methods():
    """
    Compare gradients computed by different methods to show consistency.
    """
    print("Comparison: Gradient Magnitude Analysis")
    print("=" * 50)
    
    # Use same random seed for fair comparison
    torch.manual_seed(42)
    
    # Get results from all methods
    results_inf = infinite_width_ntk_gradients()
    results_emp = empirical_ntk_gradients()
    results_man = manual_jacobian_gradients()
    
    # Compare gradient magnitudes
    print("Gradient Magnitude Comparison:")
    print(f"  Infinite-width ∂NTK/∂x1 norm: {results_inf['grad_x1'].norm():.6f}")
    print(f"  Empirical ∂NTK/∂x1 norm:      {results_emp['grad_x1'].norm():.6f}")
    print(f"  Manual ∂NTK/∂x norm:          {results_man['grad_x'].norm():.6f}")
    
    print("\nNTK Value Comparison:")
    print(f"  Infinite-width NTK norm: {results_inf['ntk'].norm():.6f}")
    print(f"  Empirical NTK norm:      {results_emp['ntk'].norm():.6f}")
    print(f"  Manual NTK norm:         {results_man['ntk'].norm():.6f}")
    
    print("\n✓ All methods successfully compute NTK gradients!")
    print("✓ Magnitudes vary due to different network architectures and widths")
    print("✓ The gradient computation methodology is consistent across all approaches")


def main():
    """
    Run all NTK gradient computation examples.
    """
    print("🧮 TORCHTANGENTS NTK GRADIENT COMPUTATION EXAMPLES")
    print("=" * 70)
    print("This example demonstrates three methods for computing NTK gradients:")
    print("1. Infinite-width NTK gradients (theoretical)")
    print("2. Empirical NTK gradients (finite-width)")
    print("3. Manual Jacobian computation (educational)")
    print("=" * 70)
    print()
    
    # Run all examples
    infinite_width_ntk_gradients()
    empirical_ntk_gradients()
    manual_jacobian_gradients()
    compare_methods()
    
    print("=" * 70)
    print("🎯 KEY TAKEAWAYS:")
    print("✅ TorchTangents supports NTK gradient computation w.r.t. inputs")
    print("✅ Both infinite-width and empirical methods work")
    print("✅ Gradients enable input optimization and sensitivity analysis")
    print("✅ Applications: adversarial analysis, input design, optimization")
    print("=" * 70)
    print("📚 For more examples, see: https://github.com/neural-tangents/neural-tangents")


if __name__ == "__main__":
    main()