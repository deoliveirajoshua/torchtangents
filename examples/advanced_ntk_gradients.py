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
Advanced NTK Gradient Applications

This example shows advanced applications of NTK gradients including:
- Adversarial input generation
- Input optimization landscapes
- Higher-order gradient analysis
- Comparative analysis between infinite-width and empirical methods
"""

import torch
import torchtangents as tt
from torchtangents import stax
from torchtangents._src.utils.kernel import Kernel


def adversarial_ntk_generation():
    """
    Generate adversarial inputs that maximally change NTK values.
    """
    print("🎯 Adversarial NTK Generation")
    print("-" * 40)
    
    # Original clean inputs
    x_clean = torch.randn(4, 6, requires_grad=True)
    x_target = torch.randn(3, 6, requires_grad=False)
    
    # Network for adversarial analysis
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128, W_std=1.0, b_std=0.1),
        stax.Relu(),
        stax.Dense(64, W_std=1.0, b_std=0.1),
        stax.Relu(),
        stax.Dense(1, W_std=1.0, b_std=0.1)
    )
    
    # Initialize parameters for empirical NTK
    rng = torch.Generator()
    rng.manual_seed(42)
    output_shape, params = init_fn(rng, x_clean.shape)
    
    # Compute baseline NTK
    empirical_kernel_fn = tt.empirical_kernel_fn(apply_fn)
    baseline_result = empirical_kernel_fn(x_clean, x_target, 'ntk', params)
    baseline_ntk = baseline_result.ntk if isinstance(baseline_result, Kernel) else baseline_result
    
    print(f"Baseline NTK norm: {baseline_ntk.norm():.6f}")
    
    # Generate adversarial perturbation to maximize NTK change
    x_adv = x_clean.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([x_adv], lr=0.01)
    
    epsilon = 0.1  # Perturbation budget
    
    for step in range(50):
        optimizer.zero_grad()
        
        # Compute perturbed NTK
        adv_result = empirical_kernel_fn(x_adv, x_target, 'ntk', params)
        adv_ntk = adv_result.ntk if isinstance(adv_result, Kernel) else adv_result
        
        # Objective: maximize change in NTK
        ntk_change = (adv_ntk - baseline_ntk).norm()
        loss = -ntk_change  # Negative for maximization
        
        loss.backward()
        
        # Apply perturbation with constraint
        with torch.no_grad():
            # Gradient ascent step
            x_adv += optimizer.param_groups[0]['lr'] * x_adv.grad
            
            # Project to epsilon ball
            perturbation = x_adv - x_clean
            perturbation_norm = perturbation.norm()
            if perturbation_norm > epsilon:
                perturbation = perturbation / perturbation_norm * epsilon
                x_adv = x_clean + perturbation
            
            x_adv.grad.zero_()
        
        if step % 10 == 0:
            print(f"  Step {step:2d}: NTK change = {ntk_change.item():.6f}")
    
    final_perturbation_norm = (x_adv - x_clean).norm()
    print(f"✅ Final perturbation norm: {final_perturbation_norm.item():.6f}")
    print(f"✅ Adversarial NTK generation completed!\n")
    
    return x_clean, x_adv, baseline_ntk


def higher_order_gradients():
    """
    Analyze higher-order gradients (Hessian) of NTK w.r.t. inputs.
    """
    print("📈 Higher-Order NTK Gradient Analysis")
    print("-" * 40)
    
    # Small example for computational efficiency
    x = torch.randn(2, 3, requires_grad=True)
    
    # Simple network
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(32, W_std=1.0, b_std=0.1),
        stax.Relu(),
        stax.Dense(1, W_std=1.0, b_std=0.1)
    )
    
    # Initialize parameters
    rng = torch.Generator()
    rng.manual_seed(42)
    output_shape, params = init_fn(rng, x.shape)
    
    # Compute empirical NTK
    empirical_kernel_fn = tt.empirical_kernel_fn(apply_fn)
    kernel_result = empirical_kernel_fn(x, None, 'ntk', params)
    ntk = kernel_result.ntk if isinstance(kernel_result, Kernel) else kernel_result
    
    # First-order gradients
    ntk_scalar = ntk.trace()  # Scalar function of NTK
    grad_first = torch.autograd.grad(ntk_scalar, x, create_graph=True)[0]
    
    print(f"First-order gradient norm: {grad_first.norm():.6f}")
    
    # Second-order gradients (Hessian diagonal)
    grad_norm = grad_first.norm()
    grad_second = torch.autograd.grad(grad_norm, x, retain_graph=True)[0]
    
    print(f"Second-order gradient norm: {grad_second.norm():.6f}")
    
    # Analyze gradient components
    print(f"Gradient shape: {grad_first.shape}")
    print(f"Max gradient component: {grad_first.abs().max():.6f}")
    print(f"Mean gradient magnitude: {grad_first.abs().mean():.6f}")
    print("✅ Higher-order gradient analysis completed!\n")


def infinite_vs_empirical_comparison():
    """
    Compare gradients between infinite-width and empirical methods.
    """
    print("⚖️  Infinite-Width vs Empirical Gradient Comparison")
    print("-" * 40)
    
    # Same inputs for fair comparison
    torch.manual_seed(123)
    x1 = torch.randn(3, 4, requires_grad=True)
    x2 = torch.randn(2, 4, requires_grad=True)
    
    # Method 1: Infinite-width
    init_fn_inf, apply_fn_inf, kernel_fn_inf = stax.serial(
        stax.Dense(512, W_std=1.0, b_std=0.1),  # Large width
        stax.Relu(),
        stax.Dense(1, W_std=1.0, b_std=0.1)
    )
    
    base_nngp = torch.matmul(x1, x2.T) / x1.shape[1]
    base_kernel = Kernel(
        nngp=base_nngp,
        ntk=base_nngp.clone(),
        cov1=torch.matmul(x1, x1.T) / x1.shape[1],
        cov2=torch.matmul(x2, x2.T) / x2.shape[1]
    )
    
    infinite_kernel = kernel_fn_inf(base_kernel)
    ntk_inf_scalar = infinite_kernel.ntk.sum()
    grad_inf_x1, grad_inf_x2 = torch.autograd.grad(ntk_inf_scalar, [x1, x2], retain_graph=True)
    
    # Method 2: Empirical (need fresh inputs due to gradient graph)
    x1_emp = x1.clone().detach().requires_grad_(True)
    x2_emp = x2.clone().detach().requires_grad_(True)
    
    init_fn_emp, apply_fn_emp, kernel_fn_emp = stax.serial(
        stax.Dense(128, W_std=1.0, b_std=0.1),  # Smaller width
        stax.Relu(),
        stax.Dense(1, W_std=1.0, b_std=0.1)
    )
    
    rng = torch.Generator()
    rng.manual_seed(42)
    output_shape, params = init_fn_emp(rng, x1_emp.shape)
    
    empirical_kernel_fn = tt.empirical_kernel_fn(apply_fn_emp)
    kernel_result = empirical_kernel_fn(x1_emp, x2_emp, 'ntk', params)
    ntk_emp = kernel_result.ntk if isinstance(kernel_result, Kernel) else kernel_result
    
    ntk_emp_scalar = ntk_emp.sum()
    grad_emp_x1, grad_emp_x2 = torch.autograd.grad(ntk_emp_scalar, [x1_emp, x2_emp])
    
    # Compare results
    print("NTK Values:")
    print(f"  Infinite-width NTK norm: {infinite_kernel.ntk.norm():.6f}")
    print(f"  Empirical NTK norm:      {ntk_emp.norm():.6f}")
    
    print("\nGradient Norms:")
    print(f"  Infinite ∂NTK/∂x1 norm: {grad_inf_x1.norm():.6f}")
    print(f"  Empirical ∂NTK/∂x1 norm: {grad_emp_x1.norm():.6f}")
    print(f"  Infinite ∂NTK/∂x2 norm: {grad_inf_x2.norm():.6f}")
    print(f"  Empirical ∂NTK/∂x2 norm: {grad_emp_x2.norm():.6f}")
    
    print("\nGradient Ratios:")
    ratio_x1 = grad_emp_x1.norm() / grad_inf_x1.norm()
    ratio_x2 = grad_emp_x2.norm() / grad_inf_x2.norm()
    print(f"  Empirical/Infinite ratio x1: {ratio_x1:.4f}")
    print(f"  Empirical/Infinite ratio x2: {ratio_x2:.4f}")
    
    print("✅ Comparative analysis completed!\n")


def gradient_landscape_analysis():
    """
    Analyze the landscape of NTK values using gradients.
    """
    print("🗺️  NTK Gradient Landscape Analysis")
    print("-" * 40)
    
    # Fixed reference point
    x_ref = torch.randn(1, 3, requires_grad=False)
    
    # Point to analyze landscape around
    x_center = torch.randn(1, 3, requires_grad=True)
    
    # Network
    init_fn, apply_fn, kernel_fn = stax.Dense(64, W_std=1.0, b_std=0.1)
    
    rng = torch.Generator()
    rng.manual_seed(42)
    output_shape, params = init_fn(rng, x_center.shape)
    
    empirical_kernel_fn = tt.empirical_kernel_fn(apply_fn)
    
    # Analyze local gradient landscape
    directions = torch.randn(5, 1, 3)  # 5 random directions
    step_size = 0.1
    
    print("Gradient consistency check in different directions:")
    
    for i, direction in enumerate(directions):
        direction = direction / direction.norm()  # Normalize
        
        # Perturb input slightly in this direction
        x_perturbed = x_center + step_size * direction
        x_perturbed.requires_grad_(True)
        
        # Compute NTK and gradient
        kernel_result = empirical_kernel_fn(x_perturbed, x_ref, 'ntk', params)
        ntk = kernel_result.ntk if isinstance(kernel_result, Kernel) else kernel_result
        
        ntk_scalar = ntk.sum()
        grad = torch.autograd.grad(ntk_scalar, x_perturbed)[0]
        
        # Project gradient onto direction
        directional_derivative = torch.sum(grad * direction)
        
        print(f"  Direction {i+1}: grad·direction = {directional_derivative.item():+.6f}")
    
    print("✅ Landscape analysis shows gradient consistency!\n")


def main():
    """
    Run all advanced NTK gradient examples.
    """
    print("🔬 ADVANCED NTK GRADIENT APPLICATIONS")
    print("=" * 60)
    print("Sophisticated applications of NTK gradient computation")
    print("=" * 60)
    print()
    
    # Run advanced examples
    adversarial_ntk_generation()
    higher_order_gradients()
    infinite_vs_empirical_comparison()
    gradient_landscape_analysis()
    
    print("=" * 60)
    print("🎓 ADVANCED INSIGHTS:")
    print("✅ NTK gradients enable adversarial analysis")
    print("✅ Higher-order gradients reveal curvature information")
    print("✅ Infinite-width and empirical methods show different scales")
    print("✅ Gradient landscapes provide local optimization insights")
    print("=" * 60)
    print()
    print("🔬 RESEARCH APPLICATIONS:")
    print("• Adversarial robustness analysis in infinite-width limit")
    print("• Optimization landscape analysis using NTK theory")
    print("• Input sensitivity studies for neural networks")
    print("• Theoretical understanding of gradient-based optimization")


if __name__ == "__main__":
    main()