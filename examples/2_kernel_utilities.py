"""2. Kernel Utilities: Analytical NTK, Empirical NTK, Gradients & Hessians."""
import torch
import torchtangents as tt

model = tt.nn.Sequential(
    tt.nn.Linear(10, 128),
    tt.nn.ReLU(),
    tt.nn.Linear(128, 5),
)

x1 = torch.randn(2, 10, requires_grad=True)
x2 = torch.randn(3, 10)

# 1. Closed-form infinite-width NNGP and NTK: shape (N1, N2)
k_inf = model.inf_kernel(x1, x2)
print(f"Infinite NNGP: {k_inf.nngp.shape}, NTK: {k_inf.ntk.shape}")

# 2. Finite-width empirical NTK from current weights: shape (N1, N2, Out, Out)
k_emp = model.empirical_ntk(x1, x2)
print(f"Empirical NTK: {k_emp.shape}")

# 3. Gradient & Hessian of the infinite-width NTK w.r.t. x1
ntk_grad = model.ntk_gradient(x1, x2, mode="all_pairs")  # (N1, N2, Features)
ntk_hess = model.ntk_hessian(x1, x2, mode="all_pairs")   # (N1, N2, Features, Features)
print(f"NTK Gradient:  {ntk_grad.shape}, NTK Hessian: {ntk_hess.shape}")
