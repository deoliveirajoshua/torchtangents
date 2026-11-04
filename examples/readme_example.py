"""Quickstart example matching README.md."""
import torch
import torch.nn as nn
import torchtangents as tt

x1 = torch.randn(2, 10, requires_grad=True)
x2 = torch.randn(3, 10)

# 1. Build a TorchTangents model (drop-in replacement for torch.nn)
model = tt.nn.Sequential(
    tt.nn.Linear(10, 128),
    tt.nn.ReLU(),
    tt.nn.Linear(128, 5),
)
out = model(x1)                                        # Standard forward pass: (2, 5)
k_inf = model.inf_kernel(x1, x2)                       # Infinite-width NNGP & NTK: (2, 3)
k_emp = model.empirical_ntk(x1, x2)                    # Finite-width empirical NTK: (2, 3, 5, 5)
grad = model.ntk_gradient(x1, x2, mode="all_pairs")    # Input Jacobian of NTK: (2, 3, 10)

# 2. Or lift an existing vanilla PyTorch model directly
vanilla_mlp = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 5))
lifted_mlp = tt.nn.lift(vanilla_mlp)
k_lifted = lifted_mlp.inf_kernel(x1, x2, get="ntk")    # Shape: (2, 3)

print(f"Forward: {out.shape} | Infinite NTK: {k_inf.ntk.shape} | Lifted NTK: {k_lifted.shape}")
