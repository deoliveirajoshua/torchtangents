"""1. Model Construction: Stateful (tt.nn) vs. Stateless (tt.stax)."""
import torch
import torchtangents as tt

x1 = torch.randn(4, 10)
x2 = torch.randn(2, 10)

# Option A: Stateful PyTorch-style module (finite forward + infinite kernel)
model = tt.nn.Sequential(
    tt.nn.Linear(10, 128),
    tt.nn.ReLU(),
    tt.nn.Linear(128, 5),
)
out = model(x1)                        # Finite-width forward pass: (4, 5)
k_nn = model.inf_kernel(x1, x2)        # Infinite-width kernels:    nngp (4, 2), ntk (4, 2)

# Option B: Stateless functional stax (infinite kernel only, no weights allocated)
kernel_fn = tt.stax.Sequential(
    tt.stax.Linear(128),
    tt.stax.ReLU(),
    tt.stax.Linear(5),
)
k_stax = kernel_fn(x1, x2)             # Identical infinite-width kernels: nngp (4, 2), ntk (4, 2)

print(f"Finite forward shape: {out.shape}")
print(f"Infinite NNGP / NTK:  {k_nn.nngp.shape}, {k_nn.ntk.shape}")
assert torch.allclose(k_nn.ntk, k_stax.ntk)
