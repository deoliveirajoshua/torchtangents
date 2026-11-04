"""6. Batched & Compiled Kernels: Scale to large datasets and accelerate with torch.compile."""
import torch
import torchtangents as tt

x_train = torch.randn(32, 16)
x_test = torch.randn(16, 16)

kernel_fn = tt.stax.Sequential(
    tt.stax.Linear(128),
    tt.stax.ReLU(),
    tt.stax.Linear(1),
)

# 1. Compute kernel in mini-batches of 8 to save GPU memory
batched_kernel_fn = tt.stax.batch(kernel_fn, batch_size=8)
k_full = kernel_fn(x_train, x_test, get=("nngp", "ntk"))
k_batched = batched_kernel_fn(x_train, x_test, get=("nngp", "ntk"))
assert torch.allclose(k_full.ntk, k_batched.ntk)
print(f"Batched NTK shape:  {k_batched.ntk.shape} (matches full kernel)")

# 2. Accelerate kernel evaluation with torch.compile (auto-enabled on Linux/macOS)
compiled_kernel_fn = tt.convert.optimize_model(kernel_fn)
k_compiled = compiled_kernel_fn(x_train, x_test, get=("nngp", "ntk"))
assert torch.allclose(k_full.ntk, k_compiled.ntk)
print(f"Compiled NTK shape: {k_compiled.ntk.shape} (matches eager execution)")
