#!/usr/bin/env python3
import math
import torch
from torch import nn, Tensor

# assume you exposed your port as "torchtangents"
from torchtangents.empirical import (
    empirical_kernel_fn,
    empirical_ntk_fn,
    empirical_nngp_fn,
    empirical_ntk_vp_fn,
    NtkImplementation,
)

# ---- tiny functional MLP -----------------------------------------------------
def init_mlp(key: int, in_dim: int, hidden: int, out_dim: int, device):
    g = torch.Generator(device=device).manual_seed(key)
    def kaiming(shape):
        fan_in = shape[1]
        std = math.sqrt(2.0 / fan_in)
        return torch.randn(*shape, generator=g, device=device) * std

    params = {
        "w1": kaiming((hidden, in_dim)),
        "b1": torch.zeros(hidden, device=device),
        "w2": kaiming((hidden, hidden)),
        "b2": torch.zeros(hidden, device=device),
        "w3": kaiming((out_dim, hidden)),
        "b3": torch.zeros(out_dim, device=device),
    }
    return params

def f(params, x: Tensor, *, training: bool = False) -> Tensor:
    # x: [N, D]
    h1 = torch.relu(x @ params["w1"].T + params["b1"])
    h2 = torch.relu(h1 @ params["w2"].T + params["b2"])
    y  = h2 @ params["w3"].T + params["b3"]
    return y  # [N, C]

# ---- main --------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    N_train, N_test, D, H, C = 16, 8, 32, 64, 10
    x_train = torch.randn(N_train, D, device=device)
    x_test  = torch.randn(N_test,  D, device=device)
    params  = init_mlp(1234, D, H, C, device)

    # trace over logits (-1) so kernels reduce to [N1, N2]
    trace_axes = (-1,)
    diagonal_axes = ()

    # --- build kernel functions (Jacobian-contraction backend) ---
    nngp_fn = empirical_nngp_fn(f, trace_axes=trace_axes, diagonal_axes=diagonal_axes)
    ntk_fn  = empirical_ntk_fn(
        f, trace_axes=trace_axes, diagonal_axes=diagonal_axes,
        vmap_axes=0, implementation=NtkImplementation.JACOBIAN_CONTRACTION
    )

    # Optional: compile the kernel fns. (PyTorch 2.1+)
    compiled_nngp = torch.compile(lambda a,b: nngp_fn(a,b,params, training=False))
    compiled_ntk  = torch.compile(lambda a,b: ntk_fn(a,b,params, training=False))

    # --- compute kernels ---
    K_nngp_tt = compiled_nngp(x_test,  None)     # test-test NNGP (vs itself)
    K_ntk_tt  = compiled_ntk (x_test,  None)     # test-test NTK
    K_ntk_tr  = compiled_ntk (x_test,  x_train)  # test-train NTK

    print("NNGP(test,test) shape:", tuple(K_nngp_tt.shape))
    print("NTK (test,test) shape:", tuple(K_ntk_tt.shape))
    print("NTK (test,train) shape:", tuple(K_ntk_tr.shape))

    # --- shape checks ---
    assert K_nngp_tt.shape == (N_test, N_test)
    assert K_ntk_tt.shape  == (N_test, N_test)
    assert K_ntk_tr.shape  == (N_test, N_train)

    # --- basic symmetry (test-test) ---
    sym_err = (K_ntk_tt - K_ntk_tt.T).abs().max().item()
    print(f"symmetry |K - K^T|_max = {sym_err:.3e}")
    assert sym_err < 1e-4, "NTK(test,test) should be ~symmetric."

    # --- NTK-vector product check ---
    # Build vp fn for (x1 = test, x2 = test) so NTK is square
    ntk_vp = empirical_ntk_vp_fn(f, x_test, None, params, training=False)
    v = torch.randn_like(f(params, x_test))  # cotangents in output space
    Kv_vp = ntk_vp(v)
    Kv_mat = K_ntk_tt @ v  # since we traced over logits, kernel is [N,N]
    vp_err = (Kv_vp - Kv_mat).abs().max().item()
    print(f"NTK-vp agreement |J J^T v - K v|_max = {vp_err:.3e}")
    assert vp_err < 5e-3, "Vector-product path should match dense kernel action."

    print("✓ torchtangents basic test passed.")

if __name__ == "__main__":
    main()
