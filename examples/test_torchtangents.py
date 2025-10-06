#!/usr/bin/env python3
import math
from turtledemo.penrose import start

import torch
from torch import nn, Tensor

import inspect, torch
from torch.utils import _pytree as pytree
print(torch.__version__)
from torchtangents import empirical_kernel_fn, empirical_ntk_fn, empirical_nngp_fn, empirical_ntk_vp_fn, NtkImplementation

test_compiled = False


# ---- tiny functional MLP -----------------------------------------------------
def init_mlp(key: int, in_dim: int, hidden: int, out_dim: int, device):
    g = torch.Generator(device=device).manual_seed(key)
    def kaiming(shape):
        fan_in = shape[1]
        std = math.sqrt(2.0 / fan_in)
        return torch.randn(*shape, generator=g, device=device) * std

    W1 = kaiming((hidden, in_dim))
    b1 = torch.zeros(hidden, device=device)
    W2 = kaiming((hidden, hidden))
    b2 = torch.zeros(hidden, device=device)
    W3 = kaiming((out_dim, hidden))
    b3 = torch.zeros(out_dim, device=device)
    return (W1, b1, W2, b2, W3, b3)  # tuple, not dict

def f(params, x: Tensor, *, training: bool = False) -> Tensor:
    # --- unwrap singleton-tuple produced by functorch vjp ---
    if isinstance(params, (tuple, list)) and len(params) == 1 and isinstance(params[0], (tuple, list, dict)):
        params = params[0]

    # --- unpack your parameter pytree ---
    if isinstance(params, dict):
        W1, b1 = params["W1"], params["b1"]
        W2, b2 = params["W2"], params["b2"]
        W3, b3 = params["W3"], params["b3"]
    else:
        # assume (W1, b1, W2, b2, W3, b3)
        W1, b1, W2, b2, W3, b3 = params

    h1 = torch.relu(x @ W1.T + b1)
    h2 = torch.relu(h1 @ W2.T + b2)
    y  = h2 @ W3.T + b3
    return y



# ---- main --------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        _ = torch.empty(0, device='cuda')
    torch.manual_seed(0)

    N_train, N_test, D, H, C = 40, 20, 32, 64, 10
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

    # -- compile kernels without compilation (for comparison) ---
    uncompiled_nngp = lambda a,b: nngp_fn(a,b,params, training=False)
    uncompiled_ntk  = lambda a,b: ntk_fn(a,b,params, training=False)
    print("Warming up uncompiled kernels...")
    if device != "cpu": start = torch.cuda.Event(enable_timing=True)
    if device != "cpu": end = torch.cuda.Event(enable_timing=True)
    if device != "cpu": start.record()
    K_nngp_tt = uncompiled_nngp(x_test,  None)
    if device != "cpu": end.record()
    if device != "cpu": torch.cuda.synchronize()
    if device != "cpu": print(f"NNGP kernel computed in {start.elapsed_time(end):.1f} ms")
    if device != "cpu": start.record()
    K_ntk_tt = uncompiled_ntk(x_test,  None)
    if device != "cpu": end.record()
    if device != "cpu": torch.cuda.synchronize()
    if device != "cpu": print(f"NTK  kernel computed in {start.elapsed_time(end):.1f} ms")
    if device != "cpu": start.record()
    K_ntk_tr = uncompiled_ntk(x_test,  x_train)
    if device != "cpu": end.record()
    if device != "cpu": torch.cuda.synchronize()
    if device != "cpu": print(f"NTK  kernel computed in {start.elapsed_time(end):.1f} ms")
    # Optional: compile the kernel fns. (PyTorch 2.1+)
    compiled_nngp = torch.compile(lambda a,b: nngp_fn(a,b,params, training=False))
    compiled_ntk  = torch.compile(lambda a,b: ntk_fn(a,b,params, training=False))

    # --- compute kernels ---
    if test_compiled:
        start.record()
        K_nngp_tt = compiled_nngp(x_test,  None)     # test-test NNGP (vs itself)
        end.record()
        torch.cuda.synchronize()
        print(f"NNGP kernel computed in {start.elapsed_time(end):.1f} ms")
        start.record()
        K_ntk_tt  = compiled_ntk (x_test,  None)     # test-test NTK
        end.record()
        torch.cuda.synchronize()
        print(f"NTK  kernel computed in {start.elapsed_time(end):.1f} ms")
        start.record()
        K_ntk_tr  = compiled_ntk (x_test,  x_train)  # test-train NTK
        end.record()
        torch.cuda.synchronize()
        print(f"NTK  kernel computed in {start.elapsed_time(end):.1f} ms")

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


if __name__ == "__main__":
    main()
