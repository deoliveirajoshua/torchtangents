"""Small demo for torchtangents usage.

Run this script to see example kernel computations and GP posterior usage.
"""
import argparse
import time
import torch
import torchtangents as tt
from typing import Optional

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def demo_mlp():
    print("--- MLP kernel demo ---")
    torch.manual_seed(0)
    X = torch.randn(5, 8)
    K0 = tt.kernel_from_inputs(X)
    seq = tt.KernelSequential([tt.DenseReLU(1.0, 0.0), tt.DenseReLU(1.0, 0.0)])
    Kf = seq.apply(K0)
    print("Input kernel shape:", K0.nngp.shape)
    print("Final NNGP shape:", Kf.nngp.shape)
    print("Final NTK shape:", Kf.ntk.shape)
    print("Sample NNGP[0,1]:", float(Kf.nngp[0, 1]))
    return Kf


def demo_conv_and_lift():
    print("\n--- Conv + lift_module demo ---")
    import torch.nn as nn

    torch.manual_seed(1)
    Ximg = torch.randn(2, 3, 16, 16)
    Kimg = tt.kernel_from_images(Ximg)
    # manual pipeline
    conv = tt.Conv2dReLUExact(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    pool = tt.GlobalAvgPool()
    manual = tt.KernelSequential([conv, pool])
    Kmanual = manual.apply(Kimg)
    print("Manual conv+pool -> nngp shape:", Kmanual.nngp.shape)

    # lifted nn.Sequential
    seq = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
    lifted = tt.lift_module(seq)
    Klift = lifted.apply(Kimg)
    print("Lifted conv+pool -> nngp shape:", Klift.nngp.shape)
    close = torch.allclose(Klift.nngp, Kmanual.nngp, atol=1e-6)
    print("Lifted vs manual close:", close)
    return Kmanual, Klift


def demo_preact_resblock_and_gp():
    print("\n--- PreActResBlock & GP demo ---")
    torch.manual_seed(2)
    X = torch.randn(4, 10)
    K0 = tt.kernel_from_inputs(X)
    block = tt.PreActResBlockDense(hidden_sigma_w2=1.0, hidden_sigma_b2=0.0, norm='layer')
    Kblock = block.transform(K0)
    print("PreActResBlock output nngp shape:", Kblock.nngp.shape)

    # small GP posterior using the block kernel
    y = torch.randn(4, 1)
    mean = tt.gp_posterior(Kblock.nngp, y, noise=1e-3)
    print("GP posterior mean shape:", mean.shape)
    return Kblock, mean


def demo_resnet():
    print("\n--- ResNet demo (conv & dense) ---")
    torch.manual_seed(3)

    # Conv ResNet (pre-activation blocks)
    Ximg = torch.randn(3, 3, 32, 32)
    Kimg = tt.kernel_from_images(Ximg)
    stem = tt.Conv2dReLUExact(kernel_size=(3,3), stride=(1,1), padding=(1,1))
    blocks = [tt.PreActResBlockConv(kernel_size=(3,3), stride=(1,1), padding=(1,1), norm='layer') for _ in range(2)]
    seq_conv = tt.KernelSequential([stem] + blocks + [tt.GlobalAvgPool()])
    Kres_conv = seq_conv.apply(Kimg)
    print("Conv ResNet pooled nngp shape:", Kres_conv.nngp.shape)

    # GP posterior on conv pooled kernel
    y = torch.randn(Kres_conv.nngp.shape[0], 1)
    mean_conv = tt.gp_posterior(Kres_conv.nngp, y, noise=1e-3)
    print("Conv ResNet GP mean shape:", mean_conv.shape)

    # Dense ResNet (pre-activation blocks)
    X = torch.randn(4, 16)
    K0 = tt.kernel_from_inputs(X)
    blocks_dense = [tt.PreActResBlockDense(hidden_sigma_w2=1.0, hidden_sigma_b2=0.0, norm='batch') for _ in range(2)]
    seq_dense = tt.KernelSequential(blocks_dense)
    Kres_dense = seq_dense.apply(K0)
    print("Dense ResNet nngp shape:", Kres_dense.nngp.shape)

    # small GP posterior on dense resnet
    y2 = torch.randn(Kres_dense.nngp.shape[0], 1)
    mean_dense = tt.gp_posterior(Kres_dense.nngp, y2, noise=1e-3)
    print("Dense ResNet GP mean shape:", mean_dense.shape)
    return Kres_conv, mean_conv, Kres_dense, mean_dense


def demo_attention_sequence():
    print("\n--- Attention sequence demo ---")
    torch.manual_seed(4)
    n, L, d = 3, 6, 8
    X = torch.randn(n, L, d)
    Kseq = tt.kernel_from_sequences(X)
    print("Input seq kernel shape:", Kseq.nngp.shape)

    # single-head analytic attention
    single = tt.AttentionAnalytic(sigma_wq2=1.0, sigma_wk2=1.0, sigma_wv2=1.0)
    K_single = single.transform(Kseq)
    print("Single-head analytic nngp shape:", K_single.nngp.shape)

    # multi-head analytic attention
    mh = tt.AttentionMultiHead(num_heads=4, sigma_wq2=1.0, sigma_wk2=1.0, sigma_wv2=1.0)
    K_mh = mh.transform(Kseq)
    print("Multi-head analytic nngp shape:", K_mh.nngp.shape)

    # lifted nn.MultiheadAttention
    import torch.nn as nn
    mha = nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
    lifted = tt.lift_module(nn.Sequential(mha))
    K_lift = lifted.apply(Kseq)
    print("Lifted MultiheadAttention nngp shape:", K_lift.nngp.shape)

    # compare multi-head analytic to lifted mapping
    close = torch.allclose(K_mh.nngp, K_lift.nngp, atol=1e-6)
    print("Multi-head analytic vs lifted equal:", close)
    return Kseq, K_single, K_mh, K_lift


def maybe_plot_kernel(K: tt.Kernel, title: str = "Kernel", show_eig: bool = False):
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return
    Kmat = K.nngp.detach().cpu()
    plt.figure(figsize=(5, 4))
    plt.imshow(Kmat, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    if show_eig:
        try:
            eigs = torch.linalg.eigvalsh(Kmat)
            plt.figure(figsize=(5, 2))
            plt.plot(eigs.cpu().numpy())
            plt.title(f"Eigenvalues: {title}")
            plt.tight_layout()
        except Exception:
            pass


def _make_device(device_arg: Optional[str]):
    if device_arg is None or device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available')
        return torch.device('cuda')
    raise ValueError(f'Unknown device: {device_arg}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=['all', 'mlp', 'conv', 'preact', 'resnet', 'attn'], default='all')
    parser.add_argument('--plot', action='store_true', help='Show plots (requires matplotlib)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--img-size', type=int, default=32)
    parser.add_argument('--n-blocks', type=int, default=2)
    parser.add_argument('--width', type=int, default=16)
    args = parser.parse_args()

    dev = _make_device(args.device)
    print(f"Using device: {dev}")

    # run selected demos with timing
    if args.demo in ('all', 'mlp'):
        t0 = time.perf_counter()
        Kf = demo_mlp()
        dur = time.perf_counter() - t0
        print(f"MLP demo time: {dur:.4f}s")
        if args.plot:
            maybe_plot_kernel(Kf, title='MLP final NNGP', show_eig=True)

    if args.demo in ('all', 'conv'):
        t0 = time.perf_counter()
        Kmanual, Klift = demo_conv_and_lift()
        dur = time.perf_counter() - t0
        print(f"Conv demo time: {dur:.4f}s")
        if args.plot:
            maybe_plot_kernel(Kmanual, title='Conv manual NNGP')

    if args.demo in ('all', 'preact'):
        t0 = time.perf_counter()
        Kblock, mean = demo_preact_resblock_and_gp()
        dur = time.perf_counter() - t0
        print(f"PreAct block demo time: {dur:.4f}s")
        if args.plot:
            maybe_plot_kernel(Kblock, title='PreAct NNGP')

    if args.demo in ('all', 'resnet'):
        t0 = time.perf_counter()
        Kres_conv, mean_conv, Kres_dense, mean_dense = demo_resnet()
        dur = time.perf_counter() - t0
        print(f"ResNet demo time: {dur:.4f}s")
        if args.plot:
            maybe_plot_kernel(Kres_conv, title='ResNet conv pooled NNGP')
            maybe_plot_kernel(Kres_dense, title='ResNet dense NNGP')

    if args.demo in ('all', 'attn'):
        t0 = time.perf_counter()
        Kseq, K_single, K_mh, K_lift = demo_attention_sequence()
        dur = time.perf_counter() - t0
        print(f"Attention demo time: {dur:.4f}s")
        if args.plot:
            maybe_plot_kernel(K_single, title='Attention single NNGP')
            maybe_plot_kernel(K_mh, title='Attention multi-head NNGP')

    if args.plot and HAS_MPL:
        plt.show()
