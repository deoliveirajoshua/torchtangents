import os
import sys
import torch
import pytest
import math

# ensure repo root is importable when running pytest from subfolders
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from torchtangents import (
    kernel_from_inputs,
    KernelSequential,
    DenseReLU,
    lift_sequential,
    gp_posterior,
    _relu_arccos,
    kernel_from_images,
    kernel_from_sequences,
    Conv2dReLU,
    GlobalAvgPool,
    Conv2dReLUExact,
    Residual,
    Kernel,
    LayerNormKernel,
    BatchNormKernel,
    PreActResBlockDense,
    PreActResBlockConv,
    lift_module,
    Dense,
    AttentionLinear,
    AttentionAnalytic,
    AttentionMultiHead,
    empirical_ntk,
    empirical_ntk_efficient,
)


def test_kernel_from_sequences_and_attention_fixed():
    # small batch of sequences
    n, L, d = 3, 5, 4
    X = torch.randn(n, L, d)
    Kseq = kernel_from_sequences(X)
    assert Kseq.nngp.shape == (n, n, L, L)

    # fixed attention: averaging over positions
    A = torch.full((L, L), 1.0 / float(L))
    attn = AttentionLinear(sigma_w2=1.0, sigma_b2=0.0, attn_mode='fixed', attn=A)
    Kout = attn.transform(Kseq)
    # manual mix for a single sample pair
    # manual Sigma_lin_{i,j,p,q} = sum_{u,v} A[p,u]*A[q,v]*Sigma_in[i,j,u,v]
    Sigma_manual = torch.einsum('pu,qv,ijuv->ijpq', A, A, Kseq.nngp)
    Sigma_manual = Sigma_manual * 1.0
    assert Kout.nngp.shape == Sigma_manual.shape
    assert torch.allclose(Kout.nngp, Sigma_manual, atol=1e-6)


def test_attention_softmax_shape_and_properties():
    n, L, d = 2, 6, 8
    X = torch.randn(n, L, d)
    Kseq = kernel_from_sequences(X)
    attn = AttentionLinear(sigma_w2=0.5, sigma_b2=0.1, attn_mode='softmax')
    Kout = attn.transform(Kseq)
    assert Kout.nngp.shape == (n, n, L, L)
    # check symmetry across sample pairs
    assert torch.allclose(Kout.nngp, Kout.nngp.permute(1, 0, 3, 2), atol=1e-6)


def test_attention_analytic_uniform_limit():
    # when sigma_wq2 or sigma_wk2 is zero, variance var=0 -> exp_m constant -> uniform attention
    n, L, d = 2, 4, 6
    X = torch.randn(n, L, d)
    Kseq = kernel_from_sequences(X)

    # set q variance to zero
    att = AttentionAnalytic(sigma_wq2=0.0, sigma_wk2=1.0, sigma_wv2=1.0)
    Kout = att.transform(Kseq)
    # attention matrix used internally would be uniform -> mixing via uniform averaging
    # resulting Sigma should equal mixing by uniform matrix A = 1/L
    A = torch.full((L, L), 1.0 / float(L), dtype=Kseq.nngp.dtype)
    Sigma_manual = torch.einsum('pu,qv,ijuv->ijpq', A, A, Kseq.nngp)
    Sigma_manual = Sigma_manual * att.sigma_wv2 + att.sigma_bv2
    assert Kout.nngp.shape == Sigma_manual.shape
    assert torch.allclose(Kout.nngp, Sigma_manual, atol=1e-6)


def test_attention_multihead_shape_and_average():
    n, L, d = 3, 5, 7
    X = torch.randn(n, L, d)
    Kseq = kernel_from_sequences(X)

    # multi-head with identical heads should equal single-head
    mh = AttentionMultiHead(num_heads=4, sigma_wq2=1.0, sigma_wk2=1.0, sigma_wv2=0.8, sigma_bv2=0.05)
    out_mh = mh.transform(Kseq)
    single = AttentionAnalytic(sigma_wq2=1.0, sigma_wk2=1.0, sigma_wv2=0.8, sigma_bv2=0.05)
    out_single = single.transform(Kseq)

    assert out_mh.nngp.shape == out_single.nngp.shape
    assert out_mh.ntk.shape == out_single.ntk.shape
    # since heads are identical, average should equal single (within numeric tolerance)
    assert torch.allclose(out_mh.nngp, out_single.nngp, atol=1e-6)
    assert torch.allclose(out_mh.ntk, out_single.ntk, atol=1e-6)


def test_lift_module_multihead():
    import torch.nn as nn

    # build a simple MultiheadAttention module
    embed_dim = 8
    num_heads = 2
    mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    # lift via lift_module
    lifted = lift_module(nn.Sequential(mha))

    # build sequence input and kernels
    n, L, d = 3, 5, embed_dim
    X = torch.randn(n, L, d)
    Kseq = kernel_from_sequences(X)

    Klift = lifted.apply(Kseq)

    # manual AttentionMultiHead with matching num_heads should produce same shape
    mh = AttentionMultiHead(num_heads=num_heads)
    Kmh = mh.transform(Kseq)

    assert Klift.nngp.shape == Kmh.nngp.shape
    assert torch.allclose(Klift.nngp, Kmh.nngp, atol=1e-6)


def test_input_kernel_shapes_and_symmetry():
    X = torch.randn(4, 6)
    K = kernel_from_inputs(X)
    assert K.nngp.shape == (4, 4)
    assert torch.allclose(K.nngp, K.nngp.t(), atol=1e-6)
    assert torch.allclose(K.ntk, torch.zeros_like(K.ntk), atol=1e-6)


def test_dense_relu_apply_and_shapes():
    X = torch.randn(5, 8)
    K0 = kernel_from_inputs(X)
    seq = KernelSequential([DenseReLU(1.0, 0.0), DenseReLU(1.0, 0.0)])
    Kf = seq.apply(K0)
    assert Kf.nngp.shape == (5, 5)
    assert Kf.ntk.shape == (5, 5)


def test_lift_sequential_adapter():
    import torch.nn as nn

    seq = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU())
    kseq = lift_sequential(seq)
    X = torch.randn(3, 8)
    K0 = kernel_from_inputs(X)
    Kf = kseq.apply(K0)
    assert Kf.nngp.shape == (3, 3)


def test_gp_posterior_shapes():
    X = torch.randn(6, 10)
    K0 = kernel_from_inputs(X)
    seq = KernelSequential([DenseReLU(1.0, 0.0)])
    Kf = seq.apply(K0)
    y = torch.randn(6, 1)
    mean = gp_posterior(Kf.nngp, y)
    assert mean.shape == (6, 1)


def test_relu_arccos_properties():
    # symmetry and bounds
    c = torch.linspace(-0.9, 0.9, steps=11)
    C, Cdot = _relu_arccos(c)
    assert C.shape == c.shape
    assert Cdot.shape == c.shape
    assert torch.all(C >= 0)
    assert torch.all(Cdot >= 0)


def test_kernel_from_images_and_conv_pool():
    # small image batch
    X = torch.randn(2, 3, 8, 8)
    Kimg = kernel_from_images(X)
    assert Kimg.nngp.shape == (2, 2, 8, 8)
    # apply a conv layer
    conv = Conv2dReLU(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), sigma_w2=1.0, sigma_b2=0.0)
    Kc = conv.transform(Kimg)
    # output should have same spatial dims due to padding
    assert Kc.nngp.shape[2:] == (8, 8)
    # pool to (n,n)
    pool = GlobalAvgPool()
    Kp = pool.transform(Kc)
    assert Kp.nngp.shape == (2, 2)


def test_conv2d_exact_matches_patch_average():
    # small batch and kernel for deterministic check
    X = torch.randn(2, 3, 6, 6)
    Kimg = kernel_from_images(X)

    conv = Conv2dReLUExact(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), sigma_w2=1.0, sigma_b2=0.0)
    Kc = conv.transform(Kimg)

    # build explicit single-sum patch-averaged pre-activation covariance from images
    n, c, H, W = X.shape
    kH, kW = 3, 3
    pad = 1
    Xpad = torch.nn.functional.pad(X, (pad, pad, pad, pad))

    # preallocate expected linear cov (n, n, H, W)
    expected_lin = torch.zeros((n, n, H, W), dtype=X.dtype)
    for i in range(H):
        for j in range(W):
            # extract patches for all images at this output location
            patches = Xpad[:, :, i:i + kH, j:j + kW].contiguous().view(n, c, -1)
            # compute per-position inner-products across channels (sum over channel*patch)
            # resulting shape (n, n)
            K_ij = torch.einsum('icp,jcp->ij', patches, patches) / float(c)
            expected_lin[:, :, i, j] = K_ij

    # expected linear after weight variance normalization (divide by kernel area)
    expected_lin = expected_lin / float(kH * kW)

    # Now apply ReLU arccos formula per-position (robust diag extraction)
    eps = 1e-6
    diag = torch.diagonal(expected_lin, dim1=0, dim2=1)
    if diag.shape[0] != n:
        axes = [i for i, s in enumerate(diag.shape) if s == n]
        if axes:
            diag = diag.permute((axes[0],) + tuple(i for i in range(len(diag.shape)) if i != axes[0]))

    diag_flat = diag.contiguous().view(n, -1).clamp(min=eps)
    s_flat = torch.sqrt(diag_flat[:, None, :] * diag_flat[None, :, :])
    s = s_flat.view(n, n, H, W)
    corr = expected_lin / (s + eps)
    C, _ = _relu_arccos(corr, eps=eps)
    expected_nngp = s * C

    assert torch.allclose(Kc.nngp, expected_nngp, atol=1e-6)


def test_conv2d_exact_stride2_no_padding():
    # Check stride=2, no padding behavior against explicit patch average
    X = torch.randn(3, 3, 7, 7)
    Kimg = kernel_from_images(X)

    conv = Conv2dReLUExact(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), sigma_w2=1.0, sigma_b2=0.0)
    Kc = conv.transform(Kimg)

    n, c, H, W = X.shape
    kH, kW = 3, 3
    stride_h, stride_w = 2, 2

    H_out = (H - kH) // stride_h + 1
    W_out = (W - kW) // stride_w + 1

    expected_lin = torch.zeros((n, n, H_out, W_out), dtype=X.dtype)
    for i_out in range(H_out):
        for j_out in range(W_out):
            i = i_out * stride_h
            j = j_out * stride_w
            patches = X[:, :, i:i + kH, j:j + kW].contiguous().view(n, c, -1)
            expected_lin[:, :, i_out, j_out] = torch.einsum('icp,jcp->ij', patches, patches) / float(c)

    expected_lin = expected_lin / float(kH * kW)

    # arccos per-position
    eps = 1e-6
    diag = torch.diagonal(expected_lin, dim1=0, dim2=1)
    if diag.shape[0] != n:
        axes = [i for i, s in enumerate(diag.shape) if s == n]
        if axes:
            diag = diag.permute((axes[0],) + tuple(i for i in range(len(diag.shape)) if i != axes[0]))

    diag_flat = diag.contiguous().view(n, -1).clamp(min=eps)
    s_flat = torch.sqrt(diag_flat[:, None, :] * diag_flat[None, :, :])
    s = s_flat.view(n, n, H_out, W_out)
    corr = expected_lin / (s + eps)
    C, _ = _relu_arccos(corr, eps=eps)
    expected_nngp = s * C

    # compare shapes and values
    assert Kc.nngp.shape == expected_nngp.shape
    assert torch.allclose(Kc.nngp, expected_nngp, atol=1e-6)


def test_residual_dense_branch():
    # Residual block with DenseReLU branch on vector inputs
    X = torch.randn(4, 10)
    K0 = kernel_from_inputs(X)
    branch = KernelSequential([DenseReLU(1.0, 0.0)])
    res = Residual(branch)
    Kout = res.transform(K0)

    # expected: input + branch(input) (branch evaluated with zero NTK for parameter contribution)
    branch_zero = Kernel(nngp=K0.nngp, ntk=torch.zeros_like(K0.nngp))
    # KernelSequential uses .apply
    bout = branch.apply(branch_zero)
    expected_nngp = K0.nngp + bout.nngp
    assert torch.allclose(Kout.nngp, expected_nngp)


def test_residual_conv_branch():
    # Residual block for images: use Conv2dReLUExact as branch and identity proj
    X = torch.randn(2, 3, 8, 8)
    Kimg = kernel_from_images(X)
    branch = Conv2dReLUExact(kernel_size=(3,3), stride=(1,1), padding=(1,1), sigma_w2=1.0, sigma_b2=0.0)
    res = Residual(branch)
    Kout = res.transform(Kimg)

    # branch evaluated with zero NTK
    branch_zero_in = Kernel(nngp=Kimg.nngp, ntk=torch.zeros_like(Kimg.nngp))
    bout = branch.transform(branch_zero_in)
    expected_nngp = Kimg.nngp + bout.nngp
    assert torch.allclose(Kout.nngp, expected_nngp, atol=1e-6)


def test_residual_composition_equivalence():
    # Applying a Residual block twice should equal a KernelSequential of two Residuals
    X = torch.randn(3, 6)
    K0 = kernel_from_inputs(X)
    branch = KernelSequential([DenseReLU(1.0, 0.0)])
    res = Residual(branch)

    K_after2 = res.transform(res.transform(K0))
    K_seq = KernelSequential([res, res]).apply(K0)
    assert torch.allclose(K_after2.nngp, K_seq.nngp, atol=1e-9)
    assert torch.allclose(K_after2.ntk, K_seq.ntk, atol=1e-9)


def _build_resnet_dense(in_dim, width, n_blocks=2, sigma_w2=1.0, sigma_b2=0.0):
    import torch.nn as nn

    layers = []
    # input projection
    layers.append(nn.Linear(in_dim, width, bias=True))
    # residual blocks: each block is Linear->ReLU with skip
    for _ in range(n_blocks):
        layers.append(nn.Linear(width, width, bias=True))
        layers.append(nn.ReLU())

    # final readout
    layers.append(nn.Linear(width, 1, bias=True))

    model = nn.Sequential(*layers)

    # initialize weights to match NTK parameterization: Normal(0, sigma_w2 / fan_in)
    for m in model:
        if isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            std = math.sqrt(sigma_w2 / float(fan_in))
            m.weight.data.normal_(0.0, std)
            if m.bias is not None:
                m.bias.data.normal_(0.0, math.sqrt(sigma_b2))
    return model


def test_finite_width_mc_ntk_matches_analytic():
    # Monte Carlo check: empirical NTK approximates analytic NTK for a small resnet
    torch.manual_seed(0)
    n = 3
    in_dim = 16
    # choose input dim equal to finite width so residual additions match dimensions
    width = in_dim
    X = torch.randn(n, in_dim)
    K0 = kernel_from_inputs(X)

    # analytic kernel via two residual blocks using DenseReLU branches
    branch = KernelSequential([DenseReLU(1.0, 0.0)])
    res = Residual(branch)
    seq = KernelSequential([res, res])
    K_analytic = seq.apply(K0)

    # Monte Carlo finite-width NTK: build finite networks that match analytic residual
    mc = 80
    ntk_sum = torch.zeros((n, n))
    import torch.nn as nn

    class SmallResNet(nn.Module):
        def __init__(self, width, n_blocks=2, sigma_w2=1.0, sigma_b2=0.0):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(width, width, bias=True) for _ in range(n_blocks)])
            self.relu = nn.ReLU()
            self.fc = nn.Linear(width, 1, bias=True)
            # initialize
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    fan_in = m.weight.shape[1]
                    std = math.sqrt(sigma_w2 / float(fan_in))
                    m.weight.data.normal_(0.0, std)
                    if m.bias is not None:
                        m.bias.data.normal_(0.0, math.sqrt(sigma_b2))

        def forward(self, x):
            h = x
            for layer in self.blocks:
                delta = self.relu(layer(h))
                h = h + delta
            return self.fc(h)

    for _ in range(mc):
        net = SmallResNet(width, n_blocks=2, sigma_w2=1.0, sigma_b2=0.0)
        net.zero_grad()
        Y = net(X).view(n)
        params = [p for p in net.parameters() if p.requires_grad]
        P = sum(p.numel() for p in params)
        J = torch.zeros((n, P))
        for i in range(n):
            net.zero_grad()
            Y[i].backward(retain_graph=True)
            grads = [p.grad.detach().view(-1).clone() for p in params]
            J[i] = torch.cat(grads)
        ntk_sum += J @ J.t()

    ntk_empirical = ntk_sum / float(mc)

    # compare analytic NTK to empirical via normalized correlation matrices
    A = K_analytic.ntk.detach()
    E = ntk_empirical.detach()
    # convert to correlation matrices
    diagA = torch.sqrt(torch.clamp(torch.diag(A), min=1e-12))
    diagE = torch.sqrt(torch.clamp(torch.diag(E), min=1e-12))
    corrA = A / (diagA[:, None] * diagA[None, :])
    corrE = E / (diagE[:, None] * diagE[None, :])
    diff = (corrA - corrE).abs().mean().item()
    # allow a fairly loose tolerance for small MC budget
    assert diff < 0.25


def test_norm_layers_and_preact_blocks():
    # LayerNorm and BatchNorm kernels on vector inputs
    X = torch.randn(4, 8)
    K0 = kernel_from_inputs(X)
    ln = LayerNormKernel()
    bn = BatchNormKernel()
    Kln = ln.transform(K0)
    Kbn = bn.transform(K0)
    # LayerNorm should normalize per-sample variances to 1 on diag
    diag_ln = torch.diag(Kln.nngp)
    assert torch.allclose(diag_ln, torch.ones_like(diag_ln), atol=1e-6)
    # BatchNorm should scale by mean variance, so mean diag becomes 1
    diag_bn = torch.diag(Kbn.nngp)
    assert torch.allclose(diag_bn.mean(), torch.tensor(1.0), atol=1e-6)

    # Pre-activation dense block shapes
    block = PreActResBlockDense(hidden_sigma_w2=1.0, hidden_sigma_b2=0.0, norm='layer')
    Kblock = block.transform(K0)
    assert Kblock.nngp.shape == K0.nngp.shape

    # Pre-activation conv block shapes
    Ximg = torch.randn(2, 3, 8, 8)
    Kimg = kernel_from_images(Ximg)
    cblock = PreActResBlockConv(kernel_size=(3,3), stride=(1,1), padding=(1,1), norm='batch')
    Kc = cblock.transform(Kimg)
    assert Kc.nngp.shape == Kimg.nngp.shape


def test_lift_module_sequential_conv_and_dense():
    import torch.nn as nn

    # conv sequential
    seq = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    Ximg = torch.randn(2, 3, 8, 8)
    Kimg = kernel_from_images(Ximg)
    lifted = lift_module(seq)
    Klift = lifted.apply(Kimg)
    # manual pipeline
    manual = KernelSequential([Conv2dReLUExact(kernel_size=(3,3), stride=(1,1), padding=(1,1)), GlobalAvgPool()])
    Kmanual = manual.apply(Kimg)
    assert Klift.nngp.shape == Kmanual.nngp.shape
    assert torch.allclose(Klift.nngp, Kmanual.nngp, atol=1e-6)

    # dense sequential
    seq2 = nn.Sequential(nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 1))
    X = torch.randn(4, 6)
    K0 = kernel_from_inputs(X)
    lifted2 = lift_module(seq2)
    Klift2 = lifted2.apply(K0)
    # manual: DenseReLU + Dense
    manual2 = KernelSequential([DenseReLU(1.0, 0.0), Dense(sigma_w2=1.0, sigma_b2=0.0)])
    Kmanual2 = manual2.apply(K0)
    assert Klift2.nngp.shape == Kmanual2.nngp.shape


def test_empirical_ntk_shape_and_symmetry():
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    X = torch.randn(5, 4)
    NTK = empirical_ntk(model, X)
    # shape
    assert NTK.shape == (5, 5)
    # symmetric
    assert torch.allclose(NTK, NTK.t(), atol=1e-6)
    # diagonal non-negative
    assert torch.all(NTK.diag() >= -1e-8)


def test_empirical_ntk_matches_manual_grad():
    import torch.nn as nn

    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    X = torch.randn(4, 4)

    # compute with library function
    NTK_lib = empirical_ntk(model, X)

    # manual computation: replicate procedure used in empirical_ntk
    model_zero = model
    model_zero.zero_grad()
    outputs = model_zero(X)
    outputs_flat = outputs.reshape(outputs.shape[0], -1)
    params = [p for p in model_zero.parameters() if p.requires_grad]

    grads = []
    for i in range(outputs_flat.shape[0]):
        pieces = []
        for k in range(outputs_flat.shape[1]):
            g = torch.autograd.grad(outputs_flat[i, k], params, retain_graph=True, create_graph=False, allow_unused=True)
            flat = []
            for t in g:
                if t is None:
                    flat.append(torch.zeros(0, device=X.device))
                else:
                    flat.append(t.detach().reshape(-1))
            pieces.append(torch.cat(flat) if flat else torch.tensor([], device=X.device))
        v = torch.cat(pieces) if pieces else torch.empty((0,), device=X.device)
        grads.append(v)

    G = torch.stack(grads, dim=0)
    NTK_manual = G @ G.t()

    assert torch.allclose(NTK_lib, NTK_manual, atol=1e-6)


def test_empirical_ntk_efficient_matches_original():
    import torch.nn as nn

    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3))
    X = torch.randn(3, 5)
    ntk1 = empirical_ntk(model, X)
    ntk2 = empirical_ntk_efficient(model, X)
    assert torch.allclose(ntk1, ntk2, atol=1e-6)


def test_empirical_ntk_efficient_param_chunking():
    import torch.nn as nn

    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 4))
    X = torch.randn(3, 6)
    # small chunk size (smaller than total P) to force multiple groups
    total_P = sum(p.numel() for p in model.parameters() if p.requires_grad)
    chunk = max(1, total_P // 3)
    ntk_no_chunk = empirical_ntk_efficient(model, X, param_chunk_size=None)
    ntk_chunk = empirical_ntk_efficient(model, X, param_chunk_size=chunk)
    assert torch.allclose(ntk_no_chunk, ntk_chunk, atol=1e-6)
