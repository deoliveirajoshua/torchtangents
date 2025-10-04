# linear.py (PyTorch port)
from __future__ import annotations

import enum
import functools
import operator as op
from typing import Callable, Optional, Sequence, Union, Iterable, Tuple

import torch
import torch.nn.functional as F

from ..utils.kernel import Kernel
from ..utils import utils  # your converted utils
from ..utils.typing import Axes, InternalLayer, InternalLayerMasked, PyTree
from .requirements import (
    layer, requires, supports_masking,
    Diagonal, Bool, mean_and_var, get_diagonal_outer_prods
)

# ---------------------------------------------------------------------------
# Enums (mirroring originals where relevant)
# ---------------------------------------------------------------------------

class Padding(enum.Enum):
  CIRCULAR = "CIRCULAR"
  SAME = "SAME"
  VALID = "VALID"

class _Pooling(enum.Enum):
  AVG = "AVG"
  SUM = "SUM"


# ---------------------------------------------------------------------------
# Simple / universal helpers
# ---------------------------------------------------------------------------

def _torch_rng_split(gen: torch.Generator, n=2) -> list[torch.Generator]:
    """Split a CPU generator deterministically into n child generators."""
    gens = []
    g = gen
    for _ in range(n):
        seed = int(torch.randint(0, 2**63 - 1, (1,), generator=g).item())
        child = torch.Generator(device="cpu")
        child.manual_seed(seed)
        gens.append(child)
    return gens

def _moveaxis(x: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    ndim = x.ndim
    src = src % ndim
    dst = dst % ndim
    perm = list(range(ndim))
    val = perm.pop(src)
    perm.insert(dst, val)
    return x.permute(perm)

def _canonicalize_axis(axis: Axes, x: Union[torch.Tensor, Tuple[int, ...]]) -> list[int]:
    if isinstance(axis, (tuple, list)):
        return [a % (x.ndim if isinstance(x, torch.Tensor) else len(x)) for a in axis]
    return [(axis % (x.ndim if isinstance(x, torch.Tensor) else len(x)))]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@layer
@supports_masking(remask_kernel=False)
def Identity() -> InternalLayer:
    def init_fn(rng: torch.Generator, input_shape: Tuple[int, ...]):
        return input_shape, ()
    def apply_fn(params, inputs, **kwargs):
        return inputs
    def kernel_fn(k: Kernel, **kwargs):
        return k
    return init_fn, apply_fn, kernel_fn


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------

@layer
@supports_masking(remask_kernel=True)
def Dense(
    out_dim: int,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    batch_axis: int = 0,
    channel_axis: int = -1,
    parameterization: str = "ntk",
    s: tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
    """
    PyTorch port of stax.Dense (with kernel_fn).
    See original docstring for parameterization details.
    """
    parameterization = parameterization.lower()

    def _init(rng: torch.Generator, input_shape: Tuple[int, ...], outd: int):
        ch_axis = channel_axis % len(input_shape)
        out_shape = input_shape[:ch_axis] + (outd,) + input_shape[ch_axis+1:]
        g_w, g_b = _torch_rng_split(rng, 2)
        in_dim = input_shape[ch_axis]
        W = torch.randn(in_dim, outd, generator=g_w)  # fan_in x fan_out
        if b_std is None:
            b = None
        else:
            b_shape = [1] * len(input_shape)
            b_shape[ch_axis] = outd
            b = torch.randn(b_shape, generator=g_b)
        return out_shape, (W, b)

    def ntk_init(rng, input_shape):
        return _init(rng, input_shape, out_dim)

    def standard_init(rng, input_shape):
        out_shape, (W, b) = _init(rng, input_shape, out_dim * s[1])
        # scale weights/bias to match standard param
        W = W * (W_std / ((input_shape[channel_axis] / s[0]) ** 0.5))
        if b is not None:
            b = b * b_std
        return out_shape, (W, b)

    if parameterization == "ntk":
        init_fn = ntk_init
    elif parameterization == "standard":
        init_fn = standard_init
    else:
        raise ValueError(f"Unsupported parameterization: {parameterization}")

    def apply_fn(params, inputs: torch.Tensor, **kwargs):
        W, b = params  # W: in_dim x out_dim
        x = inputs
        # tensordot over channel_axis with W
        x = _moveaxis(x, channel_axis, -1)  # move features last
        prod = torch.matmul(x, W)  # [..., in] @ [in, out] -> [..., out]
        if parameterization == "ntk":
            norm = W_std / (inputs.shape[channel_axis] ** 0.5)
            y = norm * prod
            if b is not None:
                y = y + b_std * b
        else:  # "standard"
            y = prod / (s[0] ** 0.5)
            if b is not None:
                y = y + b
        y = _moveaxis(y, -1, channel_axis)
        return y

    def _affine(mat: Optional[torch.Tensor], W_std_: float, b_std_: Optional[float]):
        if mat is None:
            return None
        out = mat * (W_std_ ** 2)
        if b_std_ is not None:
            out = out + (b_std_ ** 2)
        return out

    @requires(batch_axis=batch_axis,
              channel_axis=channel_axis,
              diagonal_spatial=Diagonal())
    def kernel_fn(k: Kernel, **kwargs):
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

        def fc(x):  # affine cov transform
            return _affine(x, W_std, b_std)

        if parameterization == "ntk":
            cov1 = fc(cov1)
            nngp = fc(nngp)
            cov2 = fc(cov2)
            if ntk is not None:
                ntk = (nngp if nngp is not None else 0) + (W_std ** 2) * ntk
        else:  # "standard"
            input_width = k.shape1[channel_axis] / s[0]
            if ntk is not None:
                # ntk depends on nngp and prev ntk
                ntk = input_width * nngp + (W_std ** 2) * ntk
                if b_std is not None:
                    ntk = ntk + 1.0
            cov1 = fc(cov1)
            nngp = fc(nngp)
            cov2 = fc(cov2)

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk,
                         is_gaussian=True, is_input=False)

    def mask_fn(mask: torch.Tensor, input_shape: Tuple[int, ...]):
        # True if any channel masked -> collapse channel dim
        return torch.all(mask, dim=channel_axis, keepdim=True)

    return init_fn, apply_fn, kernel_fn, mask_fn


# ---------------------------------------------------------------------------
# Convolution (shared weights; 1D/2D/3D; NCHW)
# ---------------------------------------------------------------------------

@layer
@supports_masking(remask_kernel=True)
def Conv(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    dimension_numbers: Optional[tuple[str, str, str]] = None,  # ignored in torch port; we assume NCHW/OI...
    parameterization: str = "ntk",
    s: tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
    """
    PyTorch port of stax.Conv for common cases:
    - inputs NCHW / NCL / NCDHW (channel first)
    - shared weights
    - padding: VALID or SAME or CIRCULAR (circular implemented via F.pad)
    """
    parameterization = parameterization.lower()
    strides = tuple(strides or (1,) * len(filter_shape))
    pad_enum = Padding(padding)
    nsp = len(filter_shape)

    if nsp not in (1, 2, 3):
        raise NotImplementedError("Conv only supports 1D/2D/3D in this port.")

    def _same_pad(sz, k, s):
        # PyTorch SAME for stride s: compute symmetric pad
        if s == 1:
            total = max(k - 1, 0)
        else:
            # generic; match TF 'SAME'
            out = (sz + s - 1) // s
            total = max((out - 1) * s + k - sz, 0)
        return total // 2, total - total // 2

    def _init(rng: torch.Generator, input_shape: Tuple[int, ...]):
        # Expect channel-first
        if len(input_shape) != nsp + 2:
            raise ValueError(f"Expected {nsp+2}D input (N,C,spatial...), got {input_shape}.")
        N, Cin = input_shape[0], input_shape[1]
        Cout = out_chan if parameterization == "ntk" else out_chan * s[1]
        g_w, g_b = _torch_rng_split(rng, 2)
        W = torch.randn(Cout, Cin, *filter_shape, generator=g_w)
        b = None if b_std is None else torch.randn(1, Cout, *([1]*nsp), generator=g_b)
        # infer output shape under VALID/SAME
        spatial = input_shape[2:]
        if pad_enum == Padding.VALID:
            out_sp = [ (sp - k)//strd + 1 for sp, k, strd in zip(spatial, filter_shape, strides) ]
        elif pad_enum == Padding.SAME:
            out_sp = [ (sp + strd - 1)//strd for sp, strd in zip(spatial, strides) ]
        elif pad_enum == Padding.CIRCULAR:
            # circular keeps length under VALID kernel but we pad circularly to emulate SAME result
            out_sp = [ (sp + strd - 1)//strd for sp, strd in zip(spatial, strides) ]
        else:
            raise ValueError(padding)
        out_shape = (N, Cout) + tuple(out_sp)
        return out_shape, (W, b)

    def _apply(params, x: torch.Tensor):
        W, b = params
        if parameterization == "ntk":
            norm = W_std / (W.shape[1] * int(torch.prod(torch.tensor(filter_shape))).item())**0.5
            b_rescale = b_std
        else:
            norm = 1.0 / (s[0] ** 0.5)
            b_rescale = 1.0

        # padding
        if pad_enum == Padding.VALID:
            x_pad = x
            padding_tuple = (0,) * (2 * nsp)
        elif pad_enum == Padding.SAME:
            pads = []
            for sp, k, strd in reversed(list(zip(x.shape[2:], filter_shape, strides))):
                l, r = _same_pad(sp, k, strd)
                pads.extend([l, r])
            padding_tuple = tuple(pads)
            x_pad = F.pad(x, padding_tuple)
        elif pad_enum == Padding.CIRCULAR:
            pads = []
            for sp, k, strd in reversed(list(zip(x.shape[2:], filter_shape, strides))):
                l, r = _same_pad(sp, k, strd)
                pads.extend([l, r])
            x_pad = F.pad(x, tuple(pads), mode="circular")
            padding_tuple = (0,) * (2 * nsp)
        else:
            raise ValueError(padding)

        if nsp == 1:
            y = F.conv1d(x_pad, W, bias=None, stride=strides, padding=0)
        elif nsp == 2:
            y = F.conv2d(x_pad, W, bias=None, stride=strides, padding=0)
        else:
            y = F.conv3d(x_pad, W, bias=None, stride=strides, padding=0)

        y = norm * y
        if b is not None:
            y = y + b_rescale * b
        return y

    @requires(batch_axis=0, channel_axis=1, diagonal_spatial=Diagonal(output=Bool.NO))
    def kernel_fn(k: Kernel, **kwargs):
        """
        Shared-weight conv kernel propagation.
        For now we implement the common diagonal_spatial and full_spatial paths
        by reusing your utils: conv in kernel space maps to windowed averages.
        """
        # We’ll implement a simplified diagonal-spatial path (fast & widely used).
        # Full spatial covariance path is large; if you need that next, say the word.
        def _conv_diag(lhs: Optional[torch.Tensor], batch_ndim: int):
            if lhs is None or lhs.ndim == 0:
                return lhs
            # Average over sliding windows (reduce_window analogue):
            # Compute sum over filter window via pooling; then divide by filter size.
            pool = _pool_kernel(lhs, _Pooling.AVG, filter_shape, strides, pad_enum, normalize_edges=False, batch_ndim=batch_ndim)
            return pool

        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

        if k.diagonal_spatial:
            cov1 = _conv_diag(cov1, 1 if k.diagonal_batch else 2)
            cov2 = _conv_diag(cov2, 1 if k.diagonal_batch else 2)
            nngp_u = _conv_diag(nngp, 2)
            if parameterization == "ntk":
                nngp = nngp_u
                if ntk is not None:
                    ntk = (W_std ** 2) * _conv_diag(ntk, 2) + nngp
            else:
                # standard
                if ntk is not None:
                    fan_in = k.shape1[1] * int(torch.prod(torch.tensor(filter_shape))).item()
                    ntk = (fan_in / s[0]) * nngp_u + (W_std ** 2) * _conv_diag(ntk, 2)
                    if b_std is not None:
                        ntk = ntk + 1.0
                nngp = (W_std ** 2) * nngp_u
                if b_std is not None:
                    nngp = nngp + (b_std ** 2)
        else:
            raise NotImplementedError("Full-spatial conv kernel path not yet ported in PyTorch version.")

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk,
                         is_gaussian=True, is_input=False, is_reversed=False,
                         batch_axis=0, channel_axis=1)

    def mask_fn(mask: torch.Tensor, input_shape: Tuple[int, ...]):
        # collapse channel dim of mask
        mask_c = torch.all(mask, dim=1, keepdim=True)
        # pooling the mask with same params (logical OR over window)
        return _pool_mask(mask_c, filter_shape, strides, pad_enum, batch_axis=0, channel_axis=1)

    return _init, _apply, kernel_fn, mask_fn


# ---------------------------------------------------------------------------
# Pooling (Sum / Avg)
# ---------------------------------------------------------------------------

@layer
@supports_masking(remask_kernel=True)
def AvgPool(
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    normalize_edges: bool = False,
    batch_axis: int = 0,
    channel_axis: int = 1,
) -> InternalLayerMasked:
    return _Pool(_Pooling.AVG, window_shape, strides, padding, normalize_edges, batch_axis, channel_axis)

@layer
@supports_masking(remask_kernel=True)
def SumPool(
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    batch_axis: int = 0,
    channel_axis: int = 1,
) -> InternalLayerMasked:
    return _Pool(_Pooling.SUM, window_shape, strides, padding, False, batch_axis, channel_axis)

def _Pool(
    pool_type: _Pooling,
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]],
    padding: str,
    normalize_edges: bool,
    batch_axis: int,
    channel_axis: int,
) -> InternalLayerMasked:
    window_shape = tuple(window_shape)
    strides = tuple(strides or (1,) * len(window_shape))
    pad_enum = Padding(padding)
    nsp = len(window_shape)

    def init_fn(rng, input_shape):
        # simple shape inference using torch.nn.functional pooling
        N, C, *S = input_shape
        if pad_enum == Padding.SAME:
            out_sp = [ (s + st - 1)//st for s, st in zip(S, strides) ]
        elif pad_enum in (Padding.VALID, Padding.CIRCULAR):
            # VALID semantics (CIRCULAR handled in apply by pre-pad)
            out_sp = [ (s - w)//st + 1 for s, w, st in zip(S, window_shape, strides) ]
        else:
            raise ValueError(padding)
        return (N, C, *out_sp), ()

    def apply_fn(params, x: torch.Tensor, **kwargs):
        pad_before = 0
        if pad_enum == Padding.CIRCULAR:
            # emulate SAME with circular pad for pooling
            pads = []
            for s, w, st in reversed(list(zip(x.shape[2:], window_shape, strides))):
                # SAME-style around
                total = max(( (s + st - 1)//st - 1) * st + w - s, 0)
                l = total // 2
                r = total - l
                pads.extend([l, r])
            x = F.pad(x, tuple(pads), mode="circular")
            pad_mode = 0
        elif pad_enum == Padding.SAME:
            pads = []
            for s, w, st in reversed(list(zip(x.shape[2:], window_shape, strides))):
                total = max(( (s + st - 1)//st - 1) * st + w - s, 0)
                l = total // 2
                r = total - l
                pads.extend([l, r])
            x = F.pad(x, tuple(pads))
        # apply pool
        if nsp == 1:
            out = F.avg_pool1d(x, window_shape, strides) if pool_type == _Pooling.AVG else F.max_pool1d(x, window_shape, strides) * 0 + F.avg_pool1d(x, window_shape, strides) * window_shape[0]
        elif nsp == 2:
            if pool_type == _Pooling.AVG:
                out = F.avg_pool2d(x, window_shape, strides)
            else:
                # sum pooling = avg * window_area
                out = F.avg_pool2d(x, window_shape, strides) * (window_shape[0]*window_shape[1])
        else:
            if pool_type == _Pooling.AVG:
                out = F.avg_pool3d(x, window_shape, strides)
            else:
                vol = window_shape[0]*window_shape[1]*window_shape[2]
                out = F.avg_pool3d(x, window_shape, strides) * vol
        return out

    @requires(batch_axis=batch_axis, channel_axis=channel_axis, diagonal_spatial=Diagonal(input=Bool.MAYBE))
    def kernel_fn(k: Kernel, **kwargs):
        def pool(mat: Optional[torch.Tensor], batch_ndim: int):
            if mat is None or (isinstance(mat, torch.Tensor) and mat.ndim == 0):
                return mat
            return _pool_kernel(mat, pool_type, window_shape, strides, pad_enum, normalize_edges, batch_ndim)
        nngp = pool(k.nngp, 2)
        ntk = pool(k.ntk, 2)
        cov1 = pool(k.cov1, 1 if k.diagonal_batch else 2)
        cov2 = pool(k.cov2, 1 if k.diagonal_batch else 2)
        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    def mask_fn(mask: torch.Tensor, input_shape: Tuple[int, ...]):
        return _pool_mask(mask, window_shape, strides, pad_enum, batch_axis, channel_axis)

    return init_fn, apply_fn, kernel_fn, mask_fn


# ---------------------------------------------------------------------------
# Global Pooling
# ---------------------------------------------------------------------------

@layer
@supports_masking(remask_kernel=False)
def GlobalAvgPool(batch_axis: int = 0, channel_axis: int = 1) -> InternalLayerMasked:
    return _GlobalPool(_Pooling.AVG, batch_axis, channel_axis)

@layer
@supports_masking(remask_kernel=False)
def GlobalSumPool(batch_axis: int = 0, channel_axis: int = 1) -> InternalLayerMasked:
    return _GlobalPool(_Pooling.SUM, batch_axis, channel_axis)

def _GlobalPool(pool_type: _Pooling, batch_axis: int, channel_axis: int) -> InternalLayerMasked:
    def init_fn(rng, input_shape):
        # remove spatial dims
        N, C, *S = input_shape
        return (N, C), ()
    def apply_fn(params, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs):
        # reduce over spatial axes
        if pool_type == _Pooling.AVG:
            if mask is None:
                # simple mean over all spatial dims
                dims = tuple(range(2, x.ndim))
                return x.mean(dim=dims)
            else:
                # masked mean
                dims = tuple(range(2, x.ndim))
                m = (~mask).to(x.dtype) if mask.dtype == torch.bool else mask.to(x.dtype)  # align w/ utils.mean_and_var convention if needed
                # but our supports_masking framework provides mask True==masked; we want unmasked count
                valid = (~mask).to(x.dtype)
                num = valid.sum(dim=dims).clamp_min(1)
                s = (x * valid).sum(dim=dims)
                return s / num
        else:
            dims = tuple(range(2, x.ndim))
            return x.sum(dim=dims)
    @requires(batch_axis=batch_axis, channel_axis=channel_axis, diagonal_spatial=Diagonal(input=Bool.MAYBE, output=Bool.YES))
    def kernel_fn(k: Kernel, **kwargs):
        def reduce(mat: Optional[torch.Tensor], batch_ndim: int, mask: Optional[torch.Tensor]):
            if mat is None:
                return mat
            spatial_axes = tuple(range(batch_ndim, mat.ndim))
            if pool_type == _Pooling.AVG:
                # masked mean compatible with your mean_and_var
                m, _ = mean_and_var(mat, axis=spatial_axes, mask=mask, get_var=False, keepdims=False)
                if k.diagonal_spatial:
                    # divide by spatial size if diagonal_spatial like JAX does
                    size = utils.size_at(mat, spatial_axes)
                    m = m / size
                return m
            else:
                return mat.sum(dim=spatial_axes)
        mask11, mask12, mask22 = k._get_mask_prods(k.mask1, k.mask2)
        cov1 = reduce(k.cov1, 1 if k.diagonal_batch else 2, mask11)
        cov2 = reduce(k.cov2, 1 if k.diagonal_batch else 2, mask22)
        nngp = reduce(k.nngp, 2, mask12)
        ntk = reduce(k.ntk, 2, mask12)
        # batch/channel order after spatial removal:
        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk,
                         batch_axis=0, channel_axis=1, is_reversed=False)
    def mask_fn(mask: torch.Tensor, input_shape: Tuple[int, ...]):
        # all spatial dims aggregateaggregated -> mask reduces to (N, C, 1...)->(N,C)
        dims = tuple(range(2, mask.ndim))
        return torch.all(mask, dim=dims)
    return init_fn, apply_fn, kernel_fn, mask_fn


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

@layer
@supports_masking(remask_kernel=False)
def Flatten(batch_axis: int = 0, batch_axis_out: int = 0) -> InternalLayerMasked:
    if batch_axis_out not in (0, 1, -1, -2):
        raise ValueError("batch_axis_out must be 0 or 1")
    batch_axis_out = 0 if batch_axis_out in (0, -2) else 1
    channel_axis_out = 1 - batch_axis_out

    def _out_shape(inp: Tuple[int, ...]):
        bs = inp[batch_axis]
        flat = 1
        for i, s in enumerate(inp):
            if i != batch_axis:
                flat *= s
        return (bs, flat) if batch_axis_out == 0 else (flat, bs)

    def init_fn(rng, input_shape):
        return _out_shape(input_shape), ()

    def apply_fn(params, x: torch.Tensor, **kwargs):
        shape = _out_shape(tuple(x.shape))
        x = _moveaxis(x, batch_axis, batch_axis_out if batch_axis_out == 0 else -1)
        return x.reshape(shape)

    @requires(batch_axis=batch_axis, channel_axis=None, diagonal_spatial=Diagonal(output=Bool.YES))
    def kernel_fn(k: Kernel, **kwargs):
        def trace_all(x: Optional[torch.Tensor], batch_ndim: int):
            if x is None:
                return x
            if k.diagonal_spatial:
                spatial_axes = tuple(range(batch_ndim, x.ndim))
                return x.mean(dim=spatial_axes)
            else:
                # iteratively trace last two spatial dims
                y = x
                while y.ndim > batch_ndim:
                    # trace over last two dims
                    n = y.shape[-1]
                    y = torch.einsum("...ii->...i", y) / n
                    # after einsum we lost one dim; keep tracing until batch_ndim
                return y
        cov1 = trace_all(k.cov1, 1 if k.diagonal_batch else 2)
        cov2 = trace_all(k.cov2, 1 if k.diagonal_batch else 2)
        nngp = trace_all(k.nngp, 2)
        ntk = trace_all(k.ntk, 2)
        return k.replace(cov1=cov1, cov2=cov2, nngp=nngp, ntk=ntk,
                         is_gaussian=False, is_reversed=False,
                         batch_axis=batch_axis_out, channel_axis=channel_axis_out,
                         diagonal_spatial=False)

    def mask_fn(mask: torch.Tensor, input_shape: Tuple[int, ...]):
        # broadcast to input_shape then flatten same as apply
        mask = mask.expand(input_shape)
        shape = _out_shape(input_shape)
        return mask.movedim(batch_axis, batch_axis_out).reshape(shape)

    return init_fn, apply_fn, kernel_fn, mask_fn


# ---------------------------------------------------------------------------
# LayerNorm (whiten over given axes; kernel = Gaussian-only)
# ---------------------------------------------------------------------------

@layer
@supports_masking(remask_kernel=False)
def LayerNorm(
    axis: Axes = -1,
    eps: float = 1e-12,
    batch_axis: int = 0,
    channel_axis: int = 1,
) -> InternalLayer:
    def init_fn(rng, input_shape):
        return input_shape, ()
    def apply_fn(params, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs):
        _axis = _canonicalize_axis(axis, inputs)
        mean, var = mean_and_var(inputs, _axis, keepdims=True, mask=mask, get_var=True)
        # mean/var from our helpers might be numpy/jax style; ensure torch tensor
        return (inputs - mean) / torch.sqrt(torch.as_tensor(eps, dtype=inputs.dtype, device=inputs.device) + var)

    @requires(batch_axis=batch_axis, channel_axis=channel_axis)
    def kernel_fn(k: Kernel, **kwargs):
        if not k.is_gaussian:
            raise NotImplementedError("LayerNorm kernel requires Gaussian inputs.")
        ndim = len(k.shape1)
        ch_ax = channel_axis % ndim
        _ax = list(_canonicalize_axis(axis, k.shape1))
        if ch_ax not in _ax:
            raise ValueError(f"Normalization must include channel axis {ch_ax}.")
        _ax = [a for a in _ax if a != ch_ax]

        # Build kernel-normalization masks and diagonals
        def prep_mask(m: Optional[torch.Tensor]):
            if m is None:
                return None
            if m.shape[channel_axis] != 1:
                raise NotImplementedError("Per-channel masks for LayerNorm not supported in kernel_fn.")
            return m.squeeze(channel_axis)

        kernel_axis = ()
        if (batch_axis % ndim) in _ax:
            kernel_axis = (0,)
            _ax.remove(batch_axis % ndim)

        spatial_axes = tuple(i for i in range(ndim) if i not in (ch_ax, batch_axis % ndim))
        kernel_axis += tuple(1 + (spatial_axes[::-1] if k.is_reversed else spatial_axes).index(i) for i in _ax)

        prod11, prod12, prod22 = get_diagonal_outer_prods(
            k.cov1 if k.cov1 is not None else None,
            k.cov2 if k.cov2 is not None else None,
            k.diagonal_batch,
            k.diagonal_spatial,
            op.mul,
            axis=kernel_axis,
            mask1=prep_mask(k.mask1),
            mask2=prep_mask(k.mask2),
        )

        def _safe_div(x: Optional[torch.Tensor], denom: Optional[torch.Tensor]):
            if x is None:
                return None
            return x / torch.sqrt(denom)

        nngp = _safe_div(k.nngp, prod12)
        ntk  = _safe_div(k.ntk,  prod12)
        cov1 = _safe_div(k.cov1, prod11)
        cov2 = _safe_div(k.cov2, prod22)
        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return init_fn, apply_fn, kernel_fn


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

@layer
@supports_masking(remask_kernel=False)
def Dropout(rate: float, mode: str = "train") -> InternalLayer:
    if mode not in ("test", "train"):
        raise ValueError('mode must be "test" or "train"')
    if rate <= 0.0 or rate > 1.0:
        raise ValueError("rate must be in (0, 1].")

    def init_fn(rng, input_shape):
        return input_shape, ()

    def apply_fn(params, inputs: torch.Tensor, rng: Optional[torch.Generator] = None, **kwargs):
        if mode == "test" or rate == 1.0:
            return inputs
        g = rng if rng is not None else torch.Generator(device=inputs.device.type)
        mask = (torch.rand_like(inputs, generator=g) < rate).to(inputs.dtype) / rate
        return inputs * mask

    if mode == "test":
        @requires(use_dropout=True)
        def kernel_fn(k: Kernel, **kwargs):
            return k
    else:
        @requires(use_dropout=True)
        def kernel_fn(k: Kernel, **kwargs):
            if k.is_input:
                raise ValueError("Dropout cannot be the input layer.")
            factor = 1.0 / rate
            def diag_mul(x: Optional[torch.Tensor], diagonal_batch: bool, diagonal_spatial: bool):
                if x is None:
                    return None
                if diagonal_spatial:
                    if diagonal_batch:
                        return x * factor
                    else:
                        if x.shape[0] != x.shape[1]:
                            return x
                        idx = torch.arange(x.shape[0], device=x.device)
                        out = x.clone()
                        out[idx, idx, ...] = out[idx, idx, ...] * factor
                        return out
                else:
                    # full-spatial: multiply diagonal blocks
                    out = x.clone()
                    if diagonal_batch:
                        # multiply all
                        return out * factor
                    else:
                        if out.shape[0] == out.shape[1]:
                            # multiply only sample-diagonal blocks
                            idx = torch.arange(out.shape[0], device=out.device)
                            out[idx, idx, ...] = out[idx, idx, ...] * factor
                        return out
            cov1 = diag_mul(k.cov1, k.diagonal_batch, k.diagonal_spatial)
            cov2 = diag_mul(k.cov2, k.diagonal_batch, k.diagonal_spatial)
            new_factor = factor if k.x1_is_x2 else 1.0
            nngp = diag_mul(k.nngp, False, k.diagonal_spatial)
            ntk  = diag_mul(k.ntk,  False, k.diagonal_spatial)
            return k.replace(cov1=cov1, cov2=cov2, nngp=nngp, ntk=ntk, is_gaussian=False)

    return init_fn, apply_fn, kernel_fn


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

@layer
@supports_masking(remask_kernel=False)
def Index(idx, batch_axis: int = 0, channel_axis: int = -1) -> InternalLayerMasked:
    def init_fn(rng, input_shape: Tuple[int, ...]):
        # Use numpy-like slicing on sizes:
        # We’ll compute by making a dummy tensor (sizes only)
        dummy = torch.empty(input_shape)
        out = dummy[idx]
        return tuple(out.shape), ()
    def apply_fn(params, x: torch.Tensor, **kwargs):
        return x[idx]
    def mask_fn(mask: torch.Tensor, input_shape: Tuple[int, ...]):
        return mask[idx]
    @requires(batch_axis=batch_axis, channel_axis=channel_axis)
    def kernel_fn(k: Kernel, **kwargs):
        return k[idx]
    return init_fn, apply_fn, kernel_fn, mask_fn


# ---------------------------------------------------------------------------
# Internals used by pooling/conv kernels (torch versions)
# ---------------------------------------------------------------------------

def _pool_mask(
    mask: torch.Tensor,
    window_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_axis: int,
    channel_axis: int,
) -> torch.Tensor:
    # OR over the window (True==masked); we can reuse avg_pool on float then >0
    nsp = len(window_shape)
    m = mask.to(torch.float32)
    if padding in (Padding.SAME, Padding.CIRCULAR):
        pads = []
        for s, w, st in reversed(list(zip(mask.shape[2:], window_shape, strides))):
            total = max(((s + st - 1) // st - 1) * st + w - s, 0)
            l = total // 2
            r = total - l
            pads.extend([l, r])
        mode = "circular" if padding == Padding.CIRCULAR else "constant"
        m = F.pad(m, tuple(pads), mode=mode)
    if nsp == 1:
        pooled = F.avg_pool1d(m, window_shape, stride=strides)
    elif nsp == 2:
        pooled = F.avg_pool2d(m, window_shape, stride=strides)
    else:
        pooled = F.avg_pool3d(m, window_shape, stride=strides)
    return (pooled > 0).to(mask.dtype)

def _pool_kernel(
    lhs: torch.Tensor,
    pool_type: _Pooling,
    window_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    normalize_edges: bool,
    batch_ndim: int,
) -> torch.Tensor:
    """
    Kernel-space pooling: reduce_window + optional normalization.
    We implement via average pooling (sum = avg * window_size).
    lhs layout: [N1[,N2], spatial... (zipped if full-spatial)]
    """
    nsp_total = lhs.ndim - batch_ndim
    # Treat the trailing spatial dims as pooling dims; use avg_pool with reshape.
    # We support diagonal_spatial path mainly (shape with single set of spatial dims).
    # For full-spatial we’d need window over interleaved dims; omitted for brevity.
    if nsp_total <= 0:
        return lhs
    # reshape to (B..., 1, *spatial) to reuse avg pooling
    leading = lhs.shape[:batch_ndim]
    spatial = lhs.shape[batch_ndim:]
    x = lhs.reshape((-1, 1, *spatial))
    nsp = len(window_shape)
    if padding in (Padding.SAME, Padding.CIRCULAR):
        pads = []
        for s, w, st in reversed(list(zip(spatial, window_shape, strides))):
            total = max(((s + st - 1)//st - 1)*st + w - s, 0)
            l = total // 2
            r = total - l
            pads.extend([l, r])
        mode = "circular" if padding == Padding.CIRCULAR else "constant"
        x = F.pad(x, tuple(pads), mode=mode)

    if nsp == 1:
        out = F.avg_pool1d(x, window_shape, stride=strides)
    elif nsp == 2:
        out = F.avg_pool2d(x, window_shape, stride=strides)
    else:
        out = F.avg_pool3d(x, window_shape, stride=strides)

    if pool_type == _Pooling.SUM:
        out = out * int(torch.prod(torch.tensor(window_shape)).item())

    out = out.reshape(leading + out.shape[2:])
    return out