# Copyright 2025
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

"""Requirement management for stax-like layers (PyTorch port)."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Callable, Optional, Sequence, Union
import warnings

import inspect
import torch
import numpy as np
import frozendict  # if not available, install `frozendict`

from ..utils import dataclasses as nt_dataclasses
from ..utils import utils
from ..utils.kernel import Kernel
from ..utils.typing import AnalyticKernelFn, ApplyFn, Axes, Get, InitFn, InternalLayer
from ..utils.typing import Layer, LayerKernelFn, NTTree, PyTree

# --------------------------------------------------------------------------
# Tiny NTTree helpers (lists/tuples as internal nodes, leaves otherwise)
# --------------------------------------------------------------------------

def _call_with_accepted_kwargs(fn, *args, **kwargs):
    """
    Call `fn(*args, **filtered_kwargs)` where we drop any kwargs that `fn`
    doesn't accept (unless it has **kwargs already).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # C-implemented or builtins; best effort – just call
        return fn(*args, **kwargs)

    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return fn(*args, **kwargs)

    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **accepted)


def _is_list_or_tuple(x): return type(x) in (list, tuple)

def _tree_map(fn: Callable[..., Any], *trees, is_leaf: Optional[Callable[[Any], bool]] = None):
    if not trees:
        return None
    ref = trees[0]
    leaf = (is_leaf(ref) if is_leaf is not None else not _is_list_or_tuple(ref))
    if leaf:
        return fn(*trees)
    t = type(ref)
    # zip structure
    return t(_tree_map(fn, *(xs[i] for xs in trees), is_leaf=is_leaf) for i in range(len(ref)))

def _tree_all(x) -> bool:
    if _is_list_or_tuple(x):
        return all(_tree_all(e) for e in x)
    if isinstance(x, torch.Tensor):
        return bool(x)
    if isinstance(x, np.ndarray):
        return bool(x.all())
    return bool(x)

# --------------------------------------------------------------------------
# Public decorators
# --------------------------------------------------------------------------

def layer(layer_fn: Callable[..., InternalLayer]) -> Callable[..., Layer]:
    """Decorator for public layers to add flexible kernel handling."""
    name = layer_fn.__name__

    @utils.wraps(layer_fn)
    def new_layer_fns(*args, **kwargs):
        init_fn, apply_fn, kernel_fn = layer_fn(*args, **kwargs)
        kernel_fn = _preprocess_kernel_fn(init_fn, apply_fn, kernel_fn)
        init_fn.__name__ = apply_fn.__name__ = kernel_fn.__name__ = name
        return init_fn, apply_fn, kernel_fn

    return new_layer_fns


def requires(**static_reqs):
    """Decorator adding consistency checks to a `kernel_fn`."""

    def req(kernel_fn: LayerKernelFn):
        @utils.wraps(kernel_fn)
        def new_kernel_fn(k: NTTree[Kernel], **kwargs) -> NTTree[Kernel]:
            fused_reqs = _fuse_requirements(static_reqs, {}, **kwargs)

            if isinstance(k, Kernel):
                for key, v in fused_reqs.items():
                    if v is not None:
                        if key in ('diagonal_batch', 'diagonal_spatial'):
                            if (getattr(k, key) is True and
                                (v is False or
                                 (isinstance(v, Diagonal) and v.input == Bool.NO))):
                                raise ValueError(
                                    f'{kernel_fn} requires `{key} == {v}`, but input kernel has '
                                    f'`{key} == True` (insufficient information). '
                                    f'Recompute the input kernel with `{key} == {v}`.')

                        elif key in ('batch_axis', 'channel_axis'):
                            ndim = len(k.shape1)
                            v_kernel = getattr(k, key)
                            v_pos = v % ndim
                            if v_kernel != v_pos:
                                raise ValueError(
                                    f'{kernel_fn} requires `{key} == {v_pos}`, but input kernel has '
                                    f'`{key} == {v_kernel}`, making the infinite limit ill-defined.')

                        else:
                            pass

            return kernel_fn(k, **kwargs)

        _set_req(new_kernel_fn, frozendict.frozendict(static_reqs))
        return new_kernel_fn

    return req


def supports_masking(remask_kernel: bool):
    """Decorator enabling masking support for (init_fn, apply_fn, kernel_fn)."""

    def _supports_masking(layer):
        @utils.wraps(layer)
        def layer_with_masking(*args, **kwargs) -> InternalLayer:
            layer_fns = layer(*args, **kwargs)
            init_fn, apply_fn, kernel_fn = layer_fns[:3]

            if len(layer_fns) == 3:
                _mask_fn = lambda mask, input_shape: mask
            elif len(layer_fns) == 4:
                _mask_fn = layer_fns[3]
            else:
                raise ValueError(f'Expected 3 or 4 layer functions, got {len(layer_fns)}.')

            @utils.wraps(_mask_fn)
            def mask_fn(mask, input_shape):
                if mask is None:
                    return None
                return _mask_fn(mask, input_shape)

            def apply_fn_with_masking(params, inputs, *, mask_constant=None, **kwargs):
                masked_inputs = _tree_map(
                    lambda x: _get_masked_array(x, mask_constant),
                    inputs,
                    is_leaf=lambda x: isinstance(x, (torch.Tensor, MaskedArray))
                )

                is_leaf = lambda x: isinstance(x, MaskedArray)
                inputs_unmasked = _tree_map(lambda x: x.masked_value, masked_inputs, is_leaf=is_leaf)
                mask = _tree_map(lambda x: x.mask, masked_inputs, is_leaf=is_leaf)

                outputs = apply_fn(params, inputs_unmasked, mask=mask, **kwargs)

                # derive output mask via mask_fn
                if isinstance(inputs_unmasked, torch.Tensor):
                    in_shape = inputs_unmasked.shape
                else:
                    in_shape = [i.shape for i in inputs_unmasked]  # type: ignore[index]
                outputs_mask = mask_fn(mask, in_shape)
                if outputs_mask is None:
                    return outputs
                return MaskedArray(outputs, outputs_mask)

            def kernel_fn_with_masking(k: NTTree[Kernel], **user_reqs):
                is_leaf = lambda kk: isinstance(kk, Kernel)
                mask1 = _tree_map(lambda kk: kk.mask1, k, is_leaf=is_leaf)
                shape1 = _tree_map(lambda kk: kk.shape1, k, is_leaf=is_leaf)
                mask2 = _tree_map(lambda kk: kk.mask2, k, is_leaf=is_leaf)
                shape2 = _tree_map(lambda kk: kk.shape2, k, is_leaf=is_leaf)

                mask1, mask2 = mask_fn(mask1, shape1), mask_fn(mask2, shape2)

                kout = kernel_fn(k, **user_reqs)  # type: ignore[assignment]

                if remask_kernel:
                    remask_fn = lambda kk, m1, m2: kk.mask(m1, m2)
                else:
                    remask_fn = lambda kk, m1, m2: kk.replace(mask1=m1, mask2=m2)

                kout = _tree_map(remask_fn, kout, mask1, mask2, is_leaf=is_leaf)
                return kout

            if _has_req(kernel_fn):
                _set_req(kernel_fn_with_masking, get_req(kernel_fn))

            return init_fn, apply_fn_with_masking, kernel_fn_with_masking

        return layer_with_masking

    return _supports_masking


def unmask_fn(fn: ApplyFn) -> ApplyFn:
    """Make a function returning a `MaskedArray` return a `torch.Tensor`."""
    def unmask(x: Union[MaskedArray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, MaskedArray):
            return utils.mask(x.masked_value, x.mask)
        return x

    def is_leaf(x) -> bool:
        return isinstance(x, (torch.Tensor, MaskedArray))

    @utils.wraps(fn)
    def fn_no_mask(*args, **kwargs):
        out = fn(*args, **kwargs)
        out = _tree_map(unmask, out, is_leaf=is_leaf)
        return out

    return fn_no_mask

# --------------------------------------------------------------------------
# Internal utilities / data structures
# --------------------------------------------------------------------------

@nt_dataclasses.dataclass
class MaskedArray:
    """Dataclass representing a masked torch.Tensor or PyTree thereof."""
    masked_value: PyTree
    mask: PyTree


def _get_masked_array(
    x: Union[None, torch.Tensor, float, int, MaskedArray],
    mask_constant: Optional[float] = None
) -> MaskedArray:
    """Return (masked_x, mask) where mask is bool tensor with True = masked."""
    if x is None:
        mask_mat = None

    elif isinstance(x, MaskedArray):
        x, mask_mat = x.masked_value, x.mask

    elif isinstance(x, (np.ndarray, torch.Tensor, float, int)):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x)
        if mask_constant is None:
            mask_mat = None
        else:
            if isinstance(mask_constant, float) and np.isnan(mask_constant):
                mask_mat = torch.isnan(x) if isinstance(x, torch.Tensor) else None
            else:
                mask_mat = (x == mask_constant) if isinstance(x, torch.Tensor) else None
    else:
        raise TypeError(x, type(x))

    x = utils.mask(x, mask_mat)
    return MaskedArray(x, mask_mat)

_INPUT_REQ = 'input_req'

def get_req(
    f: Callable,
    default: Optional[frozendict.frozendict] = None
) -> frozendict.frozendict:
    return getattr(f, _INPUT_REQ, default)

def _set_req(f: Callable, req: frozendict.frozendict):
    setattr(f, _INPUT_REQ, req)

def _has_req(f: Callable) -> bool:
    return hasattr(f, _INPUT_REQ)

_DEFAULT_INPUT_REQ = frozendict.frozendict(
    {
        'diagonal_batch': True,
        'diagonal_spatial': False,
        'batch_axis': 0,
        'use_dropout': False,
        'channel_axis': -1,
        'mask_constant': None
    }
)

class Bool(enum.IntEnum):
    """Trinary logic helper."""
    NO = 0
    MAYBE = 1
    YES = 2

    def __and__(self, other: 'Bool') -> 'Bool':
        return min(self, other)
    __rand__ = __and__


@dataclasses.dataclass(frozen=True)
class Diagonal:
    """Decides whether diagonal-only storage is allowed."""
    input: Bool = Bool.YES
    output: Bool = Bool.NO

    def __rshift__(self, other: 'Diagonal') -> 'Diagonal':
        if self.output == Bool.YES:
            return self
        if self.output > Bool.NO and other.input > Bool.NO:
            input_ = self.input
        elif self.output == Bool.NO and other.input < Bool.YES:
            input_ = Bool.NO
        else:
            input_ = min(self.input, other.input)
        return Diagonal(input=input_, output=other.output)

    def __and__(self, other: 'Diagonal') -> 'Diagonal':
        return Diagonal(input=self.input & other.input,
                        output=self.output & other.output)

    def __bool__(self) -> bool:
        return self.input == Bool.YES and self.output > Bool.NO

    def __lshift__(self, other: 'Diagonal') -> 'Diagonal':
        return other >> self
    __rand__ = __and__

# --------------------------------------------------------------------------
# Covariance helpers (Torch)
# --------------------------------------------------------------------------

def _cov_diag_batch_diag_spatial(
    x: torch.Tensor,
    batch_axis: int,
    channel_axis: int
) -> torch.Tensor:
    ret = torch.sum(x ** 2, dim=channel_axis)
    new_batch_axis = batch_axis - (1 if batch_axis > channel_axis else 0)
    ret = torch.movedim(ret, new_batch_axis, 0)
    return ret

def _cov_diag_batch_full_spatial(
    x: torch.Tensor,
    batch_axis: int,
    channel_axis: int
) -> torch.Tensor:
    # Move to [B, S..., C]
    x = torch.movedim(x, (batch_axis, channel_axis), (0, -1))
    b = x.shape[0]
    spatial = x.shape[1:-1]
    c = x.shape[-1]
    x2 = x.reshape(b, -1, c)                 # (B, P, C)
    print(f"x2 shape: {x2.shape}")
    ret = torch.einsum('bpc,bqc->bpq', x2, x2)  # (B, P, P)
    print(f"ret shape: {ret.shape}")
    shape = (b,) + spatial + spatial         # (B, s1,...,sk, s1,...,sk)
    ret = ret.reshape(shape)
    ret = utils.zip_axes(ret, 1)             # -> interleave spatial axes
    return ret

def _cov_full_batch_full_spatial(
    x1: torch.Tensor,
    x2: torch.Tensor,
    batch_axis: int,
    channel_axis: int
) -> torch.Tensor:
    x1 = torch.movedim(x1, (batch_axis, channel_axis), (0, -1))  # (B1, S..., C)
    x2 = torch.movedim(x2, (batch_axis, channel_axis), (0, -1))  # (B2, S..., C)
    b1, spatial, c = x1.shape[0], x1.shape[1:-1], x1.shape[-1]
    b2 = x2.shape[0]
    p = int(np.prod(spatial))
    X1 = x1.reshape(b1, p, c)
    X2 = x2.reshape(b2, p, c)
    ret = torch.einsum('bpc,Bqc->bBpq', X1, X2)  # (B1, B2, P, Q)
    ret = ret.reshape((b1, b2) + spatial + spatial)
    ret = utils.zip_axes(ret, 2)
    return ret

def _cov_full_batch_diag_spatial(
    x1: torch.Tensor,
    x2: torch.Tensor,
    batch_axis: int,
    channel_axis: int
) -> torch.Tensor:
    x1 = torch.movedim(x1, (batch_axis, channel_axis), (0, -1))  # (B1, S..., C)
    x2 = torch.movedim(x2, (batch_axis, channel_axis), (0, -1))  # (B2, S..., C)
    ret = torch.einsum('b...c,B...c->bB...', x1, x2)  # (B1, B2, S...)
    return ret


def _paired_spatial_diag(x: torch.Tensor, start_axis: int, n_spatial: int) -> torch.Tensor:
    """
    x has shape (..., s1, s1, s2, s2, ..., sK, sK, ...),
    starting at `start_axis` (i.e., interleaved pairs already).
    Take diagonal over each (si, si) and return (..., s1, s2, ..., sK, ...).

    We use torch.diagonal and move the new diag dim back to the pair's start,
    so repeated application maintains ordering.
    """
    for i in range(n_spatial):
        d1 = start_axis + 2 * i
        d2 = d1 + 1
        # take diag over the pair -> appends a new dim at the end
        x = x.diagonal(offset=0, dim1=d1, dim2=d2)
        # move the new trailing diag dim back to position d1
        x = x.movedim(-1, d1)
    return x



def _cov_diag_batch(
    x: torch.Tensor,
    diagonal_spatial: bool,
    batch_axis: int,
    channel_axis: int
) -> torch.Tensor:
    if diagonal_spatial:
        ret = _cov_diag_batch_diag_spatial(x, batch_axis, channel_axis)
    else:
        ret = _cov_diag_batch_full_spatial(x, batch_axis, channel_axis)
    return ret / x.shape[channel_axis]

def _cov(
    x1: torch.Tensor,
    x2: Optional[torch.Tensor],
    diagonal_spatial: bool,
    batch_axis: int,
    channel_axis: int
) -> Optional[torch.Tensor]:
    x2 = x1 if x2 is None else x2
    if diagonal_spatial:
        ret = _cov_full_batch_diag_spatial(x1, x2, batch_axis, channel_axis)
    else:
        ret = _cov_full_batch_full_spatial(x1, x2, batch_axis, channel_axis)
    return ret / x1.shape[channel_axis]

# --------------------------------------------------------------------------
# Inputs -> Kernel (Torch)
# --------------------------------------------------------------------------

def _inputs_to_kernel(
    x1: torch.Tensor,
    x2: Optional[torch.Tensor],
    *,
    diagonal_batch: bool,
    diagonal_spatial: Union[bool, Diagonal],
    compute_ntk: bool,
    batch_axis: int,
    channel_axis: Optional[int],
    mask_constant: Optional[float],
    eps: float = 1e-12,
    **kwargs
) -> Kernel:
    """Transforms (batches of) inputs to a `Kernel` (PyTorch)."""
    if not (isinstance(x1, torch.Tensor) and (x2 is None or isinstance(x2, torch.Tensor))):
        raise TypeError(('Wrong input types given. Found `x1` of type '
                         f'{type(x1)} and `x2` of type {type(x2)}, need both to be '
                         f'`torch.Tensor`s (`x2` can be `None`).'))

    batch_axis %= x1.ndim
    diagonal_spatial = bool(diagonal_spatial)

    if batch_axis != 0:
        warnings.warn(f'!!! Non-leading (!= 0) batch dimension in the input layer is not '
                      f'supported for batching kernels, got batch_axis = {batch_axis}. !!!')

    if channel_axis is None:
        def flatten(x):
            if x is None:
                return x
            x = torch.movedim(x, batch_axis, 0)
            return x.reshape((x.shape[0], -1))
        x1, x2 = flatten(x1), flatten(x2)
        batch_axis, channel_axis = 0, 1
        diagonal_spatial = False
    else:
        channel_axis %= x1.ndim

    def get_x_cov_mask(x):
        if x is None:
            return None, None, None
        if x.ndim < 2:
            raise ValueError(f'Inputs must be at least 2D (a batch dimension and a '
                             f'channel/feature dimension), got {x.ndim}.')
        ma = _get_masked_array(x, mask_constant)
        x_t, mask = ma.masked_value, ma.mask

        # Use a stable floating dtype (match original intent)
        x_t = x_t.to(dtype=torch.float64)

        if diagonal_batch:
            cov = _cov_diag_batch(x_t, diagonal_spatial, batch_axis, channel_axis)
        else:
            cov = _cov(x_t, x_t, diagonal_spatial, batch_axis, channel_axis)

        return x_t, cov, mask

    x1_t, cov1, mask1 = get_x_cov_mask(x1)
    x2_t, cov2, mask2 = get_x_cov_mask(x2)
    nngp = _cov(x1_t, x2_t, diagonal_spatial, batch_axis, channel_axis)

    ntk = torch.zeros((), dtype=nngp.dtype, device=nngp.device) if compute_ntk else None
    is_gaussian = False
    is_reversed = False
    x1eqx2 = utils.x1_is_x2(x1_t, x2_t, eps=eps)
    is_input = False

    return Kernel(
        cov1=cov1,
        cov2=cov2,
        nngp=nngp,
        ntk=ntk,
        x1_is_x2=x1eqx2,
        is_gaussian=is_gaussian,
        is_reversed=is_reversed,
        is_input=is_input,
        diagonal_batch=diagonal_batch,
        diagonal_spatial=diagonal_spatial,
        shape1=tuple(x1_t.shape),
        shape2=tuple(x1_t.shape) if x2_t is None else tuple(x2_t.shape),
        batch_axis=batch_axis,
        channel_axis=channel_axis,
        mask1=mask1,
        mask2=mask2,
    )

# --------------------------------------------------------------------------
# Shape propagation (Torch)
# --------------------------------------------------------------------------

def _propagate_shape(
    init_fn: InitFn,
    apply_fn: ApplyFn,
    shaped: Any,
    **kwargs
) -> Any:
    """Evaluate `init_fn` on abstract Shapes to get output shape info.

    We rely on `init_fn(rng, input_shape)` -> (output_shape, params).
    """
    # Replace RNG with a concrete torch.Generator; `init_fn` should not need real randomness
    gen = torch.Generator()
    gen.manual_seed(0)
    # shaped is an NTTree of ShapedArray-like tuples (we used ShapedArray in other module,
    # but for shape propagation we pass plain tuples)
    out_shape, _params = _call_with_accepted_kwargs(init_fn,gen, shaped, **kwargs)
    # Some apply_fns may return a MaskedArray; we only pass shapes through.
    if isinstance(out_shape, MaskedArray):
        out_shape = out_shape.masked_value
    return out_shape

def _set_shapes(
    init_fn: InitFn,
    apply_fn: ApplyFn,
    in_kernel: NTTree[Kernel],
    out_kernel: NTTree[Kernel],
    **kwargs
) -> NTTree[Kernel]:
    """Apply a kernel_fn to a Kernel and propagate shape side-information."""
    is_leaf = lambda k: isinstance(k, Kernel)

    shape1 = _tree_map(lambda k: (k.shape1), in_kernel, is_leaf=is_leaf)
    shape2 = _tree_map(lambda k: (k.shape2), in_kernel, is_leaf=is_leaf)

    kwargs1, kwargs2 = utils.split_kwargs(kwargs)

    shape1 = _propagate_shape(init_fn, unmask_fn(apply_fn), shape1, **kwargs1)
    shape2 = _propagate_shape(init_fn, unmask_fn(apply_fn), shape2, **kwargs2)

    set_shape_fn = lambda k, s1, s2: k.replace(shape1=tuple(s1), shape2=tuple(s2))
    return _tree_map(set_shape_fn, out_kernel, shape1, shape2, is_leaf=is_leaf)

# --------------------------------------------------------------------------
# Requirements fusion and kernel_fn preprocessing
# --------------------------------------------------------------------------

def _fuse_requirements(
    kernel_fn_reqs,
    default_reqs,
    **user_reqs
) -> frozendict.frozendict:
    kernel_fn_reqs = dict(kernel_fn_reqs)
    for k, v_user in user_reqs.items():
        if v_user is not None:
            if k in kernel_fn_reqs:
                v_kernel = kernel_fn_reqs[k]
                if (v_user is True and
                    (v_kernel is False or
                     (isinstance(kernel_fn_reqs[k], Diagonal) and
                      kernel_fn_reqs[k].input == Bool.NO))):
                    raise ValueError(
                        f'Asked to compute `kernel_fn` output with `{k} == {v_user}`, '
                        f'while `kernel_fn` requires `{k} == {kernel_fn_reqs[k]}`.')
            kernel_fn_reqs[k] = v_user

    for k, v_user in default_reqs.items():
        if k not in kernel_fn_reqs:
            kernel_fn_reqs[k] = v_user

    return frozendict.frozendict(kernel_fn_reqs)


def _preprocess_kernel_fn(
    init_fn: InitFn,
    apply_fn: ApplyFn,
    kernel_fn: LayerKernelFn
) -> AnalyticKernelFn:
    """Return a `kernel_fn` with additional arguments, Torch version."""
    if not _has_req(kernel_fn):
        kernel_fn = requires()(kernel_fn)

    def kernel_fn_kernel(kernel, **kwargs):
        out_kernel = kernel_fn(kernel, **kwargs)
        return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

    def kernel_fn_x1(x1, x2, get, **kwargs):
        kernel_fn_reqs = get_req(kernel_fn)
        reqs = _fuse_requirements(kernel_fn_reqs, _DEFAULT_INPUT_REQ, **kwargs)
        compute_ntk = (get is None) or ('ntk' in get)

        if x2 is None:
            x2 = _tree_map(lambda _x: None, x1)

        def input_fn(_x1, _x2):
            return _inputs_to_kernel(_x1, _x2, compute_ntk=compute_ntk, **reqs)

        kernel = _tree_map(input_fn, x1, x2)
        out_kernel = kernel_fn(kernel, **kwargs)
        return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

    @utils.get_namedtuple('AnalyticKernel')
    def kernel_fn_any(x1_or_kernel: Union[NTTree[torch.Tensor], NTTree[Kernel]],
                      x2: Optional[NTTree[torch.Tensor]] = None,
                      get: Optional[Get] = None,
                      *,
                      pattern: Optional[tuple[Optional[torch.Tensor],
                                              Optional[torch.Tensor]]] = None,
                      mask_constant: Optional[float] = None,
                      diagonal_batch: Optional[bool] = None,
                      diagonal_spatial: Optional[bool] = None,
                      **kwargs):
        """Apply kernel_fn to Tensor inputs or Kernel, returning requested info."""
        def all_of(x, cls: type) -> bool:
            def is_leaf(xx) -> bool:
                return isinstance(xx, (Kernel, torch.Tensor, np.ndarray))
            return _tree_all(_tree_map(lambda v: isinstance(v, cls), x, is_leaf=is_leaf))

        if all_of(x1_or_kernel, Kernel) and x2 is None:
            return kernel_fn_kernel(x1_or_kernel,
                                    pattern=pattern,
                                    diagonal_batch=diagonal_batch,
                                    diagonal_spatial=diagonal_spatial,
                                    **kwargs)

        return kernel_fn_x1(x1_or_kernel, x2, get,
                            pattern=pattern,
                            diagonal_batch=diagonal_batch,
                            diagonal_spatial=diagonal_spatial,
                            mask_constant=mask_constant,
                            **kwargs)

    _set_req(kernel_fn_any, get_req(kernel_fn))
    return kernel_fn_any

# --------------------------------------------------------------------------
# Diagonal extraction / outer products / mean & var (Torch)
# --------------------------------------------------------------------------

def get_diagonal(
    cov: Optional[torch.Tensor],
    diagonal_batch: bool,
    diagonal_spatial: bool
) -> Optional[torch.Tensor]:
    if cov is None:
        return cov
    batch_ndim = 1 if diagonal_batch else 2
    start_axis = 2 - batch_ndim
    end_axis = batch_ndim if diagonal_spatial else cov.ndim
    cov = utils.unzip_axes(cov, start_axis, end_axis)
    return utils.diagonal_between(cov, start_axis, end_axis)

def get_diagonal_outer_prods(
    cov1: torch.Tensor,
    cov2: Optional[torch.Tensor],
    diagonal_batch: bool,
    diagonal_spatial: bool,
    operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    axis: Sequence[int] = (),
    mask1: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = utils.canonicalize_axis(axis, cov1)

    cov1 = get_diagonal(cov1, diagonal_batch, diagonal_spatial)
    cov2 = get_diagonal(cov2, diagonal_batch, diagonal_spatial)

    cov1_mean, _ = mean_and_var(cov1, axis=axis, keepdims=True, mask=mask1)
    cov2_mean, _ = mean_and_var(cov2, axis=axis, keepdims=True, mask=mask2)

    if cov1_mean is None:
        raise ValueError('cov1 is None')

    end_axis = 1 if diagonal_spatial else cov1_mean.ndim
    prod12 = utils.outer_prod(cov1_mean, cov2_mean, 0, end_axis, operation)

    start_axis = 1 if diagonal_batch else 0
    prod11 = utils.outer_prod(cov1_mean, cov1_mean, start_axis, end_axis, operation)
    prod22 = (utils.outer_prod(cov2_mean, cov2_mean, start_axis, end_axis, operation)
              if cov2_mean is not None else prod11)

    return prod11, prod12, prod22


def mean_and_var(
    x: Optional[torch.Tensor],
    axis: Optional[Axes] = None,
    dtype: Optional[torch.dtype] = None,
    out: Optional[None] = None,
    ddof: int = 0,
    keepdims: bool = False,
    mask: Optional[torch.Tensor] = None,
    get_var: bool = False
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """torch.mean / torch.var with masking support."""
    var = None
    if x is None:
        return x, var

    if mask is None:
        mean = torch.mean(x.to(dtype) if dtype else x, dim=axis, keepdim=keepdims)
        if get_var:
            var = torch.var(x.to(dtype) if dtype else x, dim=axis, correction=ddof, keepdim=keepdims)
    else:
        axis = tuple(utils.canonicalize_axis(axis, x))
        size = utils.size_at(x, axis)
        mask_b = torch.broadcast_to(mask, x.shape) if isinstance(mask, torch.Tensor) else None
        mask_size = torch.count_nonzero(mask_b, dim=axis) if mask_b is not None else 0
        # expand mask_size over reduced axes
        if isinstance(mask_size, torch.Tensor):
            for i in axis:
                mask_size = mask_size.unsqueeze(i)
        size = size - (int(mask_size) if not isinstance(mask_size, torch.Tensor) else mask_size)
        size = torch.maximum(torch.as_tensor(1, device=x.device), size) if isinstance(size, torch.Tensor) else max(size, 1)

        mean = torch.sum(x, dim=axis, keepdim=True) / size
        if not keepdims:
            mean = mean.squeeze(dim=axis)

        if get_var:
            var = torch.sum((x - mean) ** 2, dim=axis, keepdim=True) / (size - ddof)
            if not keepdims:
                var = var.squeeze(dim=axis)

    return mean, var