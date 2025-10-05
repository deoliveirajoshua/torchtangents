# torchtangents/_src/stax/linear.py  (PyTorch port)
from __future__ import annotations

import enum
import functools
import operator as op
import string
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import warnings
import math

import torch
import torch.nn.functional as F

from ..utils.kernel import Kernel
from ..utils import utils
from ..utils.typing import Axes, InternalLayer, InternalLayerMasked, PyTree
from .requirements import (
    layer, requires, supports_masking,
    Diagonal, Bool, mean_and_var, get_diagonal_outer_prods
)

# ---------------------------------------------------------------------------
# Enums (API parity with neural_tangents.stax.linear)
# ---------------------------------------------------------------------------

class Padding(enum.Enum):
  """Type of padding in pooling and convolutional layers.

  Attributes:
    CIRCULAR:
      circular padding, as if the input were a torus.

    SAME:
      same, a.k.a. zero padding.

    VALID:
      valid, a.k.a. no padding.
  """
  CIRCULAR = 'CIRCULAR'
  SAME = 'SAME'
  VALID = 'VALID'


class _Pooling(enum.Enum):
  """Type of pooling in pooling layers.

  Attributes:
    AVG:
      average pooling, the output is normalized by the input receptive field
      size.

    SUM:
      sum pooling, no normalization.
  """
  AVG = 'AVG'
  SUM = 'SUM'


class AggregateImplementation(enum.Enum):
  """Implementation of the :obj:`Aggregate` layer.

  See :obj:`Aggregate` docstring for details.

  Attributes:
    DENSE:
      Is recommended for dense graphs, where the number of edges `E` is
      proportional to the number of vertices `V` to the power of 1.5 or more.

    SPARSE:
      Is recommended for sparse graphs, where `E ~ O(V)` or less.
  """
  DENSE = 'DENSE'
  SPARSE = 'SPARSE'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@layer
@supports_masking(remask_kernel=False)
def Identity() -> InternalLayer:
    """Identity (no-op).

    Returns:
      (init_fn, apply_fn, kernel_fn)
    """
    def init_fn(rng, input_shape: Tuple[int, ...]):
        # No parameters; output shape equals input shape.
        return input_shape, ()

    def apply_fn(params, inputs: torch.Tensor, **kwargs):
        # Pass through unchanged.
        return inputs

    def kernel_fn(k: Kernel, **kwargs) -> Kernel:
        # Pass-through in the kernel regime as well.
        return k

    return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def DotGeneral(
    *,
    lhs: Optional[Union[torch.Tensor, float]] = None,
    rhs: Optional[Union[torch.Tensor, float]] = None,
    # ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))
    dimension_numbers: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]],
                             Tuple[Tuple[int, ...], Tuple[int, ...]]] = (((), ()), ((), ())),
    precision: Optional[object] = None,  # Ignored in PyTorch; kept for API parity.
    batch_axis: int = 0,
    channel_axis: int = -1,
) -> InternalLayerMasked:
    r"""Constant (non-trainable) rhs/lhs Dot General (PyTorch version).

    Matches JAX's `lax.dot_general` semantics using `torch.einsum`.
    See NT docs for details on arguments and examples.
    """
    if rhs is not None and lhs is not None:
        raise ValueError(
            "At most one of constant `rhs` and `lhs` can be non-`None`, "
            "since the other factor is considered to be the layer `inputs`."
        )

    is_lhs = rhs is None
    other_val = lhs if is_lhs else rhs
    other = torch.as_tensor(other_val)

    # ----- helpers -----

    def _build_einsum(lhs_ndim: int, rhs_ndim: int):
        """Create an einsum equation for JAX-like dot_general.

        JAX result order: [batch dims] + [lhs non-contracting] + [rhs non-contracting]
        """
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

        # Normalize tuples (allow lists)
        lhs_contract = tuple(lhs_contract)
        rhs_contract = tuple(rhs_contract)
        lhs_batch = tuple(lhs_batch)
        rhs_batch = tuple(rhs_batch)

        # Unique labels generator (a..zA..Z)
        alphabet = list(string.ascii_lowercase + string.ascii_uppercase)
        if lhs_ndim + rhs_ndim > len(alphabet):
            raise ValueError("dot_general with >52 total dims not supported by einsum label set.")
        next_label_iter = iter(alphabet)

        # Start with unique labels per axis
        lhs_labels = [next(next_label_iter) for _ in range(lhs_ndim)]
        rhs_labels = [next(next_label_iter) for _ in range(rhs_ndim)]

        # Reuse labels for batch dims (must match and appear in output)
        for lbd, rbd in zip(lhs_batch, rhs_batch):
            lbl = lhs_labels[lbd]  # pick lhs label as canonical
            rhs_labels[rbd] = lbl

        # Reuse labels for contracting dims (must match and be reduced)
        for lcd, rcd in zip(lhs_contract, rhs_contract):
            lbl = lhs_labels[lcd]
            rhs_labels[rcd] = lbl

        # Build output label order: batch, lhs non-contracting, rhs non-contracting
        lhs_all_reduce = set(lhs_contract) | set(lhs_batch)
        rhs_all_reduce = set(rhs_contract) | set(rhs_batch)

        out_labels = []
        # batch first (in lhs order)
        out_labels += [lhs_labels[i] for i in lhs_batch]
        # lhs non-contracting
        out_labels += [lhs_labels[i] for i in range(lhs_ndim) if i not in lhs_all_reduce]
        # rhs non-contracting
        out_labels += [rhs_labels[i] for i in range(rhs_ndim) if i not in rhs_all_reduce]

        eq = f"{''.join(lhs_labels)},{''.join(rhs_labels)}->" + "".join(out_labels)
        return eq

    def _dot_general(lhs_tensor: torch.Tensor, rhs_tensor: torch.Tensor) -> torch.Tensor:
        eq = _build_einsum(lhs_tensor.ndim, rhs_tensor.ndim)
        return torch.einsum(eq, lhs_tensor, rhs_tensor)

    # Arrange args as (lhs, rhs) per JAX semantics.
    def dot_fn(x: torch.Tensor) -> torch.Tensor:
        a, b = (other.to(dtype=x.dtype, device=x.device), x) if is_lhs else (x, other.to(dtype=x.dtype, device=x.device))
        return _dot_general(a, b)

    # ----- stax-style API -----

    def init_fn(rng, input_shape: Tuple[int, ...]):
        # Infer output shape by running with a dummy tensor
        dummy = torch.empty(input_shape, dtype=other.dtype)
        out = dot_fn(dummy)
        return tuple(out.shape), ()

    def apply_fn(params, inputs: torch.Tensor, **kwargs):
        return dot_fn(inputs)

    # Kernel requirements identical to JAX version
    input_cs = dimension_numbers[0][1 if is_lhs else 0]
    diagonal_batch = (batch_axis not in input_cs) or (rhs is None and lhs is None)
    diagonal_spatial = Diagonal(
        input=Bool.YES
        if (input_cs in ((), (batch_axis,)) or (rhs is None and lhs is None))
        else Bool.NO
    )

    @requires(
        diagonal_batch=diagonal_batch,
        diagonal_spatial=diagonal_spatial,
        batch_axis=batch_axis,
        channel_axis=channel_axis,
    )
    def kernel_fn(k: Kernel, **kwargs) -> Kernel:
        # Delegate to Kernel helper which mirrors JAX semantics.
        return k.dot_general(other, other, is_lhs, dimension_numbers)

    def mask_fn(mask: torch.Tensor, input_shape: Tuple[int, ...]):
        # Follow NT convention: an output position is masked only if *all*
        # contributing inputs are masked. We invert, propagate with the linear
        # op, then invert back.
        mask_shape = list(input_shape)
        mask_shape[channel_axis] = mask.shape[channel_axis]
        broad = torch.broadcast_to(mask, mask_shape)
        inv = ~broad
        inv = inv.to(dtype=other.dtype if other.is_floating_point() else torch.float32)

        propagated = dot_fn(inv)
        # Any positive contribution => unmasked
        return ~(propagated > 0)

    return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=True)
def Aggregate(
    aggregate_axis: Optional[Axes] = None,
    batch_axis: int = 0,
    channel_axis: int = -1,
    to_dense: Optional[Callable[[torch.Tensor], torch.Tensor]] = lambda p: p,
    implementation: str = AggregateImplementation.DENSE.value,
) -> InternalLayer:
    """
    PyTorch port of the JAX Aggregate layer. See original docstring for details.
    """
    implementation = AggregateImplementation(implementation)
    if implementation == AggregateImplementation.SPARSE:
        warnings.warn(
            "Negative indices in `pattern` are considered padding (ignored)."
        )

    init_fn = lambda rng, input_shape: (input_shape, ())

    # ---------- helpers ----------

    def get_agg_axes(ndim: int) -> Tuple[Tuple[int, ...], int, int]:
        _batch_axis, _channel_axis = utils.mod((batch_axis, channel_axis), ndim)
        if aggregate_axis is None:
            agg_axes = tuple(i for i in range(ndim) if i not in (_batch_axis, _channel_axis))
        else:
            agg_axes = tuple(utils.canonicalize_axis(aggregate_axis, ndim))
        return agg_axes, _batch_axis, _channel_axis

    def get_dimension_numbers(ndim: int):
        # ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
        agg_axes, b_ax, _ = get_agg_axes(ndim)
        K = len(agg_axes)
        return (agg_axes, tuple(range(1, K + 1))), ((b_ax,), (0,))

    def _ravel_multi_index(idx: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Like jnp.ravel_multi_index with wrap=True; idx shape (..., K)."""
        K = idx.shape[-1]
        assert K == len(shape)
        # wrap
        mod = []
        for d in range(K):
            s = shape[d]
            mod.append(torch.remainder(idx[..., d], s))
        mod = torch.stack(mod, dim=-1)
        # any negative coord => mark as invalid (-1)
        invalid = (idx < 0).any(dim=-1)
        # compute strides
        strides = [1]
        for s in reversed(shape[1:]):
            strides.insert(0, strides[0] * s)
        strides = torch.tensor(strides, device=idx.device, dtype=mod.dtype)
        flat = (mod * strides).sum(dim=-1).to(torch.long)
        flat[invalid] = -1
        return flat

    def _get_senders_receivers(pattern: torch.Tensor, batch_size: int, agg_ndim: int):
        if pattern.shape[-1] != 2:
            raise ValueError(f"`pattern` must end with dim 2 (senders,receivers); got {pattern.shape[-1]}")
        s, r = pattern[..., 0], pattern[..., 1]
        # allow (B, E, 2) when K==1
        if agg_ndim == 1 and s.ndim == 2:
            s = s.unsqueeze(-1)
            r = r.unsqueeze(-1)
        if s.ndim != 3:
            raise ValueError(f"Tails and heads must be 3D; got {s.ndim}.")
        if s.shape[2] != agg_ndim:
            raise ValueError(f"Trailing dim must equal #aggregate axes ({agg_ndim}); got {s.shape[2]}.")
        if s.shape[0] != batch_size:
            raise ValueError(f"Leading dim must equal batch size; got {s.shape[0]} vs {batch_size}.")
        return s, r

    # ---------- apply_fn ----------

    def apply_fn(params,
                 inputs: torch.Tensor,
                 *,
                 pattern: Optional[torch.Tensor] = None,
                 **kwargs):
        if pattern is None:
            return inputs

        ndim = inputs.ndim
        agg_axes, b_ax, c_ax = get_agg_axes(ndim)
        K = len(agg_axes)

        if implementation == AggregateImplementation.DENSE:
            # Dense via batched matmul on flattened spatial dims.
            pattern_dense = to_dense(pattern)

            # Move to (B, *agg, C) then flatten agg -> V and get (B, C, V)
            src_axes = (b_ax,) + agg_axes + (c_ax,)
            tmp = torch.movedim(inputs, src_axes, (0,) + tuple(range(1, 1 + K)) + (-1,))
            B, *agg_shape, C = tmp.shape
            V = int(math.prod(agg_shape)) if agg_shape else 1
            x_bcv = tmp.reshape(B, V, C).transpose(1, 2)  # (B, C, V)

            # Pattern -> (B, V, V)
            pat = pattern_dense.to(dtype=inputs.dtype, device=inputs.device)
            pat_bvv = pat.reshape(B, V, V)

            out_bcv = torch.bmm(x_bcv, pat_bvv)  # (B, C, V)
            out = out_bcv.transpose(1, 2).reshape((B, *agg_shape, C))
            # Move back
            dst_axes = (b_ax,) + agg_axes + (c_ax,)
            out = torch.movedim(out, (0,) + tuple(range(1, 1 + K)) + (-1,), dst_axes)
            return out

        elif implementation == AggregateImplementation.SPARSE:
            # Sparse with index_add (segment_sum analogue).
            s, r = _get_senders_receivers(pattern, inputs.shape[b_ax], K)

            # Canonicalize axes to (B, *agg, C) -> reshape to (B, V, C)
            src_axes = (b_ax,) + agg_axes + (c_ax,)
            tmp = torch.movedim(inputs, src_axes, (0,) + tuple(range(1, 1 + K)) + (-1,))
            B, *agg_shape, C = tmp.shape
            V = int(math.prod(agg_shape)) if agg_shape else 1
            flat = tmp.reshape(B, V, C)

            # Ravel sender/receiver indices per batch
            s_flat = _ravel_multi_index(s, tuple(agg_shape))
            r_flat = _ravel_multi_index(r, tuple(agg_shape))

            out = torch.zeros_like(flat)
            for b in range(B):
                sb = s_flat[b].reshape(-1)  # (E,)
                rb = r_flat[b].reshape(-1)
                valid = (sb >= 0) & (rb >= 0)
                if valid.any():
                    sb = sb[valid]
                    rb = rb[valid]
                    msgs = flat[b, sb]  # (E', C)
                    out[b].index_add_(0, rb, msgs)
            # Restore shape and axes
            out = out.reshape((B, *agg_shape, C))
            out = torch.movedim(out, (0,) + tuple(range(1, 1 + K)) + (-1,), src_axes)
            return out

        else:
            raise ValueError(f"Unrecognized implementation == {implementation}.")

    # ---------- kernel_fn ----------

    @requires(
        batch_axis=batch_axis,
        channel_axis=channel_axis,
        diagonal_spatial=Diagonal(input=Bool.NO, output=Bool.NO),
    )
    def kernel_fn(
        k: Kernel,
        *,
        pattern: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        **kwargs,
    ):
        pattern1, pattern2 = pattern
        if pattern1 is None and pattern2 is None:
            return k
        if pattern1 is None or pattern2 is None:
            raise NotImplementedError(
                "Exactly one of pattern1/2 is None. Please provide both."
            )

        ndim = len(k.shape1)
        agg_axes, b_ax, c_ax = get_agg_axes(ndim)
        K = len(agg_axes)
        agg_shape = tuple(k.shape1[a] for a in agg_axes)
        agg_size = int(math.prod(agg_shape)) if agg_shape else 1

        def bucket_axes(_ndim: int, start_axis: int):
            """Bucket kernel axes into batch, aggregate, and non-aggregate."""
            ndim_spatial = (_ndim - start_axis) // 2
            agg_1 = tuple(a - int(b_ax < a) - int(c_ax < a) + start_axis for a in agg_axes)
            agg_2 = tuple(a + ndim_spatial for a in agg_1)
            non_agg_1 = tuple(a for a in range(start_axis, start_axis + ndim_spatial) if a not in agg_1)
            non_agg_2 = tuple(a for a in range(start_axis + ndim_spatial, _ndim) if a not in agg_2)
            return tuple(range(start_axis)), agg_1, agg_2, non_agg_1, non_agg_2

        if implementation == AggregateImplementation.DENSE:
            p1 = None if pattern1 is None else to_dense(pattern1)
            p2 = None if pattern2 is None else to_dense(pattern2)

            k2 = k.dot_general(
                other1=p1,
                other2=p2,
                is_lhs=False,
                dimension_numbers=get_dimension_numbers(ndim),
            )

            def _transpose(t: Optional[torch.Tensor], diagonal_batch: bool):
                if t is None or (hasattr(t, "ndim") and t.ndim == 0):
                    return t
                start_axis = 1 if diagonal_batch else 2
                t = utils.unzip_axes(t, start_axis)
                b, agg_1, agg_2, non_agg_1, non_agg_2 = bucket_axes(t.ndim, start_axis)
                permutation = b + non_agg_1 + agg_1 + non_agg_2 + agg_2
                # invert permutation
                inv = [0] * len(permutation)
                for i, p in enumerate(permutation):
                    inv[p] = i
                t = t.permute(inv)
                return utils.zip_axes(t, start_axis)

            return k2.replace(
                cov1=_transpose(k2.cov1, k2.diagonal_batch),
                cov2=_transpose(k2.cov2, k2.diagonal_batch),
                nngp=_transpose(k2.nngp, False),
                ntk=_transpose(k2.ntk, False),
                batch_axis=b_ax % ndim,
                channel_axis=c_ax % ndim,
            )

        elif implementation == AggregateImplementation.SPARSE:
            # Sparse pass using index_add twice (left and right aggregations).
            def _pass_messages(s1, s2, r1, r2, mat: torch.Tensor):
                v1, v2 = mat.shape[:2]

                def _send(data: torch.Tensor, r: torch.Tensor, num_segments: int):
                    out = torch.zeros((num_segments, *data.shape[1:]), dtype=data.dtype, device=data.device)
                    valid = (r >= 0)
                    if valid.any():
                        out.index_add_(0, r[valid], data[valid])
                    return out

                # Select senders along both dimensions
                x = mat[s1[:, None], s2[None, :]]
                # First sum along second axis receivers
                # We need to apply send along axis 1 for each row of axis 0
                tmp = torch.stack([_send(x[i], r2, v2) for i in range(x.shape[0])], dim=0)
                # Then sum along first axis receivers
                res = _send(tmp, r1, v1)
                return res

            def _vm_pass_self(mat, s, r):
                return _pass_messages(s, s, r, r, mat)

            def _vm_pass_cross(mat, s1, r1, s2, r2):
                # vectorize across batch dims if present
                return _pass_messages(s1, s2, r1, r2, mat)

            def _prepare_sr(pat, shape_batch, _K):
                s, r = _get_senders_receivers(pat, shape_batch, _K)
                s = _ravel_multi_index(s, agg_shape)
                r = _ravel_multi_index(r, agg_shape)
                return s, r

            s1, r1 = _prepare_sr(pattern1, k.shape1[b_ax], K)
            s2, r2 = _prepare_sr(pattern2, k.shape2[b_ax], K)

            def agg_tensor(t: Optional[torch.Tensor], diagonal_batch: bool, s1, r1, s2, r2):
                if t is None or (hasattr(t, "ndim") and t.ndim == 0):
                    return t
                start_axis = 1 if diagonal_batch else 2
                t = utils.unzip_axes(t, start_axis)
                b, agg_1, agg_2, non_agg_1, non_agg_2 = bucket_axes(t.ndim, start_axis)
                perm = b + agg_1 + agg_2 + non_agg_1 + non_agg_2
                t = t.permute(perm)
                t_shape = t.shape
                t = t.reshape(t.shape[:start_axis] + (agg_size, agg_size) + t.shape[start_axis + 2 * len(agg_axes):])
                # apply message passing
                if diagonal_batch:
                    t = _vm_pass_self(t, s1, r1)
                else:
                    t = _vm_pass_cross(t, s1, r1, s2, r2)
                t = t.reshape(t_shape)
                # invert permutation
                inv = [0] * len(perm)
                for i, p in enumerate(perm):
                    inv[p] = i
                t = t.permute(inv)
                return utils.zip_axes(t, start_axis)

            nngp = agg_tensor(k.nngp, False, s1, r1, s2, r2)
            ntk  = agg_tensor(k.ntk,  False, s1, r1, s2, r2)
            cov1 = agg_tensor(k.cov1, k.diagonal_batch, s1, r1, s1, r1)
            cov2 = agg_tensor(k.cov2, k.diagonal_batch, s2, r2, s2, r2)
            return k.replace(nngp=nngp, ntk=ntk, cov1=cov1, cov2=cov2)

        else:
            raise ValueError(f"Unrecognized implementation == {implementation}.")

    return init_fn, apply_fn, kernel_fn


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
    PyTorch port of the JAX Dense layer (fully-connected / linear).
    Follows the same API and kernel transforms as the original.
    """
    parameterization = parameterization.lower()

    # ---- init fns ----
    def _init_fn(rng, input_shape, out_dim_):
        _cax = channel_axis % len(input_shape)
        output_shape = input_shape[:_cax] + (out_dim_,) + input_shape[_cax + 1 :]

        in_dim = input_shape[_cax]
        W = torch.randn(in_dim, out_dim_)
        if b_std is None:
            b = None
        else:
            b_shape = [1] * len(input_shape)
            b_shape[channel_axis] = out_dim_
            b = torch.randn(*b_shape)
        return output_shape, (W, b)

    def ntk_init_fn(rng, input_shape):
        return _init_fn(rng, input_shape, out_dim)

    def standard_init_fn(rng, input_shape):
        output_shape, (W, b) = _init_fn(rng, input_shape, out_dim * s[1])
        in_dim_scaled = input_shape[channel_axis] / s[0]
        W *= W_std / (in_dim_scaled ** 0.5)
        b = None if b is None else b * b_std
        return output_shape, (W, b)

    if parameterization == "ntk":
        init_fn = ntk_init_fn
    elif parameterization == "standard":
        init_fn = standard_init_fn
    else:
        raise ValueError(f"Parameterization not supported: {parameterization}")

    # ---- apply fn ----
    def apply_fn(params, inputs: torch.Tensor, **kwargs):
        W, b = params
        # prod has shape like inputs but with channel dim replaced by out_dim.
        prod = torch.tensordot(W, inputs, dims=([0], [channel_axis]))
        prod = torch.movedim(prod, 0, channel_axis)

        if parameterization == "ntk":
            norm = W_std / (inputs.shape[channel_axis] ** 0.5)
            outputs = norm * prod
            if b is not None:
                outputs = outputs + b_std * b
        elif parameterization == "standard":
            outputs = prod / (s[0] ** 0.5)
            if b is not None:
                outputs = outputs + b
        else:
            raise ValueError(f"Parameterization not supported: {parameterization}")

        return outputs

    # ---- kernel fn helpers ----
    def _affine_kernel(x, w_std: float, b_std_opt: Optional[float]):
        if x is None:
            return None
        out = (w_std ** 2) * x
        if b_std_opt is not None:
            out = out + (b_std_opt ** 2)
        return out

    @requires(batch_axis=batch_axis, channel_axis=channel_axis, diagonal_spatial=Diagonal())
    def kernel_fn(k: Kernel, **kwargs):
        """Transform kernels through a Dense layer."""
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

        if parameterization == "ntk":
            cov1 = _affine_kernel(cov1, W_std, b_std)
            nngp = _affine_kernel(nngp, W_std, b_std)
            cov2 = _affine_kernel(cov2, W_std, b_std)
            if ntk is not None:
                ntk = nngp + (W_std ** 2) * ntk

        elif parameterization == "standard":
            input_width = k.shape1[channel_axis] / s[0]
            if ntk is not None:
                ntk = input_width * nngp + (W_std ** 2) * ntk
                if b_std is not None:
                    ntk = ntk + 1.0
            cov1 = _affine_kernel(cov1, W_std, b_std)
            nngp = _affine_kernel(nngp, W_std, b_std)
            cov2 = _affine_kernel(cov2, W_std, b_std)

        return k.replace(
            cov1=cov1,
            nngp=nngp,
            cov2=cov2,
            ntk=ntk,
            is_gaussian=True,
            is_input=False,
        )

    def mask_fn(mask: torch.Tensor, input_shape):
        # Collapse channel axis: all True -> keep channel; broadcast shape.
        return torch.all(mask, dim=channel_axis, keepdim=True)

    return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=True)
def Conv(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    dimension_numbers: Optional[tuple[str, str, str]] = None,
    parameterization: str = "ntk",
    s: tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
    """
    PyTorch port: general convolution wrapper (shared weights).
    Delegates to `_Conv` with `transpose=False`, `shared_weights=True`.
    """
    return _Conv(
        out_chan=out_chan,
        filter_shape=filter_shape,
        strides=strides,
        padding=padding,
        W_std=W_std,
        b_std=b_std,
        dimension_numbers=dimension_numbers,
        parameterization=parameterization,
        s=s,
        transpose=False,
        shared_weights=True,
    )

@layer
@supports_masking(remask_kernel=True)
def ConvTranspose(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    dimension_numbers: Optional[tuple[str, str, str]] = None,
    parameterization: str = "ntk",
    s: tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
    """
    PyTorch port: general transpose convolution (shared weights).
    Delegates to `_Conv` with `transpose=True`, `shared_weights=True`.
    """
    return _Conv(
        out_chan=out_chan,
        filter_shape=filter_shape,
        strides=strides,
        padding=padding,
        W_std=W_std,
        b_std=b_std,
        dimension_numbers=dimension_numbers,
        parameterization=parameterization,
        s=s,
        transpose=True,
        shared_weights=True,
    )


@layer
@supports_masking(remask_kernel=True)
def ConvLocal(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    dimension_numbers: Optional[tuple[str, str, str]] = None,
    parameterization: str = "ntk",
    s: tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
    """
    PyTorch port: locally-connected (unshared) convolution.
    Delegates to `_Conv` with `transpose=False`, `shared_weights=False`.
    """
    return _Conv(
        out_chan=out_chan,
        filter_shape=filter_shape,
        strides=strides,
        padding=padding,
        W_std=W_std,
        b_std=b_std,
        dimension_numbers=dimension_numbers,
        parameterization=parameterization,
        s=s,
        transpose=False,
        shared_weights=False,
    )


@layer
@supports_masking(remask_kernel=True)
def _Conv(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]],
    padding: str,
    W_std: float,
    b_std: Optional[float],
    dimension_numbers: Optional[tuple[str, str, str]],
    parameterization: str,
    s: tuple[int, int],
    transpose: bool,
    shared_weights: bool,
) -> InternalLayerMasked:
    """
    PyTorch port of the JAX `_Conv` building block.

    Notes / scope:
    - Supports standard conv/transposed-conv with shared weights in 1D/2D/3D.
    - Supports *locally-connected* (unshared) conv in 2D only (via unfold/fold).
      For other spatial ranks with unshared weights, raises NotImplementedError.
    - `dimension_numbers` must be a tuple like ('NCHW', 'OIHW', 'NCHW')
      (1D/3D analogues OK: 'NCL'/'OIL' and 'NCDHW'/'OIDHW'). If omitted, it
      defaults to canonical PyTorch layout for the given rank.
    - `"CIRCULAR"` padding is implemented using `F.pad(..., mode='circular')`.
    """
    parameterization = parameterization.lower()

    # ---------- helpers ----------
    def _default_dns(nd: int) -> tuple[str, str, str]:
        if nd == 1:
            return ("NCL", "OIL", "NCL")
        if nd == 2:
            return ("NCHW", "OIHW", "NCHW")
        if nd == 3:
            return ("NCDHW", "OIDHW", "NCDHW")
        raise ValueError(f"Unsupported spatial rank: {nd}")

    nd = len(filter_shape)
    if dimension_numbers is None:
        dimension_numbers = _default_dns(nd)

    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * nd
    strides = tuple(strides) if strides is not None else one

    pad_enum = Padding(padding)

    def _axis(spec: str, ch: str) -> int:
        return spec.index(ch)

    def _move_to(spec_src: str, spec_dst: str, x: torch.Tensor) -> torch.Tensor:
        # Permute x from spec_src to spec_dst (both strings of same length)
        perm = [spec_src.index(c) for c in spec_dst]
        return x.permute(*perm).contiguous()

    def _torch_padding_for_same(x_sp: Sequence[int]) -> Tuple[int, ...]:
        # PyTorch pad order is (w_right, w_left, h_right, h_left, ...)
        pads = []
        for k, s in zip(filter_shape[::-1], strides[::-1]):
            total = max(k - s, 0)
            left = total // 2
            right = total - left
            pads.extend([right, left])
        return tuple(pads)

    def _apply_circular_pad(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, _torch_padding_for_same(x.shape), mode="circular")

    def _fan_in(in_ch: int) -> int:
        fi = in_ch
        for k in filter_shape:
            fi *= k
        return fi

    # ---------- init ----------
    def _init_shared(rng, input_shape):
        # input_shape is a tuple in lhs_spec order
        in_ch = input_shape[_axis(lhs_spec, "C")]
        out_ch_param = out_chan if parameterization == "ntk" else out_chan * s[1]

        # weight shapes follow PyTorch:
        # conv:        (out_ch, in_ch, *filter_shape)
        # conv_trans:  (in_ch, out_ch, *filter_shape)
        if not transpose:
            W_shape = (out_ch_param, in_ch, *filter_shape)
        else:
            W_shape = (in_ch, out_ch_param, *filter_shape)

        # bias broadcast shape (1,C,*1s) in out_spec layout
        b_shape = None if b_std is None else tuple(
            1 if c != "C" else out_ch_param for c in out_spec
        )

        # sample params ~ N(0,1) like JAX code; scaling applied in apply_fn
        g1 = torch.Generator()
        g1.manual_seed(int(utils.as_seed(rng)))
        W = torch.randn(*W_shape, generator=g1)

        if b_std is None:
            b = None
        else:
            g2 = torch.Generator()
            g2.manual_seed(int(utils.as_seed(rng)) + 1)
            b = torch.randn(*b_shape, generator=g2)

        # compute output shape by running a dummy forward with zeros
        dummy = torch.zeros(*input_shape)
        out = _apply_shared((W, b), dummy)

        return tuple(out.shape), (W, b)

    def _init_unshared(rng, input_shape):
        # Only 2D is implemented here.
        if nd != 2:
            raise NotImplementedError("Locally-connected is only implemented for 2D.")

        N_axis = _axis(lhs_spec, "N")
        C_axis = _axis(lhs_spec, "C")
        H_axis, W_axis = [lhs_spec.index(c) for c in lhs_spec if c not in ("N", "C")]

        in_ch = input_shape[C_axis]
        H, W = input_shape[H_axis], input_shape[W_axis]
        stride_h, stride_w = strides
        kh, kw = filter_shape

        # output spatial dims for VALID/SAME (circular treated like SAME at runtime)
        if pad_enum == Padding.VALID:
            out_h = (H - kh) // stride_h + 1
            out_w = (W - kw) // stride_w + 1
        else:  # SAME or CIRCULAR
            out_h = (H + stride_h - 1) // stride_h
            out_w = (W + stride_w - 1) // stride_w

        out_ch_param = out_chan if parameterization == "ntk" else out_chan * s[1]

        # For locally-connected, weights depend on spatial position.
        # We'll store as (out_ch, out_h, out_w, in_ch*kh*kw)
        g = torch.Generator()
        g.manual_seed(int(utils.as_seed(rng)))
        W = torch.randn(out_ch_param, out_h, out_w, in_ch * kh * kw, generator=g)

        b = None
        if b_std is not None:
            b_shape = []
            for c in out_spec:
                if c == "C":
                    b_shape.append(out_ch_param)
                elif c in ("N",):
                    b_shape.append(1)
                else:
                    b_shape.append(out_h if c == out_spec[out_spec.index(c)] else out_w)
            # out_spec is typically NCHW; for general specs we’ll broadcast later.
            b = torch.randn(*b_shape)

        # Output shape matches apply on zeros.
        dummy = torch.zeros(*input_shape)
        out = _apply_unshared((W, b), dummy, out_h, out_w)
        return tuple(out.shape), (W, b)

    # ---------- apply ----------
    def _same_or_circular_pre(x: torch.Tensor) -> torch.Tensor:
        if pad_enum == Padding.CIRCULAR and not transpose:
            # circular pad to emulate SAME with wrap-around
            pads = _torch_padding_for_same(x.shape)
            return F.pad(x, pads, mode="circular")
        return x

    def _shared_conv_call(x: torch.Tensor, W: torch.Tensor, bias: Optional[torch.Tensor]):
        if nd == 1:
            fn_fwd, fn_t = F.conv1d, F.conv_transpose1d
        elif nd == 2:
            fn_fwd, fn_t = F.conv2d, F.conv_transpose2d
        elif nd == 3:
            fn_fwd, fn_t = F.conv3d, F.conv_transpose3d
        else:
            raise ValueError("Unsupported spatial rank.")

        if not transpose:
            return fn_fwd(x, W, bias=None, stride=strides,
                          padding=0 if pad_enum in (Padding.SAME, Padding.CIRCULAR) else 0)
        else:
            return fn_t(x, W, bias=None, stride=strides, padding=0,
                        output_padding=0)

    def _apply_shared(params, inputs):
        W, b = params
        # Reorder inputs to canonical N C ... per lhs_spec
        x = inputs
        if lhs_spec != _default_dns(nd)[0]:
            x = _move_to(lhs_spec, _default_dns(nd)[0], x)

        x = _same_or_circular_pre(x)

        # scaling
        if parameterization == "ntk":
            norm = W_std / (_fan_in(x.shape[1]) ** 0.5)
            b_scale = b_std
        else:
            norm = 1.0 / (s[0] ** 0.5)
            b_scale = 1.0

        y = _shared_conv_call(x, W, None)
        y = y * norm

        if b is not None:
            # bring b to N C ... (broadcast-friendly)
            b_bc = b
            if out_spec != _default_dns(nd)[2]:
                b_bc = _move_to(out_spec, _default_dns(nd)[2], b_bc)
            y = y + b_scale * b_bc

        # For transposed conv with "CIRCULAR", crop like SAME (handled by caller utils if present)
        if pad_enum == Padding.CIRCULAR and transpose:
            # Keep as-is; rely on higher-level helper if available in your utils.
            pass

        # Move back to requested out_spec
        if out_spec != _default_dns(nd)[2]:
            y = _move_to(_default_dns(nd)[2], out_spec, y)
        return y

    def _apply_unshared(params, inputs, out_h=None, out_w=None):
        # 2D locally-connected via unfold/fold
        W, b = params
        if nd != 2:
            raise NotImplementedError("Locally-connected is only implemented for 2D.")

        x = inputs
        if lhs_spec != "NCHW":
            x = _move_to(lhs_spec, "NCHW", x)

        if pad_enum == Padding.CIRCULAR:
            x = F.pad(x, _torch_padding_for_same(x.shape), mode="circular")

        kh, kw = filter_shape
        sh, sw = strides
        # For SAME, use padding that preserves ceil(H/stride)
        pad = 0 if pad_enum == Padding.VALID else (kh // 2, kw // 2)
        cols = F.unfold(x, kernel_size=(kh, kw), stride=(sh, sw), padding=pad)
        # cols: (N, C*kh*kw, L) where L = out_h*out_w
        N, CK, L = cols.shape

        if out_h is None or out_w is None:
            # infer
            out_h = int(math.sqrt(L))
            out_w = L // out_h

        # W: (out_c, out_h, out_w, CK)
        Wv = W.view(out_chan if parameterization == "ntk" else out_chan * s[1],
                    out_h * out_w, CK)  # (Cout, L, CK)
        # scale
        if parameterization == "ntk":
            norm = W_std / (_fan_in(CK // (kh * kw)) ** 0.5)
            b_scale = b_std
        else:
            norm = 1.0 / (s[0] ** 0.5)
            b_scale = 1.0

        # per-location matmul: (N, L, CK) x (Cout, L, CK)^T over CK
        colsT = cols.transpose(1, 2)  # (N, L, CK)
        y = torch.einsum("nlk,clk->ncl", colsT, Wv)  # (N, Cout, L)
        y = y * norm

        if b is not None:
            bb = b
            if out_spec != "NCHW":
                bb = _move_to(out_spec, "NCHW", bb)
            bb = bb.view(1, bb.shape[1], -1)  # (1, Cout, L)
            y = y + b_scale * bb

        y = y.view(y.shape[0], y.shape[1], out_h, out_w)  # (N, C, H, W)

        if out_spec != "NCHW":
            y = _move_to("NCHW", out_spec, y)
        return y

    # ---------- public init_fn / apply_fn ----------
    def init_fn(rng, input_shape):
        return (_init_shared if shared_weights else _init_unshared)(rng, input_shape)

    def apply_fn(params, inputs, **kwargs):
        if shared_weights:
            return _apply_shared(params, inputs)
        else:
            # 2D locally connected only
            return _apply_unshared(params, inputs)

    # ---------- kernel_fn (stays conceptually identical; relies on your utils) ----------
    @requires(batch_axis=lhs_spec.index("N"),
              channel_axis=lhs_spec.index("C"),
              diagonal_spatial=Diagonal(output=Bool.NO if shared_weights else Bool.MAYBE))
    def kernel_fn(k: Kernel, **kwargs):
        """
        Mirrors the JAX logic, delegating spatial covariance transforms to the
        existing helper routines in your port:
          - _conv_kernel_diagonal_spatial / _transpose
          - _conv_kernel_full_spatial_shared / _unshared / _transpose
        The surrounding algebra is unchanged; only JAX ops were replaced earlier.
        """
        # This block assumes your project already has these helpers ported:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        is_rev = k.is_reversed

        input_spec = tuple(c for c in lhs_spec if c not in ("N", "C"))
        conv_spec = tuple(c for c in rhs_spec if c not in ("I", "O"))
        perm = tuple(conv_spec.index(c) for c in input_spec)

        fshape_k = tuple(filter_shape[p] for p in perm)
        strides_k = tuple(strides[p] for p in perm)

        if k.diagonal_spatial:
            conv_kernel = _conv_kernel_diagonal_spatial_transpose if transpose else _conv_kernel_diagonal_spatial
        else:
            if shared_weights:
                if is_rev:
                    fshape_k = fshape_k[::-1]
                    strides_k = strides_k[::-1]
                is_rev = not is_rev
            if transpose:
                conv_kernel = _conv_kernel_full_spatial_transpose
            else:
                conv_kernel = _conv_kernel_full_spatial_shared if shared_weights else _conv_kernel_full_spatial_unshared

        def conv_unscaled(lhs, bn):
            if lhs is None:
                return None
            return conv_kernel(lhs, fshape_k, strides_k, pad_enum, bn)

        def affine(out, scale, shift, bn):
            if out is None:
                return None
            out = out * scale
            if shift is not None:
                if k.diagonal_spatial or shared_weights:
                    out = out + shift
                else:
                    # index-add along paired spatial axes (same as JAX version)
                    idx = (Ellipsis,)
                    for i in range(bn, out.ndim, 2):
                        size = out.shape[i]
                        shape = [1] * out.ndim
                        shape[i] = size
                        ar = torch.arange(size).reshape(shape)
                        idx += (ar,) * 2
                    out = out.index_put(idx, shift, accumulate=True)
            return out

        b_sq = None if b_std is None else (b_std ** 2)

        def conv_aff(lhs, bn):
            x = conv_unscaled(lhs, bn)
            return affine(x, W_std ** 2, b_sq, bn)

        cov1 = conv_aff(cov1, 1 if k.diagonal_batch else 2)
        cov2 = conv_aff(cov2, 1 if k.diagonal_batch else 2)

        if parameterization == "ntk":
            nngp_conv = conv_unscaled(nngp, 2)
            nngp = affine(nngp_conv, W_std ** 2, b_sq, 2)
            if ntk is not None:
                ntk = W_std ** 2 * conv_unscaled(ntk, 2) + nngp
        else:  # "standard"
            nngp_unscaled = conv_unscaled(nngp, 2)
            if ntk is not None:
                fin = _fan_in(k.shape1[lhs_spec.index("C")]) / s[0]
                ntk = fin * nngp_unscaled + W_std ** 2 * conv_unscaled(ntk, 2)
                if b_std is not None:
                    ntk = affine(ntk, 1.0, 1.0, 2)
            nngp = affine(nngp_unscaled, W_std ** 2, b_sq, 2)

        res = k.replace(
            cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk,
            is_gaussian=True, is_reversed=is_rev,
            batch_axis=out_spec.index("N"),
            channel_axis=out_spec.index("C"),
            is_input=False,
        )

        # reorder output spatial dims to match finite layer permutation
        out_spec_kernel = tuple(c for c in out_spec if c not in ("N", "C"))
        in_to_out = tuple(out_spec_kernel.index(c) for c in input_spec)
        res = res.transpose(in_to_out)
        return res

    # ---------- mask_fn ----------
    def mask_fn(mask: torch.Tensor, input_shape):
        b_ax, c_ax = lhs_spec.index("N"), lhs_spec.index("C")
        # collapse channel dim
        mask = torch.all(mask.bool(), dim=c_ax, keepdim=True)

        if transpose:
            if not shared_weights:
                raise NotImplementedError("Transpose + unshared mask not implemented.")
            # 1x kernel filled with ones to propagate mask as convT
            rhs_shape = list(filter_shape)
            for c in ("O", "I"):
                rhs_shape.insert(rhs_spec.index(c), 1)
            Wones = torch.ones(rhs_shape, dtype=mask.dtype, device=mask.device)

            # reorder mask to canonical N C ...
            ms = mask
            if lhs_spec != _default_dns(nd)[0]:
                ms = _move_to(lhs_spec, _default_dns(nd)[0], ms)
            out = _shared_conv_call(ms, Wones, None)
            # back to out_spec
            if out_spec != _default_dns(nd)[2]:
                out = _move_to(_default_dns(nd)[2], out_spec, out)
            return out.bool()

        else:
            pooled = _pool_mask(mask, filter_shape, strides, pad_enum, b_ax, c_ax)
            # transpose to match out_spec order
            perm = tuple(out_spec.index(c) for c in lhs_spec)
            return pooled.permute(*perm).contiguous().bool()

    return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=True)
def AvgPool(
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    normalize_edges: bool = False,
    batch_axis: int = 0,
    channel_axis: int = -1,
) -> InternalLayerMasked:
  """Average pooling.

  Based on :obj:`jax.example_libraries.stax.AvgPool`.

  Args:
    window_shape:
      The number of pixels over which pooling is to be performed.

    strides:
      The stride of the pooling window. `None` corresponds to a stride of
      `(1, 1)`.

    padding:
      Can be `VALID`, `SAME`, or `CIRCULAR` padding. Here `CIRCULAR` uses
      periodic boundary conditions on the image.

    normalize_edges:
      `True` to normalize output by the effective receptive field, `False` to
      normalize by the window size. Only has effect at the edges when `SAME`
      padding is used. Set to `True` to retain correspondence to
      :obj:`jax.example_libraries.stax.AvgPool`.

    batch_axis:
      Specifies the batch dimension. Defaults to `0`, the leading axis.

    channel_axis:
      Specifies the channel / feature dimension. Defaults to `-1`, the trailing
      axis. For `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _Pool(_Pooling.AVG, window_shape, strides, padding, normalize_edges,
               batch_axis, channel_axis)

@layer
@supports_masking(remask_kernel=True)
def SumPool(
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    batch_axis: int = 0,
    channel_axis: int = -1,
) -> InternalLayerMasked:
  """Sum pooling.

  Based on :obj:`jax.example_libraries.stax.SumPool`.

  Args:
    window_shape:
      The number of pixels over which pooling is to be performed.

    strides:
      The stride of the pooling window. `None` corresponds to a stride of
      `(1, ..., 1)`.

    padding:
      Can be `VALID`, `SAME`, or `CIRCULAR` padding. Here `CIRCULAR` uses
      periodic boundary conditions on the image.

    batch_axis:
      Specifies the batch dimension. Defaults to `0`, the leading axis.

    channel_axis:
      Specifies the channel / feature dimension. Defaults to `-1`, the trailing
      axis. For `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _Pool(_Pooling.SUM, window_shape, strides, padding, False,
               batch_axis, channel_axis)


@layer
@supports_masking(remask_kernel=True)
def _Pool(
    pool_type: _Pooling,
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]],
    padding: str,
    normalize_edges: bool,
    batch_axis: int,
    channel_axis: int,
) -> InternalLayerMasked:
    """
    PyTorch port of the generic pooling layer. Returns
    (init_fn, apply_fn, kernel_fn, mask_fn).

    Supports:
      - AVG or SUM pooling over all non-batch/non-channel spatial dims.
      - padding: "VALID", "SAME", "CIRCULAR".
      - `normalize_edges` for AVG with SAME padding (effective receptive field).
      - Arbitrary `batch_axis` and `channel_axis`.
    """
    padding = Padding(padding)
    ws = tuple(int(k) for k in window_shape)
    nd = len(ws)
    one = (1,) * nd
    strides = tuple(int(s) for s in (strides or one))

    if pool_type not in (_Pooling.AVG, _Pooling.SUM):
        raise ValueError(f"Invalid pooling type {pool_type}")

    # -------- axis helpers --------
    def _canonicalize(x: torch.Tensor) -> torch.Tensor:
        # Move (batch_axis, channel_axis) -> (N, C) leading.
        b = batch_axis % x.ndim
        c = channel_axis % x.ndim
        axes = list(range(x.ndim))
        spatial = [i for i in axes if i not in (b, c)]
        perm = [b, c] + spatial
        return x.permute(*perm).contiguous()

    def _restore(x: torch.Tensor, ndim_out: int) -> torch.Tensor:
        # Inverse of _canonicalize.
        b = batch_axis % ndim_out
        c = channel_axis % ndim_out
        axes = list(range(ndim_out))
        spatial = [i for i in axes if i not in (b, c)]
        canonical = [b, c] + spatial
        inv = [0] * len(canonical)
        for i, j in enumerate(canonical):
            inv[j] = i
        return x.permute(*inv).contiguous()

    # -------- SAME/CIRCULAR pad helpers --------
    def _same_pad_per_dim(L: int, k: int, s: int) -> tuple[int, int]:
        # TF SAME: out = ceil(L/s); total_pad = max((out-1)*s + k - L, 0)
        out = (L + s - 1) // s
        total = max((out - 1) * s + k - L, 0)
        left = total // 2
        right = total - left
        return left, right

    def _build_pad(x: torch.Tensor) -> tuple[int, ...]:
        # torch.nn.functional.pad expects pairs from last dim backwards:
        # (w_right, w_left, h_right, h_left, ...)
        pads = []
        # x is in N, C, *spatial
        for off, (k, s) in enumerate(zip(ws[::-1], strides[::-1])):
            L = x.shape[-1 - off]
            l, r = _same_pad_per_dim(L, k, s)
            pads.extend([r, l])
        return tuple(pads)

    # -------- init --------
    def init_fn(rng, input_shape):
        dummy = torch.zeros(*input_shape)
        out = apply_fn((), dummy)
        return tuple(out.shape), ()

    # -------- pooling core (apply) --------
    def _pool_sum_nd(x: torch.Tensor) -> torch.Tensor:
        # x is (N, C, *spatial). Produce SUM over window, honoring padding/stride.
        if padding == Padding.VALID:
            x_pad = x
        elif padding in (Padding.SAME, Padding.CIRCULAR):
            pads = _build_pad(x)
            mode = "circular" if padding == Padding.CIRCULAR else "constant"
            x_pad = F.pad(x, pads, mode=mode, value=0.0 if mode == "constant" else 0)
        else:
            raise ValueError(f"Unsupported padding: {padding}")

        if nd == 1:
            y_sum = F.avg_pool1d(x_pad, kernel_size=ws, stride=strides, padding=0, divisor_override=1)
        elif nd == 2:
            y_sum = F.avg_pool2d(x_pad, kernel_size=ws, stride=strides, padding=0, divisor_override=1)
        elif nd == 3:
            y_sum = F.avg_pool3d(x_pad, kernel_size=ws, stride=strides, padding=0, divisor_override=1)
        else:
            raise ValueError(f"Unsupported spatial rank: {nd}")
        return y_sum

    def _pool_avg_nd(x: torch.Tensor) -> torch.Tensor:
        y_sum = _pool_sum_nd(x)
        if normalize_edges and padding == Padding.SAME:
            # Effective-count normalization at edges.
            ones = torch.ones_like(x)
            pads = _build_pad(x)
            ones = F.pad(ones, pads, mode="constant", value=0.0)
            if nd == 1:
                counts = F.avg_pool1d(ones, kernel_size=ws, stride=strides, padding=0, divisor_override=1)
            elif nd == 2:
                counts = F.avg_pool2d(ones, kernel_size=ws, stride=strides, padding=0, divisor_override=1)
            else:
                counts = F.avg_pool3d(ones, kernel_size=ws, stride=strides, padding=0, divisor_override=1)
            return y_sum / torch.clamp(counts, min=1.0)
        else:
            denom = 1
            for ksz in ws:
                denom *= ksz
            return y_sum / float(denom)

    def apply_fn(params, inputs, **kwargs):
        del params
        x = _canonicalize(inputs)  # N, C, *spatial
        if pool_type == _Pooling.SUM:
            y = _pool_sum_nd(x)
        else:
            y = _pool_avg_nd(x)
        return _restore(y, inputs.ndim)

    # -------- kernel --------
    @requires(batch_axis=batch_axis,
              channel_axis=channel_axis,
              diagonal_spatial=Diagonal(input=Bool.MAYBE))
    def kernel_fn(k: Kernel, **kwargs) -> Kernel:
        """
        Kernel transformation for pooling. Uses project helpers:
        `_pool_kernel` and `_normalize` to match NTK behavior.
        """
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

        if k.diagonal_spatial:
            window_shape_kernel = ws
            strides_kernel = strides
        else:
            window_shape_kernel = utils.double_tuple(ws[::(-1 if k.is_reversed else 1)])
            strides_kernel = utils.double_tuple(strides[::(-1 if k.is_reversed else 1)])

        def pool(mat, batch_ndim):
            if mat is None or getattr(mat, "ndim", 0) == 0:
                return mat
            out = _pool_kernel(mat,
                               pool_type,
                               window_shape_kernel,
                               strides_kernel,
                               padding,
                               normalize_edges,
                               batch_ndim)
            if k.diagonal_spatial and pool_type == _Pooling.AVG:
                # Match edge normalization semantics for diagonal kernels.
                _window_shape = (1,) * batch_ndim + tuple(ws)
                _strides = (1,) * batch_ndim + tuple(strides)
                out = _normalize(mat, out, normalize_edges, padding, _strides, _window_shape)
            return out

        nngp = pool(nngp, 2)
        ntk = pool(ntk, 2)
        cov1 = pool(cov1, 1 if k.diagonal_batch else 2)
        cov2 = pool(cov2, 1 if k.diagonal_batch else 2)
        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    # -------- mask --------
    def mask_fn(mask, input_shape):
        _check_is_implemented(mask, channel_axis)
        return _pool_mask(mask,
                          window_shape=ws,
                          strides=strides,
                          padding=padding,
                          batch_axis=batch_axis,
                          channel_axis=channel_axis)

    return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def GlobalSumPool(
    batch_axis: int = 0,
    channel_axis: int = -1,
) -> InternalLayerMasked:
  """Global sum pooling.

  Sums over and removes (`keepdims=False`) all spatial dimensions, preserving
  the order of batch and channel axes.

  Args:
    batch_axis:
      Specifies the batch dimension. Defaults to `0`, the leading axis.

    channel_axis:
      Specifies the channel / feature dimension. Defaults to `-1`, the trailing
      axis. For `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _GlobalPool(_Pooling.SUM, batch_axis, channel_axis)


@layer
@supports_masking(remask_kernel=False)
def GlobalAvgPool(
    batch_axis: int = 0,
    channel_axis: int = -1,
) -> InternalLayerMasked:
  """Global average pooling.

  Averages over and removes (`keepdims=False`) all spatial dimensions,
  preserving the order of batch and channel axes.

  Args:
    batch_axis:
      Specifies the batch dimension. Defaults to `0`, the leading axis.

    channel_axis:
      Specifies the channel / feature dimension. Defaults to `-1`, the trailing
      axis. For `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _GlobalPool(_Pooling.AVG, batch_axis, channel_axis)


def _GlobalPool(
    pool_type: _Pooling,
    batch_axis: int,
    channel_axis: int,
) -> InternalLayerMasked:
  """General global pooling.

  Pools over and removes (`keepdims=False`) all spatial dimensions, preserving
    the order of batch and channel axes.

  Args:
    pool_type:
      specifies whether average or sum pooling should be performed.
      (`Pooling.AVG` or `Pooling.SUM`).

    batch_axis:
      Specifies the batch dimension. Defaults to `0`, the leading axis.

    channel_axis:
      Specifies the channel / feature dimension. Defaults to `-1`, the trailing
      axis. For `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """

  if pool_type == _Pooling.AVG:
    pool_fn = lambda x, axis, mask: mean_and_var(x, axis, mask=mask)[0]
  elif pool_type == _Pooling.SUM:
    pool_fn = lambda x, axis, mask: torch.sum(x, axis)
  else:
    raise ValueError(f'Invalid pooling type {pool_type}.')

  def init_fn(rng, input_shape):
    ndim = len(input_shape)
    non_spatial_axes = (batch_axis % ndim, channel_axis % ndim)
    output_shape = tuple(input_shape[i] for i in range(ndim)
                         if i in non_spatial_axes)
    return output_shape, ()

  def apply_fn(params, inputs, mask=None, **kwargs):
    non_spatial_axes = (batch_axis % inputs.ndim, channel_axis % inputs.ndim)
    spatial_axes = tuple(i for i in range(inputs.ndim)
                         if i not in non_spatial_axes)
    out = pool_fn(inputs, spatial_axes, mask)
    return out

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal(input=Bool.MAYBE, output=Bool.YES))
  def kernel_fn(k: Kernel, **kwargs):
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def _pool(mat, batch_ndim, mask=None):
      if mat is None:
        return mat
      spatial_axes = tuple(range(batch_ndim, mat.ndim))
      out = pool_fn(mat, axis=spatial_axes, mask=mask)
      if k.diagonal_spatial and pool_type == _Pooling.AVG:
        out /= utils.size_at(mat, spatial_axes)
      return out

    mask11, mask12, mask22 = k._get_mask_prods(k.mask1, k.mask2)

    cov1 = _pool(cov1, 1 if k.diagonal_batch else 2, mask11)
    cov2 = _pool(cov2, 1 if k.diagonal_batch else 2, mask22)
    nngp = _pool(nngp, 2, mask12)
    ntk = _pool(ntk, 2, mask12)

    ndim = len(k.shape1)
    batch_first = batch_axis % ndim < channel_axis % ndim
    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     batch_axis=0 if batch_first else 1,
                     channel_axis=1 if batch_first else 0,
                     is_reversed=False)

  def mask_fn(mask, input_shape):
    _check_is_implemented(mask, channel_axis)
    non_spatial_axes = (batch_axis % mask.ndim, channel_axis % mask.ndim)
    spatial_axes = tuple(i for i in range(mask.ndim)
                         if i not in non_spatial_axes)
    return torch.all(mask, spatial_axes)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def Flatten(
    batch_axis: int = 0,
    batch_axis_out: int = 0,
) -> InternalLayerMasked:
    """
    PyTorch port of Flatten with configurable input/output batch axes.
    Returns (init_fn, apply_fn, kernel_fn, mask_fn).
    """
    # Normalize output axis options to {0, 1} and set the resulting channel axis.
    if batch_axis_out in (0, -2):
        batch_axis_out = 0
        channel_axis_out = 1
    elif batch_axis_out in (1, -1):
        batch_axis_out = 1
        channel_axis_out = 0
    else:
        raise ValueError(f'`batch_axis_out` must be 0 or 1, got {batch_axis_out}.')

    def get_output_shape(input_shape):
        # input_shape is a tuple/torch.Size
        bsz = input_shape[batch_axis]
        # product of all non-batch dims
        others = list(input_shape[:batch_axis]) + list(input_shape[batch_axis + 1:])
        ch = 1
        for d in others:
            ch *= int(d)
        return (bsz, ch) if batch_axis_out == 0 else (ch, bsz)

    def init_fn(rng, input_shape):
        return get_output_shape(input_shape), ()

    def apply_fn(params, inputs, **kwargs):
        del params
        out_shape = get_output_shape(tuple(inputs.shape))
        # Move the current batch axis to the target position:
        # -0 == 0 (front), -1 == last (then we'll reshape to (C, B))
        x = torch.movedim(inputs, batch_axis, -batch_axis_out)
        return x.reshape(out_shape)

    @requires(batch_axis=batch_axis,
              channel_axis=None,
              diagonal_spatial=Diagonal(output=Bool.YES))
    def kernel_fn(k: Kernel, **kwargs):
        """Compute kernels after Flatten."""
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

        def _mean_over_spatial(x: Optional[torch.Tensor], batch_ndim: int):
            if x is None:
                return x
            if k.diagonal_spatial:
                # Average over all spatial axes (those after the batch axes)
                spatial_axes = tuple(range(x.ndim))[batch_ndim:]
                if len(spatial_axes) > 0:
                    x = x.mean(dim=spatial_axes, keepdim=False)
            else:
                # Repeatedly take (trace / size) over the last two dims
                while x.ndim > batch_ndim:
                    # trace = sum of diagonal over last two dims
                    tr = torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)
                    x = tr / x.shape[-1]
            return x

        cov1 = _mean_over_spatial(cov1, 1 if k.diagonal_batch else 2)
        cov2 = _mean_over_spatial(cov2, 1 if k.diagonal_batch else 2)
        nngp = _mean_over_spatial(nngp, 2)
        ntk = _mean_over_spatial(ntk, 2)

        return k.replace(
            cov1=cov1,
            nngp=nngp,
            cov2=cov2,
            ntk=ntk,
            is_gaussian=False,
            is_reversed=False,
            batch_axis=batch_axis_out,
            channel_axis=channel_axis_out,
            diagonal_spatial=False,
        )

    def mask_fn(mask, input_shape):
        # Ensure mask is broadcast to input shape, then move/reshape like apply_fn
        mask = torch.broadcast_to(mask, input_shape)
        out_shape = get_output_shape(tuple(mask.shape))
        mask = torch.movedim(mask, batch_axis, batch_axis_out)
        return mask.reshape(out_shape)

    return init_fn, apply_fn, kernel_fn, mask_fn


class PositionalEmbedding(enum.Enum):
  """Type of positional embeddings to use in a :obj:`GlobalSelfAttention` layer.

  Attributes:
    NONE:
      no additional positional embeddings.

    SUM:
      positional embeddings are added to activations.

    CONCAT:
      positional embeddings are concatenated with activations.
  """
  NONE = 'NONE'
  SUM = 'SUM'
  CONCAT = 'CONCAT'


class AttentionMechanism(enum.Enum):
  """Type of nonlinearity to use in a :obj:`GlobalSelfAttention` layer.

  Attributes:
    SOFTMAX:
      attention weights are computed by passing the dot product between keys
      and queries through :obj:`jax.nn.softmax`.

    IDENTITY:
      attention weights are the dot product between keys and queries.

    ABS:
      attention weights are computed by passing the dot product between keys
      and queries through :obj:`jax.numpy.abs`.

    RELU:
      attention weights are computed by passing the dot product between keys
      and queries through :obj:`jax.nn.relu`.
  """
  SOFTMAX = 'SOFTMAX'
  IDENTITY = 'IDENTITY'
  ABS = 'ABS'
  RELU = 'RELU'

  def fn(self):
    return {
        'softmax': torch.nn.functional.softmax,
        'identity': lambda x: x,
        'abs': torch.abs,
        'relu': lambda x: torch.maximum(x, 0.)
    }[self.name.lower()]


@layer
@supports_masking(remask_kernel=True)
def GlobalSelfAttention(
    n_chan_out: int,
    n_chan_key: int,
    n_chan_val: int,
    n_heads: int,
    linear_scaling: bool = True,
    W_key_std: float = 1.0,
    W_value_std: float = 1.0,
    W_query_std: float = 1.0,
    W_out_std: float = 1.0,
    b_std: Optional[float] = None,
    attention_mechanism: str = AttentionMechanism.SOFTMAX.name,
    pos_emb_type: str = PositionalEmbedding.NONE.name,
    pos_emb_p_norm: float = 2,
    pos_emb_decay_fn: Optional[Callable[[float], float]] = None,
    n_chan_pos_emb: Optional[int] = None,
    W_pos_emb_std: float = 1.0,
    val_pos_emb: bool = False,
    batch_axis: int = 0,
    channel_axis: int = -1,
) -> InternalLayerMasked:
    """
    PyTorch port of Global scaled dot-product self-attention (finite-width apply_fn).
    Returns (init_fn, apply_fn, kernel_fn, mask_fn).

    NOTE:
      • kernel_fn for the infinite-width limit is not implemented in this port.
      • apply_fn does NOT support custom `pos_emb_decay_fn` (matches original docstring).
    """
    QK_std = W_query_std * W_key_std
    OV_std = W_out_std * W_value_std

    pos_emb_type = PositionalEmbedding(pos_emb_type)
    attention_mechanism = AttentionMechanism(attention_mechanism)

    def _spatial_axes_and_shape(shape: Sequence[int],
                                b_axis: int,
                                c_axis: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return (spatial_axes, spatial_shape)."""
        b_axis = b_axis % len(shape)
        c_axis = c_axis % len(shape)
        spatial_axes = tuple(i for i in range(len(shape)) if i not in (b_axis, c_axis))
        spatial_shape = tuple(shape[i] for i in spatial_axes)
        return spatial_axes, spatial_shape

    def init_fn(rng, input_shape):
        _cax = channel_axis % len(input_shape)
        output_shape = input_shape[:_cax] + (n_chan_out,) + input_shape[_cax + 1:]

        # Number of input channels used by K/Q and V projections
        n_chan_in_keys = n_chan_in_vals = input_shape[channel_axis]

        # Random positional embeddings (if requested)
        if pos_emb_type == PositionalEmbedding.NONE:
            pos_emb = None
        else:
            _n_chan_pos = (n_chan_in_keys if n_chan_pos_emb is None else n_chan_pos_emb)
            pos_emb_shape = list(input_shape)
            pos_emb_shape[channel_axis] = _n_chan_pos
            pos_emb_shape[batch_axis] = 1
            pos_emb = torch.randn(*pos_emb_shape)

            if pos_emb_type == PositionalEmbedding.CONCAT:
                n_chan_in_keys += _n_chan_pos
                if val_pos_emb:
                    n_chan_in_vals += _n_chan_pos

        # Projection matrices (NTK parameterization uses unit-variance weights;
        # scaling is applied inside apply_fn)
        key_mats   = torch.randn(n_heads, n_chan_in_keys, n_chan_key)
        val_mats   = torch.randn(n_heads, n_chan_in_vals, n_chan_val)
        W_out      = torch.randn(n_chan_val * n_heads, n_chan_out)

        if b_std is None:
            b = None
        else:
            b_shape = [1] * len(input_shape)
            b_shape[_cax] = n_chan_out
            b = torch.randn(*b_shape)

        if linear_scaling:
            query_mats = None  # tied with keys
            warnings.warn("Linear scaling attention used -> query initialization ignored (weights tied).")
        else:
            query_mats = torch.randn(n_heads, n_chan_in_keys, n_chan_key)

        return output_shape, (query_mats, key_mats, val_mats, W_out, b, pos_emb)

    def _apply_attn_mech(G: torch.Tensor) -> torch.Tensor:
        if attention_mechanism == AttentionMechanism.SOFTMAX:
            return torch.softmax(G, dim=-1)
        if attention_mechanism == AttentionMechanism.IDENTITY:
            return G
        if attention_mechanism == AttentionMechanism.RELU:
            return F.relu(G)
        if attention_mechanism == AttentionMechanism.ABS:
            return torch.abs(G)
        raise NotImplementedError(str(attention_mechanism))

    def apply_fn(params: PyTree,
                 inputs: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        if pos_emb_decay_fn is not None:
            raise NotImplementedError("apply_fn with custom `pos_emb_decay_fn` is not supported in this PyTorch port.")

        query_mats, key_mats, val_mats, W_out, b, pos_emb = params

        spatial_axes, spatial_shape = _spatial_axes_and_shape(inputs.shape, batch_axis, channel_axis)
        n = inputs.shape[batch_axis]

        # Positional embeddings
        if pos_emb is not None:
            pe = pos_emb
            if mask is not None:
                pe = torch.where(mask, torch.zeros((), dtype=pe.dtype, device=pe.device), pe)
            pe = pe * W_pos_emb_std
        else:
            pe = None

        # Add/concat positional embeddings
        if pos_emb_type == PositionalEmbedding.SUM:
            inputs_val = None if val_pos_emb else inputs
            inputs = inputs + (pe if pe is not None else 0.0)
        elif pos_emb_type == PositionalEmbedding.CONCAT:
            inputs_val = inputs if not val_pos_emb else None
            _n_chan_pos = (inputs.shape[channel_axis] if n_chan_pos_emb is None else n_chan_pos_emb)
            _cax = channel_axis % inputs.ndim
            pe = torch.broadcast_to(
                pe,
                inputs.shape[:_cax] + (_n_chan_pos,) + inputs.shape[_cax + 1:]
            )
            inputs = torch.cat([inputs, pe], dim=channel_axis)
        else:
            inputs_val = None

        # Prepare (N, L, C) tensors
        def _prep(x: torch.Tensor) -> torch.Tensor:
            x = torch.movedim(x, (batch_axis, channel_axis), (0, -1))
            return x.reshape(n, -1, x.shape[-1])  # (N, L, C)

        x_main = _prep(inputs)
        if inputs_val is not None:
            x_val = _prep(inputs_val)
        else:
            x_val = x_main

        # Project to K, Q, V
        # shapes:
        #   x_*: (N, L, C)
        #   mats: (H, C, D) -> result (H, N, L, D)
        def _proj(x: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
            return torch.einsum('nlc,hcd->hnld', x, mats)

        keys = _proj(x_main, key_mats)
        queries = keys if linear_scaling else _proj(x_main, query_mats)
        values = _proj(x_val,   val_mats)  # (H, N, L, Dv)

        # Attention scores: (H, N, L, L)
        G = torch.matmul(queries, keys.transpose(-1, -2))
        # Scale by (C_in * (n_chan_key ** power))
        c_in = x_main.shape[-1]
        norm = float(c_in) * (float(n_chan_key) ** (1.0 if linear_scaling else 0.5))
        G = G * (QK_std / norm)

        # Masking: reduce over channel, reshape to (1, N, 1, L)
        if mask is not None:
            m = torch.all(mask, dim=channel_axis, keepdim=True)
            m = torch.movedim(m, (batch_axis, channel_axis), (0, -1)).reshape(1, n, 1, -1)
            if attention_mechanism == AttentionMechanism.SOFTMAX:
                neg_inf = torch.tensor(float('-inf'), dtype=G.dtype, device=G.device)
                G = torch.where(m, neg_inf, G)
            elif attention_mechanism in (AttentionMechanism.IDENTITY,
                                         AttentionMechanism.RELU,
                                         AttentionMechanism.ABS):
                G = torch.where(m, torch.zeros((), dtype=G.dtype, device=G.device), G)
            else:
                raise NotImplementedError(str(attention_mechanism))

        # Apply attention mechanism and weight values
        A = _apply_attn_mech(G)                  # (H, N, L, L)
        heads = torch.matmul(A, values)          # (H, N, L, Dv)

        # Concatenate heads -> (N, L, H*Dv)
        heads = torch.movedim(heads, 0, -1)      # (N, L, H, Dv)
        heads = heads.reshape(heads.shape[0], heads.shape[1], -1)

        # Output projection and scaling
        out = torch.matmul(heads, W_out)         # (N, L, C_out)
        out = out * (OV_std / (n_chan_val * n_heads * c_in) ** 0.5)

        # Restore original shape (with output channels)
        out = out.reshape((n,) + spatial_shape + (n_chan_out,))
        out = torch.movedim(out, (0, -1), (batch_axis, channel_axis))

        if b is not None:
            out = out + b_std * b
        return out

    @requires(batch_axis=batch_axis,
              channel_axis=channel_axis,
              diagonal_spatial=Diagonal(input=Bool.NO))
    def kernel_fn(k: Kernel, **kwargs):
        # Porting the full infinite-width kernel for attention is substantial.
        # In this PyTorch port, we currently do not implement kernel_fn.
        raise NotImplementedError("kernel_fn for GlobalSelfAttention is not implemented in the PyTorch port yet.")

    def mask_fn(mask, input_shape):
        # Collapse channel axis; keep dims
        return torch.all(mask, dim=channel_axis, keepdim=True)

    return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def LayerNorm(
    axis: Axes = -1,
    eps: float = 1e-12,
    batch_axis: int = 0,
    channel_axis: int = -1,
) -> InternalLayer:
  """Layer normalisation.

  Args:
    axis:
      dimensions over which to normalize.

    eps:
      (small) positive constant to be added to the variance estimates in order
      to prevent division by zero.

    batch_axis:
      batch dimension. Defaults to `0`, the leading axis.

    channel_axis:
      channel / feature dimension. Defaults to `-1`, the trailing axis. For
      `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def init_fn(rng, input_shape):
    return input_shape, ()

  def apply_fn(params, inputs, mask=None, **kwargs):
    _axis = utils.canonicalize_axis(axis, inputs)
    mean, var = mean_and_var(inputs, _axis, keepdims=True, mask=mask,
                             get_var=True)
    return (inputs - mean) / torch.sqrt(eps + var)

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def kernel_fn(k: Kernel, **kwargs):
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if not k.is_gaussian:
      raise NotImplementedError('LayerNorm only implemented for Gaussian '
                                'inputs.')

    ndim = len(k.shape1)
    _channel_axis = channel_axis % ndim
    _batch_axis = batch_axis % ndim
    _axis = utils.canonicalize_axis(axis, k.shape1)

    if _channel_axis not in _axis:
      raise ValueError(f'Normalisation over channels (axis {_channel_axis})'
                       f'necessary for convergence to an asymptotic kernel; '
                       f'got axis={_axis}.')

    _axis.remove(_channel_axis)

    spatial_axes = tuple(i for i in range(len(k.shape1))
                         if i not in (_channel_axis, batch_axis))

    # Batch axis
    if _batch_axis in _axis:
      kernel_axis = (0,)
      _axis.remove(_batch_axis)
    else:
      kernel_axis = ()

    # Spatial axes
    kernel_axis += tuple(
        1 + spatial_axes[::(-1 if k.is_reversed else 1)].index(i)
        for i in _axis)

    # Prepare masks for normalization
    def prepare_mask(m):
      if m is None:
        return m

      if m.shape[channel_axis] != 1:
        raise NotImplementedError('`LayerNorm` with different per-channel masks'
                                  'not implemented in the infinite limit.')

      m = torch.squeeze(m, channel_axis)
      if k.is_reversed:
        m = torch.moveaxis(m, range(1, m.ndim), range(m.ndim - 1, 0, -1))

      return m

    prod11, prod12, prod22 = get_diagonal_outer_prods(
        eps + cov1,
        cov2 if cov2 is None else eps + cov2,
        k.diagonal_batch,
        k.diagonal_spatial,
        op.mul,
        axis=kernel_axis,
        mask1=prepare_mask(k.mask1),
        mask2=prepare_mask(k.mask2),
        )

    nngp /= torch.sqrt(prod12)

    if ntk is not None:
      ntk /= torch.sqrt(prod12)

    cov1 /= torch.sqrt(prod11)
    if cov2 is not None:
      cov2 /= torch.sqrt(prod22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def Dropout(rate: float, mode: str = 'train') -> InternalLayer:
  """Dropout.

  Based on :obj:`jax.example_libraries.stax.Dropout`.

  Args:
    rate:
      Specifies the keep `rate`, e.g. `rate=1` is equivalent to keeping all
      neurons.

    mode:
      Either `"train"` or `"test"`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  if mode not in ('test', 'train'):
    raise ValueError('The `mode` must be either "test"  or "train".')
  if rate <= 0. or rate > 1.:
    raise ValueError('The `rate` must be > 0. and <= 1.')

  init_fn, apply_fn = ostax.Dropout(rate, mode=mode)
  kernel_fn_test = lambda k, **kwargs: k

  @requires(use_dropout=True)
  def kernel_fn_train(k: Kernel, **kwargs):
    """kernel_fn for `train` mode."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if k.is_input:
      raise ValueError('Dropout cannot be applied to the input layer.')

    factor = 1./rate

    cov1 = _diag_mul(cov1, factor, k.diagonal_batch, k.diagonal_spatial)
    cov2 = _diag_mul(cov2, factor, k.diagonal_batch, k.diagonal_spatial)

    new_factor = torch.where(k.x1_is_x2, factor, 1.)
    nngp = _diag_mul(nngp, new_factor, False, k.diagonal_spatial)
    ntk = _diag_mul(ntk, new_factor, False, k.diagonal_spatial)

    # TODO(xlc): under which condition could we leave `is_gaussian` unchanged?
    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     is_gaussian=False)

  kernel_fn = kernel_fn_test if mode == 'test' else kernel_fn_train

  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=True)
def ImageResize(
    shape: Sequence[int],
    method: Union[str, "jax.image.ResizeMethod"],
    antialias: bool = True,
    precision: "lax.Precision" = None,  # unused in PyTorch
    batch_axis: int = 0,
    channel_axis: int = -1,
) -> InternalLayerMasked:
    """
    PyTorch port of ImageResize (finite-width init/apply/mask; kernel_fn not implemented).

    Differences vs JAX version:
      • `apply_fn` uses `torch.nn.functional.interpolate` under the hood.
      • Only "nearest" and "linear" modes are supported (same as JAX note).
      • We ignore `precision` (einsum precision is JAX-specific).
      • `kernel_fn` is not implemented in this port.
    """

    # --- helpers -------------------------------------------------------------

    def _norm_method(m):
        # Accept jax ResizeMethod, or strings like "linear", "nearest", "bilinear", ...
        if hasattr(m, "name"):
            m = m.name
        m = str(m).lower()
        if m in ("linear", "bilinear", "trilinear", "triangle"):
            return "linear"
        if m in ("nearest",):
            return "nearest"
        return m  # will be caught as unsupported later

    def _target_shape(input_shape: Sequence[int]) -> Tuple[int, ...]:
        # Replace -1 with the corresponding input dimension size.
        return tuple(
            (input_shape[i] if s == -1 else s) for i, s in enumerate(shape)
        )

    def _resize_along_axis(x: torch.Tensor, axis: int, out_size: int, mode: str) -> torch.Tensor:
        """
        Resize a single axis `axis` to `out_size` using 1D interpolation.
        Works for arbitrary tensor rank by collapsing all non-axis dims
        into the channel dimension for F.interpolate.
        """
        if x.shape[axis] == out_size:
            return x

        # Move the axis to the last position: (..., L)
        axis = axis % x.ndim
        perm = [i for i in range(x.ndim) if i != axis] + [axis]
        x_perm = x.permute(perm)
        *lead, L = x_perm.shape

        # Collapse all leading dims into "channels" for interpolate: (N=1, C=prod(lead), L)
        C = 1
        for d in lead:
            C *= d
        x_flat = x_perm.reshape(1, C, L)

        # Interpolate
        if mode == "linear":
            # align_corners=False matches typical framework defaults for linear
            y_flat = F.interpolate(x_flat, size=out_size, mode="linear", align_corners=False, antialias=antialias)
        elif mode == "nearest":
            y_flat = F.interpolate(x_flat, size=out_size, mode="nearest")
        else:
            raise NotImplementedError(f"Unsupported resize method '{mode}' in PyTorch port.")

        # Restore original rank and permute back
        y_perm = y_flat.reshape(*lead, out_size)
        # Inverse permutation
        inv = [0] * x.ndim
        for i, p in enumerate(perm):
            inv[p] = i
        y = y_perm.permute(inv)
        return y

    def _resize_general(x: torch.Tensor, out_shape: Sequence[int], mode: str) -> torch.Tensor:
        """
        Resize a tensor to a fully-specified out_shape by sequentially
        resizing individual axes. Supports any rank by using 1D interpolation per axis.
        """
        assert len(out_shape) == x.ndim, "`shape` must have same rank as inputs."
        y = x
        for ax in range(y.ndim):
            tgt = out_shape[ax]
            if tgt != y.shape[ax]:
                y = _resize_along_axis(y, ax, tgt, mode)
        return y

    # Pre-parse/normalize axes once for mask/kernel annotations
    def _mod_axes(ndim: int) -> Tuple[int, int]:
        return utils.mod((batch_axis, channel_axis), ndim)

    # --- init / apply / mask / kernel ---------------------------------------

    def init_fn(rng, input_shape):
        return _target_shape(input_shape), ()

    def apply_fn(params, x: torch.Tensor, **kwargs):
        mode = _norm_method(method)
        if mode not in ("linear", "nearest"):
            raise NotImplementedError(f"Only 'linear' and 'nearest' are supported in this PyTorch port, got '{mode}'.")
        return _resize_general(x, _target_shape(tuple(x.shape)), mode)

    def mask_fn(mask: torch.Tensor, input_shape):
        """
        Follow the NT convention: an output location is masked only if
        *all* contributing inputs are masked. Achieve this by inverting
        the mask before resize and inverting back after.
        """
        mode = _norm_method(method)
        inv = (~mask).to(dtype=torch.float32)
        resized = _resize_general(inv, _target_shape(tuple(mask.shape)), mode)
        # any positive contribution means "unmasked"; convert back to bool and invert
        return ~(resized > 0)

    # Infer modded axes relative to the provided `shape` tuple (same convention as JAX code).
    _bax, _cax = utils.mod((batch_axis, channel_axis), shape)

    diagonal_batch = shape[_bax] == -1
    diagonal_spatial = Diagonal(
        input=Bool.NO
        if any(shape[i] != -1 for i in range(len(shape)) if i not in (_bax, _cax))
        else Bool.YES
    )

    @requires(
        batch_axis=_bax,
        channel_axis=_cax,
        diagonal_batch=diagonal_batch,
        diagonal_spatial=diagonal_spatial,
    )
    def kernel_fn(k: Kernel, **kwargs) -> Kernel:
        # The JAX version provides an analytic kernel update for "linear"/"nearest".
        # Porting that logic precisely to PyTorch would require re-implementing the
        # same kernel-space resize. To keep this port tight and correct, we defer it.
        raise NotImplementedError("kernel_fn for ImageResize is not implemented in the PyTorch port yet.")

    return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def Index(
    idx: utils.SliceType,
    batch_axis: int = 0,
    channel_axis: int = -1,
) -> InternalLayerMasked:
  """Index into the array mimicking :class:`numpy.ndarray` indexing.

  Args:
    idx:
      a `slice` object that would result from indexing an array as `x[idx]`.
      To create this object, use the helper object :obj:`Slice`, i.e. pass
      `idx=stax.Slice[1:10, :, ::-1]` (which is equivalent to passing an
      explicit `idx=(slice(1, 10, None), slice(None), slice(None, None, -1)`.

    batch_axis:
      batch axis for `inputs`. Defaults to `0`, the leading axis.

    channel_axis:
      channel axis for `inputs`. Defaults to `-1`, the trailing axis. For
      `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.

  Raises:
    NotImplementedError:
      If the `channel_axis` (infinite width) is indexed
      (except for `:` or `...`) in the kernel regime (`kernel_fn`).

    NotImplementedError:
      If the `batch_axis` is indexed with an integer (as opposed to a tuple or
      slice) in the kernel regime (`kernel_fn`), since the library currently
      requires there always to be `batch_axis` in  the kernel regime (while
      indexing with integers removes the respective axis).

    ValueError:
      If `init_fn` is called on a shape with dummy axes (with sizes like `-1`
      or `None`), that are indexed with non-trivial (not `:` or `...`) slices.
      For indexing, the size of the respective axis needs to be specified.

  Example:
    >>> from neural_tangents import stax
    >>> #
    >>> init_fn, apply_fn, kernel_fn = stax.serial(
    >>>     stax.Conv(128, (3, 3)),
    >>>     stax.Relu(),
    >>>     # Select every other element from the batch (leading axis), cropped
    >>>     # to the upper-left 4x4 corner.
    >>>     stax.Index(idx=stax.Slice[::2, :4, :4])
    >>>     stax.Conv(128, (2, 2)),
    >>>     stax.Relu(),
    >>>     # Select the first row. Notice that the image becomes 1D.
    >>>     stax.Index(idx=stax.Slice[:, 0, ...])
    >>>     stax.Conv(128, (2,))
    >>>     stax.GlobalAvgPool(),
    >>>     stax.Dense(10)
    >>> )
  """
  def init_fn(rng, input_shape):
    return utils.slice_shape(input_shape, idx), ()

  def apply_fn(params, x, **kwargs):
    return x[idx]

  def mask_fn(mask, input_shape):
    return mask[idx]

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def kernel_fn(k: Kernel, **kwargs) -> Kernel:
    return k[idx]

  return init_fn, apply_fn, kernel_fn, mask_fn


class _Slice:
  def __getitem__(self, idx: utils.SliceType) -> utils.SliceType:
    return idx


Slice = _Slice()

_CONV_KERNEL_DIMENSION_NUMBERS = ('NCHW', 'OIHW', 'NCHW')


def _affine(
    mat: Optional[jnp.ndarray],
    W_std: float,
    b_std: Optional[float],
) -> Optional[jnp.ndarray]:
  """Get covariances of affine outputs if inputs have covariances `nngp`.

  The output is assumed to be `xW + b`, where `x` is the input, `W` is a matrix
  of i.i.d. Gaussian weights with std `W_std`, `b` is a vector of i.i.d.
  Gaussian biases with std `b_std`.

  Args:
    mat:
      a `jnp.ndarray` containing sample-[sample-]position[-position] covariances
      of inputs.

    W_std:
      standard deviation of a fully-connected layer weights.

    b_std:
      standard deviation of a fully-connected layer biases.
      `None` means no bias.

  Returns:
    a `jnp.ndarray` containing sample-[sample-]position[-position] covariances
    of FC outputs. Has the same shape as `nngp`.
  """
  if mat is not None:
    mat *= W_std**2

    if b_std is not None:
      mat += b_std**2

  return mat


def _same_pad_for_filter_shape(
    x: torch.Tensor,
    filter_shape: Sequence[int],
    strides: Sequence[int],
    axes: Sequence[int],
    mode: str = "wrap",
) -> torch.Tensor:
    """
    PyTorch version of SAME-padding helper used to emulate SAME via VALID.

    Pads `x` along `axes` so that a VALID conv with `filter_shape` and `strides`
    produces the same spatial shape as the (un-padded) input `x` (i.e. TF/JAX
    "SAME" behavior). Default `mode="wrap"` maps to PyTorch's `circular`.

    Args:
      x: input tensor (e.g., NHWC or NCHW, any rank).
      filter_shape: spatial kernel sizes per `axes`.
      strides: spatial strides per `axes`.
      axes: which tensor axes are spatial and should be padded.
      mode: only "wrap" is supported here (mapped to `circular`).

    Returns:
      Padded tensor.
    """
    if mode != "wrap":
        raise NotImplementedError("Only mode='wrap' (circular) is supported in this port.")

    if not (len(filter_shape) == len(strides) == len(axes)):
        raise ValueError("`filter_shape`, `strides`, and `axes` must have the same length.")

    # Compute (pad_before, pad_after) per dimension, defaulting to (0, 0).
    pads_per_dim: list[tuple[int, int]] = [(0, 0)] * x.ndim

    for i, axis in enumerate(axes):
        n = int(x.shape[axis])
        f = int(filter_shape[i])
        s = int(strides[i])

        # Output size for SAME padding
        out_size = math.ceil(n / s)
        # Total padding needed to hit SAME output size with VALID conv afterwards
        total_pad = max((out_size - 1) * s + f - n, 0)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pads_per_dim[axis] = (pad_before, pad_after)

    # torch.nn.functional.pad expects padding for the last dimension first:
    # (pad_left_last, pad_right_last, pad_left_prev, pad_right_prev, ...)
    pad_seq: list[int] = []
    for dim in reversed(range(x.ndim)):
        left, right = pads_per_dim[dim]
        pad_seq.extend([left, right])

    return F.pad(x, pad_seq, mode="circular")


def _same_pad_for_filter_shape_transpose(
    x: torch.Tensor,
    axes: Sequence[int],
    out_shape: Sequence[int],
) -> torch.Tensor:
    """Transpose of `_same_pad_for_filter_shape` in PyTorch.

    Unpads (crops) the tensor and fills each coordinate with the sum of all
    elements at positions where the current element would appear during
    circular padding. Mirrors the JAX version that uses `lax.reduce_window`
    with SAME padding and (potentially large) window dilation.

    Args:
      x: Tensor to transform (e.g., NHWC-like layout with arbitrary axes).
      axes: Indices of spatial axes in `x` (e.g., (1, 2) for NHWC).
      out_shape: Target full output shape after cropping (same rank as `x`).

    Returns:
      Tensor with shape exactly `out_shape`.
    """
    n_spatial = len(axes)
    if n_spatial < 1 or n_spatial > 3:
        raise ValueError(
            f"_same_pad_for_filter_shape_transpose supports 1D/2D/3D only; got {n_spatial}D."
        )

    # Compute per-dimension window sizes (odd) and dilations exactly as in JAX.
    # window_dimensions[i] = ceil(x.shape[i] / out_shape[i]) made odd by 2*k+1
    window_dims = []
    dilations = []
    for i in range(x.ndim):
        if i in axes:
            w = int(math.ceil(x.shape[i] / out_shape[i]))
            w = (w // 2) * 2 + 1  # make odd: 2*k + 1
            d = int(out_shape[i])
        else:
            w = 1
            d = 1
        window_dims.append(w)
        dilations.append(d)

    # Move spatial axes to the end to get shape [..., *spatial]
    x = x.movedim(axes, list(range(-n_spatial, 0)))
    split = (-n_spatial) if n_spatial != 0 else x.ndim
    pre = x.shape[:split]

    # Collapse non-spatial dims, add singleton channel, so shape is (B, 1, *spatial)
    x = x.reshape((-1, 1, *x.shape[split:]))

    # Build an all-ones kernel of spatial shape = window_dims over spatial dims only.
    k_spatial = tuple(window_dims[-n_spatial:])
    weight = torch.ones(
        (1, 1, *k_spatial), dtype=x.dtype, device=x.device
    )

    # SAME padding for conv with dilation d and kernel k: pad = floor((d*(k-1)+1 - 1)/2)
    eff_k = [d * (k - 1) + 1 for k, d in zip(k_spatial, dilations[-n_spatial:])]
    pad = tuple((ek - 1) // 2 for ek in eff_k)

    # Choose convNd
    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[n_spatial]

    # Perform sum-reduce via conv with stride=1, SAME padding, dilation as computed.
    x = conv(
        x,
        weight,
        bias=None,
        stride=tuple(1 for _ in range(n_spatial)),
        padding=pad,
        dilation=tuple(dilations[-n_spatial:])
    )

    # Restore non-spatial dims
    x = x.reshape((*pre, *x.shape[2:]))

    # If needed, crop centrally to match `out_shape` (JAX does this when SAME produces off-by-one).
    if tuple(x.shape) != tuple(out_shape):
        # compute symmetric crops for spatial axes; keep non-spatial untouched
        slices = []
        for i in range(x.ndim):
            if i in axes:
                total = x.shape[i] - out_shape[i]
                if total < 0:
                    raise ValueError(
                        f"Computed output larger than requested along dim {i}: "
                        f"{x.shape[i]} vs {out_shape[i]}"
                    )
                left = total // 2
                right = total - left
                slices.append(slice(left, x.shape[i] - right))
            else:
                slices.append(slice(None))
        x = x[tuple(slices)]

    return x


def _pool_transpose(
    x: torch.Tensor,
    filter_shape: Sequence[int],
    strides: Sequence[int],
    axes: Sequence[int],
    padding: Padding,
) -> torch.Tensor:
    """Transpose convolution with an all-ones filter (PyTorch).

    This mirrors the JAX version:
      1) move `axes` to the end,
      2) collapse non-spatial dims into batch, add a singleton channel,
      3) apply conv_transpose with an all-ones kernel,
      4) reshape back and restore axes.
    """
    n_spatial = len(axes)

    # Move spatial axes to the end so x is [..., *spatial]
    x = x.movedim(axes, list(range(-n_spatial, 0)))
    split = (-n_spatial) if n_spatial != 0 else x.ndim
    x_preshape = x.shape[:split]

    # Collapse non-spatial into batch and add a singleton channel
    x = x.reshape((-1, 1, *x.shape[split:]))

    # All-ones kernel; PyTorch conv_transpose expects (C_in, C_out, *k)
    weight = torch.ones(
        (1, 1, *filter_shape), dtype=x.dtype, device=x.device
    )

    # Choose the appropriate conv_transposeNd
    conv_t = {
        1: F.conv_transpose1d,
        2: F.conv_transpose2d,
        3: F.conv_transpose3d,
    }.get(n_spatial, None)
    if conv_t is None:
        raise ValueError(f"_pool_transpose: only 1D/2D/3D supported, got {n_spatial}D.")

    # Map JAX-like padding to PyTorch args.
    # VALID -> padding=0, output_padding=0
    # SAME  -> choose integer padding p = ceil((k - s)/2),
    #          then set output_padding = s - (k - 2p) to hit output = in*s
    if padding == Padding.VALID:
        pad = tuple(0 for _ in filter_shape)
        out_pad = tuple(0 for _ in filter_shape)
    elif padding == Padding.SAME:
        pad = tuple(int(math.ceil((k - s) / 2)) for k, s in zip(filter_shape, strides))
        out_pad = tuple(int(s - (k - 2 * p)) for k, s, p in zip(filter_shape, strides, pad))
        # guard: output_padding must be in [0, stride-1]
        out_pad = tuple(max(0, min(s - 1, op)) for s, op in zip(strides, out_pad))
    else:
        # CIRCULAR is handled elsewhere in the original code via pre/post ops;
        # here we match the JAX helper which never passes CIRCULAR directly.
        raise ValueError(f"_pool_transpose does not support padding={padding} directly.")

    # Perform transpose conv
    x = conv_t(x, weight, bias=None, stride=tuple(strides), padding=pad, output_padding=out_pad)

    # Restore original batch dims and axes order
    x = x.reshape((*x_preshape, *x.shape[2:]))
    x = x.movedim(list(range(-n_spatial, 0)), axes)
    return x


def _get_dimension_numbers(
    n: int,
    channels_first: bool = True
) -> tuple[str, str, str]:
    """
    PyTorch-friendly helper that mirrors JAX's conv `dimension_numbers` layout.
    Returns (lhs_spec, rhs_spec, out_spec) strings for bookkeeping.

    Args:
        n: number of spatial dimensions (e.g., 2 for images).
        channels_first: if True use NCHW... style, else NHWC...

    Returns:
        (lhs_spec, rhs_spec, out_spec)
          lhs_spec:  e.g. 'NCHW' or 'NHWC'
          rhs_spec:  spatial dims + 'IO', e.g. 'HWIO'
          out_spec:  same as lhs_spec
    """
    spatial_dims = ''.join(c for c in string.ascii_uppercase
                           if c not in ('N', 'C', 'I', 'O'))[:n]
    lhs_spec = ('NC' + spatial_dims) if channels_first else ('N' + spatial_dims + 'C')
    rhs_spec = spatial_dims + 'IO'
    out_spec = lhs_spec
    return lhs_spec, rhs_spec, out_spec



def _conv_kernel_full_spatial_shared(
    lhs: Optional[torch.Tensor],
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_ndim: int,
) -> Optional[torch.Tensor]:
    """
    PyTorch port of `_conv_kernel_full_spatial_shared`.

    Computes the covariance of shared-weight (regular) CNN outputs given
    full-spatial input covariances (`kernel.diagonal_spatial == False`).
    """
    if lhs is None or lhs.ndim == 0:
        return lhs

    # If circular padding was requested, emulate SAME-padding up front and
    # switch to VALID for the inner convs (to mirror the JAX path).
    if padding == Padding.CIRCULAR:
        spatial_axes = tuple(range(batch_ndim, lhs.ndim))
        total_filter_shape = utils.double_tuple(filter_shape)
        total_strides = utils.double_tuple(strides)
        lhs = _same_pad_for_filter_shape(
            x=lhs,
            filter_shape=total_filter_shape,
            strides=total_strides,
            axes=spatial_axes,
        )

    # ----- Inner helpers (PyTorch versions) ---------------------------------

    def _lax_conv_pt(
        x4d: torch.Tensor, w: torch.Tensor, stride_2d: Tuple[int, int], pad_enum: Padding
    ) -> torch.Tensor:
        # Depthwise groups: one group per channel.
        groups = x4d.shape[1]
        # Map padding enum -> PyTorch padding arg
        if pad_enum == Padding.VALID:
            pad_arg: Union[int, str, Tuple[int, int]] = 0
        elif pad_enum == Padding.SAME:
            pad_arg = "same"  # PyTorch supports string "same"
        else:
            # We shouldn't see CIRCULAR here; it is handled above.
            raise ValueError(f"Unexpected padding inside conv: {pad_enum}")
        return F.conv2d(x4d, w, bias=None, stride=stride_2d, padding=pad_arg, groups=groups)

    def _get_n_channels(batch_and_channels: int) -> int:
        """
        JAX version picks a TPU-friendly channel count; for CPU/GPU we keep 1
        (i.e., minimal channels), which is correct and generally fastest.
        """
        # CPU/GPU default
        return 1

    def _conv_kernel_full_spatial_loop_pt(
        lhs_t: torch.Tensor,
        filt_shape: Sequence[int],
        stride_seq: Sequence[int],
        pad_enum: Padding,
        lax_conv_fn: Callable[[torch.Tensor, torch.Tensor, Tuple[int, int], Padding], torch.Tensor],
        get_c: Callable[[int], int],
    ) -> torch.Tensor:
        # JAX path switches CIRCULAR -> VALID earlier; keep as-is here.
        # Iterate over paired spatial axes from the end inward.
        batch_and_pairs = lhs_t.ndim - len(filt_shape) * 2

        x = lhs_t
        for i in range(x.ndim - 1, batch_and_pairs, -2):
            spatial_i = (i - batch_and_pairs) // 2  # which spatial dim (from last)
            k = int(filt_shape[spatial_i])
            s = int(stride_seq[spatial_i])

            # Move the current spatial pair next to each other at the end.
            x = x.movedim((i - 1, i), (-2, -1))
            preshape = x.shape[:-2]  # everything but the last pair
            n_channels = get_c(utils.size_at(preshape))

            # Flatten preshape into batch*, make a (B*, C, H, W) tensor.
            x4d = x.reshape(-1, n_channels, x.shape[-2], x.shape[-1])

            # Build a depthwise kernel whose per-channel spatial kernel is a
            # k x k diagonal (values 1/k on the diagonal, zeros elsewhere).
            eye_k = torch.eye(k, dtype=x4d.dtype, device=x4d.device) / float(k)
            w = eye_k.unsqueeze(0).unsqueeze(0)                  # (1,1,k,k)
            w = w.repeat(n_channels, 1, 1, 1)                    # (C,1,k,k)

            # Depthwise conv on the current pair, stride only along that pair.
            x4d = lax_conv_fn(x4d, w, (s, s), pad_enum)

            # Restore rank with the new last-2 dims replacing the pair.
            x = x4d.reshape(*preshape, x4d.shape[-2], x4d.shape[-1])

        return x

    # Run the shared full-spatial loop in PyTorch.
    out = _conv_kernel_full_spatial_loop_pt(
        lhs_t=lhs,
        filt_shape=filter_shape,
        stride_seq=strides,
        pad_enum=(Padding.VALID if padding == Padding.CIRCULAR else padding),
        lax_conv_fn=_lax_conv_pt,
        get_c=_get_n_channels,
    )

    return out


def _conv_kernel_full_spatial_unshared(
    lhs: Optional[torch.Tensor],
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_ndim: int,
) -> Optional[torch.Tensor]:
    """
    PyTorch port of `_conv_kernel_full_spatial_unshared`.

    Computes the covariance of an *unshared* (locally-connected) conv when
    `kernel.diagonal_spatial == False`. Off-diagonal spatial entries are zero;
    spatial-diagonal entries reuse `_conv_kernel_diagonal_spatial`.
    """
    if lhs is None or lhs.ndim == 0:
        return lhs

    # Separate paired spatial axes.
    lhs_unzipped = utils.unzip_axes(lhs, batch_ndim)

    # Take spatial diagonals (same-position covariances).
    lhs_diag = utils.diagonal_between(lhs_unzipped, batch_ndim)

    # Convolve those diagonals with the diagonal-spatial kernel.
    out_diag = _conv_kernel_diagonal_spatial(
        lhs=lhs_diag,
        filter_shape=filter_shape,
        strides=strides,
        padding=padding,
        batch_ndim=batch_ndim,
    )
    if out_diag is None:
        raise ValueError("_conv_kernel_diagonal_spatial returned None")

    # For each batch entry, turn the flattened diagonal into a diagonal matrix.
    b_flat = int(math.prod(out_diag.shape[:batch_ndim]))
    out_diag_flat = out_diag.reshape(b_flat, -1)               # (B*, M)
    out_flat = torch.diag_embed(out_diag_flat)                 # (B*, M, M)

    spatial = out_diag.shape[batch_ndim:]
    out = out_flat.reshape(*out_diag.shape[:batch_ndim], *spatial, *spatial)

    # Zip spatial axes back to the (…, X, X, Y, Y, …) layout.
    out = utils.zip_axes(out, batch_ndim)
    return out


def _conv_kernel_full_spatial_transpose(
    lhs: Optional[torch.Tensor],
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_ndim: int,
) -> Optional[torch.Tensor]:
    """
    PyTorch port of `_conv_kernel_full_spatial_transpose`.

    Computes the covariance of a CNN *transpose* (a.k.a. deconvolution) when
    `kernel.diagonal_spatial == False` (full spatial covariance).

    Args:
      lhs: (2*S + batch_ndim)-D tensor of input covariances. Shape like
           (B1[, B2], H, H, W, W, ...) for S spatial dims.
      filter_shape: spatial kernel sizes per dim, e.g. (kH, kW).
      strides: spatial strides per dim, e.g. (sH, sW).
      padding: Padding enum. If CIRCULAR, we emulate by using VALID in the
               conv loop then crop with `_same_pad_for_filter_shape_transpose`.
      batch_ndim: number of batch dims in `lhs` (1 or 2).

    Returns:
      Tensor with same rank as `lhs`, with spatial pairs transposed-convolved.
    """
    if lhs is None or lhs.ndim == 0:
        return lhs

    # --- Core loop: like `_conv_kernel_full_spatial_loop`, but with conv_transpose2d
    def _rhs_kernel(n_channels: int, k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # Diagonal k×k kernel scaled by 1/k, duplicated across channels.
        diag = torch.eye(k, dtype=dtype, device=device) / float(k)   # (k, k)
        weight = diag.view(1, 1, k, k).expand(n_channels, 1, k, k).contiguous()  # (C,1,k,k)
        return weight

    # Transpose conv loop uses VALID when padding is CIRCULAR (circular handled after).
    loop_padding = Padding.VALID if padding == Padding.CIRCULAR else padding

    out = lhs
    # Walk spatial pairs from the end toward the batch dims: (..., Xi, Xi, ...)
    for i in range(out.ndim - 1, batch_ndim, -2):
        spatial_i = (i - batch_ndim) // 2  # 0-based spatial dim index

        # Bring current spatial pair to last two dims -> [..., H, W]
        out = out.movedim((i - 1, i), (-2, -1))
        preshape = out.shape[:-2]

        # For transpose path we force "channels" to 1 (no depthwise groups).
        n_channels = 1

        # Collapse leading dims to a big batch, expose fake channels, then 2D
        out = out.reshape(-1, n_channels, out.shape[-2], out.shape[-1])  # (B*, 1, H, W)

        k = int(filter_shape[spatial_i])
        s = int(strides[spatial_i])
        weight = _rhs_kernel(n_channels, k, out.dtype, out.device)        # (1,1,k,k)

        # Map our enum to torch's padding arg
        if loop_padding == Padding.VALID:
            pt_padding = 0
        elif loop_padding == Padding.SAME:
            # conv_transpose2d doesn't accept "same"; emulate SAME with p = k//2
            pt_padding = k // 2
        else:
            raise ValueError(f"Unsupported padding in transpose loop: {loop_padding}")

        out = F.conv_transpose2d(
            out,
            weight,
            bias=None,
            stride=(s, s),
            padding=pt_padding,
            groups=n_channels,
        )

        # Restore leading shape: [..., new_H, new_W]
        out = out.reshape(*preshape, out.shape[-2], out.shape[-1])

    # --- If circular padding was requested, crop/aggregate to SAME-circular shape.
    if padding == Padding.CIRCULAR:
        spatial_axes = tuple(range(batch_ndim, out.ndim))

        # Target shape that corresponds to "SAME" for the full-spatial (doubled) axes.
        total_strides = utils.double_tuple(strides)  # (s1, s1, s2, s2, ...)
        target = list(out.shape)

        # Use transposed-conv SAME-size formula: out = (in - 1) * stride + 1 (for odd, symmetric kernels).
        for ax, s in zip(spatial_axes, total_strides):
            in_sz = lhs.shape[ax]
            target[ax] = (in_sz - 1) * int(s) + 1

        out = _same_pad_for_filter_shape_transpose(
            x=out,
            axes=spatial_axes,
            out_shape=tuple(target),
        )

    return out


def _conv_kernel_full_spatial_loop(
    lhs: torch.Tensor,
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    lax_conv: Callable[[torch.Tensor, torch.Tensor, Tuple[int, ...], str], torch.Tensor],
    get_n_channels: Callable[[int], int],
) -> torch.Tensor:
    """
    PyTorch port of `_conv_kernel_full_spatial_loop`.

    Iteratively applies a depthwise 2D convolution over each *pair* of spatial
    axes of a full-spatial covariance tensor `lhs`. Each pass uses a fixed
    k×k diagonal kernel (scaled by 1/k) with stride (s, s). When the original
    JAX code requests `Padding.CIRCULAR`, we emulate it earlier in the pipeline
    (as in the JAX version) and use VALID here; otherwise we pass through the
    given padding.

    Args:
      lhs: full-spatial covariance, shape = (batch_dims..., X1, X1, X2, X2, ...)
      filter_shape: tuple of kernel sizes per spatial dimension (k1, k2, ...)
      strides: tuple of strides per spatial dimension (s1, s2, ...)
      padding: Padding enum (VALID or SAME here; CIRCULAR already handled)
      lax_conv: ignored in this PyTorch port (kept for API parity)
      get_n_channels: function mapping the product of pre-spatial dims to #ch

    Returns:
      Tensor with the same rank as `lhs`, convolved along each spatial pair.
    """
    # In JAX they convert CIRCULAR -> VALID for this loop (circular was handled earlier).
    padding = Padding.VALID if padding == Padding.CIRCULAR else padding

    def _rhs_kernel(n_channels: int, k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # Diagonal k×k kernel scaled by 1/k, duplicated depthwise across channels.
        diag = torch.eye(k, dtype=dtype, device=device) / float(k)  # (k, k)
        weight = diag.view(1, 1, k, k).expand(n_channels, 1, k, k).contiguous()  # (C,1,k,k) for depthwise
        return weight

    batch_ndim = lhs.ndim - 2 * len(filter_shape)

    out = lhs
    # Walk the spatial pairs from the end toward the batch dims: (..., Xi, Xi, ...)
    for i in range(out.ndim - 1, batch_ndim, -2):
        spatial_i = (i - batch_ndim) // 2  # which spatial dim (0-based)

        # Bring the current spatial pair to the end so it's ... , H, W
        out = out.movedim((i - 1, i), (-2, -1))  # [..., H, W]
        preshape = out.shape[:-2]                # batch-like leading dims
        n_channels = get_n_channels(int(torch.tensor(preshape).prod().item()) if len(preshape) > 0 else 1)

        # Collapse leading dims to a big batch, expose "channels", then 2D
        out = out.reshape(-1, n_channels, out.shape[-2], out.shape[-1])  # (B*, C, H, W)

        k = int(filter_shape[spatial_i])
        s = int(strides[spatial_i])

        weight = _rhs_kernel(n_channels, k, out.dtype, out.device)       # (C,1,k,k)

        # Map Padding enum to PyTorch conv2d padding argument
        if padding == Padding.VALID:
            pt_padding = 0
        elif padding == Padding.SAME:
            pt_padding = "same"
        else:
            raise ValueError(f"Unsupported padding in this loop: {padding}")

        # Depthwise conv: groups = C, stride = (s, s)
        out = F.conv2d(out, weight, bias=None, stride=(s, s), padding=pt_padding, groups=n_channels)

        # Restore shape: [..., new_H, new_W]
        out = out.reshape(*preshape, out.shape[-2], out.shape[-1])

    return out


def _conv_kernel_diagonal_spatial(
    lhs: Optional[torch.Tensor],
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_ndim: int,
) -> Optional[torch.Tensor]:
    """
    PyTorch port of `_conv_kernel_diagonal_spatial`.

    Computes the output covariance of a (shared-weights) convolution when the
    kernel has *diagonal* spatial structure (`kernel.diagonal_spatial == True`).
    """
    if lhs is None or lhs.ndim == 0:
        return lhs

    spatial_axes: Tuple[int, ...] = tuple(range(batch_ndim, lhs.ndim))
    apply_padding = Padding.VALID if padding == Padding.CIRCULAR else padding

    # Emulate circular padding by pre-padding with `_same_pad_for_filter_shape`
    # and then using VALID below, matching the JAX implementation.
    if padding == Padding.CIRCULAR:
        lhs = _same_pad_for_filter_shape(lhs, filter_shape, strides, spatial_axes)

    def _sum_pool_along_axis(
        x: torch.Tensor,
        axis: int,
        k: int,
        s: int,
        pad_kind: Padding,
    ) -> torch.Tensor:
        """
        1D box (sum) pooling along a single axis using conv_transforms.
        Implements JAX reduce_window(sum, window=k, stride=s, padding=pad_kind)
        for that axis. Other dimensions are treated as batch.
        """
        # Move target axis to the end; collapse the rest into batch.
        perm = [i for i in range(x.ndim) if i != axis] + [axis]
        inv = [0] * x.ndim
        for i, p in enumerate(perm):
            inv[p] = i
        x = x.permute(*perm)
        *flat_batch, L = x.shape
        x = x.reshape(-1, 1, L)

        # Compute SAME/VALID padding like JAX for 1D reduce_window.
        if pad_kind == Padding.VALID:
            pad_left = pad_right = 0
        elif pad_kind == Padding.SAME:
            out_len = math.ceil(L / s)
            total_pad = max((out_len - 1) * s + k - L, 0)
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
        else:
            raise ValueError(f"Unsupported padding for pooling: {pad_kind}")

        if pad_left or pad_right:
            x = F.pad(x, (pad_left, pad_right))

        # Sum pooling == conv with all-ones kernel.
        weight = torch.ones(1, 1, k, dtype=x.dtype, device=x.device)
        y = F.conv1d(x, weight, bias=None, stride=s, padding=0)

        # Restore original layout.
        y = y.reshape(*flat_batch, y.shape[-1]).permute(*inv)
        return y

    # Perform separable box filtering: sequential 1D sum pooling per spatial axis.
    out = lhs
    for ax, k, s in zip(spatial_axes, filter_shape, strides):
        out = _sum_pool_along_axis(out, ax, int(k), int(s), apply_padding)

    # Normalize by the filter's total size (AVG pooling behavior in kernel space).
    filter_size = functools.reduce(op.mul, filter_shape, 1)
    return out / filter_size


def _conv_kernel_diagonal_spatial_transpose(
    lhs: Optional[torch.Tensor],
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_ndim: int,
) -> Optional[torch.Tensor]:
    """
    PyTorch port of `_conv_kernel_diagonal_spatial_transpose`.

    Computes the covariance of a transpose-convolution layer when the kernel
    has diagonal spatial structure (i.e., `kernel.diagonal_spatial == True`).
    """
    if lhs is None or lhs.ndim == 0:
        return lhs

    spatial_axes: Tuple[int, ...] = tuple(range(batch_ndim, lhs.ndim))
    apply_padding = Padding.VALID if padding == Padding.CIRCULAR else padding

    def _pool_transpose(
        x: torch.Tensor,
        win: Sequence[int],
        st: Sequence[int],
        axes: Sequence[int],
        pad_kind: Padding,
    ) -> torch.Tensor:
        """
        Separable transpose of SUM pooling over the given spatial `axes`.
        Each axis is handled by a 1D transposed convolution with a kernel of ones.
        """
        def transposed_sum_along(_x: torch.Tensor, axis: int, k: int, s: int, pk: Padding) -> torch.Tensor:
            # Move target axis to the end; collapse all other dims into a batch.
            perm = [i for i in range(_x.ndim) if i != axis] + [axis]
            inv = [0] * _x.ndim
            for i, p in enumerate(perm):
                inv[p] = i
            _x = _x.permute(*perm)
            *nc, L = _x.shape
            _x = _x.reshape(-1, 1, L)

            # padding for SAME vs VALID
            if pk == Padding.SAME:
                pad = k // 2
            elif pk == Padding.VALID:
                pad = 0
            else:
                raise ValueError(f"Unsupported padding for pool transpose: {pk}")

            weight = torch.ones(1, 1, k, dtype=_x.dtype, device=_x.device)
            y = F.conv_transpose1d(_x, weight, bias=None, stride=s, padding=pad)

            # Restore original layout.
            y = y.reshape(*nc, y.shape[-1]).permute(*inv)
            return y

        y = x
        for ax in axes:
            y = transposed_sum_along(y, ax, int(win[ax]), int(st[ax]), pad_kind)
        return y

    # Do the transposed pooling with VALID/SAME (CIRCULAR handled below).
    out = _pool_transpose(lhs, filter_shape, strides, spatial_axes, apply_padding)

    # If original padding was CIRCULAR, emulate the transpose of circular SAME padding.
    if padding == Padding.CIRCULAR:
        out_shape = _pool_transpose(lhs, filter_shape, strides, spatial_axes, Padding.SAME).shape
        out = _same_pad_for_filter_shape_transpose(out, spatial_axes, out_shape)

    filter_size = functools.reduce(op.mul, filter_shape, 1)
    return out / filter_size


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
    PyTorch port of `_pool_kernel`.

    Computes covariances of pooling outputs given input covariances `lhs`
    by performing sliding-window *sum* reductions (optionally normalized for
    average pooling).

    Supports arbitrary number of spatial axes (including the doubled spatial
    layout used for kernel tensors) by applying separable 1D pooling along
    each spatial axis in sequence.
    """
    assert batch_ndim in (1, 2), "batch_ndim must be 1 or 2."

    # If circular padding, emulate SAME-padding with wrap and then switch to VALID.
    if padding == Padding.CIRCULAR:
        spatial_axes = tuple(range(batch_ndim, lhs.dim()))
        # SAME-pad (with wrap) so that a subsequent VALID reduction matches input size.
        def same_pad_sizes(L_in: int, k: int, s: int) -> tuple[int, int]:
            out_len = math.ceil(L_in / s)
            total = max((out_len - 1) * s + k - L_in, 0)
            return total // 2, total - total // 2

        # Apply per-axis circular pad
        pads = []
        for ax in reversed(range(lhs.dim())):
            if ax in spatial_axes:
                k = window_shape[ax]
                s = strides[ax]
                p0, p1 = same_pad_sizes(lhs.size(ax), k, s)
                pads.extend([p0, p1])
            else:
                pads.extend([0, 0])
        # torch.nn.functional.pad expects pads for last dim first
        lhs = F.pad(lhs, tuple(pads), mode="circular")
        padding = Padding.VALID

    # Extend window/stride with ones for batch dims (already provided by caller).
    win = tuple(window_shape)
    st = tuple(strides)

    # Helper: separable 1D sum-pooling along a given axis.
    def sum_pool_along(x: torch.Tensor, axis: int, k: int, s: int, pad_kind: Padding) -> torch.Tensor:
        # Move target axis to the end and collapse the rest to N*C for pooling.
        perm = [i for i in range(x.dim()) if i != axis] + [axis]
        inv = [0] * x.dim()
        for i, p in enumerate(perm):
            inv[p] = i
        x = x.permute(*perm)
        *nc, L = x.shape
        x = x.reshape(-1, 1, L)

        # SAME padding (zero) emulation for sums (reduce_window with init=0).
        if pad_kind == Padding.SAME:
            out_len = math.ceil(L / s)
            total = max((out_len - 1) * s + k - L, 0)
            p0, p1 = total // 2, total - total // 2
            x = F.pad(x, (p0, p1))

        # Use avg_pool1d with divisor_override=1 to compute SUM.
        y = F.avg_pool1d(x, kernel_size=k, stride=s, padding=0, divisor_override=1)

        # Restore original shape and axis order.
        y = y.reshape(*nc, y.shape[-1]).permute(*inv)
        return y

    # Apply separable pooling across all spatial axes (everything after batch dims).
    out = lhs
    for ax in range(batch_ndim, out.dim()):
        out = sum_pool_along(out, ax, win[ax], st[ax], padding)

    # If average pooling, apply normalization as in JAX (edge-aware for SAME).
    if pool_type == _Pooling.AVG:
        out = _normalize(lhs, out, normalize_edges, padding, st, win)

    return out


def _normalize(lhs: torch.Tensor,
               out: torch.Tensor,
               normalize_edges: bool,
               padding,
               strides: Sequence[int],
               window_shape: Sequence[int]) -> torch.Tensor:
    """
    PyTorch port of `_normalize`.

    - If `padding == Padding.SAME` and `normalize_edges` is True, we divide by the
      *effective* window size at each position (smaller near edges) by running a
      sum-pool over an all-ones tensor with SAME padding.
    - Otherwise, divide by the full window volume (product of `window_shape`).
    """
    if padding == Padding.SAME and normalize_edges:
        ones = torch.ones_like(lhs, dtype=lhs.dtype)

        # Assume N (batch), C (channel), then D spatial dims (PyTorch convention).
        D = ones.dim() - 2
        if D < 1 or D > 3:
            raise NotImplementedError("Only 1D/2D/3D pooling is supported.")

        k = tuple(window_shape[-D:])
        s = tuple(strides[-D:])

        def same_pad_sizes(L_in: int, kernel: int, stride: int) -> tuple[int, int]:
            out_len = math.ceil(L_in / stride)
            total = max((out_len - 1) * stride + kernel - L_in, 0)
            return total // 2, total - total // 2

        # Build asymmetric SAME pad for each spatial dim, then `F.pad` before pooling.
        if D == 1:
            L = ones.size(2)
            p0, p1 = same_pad_sizes(L, k[0], s[0])
            padded = F.pad(ones, (p0, p1))
            window_sizes = F.avg_pool1d(padded, kernel_size=k, stride=s, padding=0,
                                        divisor_override=1)
        elif D == 2:
            H, W = ones.size(2), ones.size(3)
            ph0, ph1 = same_pad_sizes(H, k[0], s[0])
            pw0, pw1 = same_pad_sizes(W, k[1], s[1])
            # F.pad order: (w_left, w_right, h_top, h_bottom)
            padded = F.pad(ones, (pw0, pw1, ph0, ph1))
            window_sizes = F.avg_pool2d(padded, kernel_size=k, stride=s, padding=0,
                                        divisor_override=1)
        else:  # D == 3
            Dd, H, W = ones.size(2), ones.size(3), ones.size(4)
            pd0, pd1 = same_pad_sizes(Dd, k[0], s[0])
            ph0, ph1 = same_pad_sizes(H,  k[1], s[1])
            pw0, pw1 = same_pad_sizes(W,  k[2], s[2])
            # F.pad order: (w_left, w_right, h_top, h_bottom, d_front, d_back)
            padded = F.pad(ones, (pw0, pw1, ph0, ph1, pd0, pd1))
            window_sizes = F.avg_pool3d(padded, kernel_size=k, stride=s, padding=0,
                                        divisor_override=1)

        out = out / window_sizes
    else:
        out = out / float(math.prod(window_shape))

    return out


def _diag_mul_full_spatial(
    x: torch.Tensor,
    factor: float,
    diagonal_batch: bool,
) -> torch.Tensor:
    """
    Multiply `x` by `factor` only on entries that are on the diagonal of each
    paired dimension (full-spatial layout).

    - If `diagonal_batch` is True: keep the single batch dim intact and multiply
      where each spatial pair is equal (i.e., diag over each spatial pair).
    - If `diagonal_batch` is False: require the two batch dims to be equal
      (square); then also multiply where every pair (including batch pair)
      is on its diagonal.
    """
    if diagonal_batch:
        idx = [slice(None)]  # keep the single batch dim
        batch_ndim = 1
    else:
        if x.size(0) != x.size(1):
            return x
        idx = []
        batch_ndim = 2

    ndims = x.ndim // 2  # number of dimension *pairs*
    # Build advanced index picking diagonals in each pair
    for i in range(ndims):
        size = x.size(2 - batch_ndim + 2 * i)
        shape = [1] * ndims
        shape[i] = size
        ar = torch.arange(size, device=x.device).view(*shape)
        idx.extend([ar, ar])  # same index for both dims in the pair

    x[tuple(idx)] *= factor
    return x


def _diag_mul_diagonal_spatial(
    x: torch.Tensor,
    factor: float,
    diagonal_batch: bool,
) -> torch.Tensor:
    """
    Multiply by `factor`:
      - if `diagonal_batch` is True: scale the whole tensor;
      - else: scale only the diagonal over the first two (batch) dims,
        leaving off-diagonals unchanged. If those two dims aren't square,
        return `x` unchanged.
    """
    if diagonal_batch:
        x = x * factor
    else:
        if x.size(0) != x.size(1):
            return x
        # diagonal view over the first two dims; supports broadcasting over trailing dims
        d = x.diagonal(offset=0, dim1=0, dim2=1)
        d *= factor
    return x



def _diag_mul(
    x: Optional[torch.Tensor],
    factor: float,
    diagonal_batch: bool,
    diagonal_spatial: bool,
) -> Optional[torch.Tensor]:
  if x is None:
    return x

  if diagonal_spatial:
    return _diag_mul_diagonal_spatial(x, factor, diagonal_batch)

  return _diag_mul_full_spatial(x, factor, diagonal_batch)


def _vmap_2d(
    fn: Callable[[float, float, float], float],
    cov12: torch.Tensor,
    var1: torch.Tensor,
    var2: Optional[torch.Tensor],
    diagonal_batch: bool,
    diagonal_spatial: bool,
) -> torch.Tensor:
    """
    Effectively a "2D vmap" of `fn(cov12, var1, var2)` with the same
    layout/shape semantics as the JAX version.

    NOTE: `fn` should support Torch tensor inputs (0-d tensors). If it only
    uses Python math, wrap those with torch ops.
    """
    try:
        from torch.func import vmap  # PyTorch 2.x
    except Exception:                # functorch fallback
        from functorch import vmap

    batch_ndim = 1 if diagonal_batch else 2
    start = 2 - batch_ndim
    cov_end = batch_ndim if diagonal_spatial else cov12.ndim

    # Flatten the pair of spatial axes we map over into a single axis at `start`.
    _cov12 = utils.make_2d(cov12, start, cov_end)

    var_end = 1 if diagonal_spatial else var1.ndim
    var1 = var1.reshape(var1.shape[:start] + (-1,) + var1.shape[var_end:])
    var2 = var1 if var2 is None else var2.reshape(
        var2.shape[:start] + (-1,) + var2.shape[var_end:]
    )

    # Wrap `fn` so it always returns a tensor on the correct device/dtype.
    def _fn_wrapped(c, v1, v2):
        out = fn(c, v1, v2)
        if not torch.is_tensor(out):
            out = torch.as_tensor(out, dtype=c.dtype, device=c.device)
        return out

    # Inner vmap maps over the "j" axis; outer vmap maps over the "i" axis.
    inner = vmap(_fn_wrapped, in_dims=(start, None, start), out_dims=start)
    outer = vmap(inner,       in_dims=(start, start, None), out_dims=start)
    out = outer(_cov12, var1, var2)

    # Restore original `cov12` shape.
    out_shape = (
        cov12.shape[:start]
        + cov12.shape[start:cov_end:2]
        + cov12.shape[start + 1:cov_end:2]
        + cov12.shape[cov_end:]
    )
    out = out.reshape(out_shape)
    out = utils.zip_axes(out, start, cov_end)
    return out




_NEG_INF = -1e20  # softmax raises an error if all entries are -jnp.inf


def _check_is_implemented(mask: torch.Tensor, channel_axis: int) -> None:
  if mask.shape[channel_axis] != 1:
    raise NotImplementedError(
        'Different channel-wise masks as inputs to '
        'pooling layers are not yet supported. Please '
        'let us know about your use case at '
        'https://github.com/google/neural-tangents/issues/new')


def _pool_mask(
    mask: torch.Tensor,
    window_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_axis: int,
    channel_axis: int,
) -> torch.Tensor:
    """
    PyTorch port of the JAX `_pool_mask`.

    It **does not** actually pool; it computes the mask indices that survive a
    reduce-window w/ the given kernel/stride/padding and slices the input mask
    so its shape matches the pooled output shape (mirroring the JAX behavior).
    """
    # Normalize shapes and insert singleton entries for non-spatial axes.
    window = list(map(int, window_shape))
    stride = list(map(int, strides))

    non_spatial_axes = utils.canonicalize_axis((batch_axis, channel_axis), mask)
    for i in sorted(non_spatial_axes):
        window.insert(i, 1)
        stride.insert(i, 1)

    in_shape = list(mask.shape)
    ndim = len(in_shape)

    pad_name = padding.name if hasattr(padding, "name") else str(padding).upper()

    def _same_pads(in_sz: int, k: int, s: int) -> tuple[int, int]:
        # SAME padding (XLA/JAX semantics)
        out_sz = (in_sz + s - 1) // s  # ceil(in/s)
        total = max((out_sz - 1) * s + k - in_sz, 0)
        return total // 2, total - total // 2

    def _out_dim(in_sz: int, k: int, s: int, pad_mode: str) -> int:
        if pad_mode == "VALID":
            return 0 if in_sz < k else (in_sz - k) // s + 1
        elif pad_mode == "SAME":
            return (in_sz + s - 1) // s  # ceil(in/s)
        else:
            raise ValueError(f"Unsupported padding: {pad_mode}")

    # Compute the reduce_window output shape.
    out_shape = [0] * ndim
    for i in range(ndim):
        out_shape[i] = _out_dim(in_shape[i], window[i], stride[i], pad_name)

    out_shape = tuple(out_shape)

    # If shapes don't match, slice (stride) through the mask with the same
    # offsets JAX uses.
    if tuple(in_shape) != out_shape:
        if pad_name == "VALID":
            pads = [(0, 0)] * ndim
        elif pad_name == "SAME":
            pads = [_same_pads(in_shape[i], window[i], stride[i]) for i in range(ndim)]
        else:
            raise ValueError(f"Unsupported padding: {pad_name}")

        slices = []
        for i in range(ndim):
            start = -pads[i][0] + (window[i] - 1) // 2
            end = start + 1 + (out_shape[i] - 1) * stride[i]
            slices.append(slice(start, end, stride[i]))

        mask = mask[tuple(slices)]
        if tuple(mask.shape) != out_shape:
            raise ValueError(
                f"Mask shape must be {out_shape}, but got {tuple(mask.shape)}; "
                "please submit a bug to https://github.com/google/neural-tangents/issues/new."
            )

    return mask


def _pooling_layer(reducer, init_val, rescaler=None):
    """PyTorch version adapted from :obj:`jax.example_libraries.stax`.

    This helper is only used in our codepath with `reducer == lax.add` and
    `init_val == 0.` (i.e., sum-pooling) and an optional `rescaler` that
    divides by the window size for AVG pooling with fixed normalization.
    """

    def _compute_same_pads(in_size: int, k: int, s: int) -> tuple[int, int]:
        # SAME padding formula consistent with XLA/JAX:
        # out = ceil(in / s)
        # total_pad = max((out - 1) * s + k - in, 0)
        # pad_before = total_pad // 2, pad_after = total_pad - pad_before
        out = (in_size + s - 1) // s
        total = max((out - 1) * s + k - in_size, 0)
        return total // 2, total - total // 2

    def _reduce_window_out_dim(in_size: int, k: int, s: int, padding: str) -> int:
        if padding == "VALID":
            # floor((in - k)/s) + 1, with clamp at 0 if k > in.
            return 0 if in_size < k else (in_size - k) // s + 1
        elif padding == "SAME":
            # ceil(in / s)
            return (in_size + s - 1) // s
        else:
            raise ValueError(f"Unsupported padding: {padding}")

    def PoolingLayer(window_shape, strides=None, padding="VALID", spec=None):
        """Pooling."""
        window_shape = tuple(int(x) for x in window_shape)
        strides = tuple(int(x) for x in (strides or (1,) * len(window_shape)))
        padding = padding.upper()

        rescale = rescaler(window_shape, strides, padding) if rescaler else None

        # Figure out positions of batch and channel axes in the provided `spec`
        # (e.g., "NCHW" -> batch at 0, channel at 1; "NHWC" -> 0 and -1).
        if spec is None:
            # Default matches stax: batch first, channel last.
            non_spatial_axes = (0, len(window_shape) + 1)
        else:
            non_spatial_axes = (spec.index("N"), spec.index("C"))

        # Build expanded window/stride with 1s inserted for non-spatial axes.
        k_full = list(window_shape)
        s_full = list(strides)
        for i in sorted(non_spatial_axes):
            k_full = k_full[:i] + [1] + k_full[i:]
            s_full = s_full[:i] + [1] + s_full[i:]
        k_full = tuple(k_full)
        s_full = tuple(s_full)

        def init_fun(rng, input_shape):
            # Compute output shape using XLA/JAX semantics.
            in_shape = tuple(int(d) for d in input_shape)
            out_shape = list(in_shape)

            # Spatial axes are those NOT in non_spatial_axes
            spatial_axes = [i for i in range(len(in_shape)) if i not in non_spatial_axes]
            for ax, (k, s) in zip(spatial_axes, zip(window_shape, strides)):
                out_shape[ax] = _reduce_window_out_dim(in_shape[ax], k, s, padding)

            return tuple(out_shape), ()

        def apply_fun(params, inputs, **kwargs):
            x = inputs
            ndim = x.ndim

            # Determine channel and batch axis per spec and permute to NCH... order that
            # PyTorch conv/avg ops expect (N, C, *spatial*).
            if spec is None:
                b_ax, c_ax = 0, x.ndim - 1
            else:
                b_ax, c_ax = spec.index("N"), spec.index("C")

            spatial_axes = [i for i in range(ndim) if i not in (b_ax, c_ax)]

            # Permute to (N, C, *spatial*)
            perm = [b_ax, c_ax] + spatial_axes
            x = x.permute(perm)

            N, C, *spatial = x.shape
            num_spatial = len(spatial)

            if num_spatial not in (1, 2, 3):
                raise NotImplementedError(
                    f"Pooling only implemented for 1D/2D/3D, got {num_spatial}D."
                )

            # Compute asymmetric pads for SAME; VALID -> zeros.
            pads_before = []
            pads_after = []
            if padding == "VALID":
                pads_before = [0] * num_spatial
                pads_after = [0] * num_spatial
            elif padding == "SAME":
                for in_sz, k, s in zip(spatial, window_shape, strides):
                    p0, p1 = _compute_same_pads(int(in_sz), int(k), int(s))
                    pads_before.append(p0)
                    pads_after.append(p1)
            else:
                raise ValueError(f"Unsupported padding: {padding}")

            # torch.nn.functional.pad expects pads in reverse spatial order:
            # For 1D: (pad_right, pad_left)
            # For 2D: (pad_w_right,pad_w_left, pad_h_right,pad_h_left)
            # For 3D: (..., pad_d_right,pad_d_left)
            if any(p != 0 for p in pads_before + pads_after):
                pad_tuple = []
                for p0, p1 in zip(reversed(pads_before), reversed(pads_after)):
                    pad_tuple.extend([p1, p0])
                x = F.pad(x, tuple(pad_tuple), mode="constant", value=float(init_val))

            # Implement sum-pooling via depthwise conv with ones kernel.
            # Note: In our use, `reducer` is `lax.add` and `init_val` is 0.
            kernel_size = window_shape
            groups = C

            if num_spatial == 1:
                weight = torch.ones((C, 1, kernel_size[0]), dtype=x.dtype, device=x.device)
                out = F.conv1d(x, weight=weight, bias=None, stride=strides, padding=0, groups=groups)
            elif num_spatial == 2:
                weight = torch.ones(
                    (C, 1, kernel_size[0], kernel_size[1]),
                    dtype=x.dtype,
                    device=x.device,
                )
                out = F.conv2d(x, weight=weight, bias=None, stride=strides, padding=0, groups=groups)
            else:  # num_spatial == 3
                weight = torch.ones(
                    (C, 1, kernel_size[0], kernel_size[1], kernel_size[2]),
                    dtype=x.dtype,
                    device=x.device,
                )
                out = F.conv3d(x, weight=weight, bias=None, stride=strides, padding=0, groups=groups)

            # Optional rescale (e.g., to get fixed-normalization AVG pool).
            if rescale is not None:
                # rescaler has signature: rescale(outputs, inputs, spec)
                # It expects original layout, so temporarily convert back.
                inv_perm = [perm.index(i) for i in range(len(perm))]
                out_for_rescale = out.permute(inv_perm)
                x_for_rescale = inputs
                out_rescaled = rescale(out_for_rescale, x_for_rescale, spec)
                out = out_rescaled.permute(perm)

            # Permute back to original axis order.
            inv_perm = [perm.index(i) for i in range(len(perm))]
            return out.permute(inv_perm)

        return init_fun, apply_fun

    return PoolingLayer


def _pos_emb_identity(shape: Sequence[int]) -> torch.Tensor:
  size = utils.size_at(shape)  # pytype: disable=wrong-arg-types  # jax-ndarray
  R = torch.eye(size).reshape(tuple(shape) * 2)
  R = utils.zip_axes(R)
  return R

def _pos_emb_pdist(
    shape: Sequence[int],
    pos_emb_p_norm: Optional[float],
    pos_emb_decay_fn: Optional[Callable[[float], float]],
) -> torch.Tensor:
    if pos_emb_decay_fn is None:
        # Identity / one-hot positional embeddings.
        return _pos_emb_identity(shape)

    # Pairwise distance-based positional embeddings.
    ndim = len(shape)
    R = torch.zeros((1,) * (ndim * 2), dtype=torch.get_default_dtype())

    for axis in range(ndim):
        d = torch.arange(shape[axis], dtype=R.dtype, device=R.device)  # (S,)
        pd = d[:, None] - d[None, :]                                   # (S, S)
        pd = pd.reshape(
            (1,) * (2 * axis) + (shape[axis], shape[axis]) + (1,) * (2 * (ndim - axis - 1))
        )
        R = R + torch.abs(pd) ** pos_emb_p_norm

    R = pos_emb_decay_fn(R)
    return R


def _get_all_pos_emb(
    k: Kernel,
    pos_emb_type: PositionalEmbedding,
    pos_emb_p_norm: float,
    pos_emb_decay_fn: Optional[Callable[[float], float]],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if pos_emb_type == PositionalEmbedding.NONE:
        return None, None, None

    shape, _ = _shape_and_axes(k.shape1, (k.batch_axis, k.channel_axis))
    R = _pos_emb_pdist(shape, pos_emb_p_norm, pos_emb_decay_fn)  # (X1, X1, ..., XN, XN) interleaved

    if k.is_reversed:
        R = utils.reverse_zipped(R)

    batch_ndim = 1 if k.diagonal_batch else 2

    # Add leading batch axes.
    R11 = R
    for _ in range(batch_ndim):
        R11 = R11.unsqueeze(0)

    # Cross-batch version gets two batch dims if needed.
    R12 = R11 if batch_ndim == 2 else R11.unsqueeze(0).unsqueeze(1)
    R22 = None if k.cov2 is None else R11

    mask11, mask12, mask22 = k._get_mask_prods(k.mask1, k.mask2)
    R11 = utils.mask(R11, mask11)
    R12 = utils.mask(R12, mask12)
    R22 = utils.mask(R22, mask22)

    return R11, R12, R22


def _shape_and_axes(
    x: tuple[int, ...],
    ignore_axes: Iterable[int] = (),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
  ndim = len(x)
  ignore_axes = tuple(i % ndim for i in ignore_axes)
  axes = tuple(i for i in range(ndim) if i not in ignore_axes)
  shape = tuple(x[i] for i in axes)
  return shape, axes