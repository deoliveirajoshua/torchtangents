# Copyright 2019 Google LLC
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

"""Branching functions (PyTorch port).

These layers split an input into multiple branches or fuse multiple inputs from
several branches into one.
"""

import functools
from typing import Callable, Iterable, Optional, Sequence, List, Tuple
import warnings

import torch

from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer, InternalLayerMasked, Kernels
from .requirements import layer
from .requirements import supports_masking


# ----------------------------- FanOut -----------------------------------------

@layer
def FanOut(num: int) -> InternalLayer:
    """Fan-out.

    This layer takes an input and produces `num` copies that can be fed into
    different branches of a neural network (for example with residual
    connections).

    Args:
      num: The number of outgoing edges to fan out into.

    Returns:
      `(init_fn, apply_fn, kernel_fn)`.
    """
    # stax.FanOut returns (init_fn, apply_fn). We mimic the behavior:
    def init_fn(rng, input_shape):
        # No parameters, output shapes are `num` copies of input_shape.
        params = ()
        out_shape = tuple(input_shape for _ in range(num))
        return params, out_shape

    def apply_fn(params, inputs, **kwargs):
        # inputs is a single tensor; return list/tuple of identical references
        # (explicit clones are unnecessary; call .clone() if mutation is a risk).
        return tuple(inputs for _ in range(num))

    kernel_fn = lambda k, **kwargs: [k] * num
    return init_fn, apply_fn, kernel_fn


# ---------------------------- FanInSum ----------------------------------------

@layer
@supports_masking(remask_kernel=False)
def FanInSum() -> InternalLayerMasked:
    """Fan-in sum.

    Sums a list of inputs (e.g., produced by `FanOut`) into a single output.
    """

    def init_fn(rng, input_shapes):
        # Pass-through: sum doesn’t change the shape.
        if not isinstance(input_shapes, (list, tuple)) or len(input_shapes) == 0:
            raise ValueError("FanInSum expects a non-empty sequence of input shapes.")
        params = ()
        # Validate equal shapes
        first = input_shapes[0]
        for s in input_shapes[1:]:
            if s != first:
                raise ValueError("All input shapes should be equal in `FanInSum`.")
        return params, first

    def apply_fn(params, inputs, **kwargs):
        # inputs: Sequence[Tensor]
        if len(inputs) == 0:
            raise ValueError("FanInSum expects a non-empty sequence of inputs.")
        out = functools.reduce(torch.add, inputs)
        return out

    def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
        ks, is_reversed = _preprocess_kernels_for_fan_in(ks)
        if not all([k.shape1 == ks[0].shape1 and k.shape2 == ks[0].shape2 for k in ks[1:]]):
            raise ValueError('All shapes should be equal in `FanInSum/FanInProd`, '
                             f'got `x1.shape`s of {[k.shape1 for k in ks]}, '
                             f'`x2.shape`s of {[k.shape2 for k in ks]}.')

        is_gaussian = all(k.is_gaussian for k in ks)
        if not is_gaussian and len(ks) != 1:
            # As in JAX version: only implemented for mean-zero Gaussians.
            raise NotImplementedError('`FanInSum` is only implemented when all inputs '
                                      'are mean-zero Gaussian (`is_gaussian=True`).')

        def _mats_sum(mats):
            return None if mats[0] is None else functools.reduce(torch.add, mats)

        cov1s = [k.cov1 for k in ks]
        cov2s = [k.cov2 for k in ks]
        nngps = [k.nngp for k in ks]
        ntks  = [k.ntk  for k in ks]
        cov1, cov2, nngp, ntk = map(_mats_sum, (cov1s, cov2s, nngps, ntks))

        return Kernel(
            cov1=cov1,
            cov2=cov2,
            nngp=nngp,
            ntk=ntk,
            x1_is_x2=ks[0].x1_is_x2,
            is_gaussian=is_gaussian,
            is_reversed=is_reversed,
            is_input=ks[0].is_input,
            diagonal_batch=ks[0].diagonal_batch,
            diagonal_spatial=ks[0].diagonal_spatial,
            shape1=ks[0].shape1,
            shape2=ks[0].shape2,
            batch_axis=ks[0].batch_axis,
            channel_axis=ks[0].channel_axis,
            mask1=None,
            mask2=None,
        )  # pytype:disable=wrong-keyword-args

    def mask_fn(mask, input_shape):
        return _sum_masks(mask)

    return init_fn, apply_fn, kernel_fn, mask_fn


# ---------------------------- FanInProd ---------------------------------------

@layer
@supports_masking(remask_kernel=False)
def FanInProd() -> InternalLayerMasked:
    """Fan-in product.

    Elementwise product of a list of inputs into a single output.
    """

    def init_fn(rng, input_shapes):
        if not isinstance(input_shapes, (list, tuple)) or len(input_shapes) == 0:
            raise ValueError("FanInProd expects a non-empty sequence of input shapes.")
        first = input_shapes[0]
        for s in input_shapes[1:]:
            if s != first:
                raise ValueError('All shapes should be equal in `FanInProd`.')
        params = ()
        return params, first  # shape unchanged

    def apply_fn(params, inputs, **kwargs):
        return functools.reduce(torch.mul, inputs)

    def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
        ks, is_reversed = _preprocess_kernels_for_fan_in(ks)
        if not all([k.shape1 == ks[0].shape1 and k.shape2 == ks[0].shape2 for k in ks[1:]]):
            raise ValueError('All shapes should be equal in `FanInProd`.')

        is_gaussian = (len(ks) == 1) and ks[0].is_gaussian

        def _mats_prod(nngps, ntks):
            if any(x is None for x in ntks):
                return functools.reduce(torch.mul, nngps), None

            # Product rule: d(∏ f_i) = ∑ ( (∏_{j≠i} f_j) df_i )
            nngp_prod = torch.ones_like(nngps[0])
            ntk_prod  = torch.zeros_like(ntks[0])
            for nngp, ntk in zip(nngps, ntks):
                ntk_prod = ntk_prod * nngp + nngp_prod * ntk
                nngp_prod = nngp_prod * nngp
            return nngp_prod, ntk_prod

        cov1s = [k.cov1 for k in ks]
        cov2s = [k.cov2 for k in ks]
        nngps = [k.nngp for k in ks]
        ntks  = [k.ntk  for k in ks]

        cov1 = functools.reduce(torch.mul, cov1s)
        cov2 = None if any(c is None for c in cov2s) else functools.reduce(torch.mul, cov2s)
        nngp, ntk = _mats_prod(nngps, ntks)

        return Kernel(
            cov1=cov1,
            cov2=cov2,
            nngp=nngp,
            ntk=ntk,
            x1_is_x2=ks[0].x1_is_x2,
            is_gaussian=is_gaussian,
            is_reversed=is_reversed,
            is_input=ks[0].is_input,
            diagonal_batch=ks[0].diagonal_batch,
            diagonal_spatial=ks[0].diagonal_spatial,
            shape1=None,
            shape2=None,
            batch_axis=ks[0].batch_axis,
            channel_axis=ks[0].channel_axis,
            mask1=None,
            mask2=None,
        )  # pytype:disable=wrong-keyword-args

    def mask_fn(mask, input_shape):
        return _sum_masks(mask)

    return init_fn, apply_fn, kernel_fn, mask_fn


# --------------------------- FanInConcat --------------------------------------

@layer
@supports_masking(remask_kernel=False)
def FanInConcat(axis: int = -1) -> InternalLayerMasked:
    """Fan-in concatenation.

    Concatenates a list of inputs along `axis` into a single output.
    """

    def init_fn(rng, input_shapes):
        if not isinstance(input_shapes, (list, tuple)) or len(input_shapes) == 0:
            raise ValueError("FanInConcat expects a non-empty sequence of input shapes.")
        ndim = len(input_shapes[0])
        _axis = axis % ndim
        # Validate non-axis dims equal; compute concatenated shape.
        base = list(input_shapes[0])
        for s in input_shapes[1:]:
            if any(s[i] != base[i] for i in range(ndim) if i != _axis):
                raise ValueError('Non-`axis` shapes should be equal in `FanInConcat`.')
            base[_axis] += s[_axis]
        params = ()
        return params, tuple(base)

    def apply_fn(params, inputs, **kwargs):
        return torch.cat(inputs, dim=axis)

    def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
        ks, is_reversed = _preprocess_kernels_for_fan_in(ks)

        diagonal_batch = ks[0].diagonal_batch
        diagonal_spatial = ks[0].diagonal_spatial

        shape1, shape2 = ks[0].shape1, ks[0].shape2

        ndim = len(shape1)
        _axis = axis % ndim
        batch_axis = ks[0].batch_axis
        channel_axis = ks[0].channel_axis

        new_shape1 = shape1[:_axis] + shape1[_axis + 1:]
        new_shape2 = shape2[:_axis] + shape2[_axis + 1:]
        for k in ks:
            k_shape1 = k.shape1[:_axis] + k.shape1[_axis + 1:]
            k_shape2 = k.shape2[:_axis] + k.shape2[_axis + 1:]
            if k_shape1 != new_shape1 or k_shape2 != new_shape2:
                raise ValueError('Non-`axis` shapes should be equal in `FanInConcat`.')

        # Check if inputs are independent Gaussians.
        if _axis != channel_axis:
            is_gaussian = all(k.is_gaussian for k in ks)
            if not is_gaussian:
                raise NotImplementedError(
                    '`FanInConcat` along a non-channel axis requires all inputs '
                    'to be mean-zero Gaussian (`is_gaussian=True`).'
                )
        else:
            # TODO(romann): allow nonlinearity after channelwise concatenation.
            # TODO(romann): support concatenating different channelwise masks.
            is_gaussian = False

        if _axis == batch_axis:
            warnings.warn(f'Concatenation along the batch axis ({_axis}) gives '
                          f'inconsistent covariances when batching - '
                          f'proceed with caution.')

        spatial_axes = tuple(i for i in range(ndim) if i not in (channel_axis, batch_axis))
        # Change spatial axis according to the kernel `is_reversed`.
        if _axis in spatial_axes and is_reversed:
            _axis = spatial_axes[::-1][spatial_axes.index(_axis)]

        # Map activation tensor axis to the covariance tensor axis.
        tensor_axis_to_kernel_axis = {
            **{batch_axis: 0, channel_axis: -1},
            **{sp: idx + 1 for idx, sp in enumerate(spatial_axes)}
        }

        _axis = tensor_axis_to_kernel_axis[_axis]
        widths = [k.shape1[channel_axis] for k in ks]

        cov1 = _concat_kernels([k.cov1 for k in ks], _axis,
                               diagonal_batch, diagonal_spatial, widths)
        cov2 = _concat_kernels([k.cov2 for k in ks], _axis,
                               diagonal_batch, diagonal_spatial, widths)
        nngp = _concat_kernels([k.nngp for k in ks], _axis,
                               False, diagonal_spatial, widths)
        ntk  = _concat_kernels([k.ntk  for k in ks], _axis,
                               False, diagonal_spatial, widths)

        return Kernel(
            cov1=cov1,
            cov2=cov2,
            nngp=nngp,
            ntk=ntk,
            x1_is_x2=ks[0].x1_is_x2,
            is_gaussian=is_gaussian,
            is_reversed=is_reversed,
            is_input=ks[0].is_input,
            diagonal_batch=diagonal_batch,
            diagonal_spatial=diagonal_spatial,
            shape1=None,
            shape2=None,
            batch_axis=batch_axis,
            channel_axis=channel_axis,
            mask1=None,
            mask2=None,
        )  # pytype:disable=wrong-keyword-args

    def mask_fn(mask, input_shape):
        return _concat_masks(mask, input_shape, axis)

    return init_fn, apply_fn, kernel_fn, mask_fn


# --------------------------- INTERNAL UTILITIES -------------------------------

def _map_tuples(fn: Callable, tuples: Iterable[tuple]) -> tuple:
    return tuple(map(fn, zip(*(t for t in tuples))))


def _sum_masks(masks: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    def add_two_masks(mask1, mask2):
        if mask1 is None: return mask2
        if mask2 is None: return mask1
        return mask1 & mask2
    mask = functools.reduce(add_two_masks, masks, None)
    return mask


def _concat_masks(
    masks: List[Optional[torch.Tensor]],
    input_shapes: Sequence[Sequence[int]],
    axis: int
) -> Optional[torch.Tensor]:
    """Returns a mask which is a concatenation of `masks`.

    Mirrors the JAX semantics including broadcasting before concatenation.
    """
    if len(masks) != len(input_shapes):
        raise ValueError(f'Number of masks ({len(masks)}) and inputs '
                         f'({len(input_shapes)}) do not match.')

    if all(m is None for m in masks):
        return None

    axis %= len(input_shapes[0])

    # Get device/dtype from first non-None mask
    dev = next((m.device for m in masks if m is not None), torch.device('cpu'))
    dtype = torch.bool

    # Expand the concatenation dimension of each mask to match its input.
    expanded = []
    for i, m in enumerate(masks):
        if m is None:
            expanded.append(None)
        else:
            target_shape = m.shape[:axis] + (input_shapes[i][axis],) + m.shape[axis+1:]
            expanded.append(torch.broadcast_to(m, target_shape))

    # Max shape to broadcast all masks to along non-concat dimensions.
    shapes_present = [m.shape for m in expanded if m is not None]
    max_shape = _map_tuples(max, shapes_present)

    # Shapes for replacing None masks.
    max_shapes = [tuple(map(min, max_shape, i)) for i in input_shapes]

    # Broadcast each mask to max_shape except along concat axis where we keep its size.
    final_masks = []
    for i, m in enumerate(expanded):
        if m is not None:
            shape = max_shape[:axis] + (m.shape[axis],) + max_shape[axis+1:]
            final_masks.append(torch.broadcast_to(m, shape))
        else:
            final_masks.append(torch.zeros(max_shapes[i], dtype=dtype, device=dev))

    return torch.cat(final_masks, dim=axis)


def _preprocess_kernels_for_fan_in(ks: Kernels) -> Tuple[List[Kernel], bool]:
    # Check diagonal requirements.
    if not all(k.diagonal_batch == ks[0].diagonal_batch and
               k.diagonal_spatial == ks[0].diagonal_spatial and
               k.batch_axis == ks[0].batch_axis and
               k.channel_axis == ks[0].channel_axis
               for k in ks[1:]):
        raise NotImplementedError('`FanIn` layers require all inputs to output the '
                                  'same covariance layout (matching diagonal_* and axes).')

    # If kernels have different spatial axes order, transpose some of them.
    n_kernels = len(ks)
    n_reversed = sum(ker.is_reversed for ker in ks)
    ks = list(ks)

    if n_reversed > n_kernels / 2:
        is_reversed = True
        for i in range(n_kernels):
            if not ks[i].is_reversed:
                ks[i] = ks[i].reverse()
    else:
        is_reversed = False
        for i in range(n_kernels):
            if ks[i].is_reversed:
                ks[i] = ks[i].reverse()

    # Warnings.
    warnings.warn('`FanIn` layers assume independent inputs which is not verified '
                  'in code. Ensure at least one mixing layer (Dense/Conv/GSA, etc.) '
                  'in each branch.')

    return ks, is_reversed


def _concat_kernels(
    mats: Sequence[Optional[torch.Tensor]],
    axis: int,
    diagonal_batch: bool,
    diagonal_spatial: bool,
    widths: Sequence[int]
) -> Optional[torch.Tensor]:
    """Compute the covariance of concatenated activations with given covariances.

    Mirrors the JAX implementation, including the 0-block insertion case.
    """
    if mats[0] is None:
        return None

    n_mats = len(mats)
    mat_ndim = mats[0].ndim

    # Averaging if concatenating along features or a diagonalized dimension.
    if axis == -1:
        if all(w == widths[0] for w in widths):
            widths = [1] * len(widths)
        # Weighted average
        acc = None
        total_w = sum(widths)
        for m, w in zip(mats, widths):
            acc = m * w if acc is None else (acc + m * w)
        mat = acc / total_w

    # Simple concatenation along the axis if the axis is not duplicated.
    elif ((axis == 0 and diagonal_batch) or
          (axis != 0 and diagonal_spatial)):
        concat_axis = axis + (0 if diagonal_batch else 1)
        mat = torch.cat(mats, dim=concat_axis)

    # 2D concatenation with insertion of 0-blocks if the axis is present twice.
    else:
        rows = []
        pad_axis = max(0, 2 * axis - (1 if diagonal_batch else 0))

        for i, m in enumerate(mats):
            # Build target shape after padding
            pre = sum(mats[j].shape[pad_axis] for j in range(i))
            post = sum(mats[j].shape[pad_axis] for j in range(i + 1, n_mats))
            target_shape = list(m.shape)
            target_shape[pad_axis] = pre + m.shape[pad_axis] + post

            # Allocate zeros and copy m into the centered slice
            padded = m.new_zeros(target_shape)
            # Build slice objects for assignment
            slices = [slice(None)] * mat_ndim
            slices[pad_axis] = slice(pre, pre + m.shape[pad_axis])
            padded[tuple(slices)] = m
            rows.append(padded)

        mat = torch.cat(rows, dim=pad_axis + 1)

    return mat
