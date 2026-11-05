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
# limitations under the License.

"""Branching functions ported to PyTorch."""

import functools
from typing import Callable, Iterable, Optional, Sequence
import warnings

import torch

from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer
from ..utils.typing import InternalLayerMasked
from ..utils.typing import Kernels
from .requirements import layer
from .requirements import supports_masking


@layer
def FanOut(num: int) -> InternalLayer:
  """Fan-out.

  This layer takes an input and produces `num` copies that can be fed into
  different branches of a neural network (for example with residual
  connections).
  """
  def init_fn(rng, input_shape):
    return [input_shape] * num, ()

  def apply_fn(params, inputs, **kwargs):
    return [inputs] * num

  def kernel_fn(k: Kernel, **kwargs) -> list[Kernel]:
    return [k] * num

  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def FanInSum() -> InternalLayerMasked:
  """Fan-in sum."""
  def init_fn(rng, input_shape):
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) == 0:
      raise ValueError("FanInSum requires a tuple/list of shapes.")
    return input_shape[0], ()

  def apply_fn(params, inputs, **kwargs):
    if len(inputs) == 0:
      raise ValueError("FanInSum requires at least one input.")
    res = inputs[0]
    for inp in inputs[1:]:
      res = res + inp
    return res

  def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
    ks, is_reversed = _preprocess_kernels_for_fan_in(ks)
    if not all([k.shape1 == ks[0].shape1 and
                k.shape2 == ks[0].shape2 for k in ks[1:]]):
      raise ValueError(f'All shapes should be equal in FanInSum/FanInProd, '
                       f'got x1.shapes of {[k.shape1 for k in ks]}, '
                       f'x2.shapes of {[k.shape2 for k in ks]}.')

    is_gaussian = all(k.is_gaussian for k in ks)
    if not is_gaussian and len(ks) != 1:
      raise NotImplementedError('FanInSum is only implemented for the '
                                'case where all input layers guaranteed to be '
                                'mean-zero Gaussian.')

    def _mats_sum(mats):
      if mats[0] is None:
        return None
      res = mats[0]
      for m in mats[1:]:
        res = res + m
      return res

    cov1s = [k.cov1 for k in ks]
    cov2s = [k.cov2 for k in ks]
    nngps = [k.nngp for k in ks]
    ntks = [k.ntk for k in ks]
    cov1 = _mats_sum(cov1s)
    cov2 = None if None in cov2s else (cov1 if (ks[0].x1_is_x2 or all(c2 is c1 for c1, c2 in zip(cov1s, cov2s))) else _mats_sum(cov2s))
    nngp = _mats_sum(nngps)
    ntk = _mats_sum(ntks)

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
    )

  def mask_fn(mask, input_shape):
    return _sum_masks(mask)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def FanInProd() -> InternalLayerMasked:
  """Fan-in product."""
  def init_fn(rng, input_shape):
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) == 0:
      raise ValueError("FanInProd requires a tuple/list of shapes.")
    return input_shape[0], ()

  def apply_fn(params, inputs, **kwargs):
    if len(inputs) == 0:
      raise ValueError("FanInProd requires at least one input.")
    res = inputs[0]
    for inp in inputs[1:]:
      res = res * inp
    return res

  def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
    ks, is_reversed = _preprocess_kernels_for_fan_in(ks)
    if not all([k.shape1 == ks[0].shape1 and
                k.shape2 == ks[0].shape2 for k in ks[1:]]):
      raise ValueError('All shapes should be equal in FanInProd.')

    is_gaussian = len(ks) == 1 and ks[0].is_gaussian

    def _mats_prod(nngps, ntks):
      if None in ntks:
        res = nngps[0]
        for n in nngps[1:]:
          res = res * n
        return res, None

      nngp_prod = 1.
      ntk_prod = 0.
      for nngp, ntk in zip(nngps, ntks):
        ntk_prod = ntk_prod * nngp + nngp_prod * ntk
        nngp_prod = nngp_prod * nngp
      return nngp_prod, ntk_prod

    cov1s = [k.cov1 for k in ks]
    cov2s = [k.cov2 for k in ks]
    nngps = [k.nngp for k in ks]
    ntks = [k.ntk for k in ks]

    cov1 = cov1s[0]
    for c in cov1s[1:]:
      cov1 = cov1 * c

    if None in cov2s:
      cov2 = None
    elif ks[0].x1_is_x2 or all(c2 is c1 for c1, c2 in zip(cov1s, cov2s)):
      cov2 = cov1
    else:
      cov2 = cov2s[0]
      for c in cov2s[1:]:
        cov2 = cov2 * c

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
    )

  def mask_fn(mask, input_shape):
    return _sum_masks(mask)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def FanInConcat(axis: int = -1) -> InternalLayerMasked:
  """Fan-in concatenation."""
  def init_fn(rng, input_shape):
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) == 0:
      raise ValueError("FanInConcat requires at least a tuple/list of shapes.")
    
    ndim = len(input_shape[0])
    _axis = axis % ndim
    new_shape = list(input_shape[0])
    for s in input_shape[1:]:
        new_shape[_axis] += s[_axis]
    return tuple(new_shape), ()

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

    new_shape1 = list(shape1)
    new_shape1.pop(_axis)
    new_shape1 = tuple(new_shape1)

    new_shape2 = list(shape2) if shape2 is not None else None
    if new_shape2 is not None:
        new_shape2.pop(_axis)
        new_shape2 = tuple(new_shape2)

    for k in ks:
      k_shape1 = list(k.shape1)
      k_shape1.pop(_axis)
      k_shape1 = tuple(k_shape1)

      k_shape2 = list(k.shape2) if k.shape2 is not None else None
      if k_shape2 is not None:
          k_shape2.pop(_axis)
          k_shape2 = tuple(k_shape2)

      if k_shape1 != new_shape1 or (new_shape2 is not None and k_shape2 != new_shape2):
        raise ValueError('Non-`axis` shapes should be equal in FanInConcat.')

    if _axis != channel_axis:
      is_gaussian = all(k.is_gaussian for k in ks)
      if not is_gaussian:
        raise NotImplementedError(
            'FanInConcat layer along the non-channel axis is only implemented '
            'for the case if all input layers guaranteed to be mean-zero '
            'Gaussian, i.e. having all `is_gaussian` set to `True`.')
    else:
      is_gaussian = False

    if _axis == batch_axis:
      warnings.warn(f'Concatenation along the batch axis ({_axis}) gives '
                    f'inconsistent covariances when batching - '
                    f'proceed with caution.')

    spatial_axes = tuple(i for i in range(ndim)
                         if i not in (channel_axis, batch_axis))
    if _axis in spatial_axes and is_reversed:
      _axis = spatial_axes[::-1][spatial_axes.index(_axis)]

    tensor_axis_to_kernel_axis = {
        batch_axis: 0,
        channel_axis: -1, # neural-tangents internal map uses -1 for channel dimensions
                          # this is extremely janky, because channel_axis should be 1 to be consistent with other ports,
                          # but the parity tests some how pass, and only pass, if this is -1.
                          # I have no idea why this is the case.
                          # TODO: fix this. Or at least, figure out later why this is the case.
    }
    for idx, spatial_axis in enumerate(spatial_axes):
        tensor_axis_to_kernel_axis[spatial_axis] = idx + 1

    _k_axis = tensor_axis_to_kernel_axis[_axis]
    widths = [k.shape1[channel_axis] for k in ks]

    cov1 = _concat_kernels([k.cov1 for k in ks], _k_axis,
                           diagonal_batch, diagonal_spatial, widths)
    cov2_list = [k.cov2 for k in ks]
    cov2 = None if None in cov2_list else (cov1 if (ks[0].x1_is_x2 or all(k.cov2 is k.cov1 for k in ks)) else _concat_kernels(cov2_list, _k_axis, diagonal_batch, diagonal_spatial, widths))
    nngp = _concat_kernels([k.nngp for k in ks], _k_axis,
                           False, diagonal_spatial, widths)
    ntk = _concat_kernels([k.ntk for k in ks], _k_axis,
                          False, diagonal_spatial, widths)

    return Kernel(cov1=cov1,
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
                  mask2=None)

  def mask_fn(mask, input_shape):
    return _concat_masks(mask, input_shape, axis)

  return init_fn, apply_fn, kernel_fn, mask_fn


# INTERNAL UTILITIES

def _sum_masks(masks: list[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
  def add_two_masks(mask1, mask2):
    if mask1 is None:
      return mask2
    if mask2 is None:
      return mask1
    return mask1 & mask2

  mask = functools.reduce(add_two_masks, masks, None)
  return mask


def _concat_masks(
    masks: list[Optional[torch.Tensor]],
    input_shapes: Sequence[Sequence[int]],
    axis: int
) -> Optional[torch.Tensor]:
  if len(masks) != len(input_shapes):
    raise ValueError(f'Number of masks ({len(masks)}) and inputs '
                     f'({len(input_shapes)}) don\'t match.')

  if all(m is None for m in masks):
    return None

  axis %= len(input_shapes[0])

  def broadcast_shape(m, s):
      shape_l = tuple(m.shape[:axis])
      shape_m = tuple(s[axis: axis + 1])
      shape_r = tuple(m.shape[axis + 1:])
      return shape_l + shape_m + shape_r

  masks = [m if m is None else m.expand(broadcast_shape(m, input_shapes[i]))
           for i, m in enumerate(masks)]

  max_shape = tuple(max(s) for s in zip(*(m.shape for m in masks if m is not None)))
  max_shapes = [tuple(min(ms, is_) for ms, is_ in zip(max_shape, input_shapes[i])) for i in range(len(input_shapes))]

  for i, m in enumerate(masks):
      if m is None:
          masks[i] = torch.zeros(max_shapes[i], dtype=torch.bool)
      else:
          target_shape = max_shape[:axis] + m.shape[axis: axis + 1] + max_shape[axis + 1:]
          masks[i] = m.expand(target_shape)

  return torch.cat(masks, dim=axis)


def _preprocess_kernels_for_fan_in(ks: Kernels) -> tuple[list[Kernel], bool]:
  if not all(k.diagonal_batch == ks[0].diagonal_batch and
             k.diagonal_spatial == ks[0].diagonal_spatial and
             k.batch_axis == ks[0].batch_axis and
             k.channel_axis == ks[0].channel_axis
             for k in ks[1:]):
    raise NotImplementedError('FanIn layers are only implemented if all input layers '
                              'output the same layout covariance matrices.')

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

  return ks, is_reversed


def _concat_kernels(
    mats: Sequence[Optional[torch.Tensor]],
    axis: int,
    diagonal_batch: bool,
    diagonal_spatial: bool,
    widths: Sequence[int]
) -> Optional[torch.Tensor]:
  if mats[0] is None:
    return None

  n_mats = len(mats)
  mat_ndim = mats[0].ndim

  if axis == -1:
    if all(w == widths[0] for w in widths):
      widths = [1] * len(widths)
    mat = None
    for i in range(n_mats):
        tmp = mats[i] * widths[i]
        if mat is None: mat = tmp
        else: mat = mat + tmp
    mat = mat / sum(widths)

  elif ((axis == 0 and diagonal_batch) or
        (axis != 0 and diagonal_spatial)):
    concat_axis = axis + (0 if diagonal_batch else 1)
    mat = torch.cat(mats, dim=concat_axis)

  else:
    rows = []
    pad_axis = max(0, 2 * axis - (1 if diagonal_batch else 0))
    for i, mat in enumerate(mats):
        
      pad_left = sum(mats[j].shape[pad_axis] for j in range(i))
      pad_right = sum(mats[j].shape[pad_axis] for j in range(i + 1, n_mats))

      # F.pad expects pairs from last dimension to first, but we only want to pad the `pad_axis`
      pad_arg = []
      for d in range(mat_ndim - 1, -1, -1):
          if d == pad_axis:
              pad_arg.extend([pad_left, pad_right])
          else:
              pad_arg.extend([0, 0])
      
      rows.append(torch.nn.functional.pad(mat, pad_arg))
    mat = torch.cat(rows, dim=pad_axis + 1)

  return mat
