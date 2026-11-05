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

"""Linear functions ported to PyTorch."""

import functools
import operator as op
import operator
import math
import string
from typing import Any, Callable, Iterable, Optional, Sequence, Union
import warnings

import enum
import torch
from torch.nn.functional import avg_pool1d, avg_pool2d, avg_pool3d, pad, conv2d, conv_transpose2d
import numpy as np

from ..utils import utils
from ..utils.kernel import Kernel
from ..utils.typing import Axes
from ..utils.typing import InternalLayer
from ..utils.typing import InternalLayerMasked
from ..utils.typing import PyTree
from .requirements import Bool
from .requirements import Diagonal
from .requirements import layer
from .requirements import requires
from .requirements import supports_masking
from .requirements import get_diagonal
from . import requirements as req


_EYE_FILTER_CACHE = {}

def _get_identity_filter(K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (K, str(device), str(dtype))
    w = _EYE_FILTER_CACHE.get(key)
    if w is None or w.device != device or w.dtype != dtype:
        w = torch.eye(K, device=device, dtype=dtype).view(1, 1, K, K) / K
        _EYE_FILTER_CACHE[key] = w
    return w


def _affine(
    mat: Optional[torch.Tensor],
    W_std: float,
    b_std: Optional[float],
) -> Optional[torch.Tensor]:
  if mat is not None:
    if W_std == 1.0 and (b_std is None or b_std == 0.0):
      return mat
    mat = mat * (W_std ** 2)

    if b_std is not None:
      mat = mat + (b_std ** 2)

  return mat


@layer
@supports_masking(remask_kernel=False)
def Identity() -> InternalLayer:
  """Identity (no-op)."""
  def init_fn(rng, input_shape):
    return input_shape, ()
  
  def apply_fn(params, inputs, **kwargs):
    return inputs

  kernel_fn = lambda k, **kwargs: k
  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def Flatten(
    batch_axis: int = 0,
    batch_axis_out: int = 0,
) -> InternalLayerMasked:
  if batch_axis_out in (0, -2):
    batch_axis_out = 0
    channel_axis_out = 1
  elif batch_axis_out in (1, -1):
    batch_axis_out = 1
    channel_axis_out = 0
  else:
    raise ValueError(
        f'`batch_axis_out` must be 0 or 1, got {batch_axis_out}.')

  def get_output_shape(input_shape):
    batch_size = input_shape[batch_axis]
    channel_size = functools.reduce(
        op.mul,
        input_shape[:batch_axis] + input_shape[(batch_axis + 1)
                                               or len(input_shape):],
        1
    )
    if batch_axis_out == 0:
      return batch_size, channel_size
    return channel_size, batch_size

  def init_fn(rng, input_shape):
    output_shape = get_output_shape(input_shape)
    return output_shape, ()

  def apply_fn(params, inputs, **kwargs):
    output_shape = get_output_shape(inputs.shape)
    inputs = torch.movedim(inputs, batch_axis, -batch_axis_out if batch_axis_out != 0 else 0)
    # neural tangents uses jnp.moveaxis(inputs, batch_axis, -batch_axis_out)
    # when batch_axis_out == 0, jnp.moveaxis(..., 0, 0)
    if batch_axis_out == 0:
        inputs = torch.movedim(inputs, batch_axis, 0)
    else:
        # need to move batch_axis to the end
        inputs = torch.movedim(inputs, batch_axis, -batch_axis_out)
    return inputs.reshape(output_shape)

  @requires(batch_axis=batch_axis,
            channel_axis=None,
            diagonal_spatial=Diagonal(output=Bool.YES))
  def kernel_fn(k: Kernel, **kwargs):
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def trace(x, batch_ndim):
      if x is None:
        return x

      if k.diagonal_spatial:
        spatial_axes = tuple(range(batch_ndim, x.ndim))
        # mean over spatial axes
        if spatial_axes:
             x = torch.mean(x, dim=spatial_axes)
      else:
        while x.ndim > batch_ndim:
          # neural tangents uses x = jnp.trace(x, axis1=-2, axis2=-1) / x.shape[-1]
          diag = torch.diagonal(x, offset=0, dim1=-2, dim2=-1)
          x = torch.sum(diag, dim=-1) / x.shape[-1]

      return x

    cov1 = trace(cov1, 1 if k.diagonal_batch else 2)
    cov2 = None if cov2 is None else (cov1 if (cov2 is cov1 or k.x1_is_x2) else trace(cov2, 1 if k.diagonal_batch else 2))
    nngp = trace(nngp, 2)
    ntk = trace(ntk, 2)

    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     is_gaussian=False,
                     is_reversed=False,
                     batch_axis=batch_axis_out,
                     channel_axis=channel_axis_out,
                     diagonal_spatial=False)

  def mask_fn(mask, input_shape):
    mask = mask.expand(input_shape)
    output_shape = get_output_shape(mask.shape)
    if batch_axis_out == 0:
        mv = torch.movedim(mask, batch_axis, 0)
    else:
        mv = torch.movedim(mask, batch_axis, -batch_axis_out)
    return mv.reshape(output_shape)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=True)
def Linear(
    out_dim: int,
    W_std: float = 1.,
    b_std: Optional[float] = None,
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
    parameterization: str = 'ntk',
    s: tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
  """
  A fully connected (Dense) layer.
  Renamed to Linear to be consistent with PyTorch naming.
  """
  parameterization = parameterization.lower()

  def _init_fn(rng, input_shape, out_dim):
    _channel_axis = channel_axis % len(input_shape)
    output_shape = (input_shape[:_channel_axis] + (out_dim,)
                    + input_shape[_channel_axis + 1:])
    if rng is None:
      return output_shape, (None, None)
    W = torch.randn(input_shape[_channel_axis], out_dim, generator=rng)

    if b_std is None:
      b = None
    else:
      b_shape = [1] * len(input_shape)
      b_shape[channel_axis] = out_dim
      b = torch.randn(b_shape, generator=rng)

    return output_shape, (W, b)

  def ntk_init_fn(rng, input_shape):
    return _init_fn(rng, input_shape, out_dim)

  def standard_init_fn(rng, input_shape):
    output_shape, (W, b) = _init_fn(rng, input_shape, out_dim * s[1])
    W = W * (W_std / (input_shape[channel_axis] / s[0])**0.5)
    b = None if b is None else b * b_std
    return output_shape, (W, b)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError(f'Parameterization not supported: {parameterization}')

  def apply_fn(params, inputs, **kwargs):
    W, b = params
    prod = torch.movedim(torch.tensordot(inputs, W, dims=([channel_axis], [0])),
                        -1, channel_axis)
    # neural-tangents was jnp.tensordot(W, inputs, (0, channel_axis)), output shape had channel first.

    if parameterization == 'ntk':
      norm = W_std / inputs.shape[channel_axis]**0.5
      outputs = norm * prod
      if b is not None:
        outputs = outputs + b_std * b
    elif parameterization == 'standard':
      outputs = prod / s[0]**0.5
      if b is not None:
        outputs = outputs + b
    else:
      raise ValueError(f'Parameterization not supported: {parameterization}')

    return outputs

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal(input=Bool.MAYBE, output=Bool.NO))
  def kernel_fn(k: Kernel, **kwargs):
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def fc(x):
      return _affine(x, W_std, b_std)

    if parameterization == 'ntk':
      cov1 = fc(cov1)
      nngp = fc(nngp)
      cov2 = cov1 if (cov2 is cov1 or cov2 is None) else fc(cov2)
      if ntk is not None:
        ntk = (nngp + ntk) if W_std == 1.0 else (nngp + W_std**2 * ntk)
    elif parameterization == 'standard':
      input_width = k.shape1[channel_axis] / s[0]
      if ntk is not None:
        ntk = input_width * nngp + W_std**2 * ntk
        if b_std is not None:
          ntk = ntk + 1.
      cov1 = fc(cov1)
      nngp = fc(nngp)
      cov2 = cov1 if (cov2 is cov1 or cov2 is None) else fc(cov2)

    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     is_gaussian=True,
                     is_input=False)

  def mask_fn(mask, input_shape):
    return torch.all(mask, dim=channel_axis, keepdim=True)

  return init_fn, apply_fn, kernel_fn, mask_fn


Dense = Linear


def _diag_mul_full_spatial(
    x: torch.Tensor,
    factor: Union[float, torch.Tensor],
    diagonal_batch: bool,
) -> torch.Tensor:
  if diagonal_batch:
    idx = (slice(None),)
    batch_ndim = 1
  else:
    if x.shape[0] != x.shape[1]:
      return x
    idx = ()
    batch_ndim = 2

  ndims = x.ndim // 2
  for i in range(ndims):
    shape = [1] * ndims
    size = x.shape[2 - batch_ndim + 2 * i]
    shape[i] = size
    r = torch.arange(size, device=x.device).reshape(shape)
    idx += (r, r)

  x = x.clone()
  x[idx] = x[idx] * factor
  return x


def _diag_mul_diagonal_spatial(
    x: torch.Tensor,
    factor: Union[float, torch.Tensor],
    diagonal_batch: bool,
) -> torch.Tensor:
  if diagonal_batch:
    x = x * factor

  else:
    if x.shape[0] != x.shape[1]:
      return x
    r = torch.arange(x.shape[0], device=x.device)
    x = x.clone()
    x[r, r] = x[r, r] * factor

  return x


def _diag_mul(
    x: Optional[torch.Tensor],
    factor: Union[float, torch.Tensor],
    diagonal_batch: bool,
    diagonal_spatial: bool,
) -> Optional[torch.Tensor]:
  if x is None:
    return x
  if diagonal_spatial:
    return _diag_mul_diagonal_spatial(x, factor, diagonal_batch)
  return _diag_mul_full_spatial(x, factor, diagonal_batch)


@layer
@supports_masking(remask_kernel=False)
def Dropout(rate: float, mode: str = 'train') -> InternalLayer:
  """Dropout.
  Based on :obj:`jax.example_libraries.stax.Dropout`.
  """
  if mode not in ('test', 'train'):
    raise ValueError('The `mode` must be either "test"  or "train".')
  if rate <= 0. or rate > 1.:
    raise ValueError('The `rate` must be > 0. and <= 1.')

  def init_fn(rng, input_shape):
    return input_shape, ()
  
  def apply_fn(params, inputs, **kwargs):
    if mode == 'train':
      return torch.nn.functional.dropout(inputs, p=1.0 - rate, training=True)
    return inputs

  @requires(use_dropout=True)
  def kernel_fn(k: Kernel, **kwargs):
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if k.is_input:
      raise ValueError('Dropout cannot be applied to the input layer.')

    factor = 1. / rate

    cov1 = _diag_mul(cov1, factor, k.diagonal_batch, k.diagonal_spatial)
    cov2 = None if cov2 is None else (cov1 if (cov2 is cov1 or k.x1_is_x2) else _diag_mul(cov2, factor, k.diagonal_batch, k.diagonal_spatial))

    new_factor = factor if k.x1_is_x2 else 1.0
    
    nngp = _diag_mul(nngp, new_factor, False, k.diagonal_spatial)
    ntk = _diag_mul(ntk, new_factor, False, k.diagonal_spatial)

    return k.replace(cov1=cov1, cov2=cov2, nngp=nngp, ntk=ntk)

  return init_fn, apply_fn, kernel_fn

class Padding(str, enum.Enum):
  CIRCULAR = 'CIRCULAR'
  SAME = 'SAME'
  VALID = 'VALID'

class AggregateImplementation(str, enum.Enum):
  DENSE = 'DENSE'
  SPARSE = 'SPARSE'

@layer
@supports_masking(remask_kernel=True)
def Conv(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
    parameterization: str = 'ntk',
    s: tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
    return _Conv(out_chan, filter_shape, strides, padding, W_std, b_std, batch_axis, channel_axis, parameterization, s, transpose=False, shared_weights=True)

@layer
@supports_masking(remask_kernel=True)
def ConvTranspose(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
    parameterization: str = 'ntk',
    s: tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
    return _Conv(out_chan, filter_shape, strides, padding, W_std, b_std, batch_axis, channel_axis, parameterization, s, transpose=True, shared_weights=True)

@layer
@supports_masking(remask_kernel=True)
def ConvLocal(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
    parameterization: str = 'ntk',
    s: tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
    return _Conv(out_chan, filter_shape, strides, padding, W_std, b_std, batch_axis, channel_axis, parameterization, s, transpose=False, shared_weights=False)


# Note: visualization reference for conv operations: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
def _Conv(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    batch_axis: int = 0,
    channel_axis: int = 1,
    parameterization: str = 'ntk',
    s: tuple[int, int] = (1, 1),
    transpose: bool = False,
    shared_weights: bool = True,
) -> InternalLayerMasked:
  parameterization = parameterization.lower()
  
  if strides is None:
    strides = (1,) * len(filter_shape)

  def init_fn(rng, input_shape):
    _channel_axis = channel_axis % len(input_shape)
    _batch_axis = batch_axis % len(input_shape)
    spatial_axes = tuple(i for i in range(len(input_shape)) if i != _channel_axis and i != _batch_axis)
    if len(spatial_axes) != len(filter_shape):
        raise ValueError(f"Filter shape {filter_shape} does not match spatial dimensions {len(spatial_axes)}")

    out_shape = list(input_shape)
    
    for i, ax in enumerate(spatial_axes):
      if not transpose:
        if padding == Padding.VALID:
          out_shape[ax] = (input_shape[ax] - filter_shape[i]) // strides[i] + 1
        elif padding == Padding.SAME or padding == Padding.CIRCULAR:
          out_shape[ax] = math.ceil(input_shape[ax] / strides[i])
      else:
        if padding == Padding.VALID:
          out_shape[ax] = (input_shape[ax] - 1) * strides[i] + filter_shape[i]
        elif padding == Padding.SAME or padding == Padding.CIRCULAR:
          out_shape[ax] = input_shape[ax] * strides[i]

    in_chan = input_shape[_channel_axis]
    
    if not transpose:
      W_shape = (out_chan, in_chan) + tuple(filter_shape)
    else:
      W_shape = (in_chan, out_chan) + tuple(filter_shape)

    if not shared_weights:
      out_spatial_size = math.prod([out_shape[ax] for ax in spatial_axes])
      W_shape = (out_spatial_size,) + W_shape

    if parameterization == 'ntk':
      out_shape[_channel_axis] = out_chan
      if rng is None:
        return tuple(out_shape), (None, None)
      W = torch.randn(W_shape, generator=rng)
      b = None if b_std is None else torch.randn((out_chan,), generator=rng)
    elif parameterization == 'standard':
      curr_out_chan = out_chan * s[1]
      out_shape[_channel_axis] = curr_out_chan
      if rng is None:
        return tuple(out_shape), (None, None)
      if not transpose:
        W_shape_s = (curr_out_chan, in_chan * s[0]) + tuple(filter_shape)
      else:
        W_shape_s = (in_chan * s[0], curr_out_chan) + tuple(filter_shape)
      
      if not shared_weights:
        out_spatial_size = math.prod([out_shape[ax] for ax in spatial_axes])
        W_shape_s = (out_spatial_size,) + W_shape_s
        
      W = torch.randn(W_shape_s, generator=rng)
      fan_in = in_chan * math.prod(filter_shape)
      W = W * (W_std / (fan_in / s[0])**0.5)
      b = None if b_std is None else torch.randn((curr_out_chan,), generator=rng) * b_std
    else:
      raise ValueError(f'Parameterization not supported: {parameterization}')

    return tuple(out_shape), (W, b)

  def apply_fn(params, inputs, **kwargs):
    W, b = params
    in_chan = inputs.shape[channel_axis]
    
    if parameterization == 'ntk':
      fan_in = in_chan * math.prod(filter_shape)
      norm = W_std / fan_in**0.5
      b_scale = b_std
    elif parameterization == 'standard':
      norm = 1. / s[0]**0.5
      b_scale = 1.
      
    ndim = inputs.ndim
    _batch = batch_axis % ndim
    _chan = channel_axis % ndim
    spatial_axes = tuple(i for i in range(ndim) if i != _chan and i != _batch)

    perm = (_batch, _chan) + spatial_axes
    x = inputs.permute(perm)

    def _periodic_wrap_sum(res, target_shape, axes):
      for i, ax in enumerate(axes):
        target = target_shape[i]
        curr = res.shape[ax]
        if curr > target:
          out = res.narrow(ax, 0, target).clone()
          for start in range(target, curr, target):
            chunk_len = min(target, curr - start)
            out.narrow(ax, 0, chunk_len).add_(res.narrow(ax, start, chunk_len))
          res = out
      return res

    if not shared_weights:
      # ConvLocal implementation
      # PyTorch doesn't have a direct ConvLocal layer, so we use unfold + einsum
      if len(spatial_axes) == 1:
        patches = torch.nn.functional.unfold(x.unsqueeze(-1), kernel_size=(filter_shape[0], 1), stride=(strides[0], 1), padding=0)
        # patches: (N, InC*Vol, L)
        patches = patches.transpose(1, 2) # (N, L, InC*Vol)
        # W: (L, OutC, InC*Vol)
        out = torch.einsum('blk, lok -> bol', patches, W)
        out_spatial_shape = [(x.shape[2] - filter_shape[0]) // strides[0] + 1]
        out = out.reshape((x.shape[0], out.shape[1], *out_spatial_shape))
      elif len(spatial_axes) == 2:
        if padding == Padding.SAME:
            pad_h = max((math.ceil(x.shape[2] / strides[0]) - 1) * strides[0] + filter_shape[0] - x.shape[2], 0)
            pad_w = max((math.ceil(x.shape[3] / strides[1]) - 1) * strides[1] + filter_shape[1] - x.shape[3], 0)
            x = torch.nn.functional.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        
        patches = torch.nn.functional.unfold(x, kernel_size=filter_shape, stride=strides, padding=0)
        patches = patches.transpose(1, 2)
        out = torch.einsum('blk, lok -> bol', patches, W)
        out_spatial_shape = [((x.shape[2+i] - filter_shape[i]) // strides[i] + 1) for i in range(len(spatial_axes))]
        out = out.reshape((x.shape[0], out.shape[1], *out_spatial_shape))
      else:
        raise NotImplementedError("ConvLocal only supported for 1D and 2D spatial.")
    else:
      # Conv or ConvTranspose
      target_spatial_shape = [inputs.shape[ax] * strides[i] if transpose else math.ceil(inputs.shape[ax] / strides[i]) for i, ax in enumerate(spatial_axes)]
      
      cur_padding = 0
      if padding == Padding.SAME and not transpose:
          padding_tuple = []
          for i in range(len(spatial_axes)):
              pad_val = max((math.ceil(x.shape[2+i] / strides[i]) - 1) * strides[i] + filter_shape[i] - x.shape[2+i], 0)
              padding_tuple.insert(0, pad_val // 2)
              padding_tuple.insert(0, pad_val - pad_val // 2)
          x = torch.nn.functional.pad(x, tuple(padding_tuple))
      elif padding == Padding.CIRCULAR and not transpose:
          padding_tuple = []
          for i in range(len(spatial_axes)):
              pad_val = max((math.ceil(x.shape[2+i] / strides[i]) - 1) * strides[i] + filter_shape[i] - x.shape[2+i], 0)
              padding_tuple.insert(0, pad_val // 2)
              padding_tuple.insert(0, pad_val - pad_val // 2)
          x = torch.nn.functional.pad(x, tuple(padding_tuple), mode='circular')

      if not transpose:
        conv_fn = {1: torch.nn.functional.conv1d, 2: torch.nn.functional.conv2d, 3: torch.nn.functional.conv3d}[len(spatial_axes)]
        out = conv_fn(x, W, bias=None, stride=strides, padding=cur_padding)
      else:
        conv_fn = {1: torch.nn.functional.conv_transpose1d, 2: torch.nn.functional.conv_transpose2d, 3: torch.nn.functional.conv_transpose3d}[len(spatial_axes)]
        out = conv_fn(x, W, bias=None, stride=strides, padding=0)
        
        if padding == Padding.SAME:
          for i in range(len(spatial_axes)):
            ax = 2 + i
            diff = filter_shape[i] - strides[i]
            p_low = diff // 2
            out = out.narrow(ax, p_low, target_spatial_shape[i])
        elif padding == Padding.CIRCULAR:
          out = _periodic_wrap_sum(out, target_spatial_shape, [2 + i for i in range(len(spatial_axes))])

    out = out * norm
    if b is not None:
      b_reshape = [1]*out.ndim
      b_reshape[1] = out.size(1)
      out = out + b.view(b_reshape) * b_scale

    inv_perm = [0]*ndim
    for i, p in enumerate(perm):
      inv_perm[p] = i
    return out.permute(inv_perm)

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal(output=Bool.NO if shared_weights else Bool.YES))
  def kernel_fn(k: Kernel, **kwargs):
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
    is_reversed = k.is_reversed

    if not shared_weights and not k.diagonal_spatial:
        nngp = get_diagonal(nngp, False, False)
        if ntk is not None:
            ntk = get_diagonal(ntk, False, False)
        if cov1 is not None:
            cov1 = get_diagonal(cov1, k.diagonal_batch, False)
        if cov2 is not None:
            cov2 = cov1 if (cov2 is cov1 or k.x1_is_x2) else get_diagonal(cov2, k.diagonal_batch, False)
        k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk,
                      diagonal_spatial=True)

    def _periodic_wrap_sum_kernel(res, target):
        ndim = res.ndim
        # res has shape (..., H1, H2) at the point of call in loop
        curr = res.shape[-1]
        if curr > target:
            # Pad to multiple
            pad_len = (target - (curr % target)) % target
            res = torch.nn.functional.pad(res, (0, pad_len, 0, pad_len))
            # Reshape and sum
            new_curr = res.shape[-1]
            num_blocks = new_curr // target
            new_shape = res.shape[:-2] + (num_blocks, target, num_blocks, target)
            res = res.reshape(new_shape).sum(dim=(-4, -2))
        return res

    def conv_unscaled(mat, b_ndim):
      if mat is None:
        return mat
      if mat.ndim == 0:
        return torch.zeros_like(nngp) if id(mat) != id(nngp) else torch.zeros([], device=mat.device, dtype=mat.dtype)
      
      ndim = mat.ndim
      
      cur_filter = filter_shape
      cur_strides = strides
      if is_reversed and not k.diagonal_spatial:
          cur_filter = cur_filter[::-1]
          cur_strides = cur_strides[::-1]

      if k.diagonal_spatial:
        sp_axes_len = (ndim - b_ndim)
        
        flat_b = math.prod(mat.shape[:b_ndim])
        x = mat.reshape((flat_b, 1) + mat.shape[b_ndim:])

        if padding == Padding.SAME:
            padding_tuple = []
            for i in range(sp_axes_len):
              pad_val = max((math.ceil(x.shape[2+i] / cur_strides[i]) - 1) * cur_strides[i] + cur_filter[i] - x.shape[2+i], 0)
              padding_tuple.insert(0, pad_val - pad_val // 2)
              padding_tuple.insert(0, pad_val // 2)
            x = pad(x, tuple(padding_tuple))
        elif padding == Padding.CIRCULAR:
            padding_tuple = []
            for i in range(sp_axes_len):
              pad_val = max((math.ceil(x.shape[2+i] / cur_strides[i]) - 1) * cur_strides[i] + cur_filter[i] - x.shape[2+i], 0)
              padding_tuple.insert(0, pad_val - pad_val // 2)
              padding_tuple.insert(0, pad_val // 2)
            x = pad(x, tuple(padding_tuple), mode='circular')

        pool_fn = {1: avg_pool1d, 2: avg_pool2d, 3: avg_pool3d}[sp_axes_len]
        f_size = math.prod(cur_filter)
        out = pool_fn(x, kernel_size=cur_filter, stride=cur_strides, padding=0, divisor_override=f_size)
        
        return out.reshape(mat.shape[:b_ndim] + out.shape[2:])
      else:
        out = mat
        for i in range(ndim - 1, b_ndim, -2):
          sp_i = (i - b_ndim) // 2
          out = torch.movedim(out, (i - 1, i), (-2, -1))
          
          preshape = out.shape[:-2]
          x = out.contiguous().reshape(-1, 1, out.shape[-2], out.shape[-1])
          
          K = cur_filter[sp_i]
          S = cur_strides[sp_i]
          
          L_in = out.shape[-2]
          cur_target = int(L_in * S if transpose else math.ceil(L_in / S))

          if not transpose:
              if padding == Padding.SAME or padding == Padding.CIRCULAR:
                  pad_val1 = max((math.ceil(x.shape[-2] / S) - 1) * S + K - x.shape[-2], 0)
                  pad_val2 = max((math.ceil(x.shape[-1] / S) - 1) * S + K - x.shape[-1], 0)
                  padding_tuple = (pad_val2 // 2, pad_val2 - pad_val2 // 2, pad_val1 // 2, pad_val1 - pad_val1 // 2)
                  x = pad(x, padding_tuple, mode='circular' if padding == Padding.CIRCULAR else 'constant')
              
              w = _get_identity_filter(K, x.device, x.dtype)
              x = conv2d(x, w, stride=(S, S))
          else:
              # Transpose
              w = _get_identity_filter(K, x.device, x.dtype)
              x = conv_transpose2d(x, w, stride=(S, S), padding=0)
              
              if padding == Padding.SAME:
                  diff = K - S
                  p_low = diff // 2
                  x = x.narrow(-2, p_low, cur_target).narrow(-1, p_low, cur_target)
              elif padding == Padding.CIRCULAR:
                  x = _periodic_wrap_sum_kernel(x, cur_target)
          
          out = x.reshape(preshape + x.shape[-2:])
        return out

    def affine(out, scale, shift, b_ndim):
      if out is not None:
        if scale != 1.0:
          out = out * scale
        if shift is not None and shift != 0.0:
          out = out + shift
      return out
      
    b_std_sq = b_std**2 if b_std is not None else None
    def conv(lhs, b_ndim):
       unscaled = conv_unscaled(lhs, b_ndim)
       return affine(unscaled, W_std**2, b_std_sq, b_ndim)

    cov1 = conv(cov1, 1 if k.diagonal_batch else 2)
    if cov2 is not None:
      cov2 = cov1 if (cov2 is cov1 or k.x1_is_x2) else conv(cov2, 1 if k.diagonal_batch else 2)
    
    if parameterization == 'ntk':
      nngp = conv(nngp, 2)
      if ntk is not None:
        ntk = (conv_unscaled(ntk, 2) + nngp) if W_std == 1.0 else (W_std**2 * conv_unscaled(ntk, 2) + nngp)
    elif parameterization == 'standard':
      nngp_unscaled = conv_unscaled(nngp, 2)
      if ntk is not None:
        fan_in = k.shape1[channel_axis] * math.prod(filter_shape)
        ntk = (fan_in / s[0] * nngp_unscaled + W_std ** 2 * conv_unscaled(ntk, 2))
        if b_std is not None:
          ntk = affine(ntk, 1, 1., 2)
      nngp = affine(nngp_unscaled, W_std**2, b_std_sq, 2)

    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     is_gaussian=True,
                     is_reversed=not is_reversed if shared_weights and not k.diagonal_spatial else is_reversed,
                     is_input=False)



  def mask_fn(mask, input_shape):
    return torch.all(mask, dim=channel_axis, keepdim=True)

  return init_fn, apply_fn, kernel_fn, mask_fn



class PositionalEmbedding(enum.Enum):
  NONE = 'NONE'
  SUM = 'SUM'
  CONCAT = 'CONCAT'


class AttentionMechanism(enum.Enum):
  SOFTMAX = 'SOFTMAX'
  IDENTITY = 'IDENTITY'
  ABS = 'ABS'
  RELU = 'RELU'

  def fn(self):
    def _softmax(x, axis=-1):
        if axis == -1 or isinstance(axis, int):
            return torch.nn.functional.softmax(x, dim=axis)
        else:
            x_max = torch.amax(x, dim=axis, keepdim=True)
            unnorm = torch.exp(x - x_max.detach())
            return unnorm / torch.sum(unnorm, dim=axis, keepdim=True)

    return {
        'softmax': _softmax,
        'identity': lambda x: x,
        'abs': torch.abs,
        'relu': torch.nn.functional.relu
    }[self.name.lower()]


def _pos_emb_identity(shape: Sequence[int]) -> torch.Tensor:
  size = utils.size_at(shape)
  R = torch.eye(size).reshape(tuple(shape) * 2)
  R = utils.zip_axes(R)
  return R


def _pos_emb_pdist(
    shape: Sequence[int],
    pos_emb_p_norm: Optional[float],
    pos_emb_decay_fn: Optional[Callable[[float], float]],
) -> torch.Tensor:
  if pos_emb_decay_fn is None:
    return _pos_emb_identity(shape)

  ndim = len(shape)
  # Default to CPU if we have no other info, but this is usually called with meta device during init
  dev = 'cpu'
  R = torch.zeros((1,) * (ndim * 2), device=dev)
  for axis in range(ndim):
    d = torch.arange(shape[axis], dtype=torch.float32, device=dev)
    pd = utils.outer_prod(d, d, 0, d.ndim, op.sub)
    pd = pd.reshape((1,) * (2 * axis) +
                    pd.shape +
                    (1,) * (2 * (ndim - axis - 1)))
    R = R + torch.abs(pd) ** pos_emb_p_norm

  R = pos_emb_decay_fn(R)
  return R


def _shape_and_axes(
    x: tuple[int, ...],
    ignore_axes: Iterable[int] = (),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
  ndim = len(x)
  ignore_axes = tuple(i % ndim for i in ignore_axes)
  axes = tuple(i for i in range(ndim) if i not in ignore_axes)
  shape = tuple(x[i] for i in axes)
  return shape, axes


def _get_all_pos_emb(
    k: Kernel,
    pos_emb_type: PositionalEmbedding,
    pos_emb_p_norm: float,
    pos_emb_decay_fn: Optional[Callable[[float], float]],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
  if pos_emb_type == PositionalEmbedding.NONE:
    return None, None, None

  shape, _ = _shape_and_axes(k.shape1, (k.batch_axis, k.channel_axis))
  R = _pos_emb_pdist(shape, pos_emb_p_norm, pos_emb_decay_fn)
  R = R.to(k.cov1.device)

  if k.is_reversed:
    R = utils.reverse_zipped(R)

  batch_ndim = 1 if k.diagonal_batch else 2
  R11 = R.view((1,) * batch_ndim + R.shape) if batch_ndim > 0 else R
  R12 = R11 if batch_ndim == 2 else R.view((1, 1) + R.shape)
  R22 = None if k.cov2 is None else R11

  mask11, mask12, mask22 = k._get_mask_prods(k.mask1, k.mask2)
  R11 = utils.mask(R11, mask11)
  R12 = utils.mask(R12, mask12)
  R22 = utils.mask(R22, mask22)
  return R11, R12, R22


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
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
) -> InternalLayerMasked:
  QK_std = W_query_std * W_key_std
  OV_std = W_out_std * W_value_std

  pos_emb_type = PositionalEmbedding(pos_emb_type)
  attention_mechanism = AttentionMechanism(attention_mechanism)

  @functools.lru_cache(1)
  def get_pos_emb_L(spatial_shape):
    size = utils.size_at(spatial_shape)
    R = _pos_emb_pdist(spatial_shape, pos_emb_p_norm, pos_emb_decay_fn)
    R = utils.unzip_axes(R)
    L = torch.linalg.cholesky(R.reshape(size, size)).reshape(R.shape)
    return L

  def init_fn(rng, input_shape):
    _channel_axis = channel_axis % len(input_shape)
    output_shape = (input_shape[:_channel_axis] + (n_chan_out,) +
                    input_shape[_channel_axis + 1:])
    if rng is None:
      return tuple(output_shape), (None, None, None, None, None, None)

    n_chan_in_keys = n_chan_in_vals = input_shape[channel_axis]

    if pos_emb_type == PositionalEmbedding.NONE:
      pos_emb = None
    else:
      _n_chan_pos_emb = (n_chan_in_keys if n_chan_pos_emb is None
                         else n_chan_pos_emb)

      pos_emb_shape = list(input_shape)
      pos_emb_shape[channel_axis] = _n_chan_pos_emb
      pos_emb_shape[batch_axis] = 1
      pos_emb = torch.randn(pos_emb_shape, generator=rng)

      if pos_emb_type == PositionalEmbedding.CONCAT:
        n_chan_in_keys += _n_chan_pos_emb
        if val_pos_emb:
          n_chan_in_vals += _n_chan_pos_emb

    key_matrices = torch.randn((n_heads, n_chan_in_keys, n_chan_key), generator=rng)
    val_matrices = torch.randn((n_heads, n_chan_in_vals, n_chan_val), generator=rng)
    W_out = torch.randn((n_chan_val * n_heads, n_chan_out), generator=rng)

    if b_std is None:
      b = None
    else:
      b_shape = [1] * len(input_shape)
      b_shape[_channel_axis] = n_chan_out
      b = torch.randn(b_shape, generator=rng)

    if linear_scaling:
      query_matrices = None
      warnings.warn('Linear scaling attention used -> query initialization '
                    'ignored, tying the weights '
                    '(see docstring for more details).')
    else:
      query_matrices = torch.randn((n_heads, n_chan_in_keys, n_chan_key), generator=rng)

    return (tuple(output_shape),
            (query_matrices, key_matrices, val_matrices, W_out, b, pos_emb))

  def apply_fn(params: PyTree,
               inputs: torch.Tensor,
               mask: Optional[torch.Tensor] = None,
               **kwargs) -> torch.Tensor:
    query_matrices, key_matrices, val_matrices, W_out, b, pos_emb = params

    spatial_shape, spatial_axes = _shape_and_axes(inputs.shape,
                                                  (batch_axis, channel_axis))
    n = inputs.shape[batch_axis]

    if pos_emb is not None:
      if pos_emb_decay_fn is not None:
        L = get_pos_emb_L(spatial_shape).to(inputs.device)
        first = tuple(range(L.ndim // 2))
        last = tuple(range(L.ndim // 2, L.ndim))
        pos_emb = torch.tensordot(L, pos_emb, dims=(last, spatial_axes))
        pos_emb = torch.movedim(pos_emb, first, spatial_axes)

      if mask is not None:
        pos_emb = torch.where(mask.to(torch.bool), torch.zeros((), dtype=pos_emb.dtype, device=pos_emb.device), pos_emb)

      pos_emb = pos_emb * W_pos_emb_std

    if pos_emb_type == PositionalEmbedding.SUM:
      inputs_val = None if val_pos_emb else inputs
      inputs = pos_emb + inputs

    elif pos_emb_type == PositionalEmbedding.CONCAT:
      inputs_val = inputs if not val_pos_emb else None
      _n_chan_pos_emb = (inputs.shape[channel_axis] if n_chan_pos_emb is None
                         else n_chan_pos_emb)
      _channel_axis = channel_axis % inputs.ndim
      
      pos_emb_shape = inputs.shape[:_channel_axis] + (_n_chan_pos_emb,) + inputs.shape[_channel_axis + 1:]
      pos_emb = pos_emb.expand(pos_emb_shape)
      inputs = torch.cat([inputs, pos_emb], dim=channel_axis)

    elif pos_emb_type == PositionalEmbedding.NONE:
      inputs_val = None

    if inputs_val is not None:
       inputs_val = torch.movedim(inputs_val, (batch_axis, channel_axis), (0, -1))
       inputs_val = inputs_val.reshape((n, -1, inputs_val.shape[-1]))

    inputs = torch.movedim(inputs, (batch_axis, channel_axis), (0, -1))
    inputs = inputs.reshape((n, -1, inputs.shape[-1]))

    def _inputs_dot(matrices, _inputs=inputs):
      ret = torch.tensordot(_inputs, matrices, dims=([-1], [-2]))
      return torch.movedim(ret, 2, 0)

    if inputs_val is not None:
      values = _inputs_dot(val_matrices, inputs_val)
      n_chan_in = inputs_val.shape[-1]
    else:
      values = _inputs_dot(val_matrices)
      n_chan_in = inputs.shape[-1]

    keys = _inputs_dot(key_matrices)
    if linear_scaling:
      queries = keys
    else:
      queries = _inputs_dot(query_matrices)

    G_mat = torch.matmul(queries, torch.movedim(keys, -1, -2))
    norm = inputs.shape[-1] * n_chan_key ** (1 if linear_scaling else 0.5)
    G_mat = G_mat * (QK_std / norm)

    if mask is not None:
      mask = torch.all(mask.to(torch.bool), dim=channel_axis, keepdim=True)
      mask = torch.movedim(mask, (batch_axis, channel_axis), (0, -1))
      mask = mask.reshape((1, mask.shape[0], 1, -1))

      if attention_mechanism == AttentionMechanism.SOFTMAX:
        G_mat = torch.where(mask, torch.tensor(-float('inf'), dtype=G_mat.dtype, device=G_mat.device), G_mat)
      elif attention_mechanism in (AttentionMechanism.IDENTITY,
                                   AttentionMechanism.RELU,
                                   AttentionMechanism.ABS):
        G_mat = torch.where(mask, torch.zeros((), dtype=G_mat.dtype, device=G_mat.device), G_mat)
      else:
        raise NotImplementedError(attention_mechanism, mask)

    G_mat = attention_mechanism.fn()(G_mat)
    
    heads = torch.matmul(G_mat, values)
    heads = torch.movedim(heads, 0, -1)
    heads = heads.reshape(heads.shape[:-2] + (-1,))

    outputs = torch.matmul(heads, W_out)
    outputs = outputs * (OV_std / (n_chan_val * n_heads * n_chan_in) ** 0.5)

    outputs = outputs.reshape((n,) + spatial_shape + (n_chan_out,))
    outputs = torch.movedim(outputs, (0, -1), (batch_axis, channel_axis))
    if b is not None:
      outputs = outputs + b_std * b
    return outputs

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal(input=Bool.NO))
  def kernel_fn(k: Kernel, **kwargs):
    R1, R12, R2 = _get_all_pos_emb(k, pos_emb_type, pos_emb_p_norm, pos_emb_decay_fn)

    def _get_interpolation_coefficients():
      input_weight, pos_emb_weight = 1., W_pos_emb_std**2

      if pos_emb_type == PositionalEmbedding.CONCAT:
        n_chan_input = k.shape1[channel_axis]
        _n_chan_pos_emb = (k.shape1[channel_axis] if n_chan_pos_emb is None
                           else n_chan_pos_emb)
        n_chan_total = n_chan_input + _n_chan_pos_emb

        input_weight *= n_chan_input / n_chan_total
        pos_emb_weight *= _n_chan_pos_emb / n_chan_total

      return input_weight, pos_emb_weight

    def weighted_sum(x, y, x_weight, y_weight):
      if x is None or y is None:
        return x
      return x_weight * x + y_weight * y.to(x.device)

    kern_weight, pos_emb_weight = _get_interpolation_coefficients()

    cov1_interp = weighted_sum(k.cov1, R1, kern_weight, pos_emb_weight)
    cov2_interp = None if k.cov2 is None else (cov1_interp if (k.cov2 is k.cov1 or k.x1_is_x2) else weighted_sum(k.cov2, R2, kern_weight, pos_emb_weight))

    if val_pos_emb or (not linear_scaling and
                       attention_mechanism == AttentionMechanism.IDENTITY):
      nngp_interp = weighted_sum(k.nngp, R12, kern_weight, pos_emb_weight)
      ntk_interp = weighted_sum(k.ntk, R12, kern_weight, pos_emb_weight)

    if linear_scaling:
      def _get_weighting(mat, mask):
        if mat is None:
          return None

        if not k.diagonal_batch:
          mat = torch.movedim(torch.diagonal(mat, offset=0, dim1=0, dim2=1), -1, 0)

        if mask is not None:
          mask = torch.all(mask.to(torch.bool), dim=channel_axis, keepdim=True)
          mask = torch.squeeze(torch.movedim(mask, (batch_axis, channel_axis), (0, -1)), dim=-1)
          if k.is_reversed:
             mask = torch.movedim(mask, tuple(range(1, mask.ndim)), tuple(range(mask.ndim -1, 0, -1)))
          mask = utils.interleave_ones(mask, 1, mask.ndim, x_first=False)
          
          if attention_mechanism == AttentionMechanism.SOFTMAX:
            mat = torch.where(mask, torch.tensor(-float('inf'), dtype=mat.dtype, device=mat.device), mat)
          else:
            mat = torch.where(mask, torch.zeros((), dtype=mat.dtype, device=mat.device), mat)

        if attention_mechanism == AttentionMechanism.SOFTMAX:
          axes = tuple(range(mat.ndim))
          return attention_mechanism.fn()(QK_std * mat, axis=axes[2::2])
        else:
          return attention_mechanism.fn()(QK_std * mat)

      def _weigh_kernel(mat, G1, G2=None):
        if mat is not None and mat.ndim != 0 and G1 is not None:
          G2 = G1 if G2 is None else G2

          G1_dims = tuple(range(1, G1.ndim))
          G2_dims = tuple(range(G1.ndim, G1.ndim + G2.ndim - 1))
          mat_dims = tuple(utils.zip_flat(G1_dims[1::2], G2_dims[1::2]))
          res_dims = tuple(utils.zip_flat(G1_dims[::2], G2_dims[::2]))

          G1_dims = (0,) + G1_dims

          if mat.ndim % 2:
            G2_dims = (0,) + G2_dims
            mat_dims = (0,) + mat_dims
            res_dims = (0,) + res_dims
          else:
            G2_dims = (-1,) + G2_dims
            mat_dims = (0, -1) + mat_dims
            res_dims = (0, -1) + res_dims

          def make_einsum_equation(*dim_lists):
             alphabet = string.ascii_letters
             all_dims = []
             for lst in dim_lists:
                 for d in lst:
                     if d not in all_dims:
                         all_dims.append(d)
             dim_map = {d: alphabet[i] for i, d in enumerate(all_dims)}
             eq = ""
             for i, lst in enumerate(dim_lists[:-1]):
                 eq += "".join(dim_map[d] for d in lst)
                 if i < len(dim_lists) - 2:
                     eq += ","
             eq += "->" + "".join(dim_map[d] for d in dim_lists[-1])
             return eq

          equation = make_einsum_equation(G1_dims, mat_dims, G2_dims, res_dims)
          mat = torch.einsum(equation, G1, mat, G2)
        return _affine(mat, OV_std, b_std)

      G1 = _get_weighting(cov1_interp, k.mask1)
      G2 = None if k.cov2 is None else (G1 if (k.cov2 is k.cov1 or k.x1_is_x2) else _get_weighting(cov2_interp, k.mask2))

      cov1 = _weigh_kernel(cov1_interp if val_pos_emb else k.cov1, G1)
      cov2 = None if k.cov2 is None else (cov1 if (k.cov2 is k.cov1 or k.x1_is_x2) else _weigh_kernel(cov2_interp if val_pos_emb else k.cov2, G2))
      nngp = _weigh_kernel(nngp_interp if val_pos_emb else k.nngp, G1, G2)
      
      if k.ntk is None:
        ntk = None
      else:
        ntk_to_weigh = ntk_interp if val_pos_emb else k.ntk
        if ntk_to_weigh is not None:
             ntk = _weigh_kernel(ntk_to_weigh, G1, G2)
             if b_std is None:
                 ntk = ntk + 2 * nngp
             else:
                 ntk = ntk + 2 * (nngp - b_std**2)

    elif attention_mechanism == AttentionMechanism.IDENTITY:
      def dot(lhs, rhs, diagonal_batch=k.diagonal_batch):
        if lhs is None:
          return None

        c_axes = tuple(range(1 if diagonal_batch else 2, lhs.ndim))
        if rhs is None:
          return torch.sum(lhs ** 2, dim=c_axes, keepdim=True)

        rhs = rhs.expand(lhs.shape)
        res = torch.sum(lhs * rhs, dim=c_axes)
        return res.reshape(res.shape + (1,) * len(c_axes))

      dot11 = dot(cov1_interp, None if val_pos_emb else k.cov1)
      dot12 = dot(nngp_interp, None if val_pos_emb else k.nngp, False)
      dot22 = None if k.cov2 is None else (dot11 if (k.cov2 is k.cov1 or k.x1_is_x2) else dot(cov2_interp, None if val_pos_emb else k.cov2))

      std = QK_std * OV_std

      nngp = _affine(dot12 * nngp_interp, std, b_std)
      cov1 = _affine(dot11 * cov1_interp, std, b_std)
      cov2 = None if k.cov2 is None else (cov1 if (k.cov2 is k.cov1 or k.x1_is_x2) else _affine(None if dot22 is None else dot22 * cov2_interp, std, b_std))

      if ntk_interp is not None:
        if val_pos_emb or pos_emb_type == PositionalEmbedding.NONE:
          nngp_dot_ntk = dot(nngp_interp, ntk_interp, False)
          ntk = 2 * nngp_dot_ntk
        else:
          nngp_dot_ntk_1 = dot(nngp_interp, k.ntk, False)
          nngp_dot_ntk_2 = dot(k.nngp, ntk_interp, False)
          ntk = (nngp_dot_ntk_1 + nngp_dot_ntk_2)

        ntk = _affine(
            ntk * nngp_interp + dot12 * (ntk_interp + 4 * nngp_interp),
            std,
            b_std)
      else:
        ntk = None
    else:
      raise NotImplementedError(f'No known closed form expression for square '
                                f'root scaling and {attention_mechanism} '
                                f'attention mechanism.')

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk, is_gaussian=True)

  def mask_fn(mask, input_shape):
    return torch.all(mask.to(torch.bool), dim=channel_axis, keepdim=True)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=True)
def ImageResize(
    shape: Sequence[int],
    method: str = 'bilinear',
    antialias: bool = True,
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
) -> InternalLayerMasked:
  def _shape(input_shape):
    return tuple(s if s != -1 else input_shape[i] for i, s in enumerate(shape))

  def init_fn(rng, input_shape):
    return _shape(input_shape), ()

  def apply_fn(params, x, **kwargs):
    x_shape = _shape(x.shape)
    spatial_size = tuple(x_shape[i] for i in range(len(x_shape)) if i != batch_axis % x.ndim and i != channel_axis % x.ndim)
    x = torch.movedim(x, (batch_axis, channel_axis), (0, -1))
    
    original_ndim = x.ndim
    if x.ndim == 3:
      x = x.unsqueeze(1)
    elif x.ndim == 4:
      pass
    elif x.ndim == 5:
      pass
    else:
      raise NotImplementedError(f"ImageResize apply_fn does not support {x.ndim}D tensor yet.")
    
    x = x.movedim(-1, 1)

    # PyTorch's interpolate needs float and 3D/4D/5D shapes
    original_dtype = x.dtype
    x = x.float()

    if len(spatial_size) == 1:
        bmode = "linear" if method == "bilinear" else method
    elif len(spatial_size) == 2:
        bmode = "bilinear" if method == "bilinear" else method
    elif len(spatial_size) == 3:
        bmode = "trilinear" if method == "bilinear" else method
    else:
        bmode = method

    x = torch.nn.functional.interpolate(x, size=spatial_size, mode=bmode, antialias=antialias if bmode in ['bilinear', 'bicubic'] else False)
    x = x.to(dtype=original_dtype)
    x = x.movedim(1, -1)
    
    if original_ndim == 3:
        x = x.squeeze(1)

    x = torch.movedim(x, (0, -1), (batch_axis, channel_axis))
    return x

  def mask_fn(mask, input_shape):
    raise NotImplementedError('Masking with ImageResize in Torchtangents is yet to be fully ported.')

  batch_axis, channel_axis = utils.mod((batch_axis, channel_axis), shape)

  diagonal_batch = shape[batch_axis] == -1
  diagonal_spatial = Diagonal(
      input=Bool.NO
      if any(shape[i] != -1 for i in range(len(shape))
             if i not in (batch_axis, channel_axis))
      else Bool.YES)

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_batch=diagonal_batch,
            diagonal_spatial=diagonal_spatial)
  def kernel_fn(k: Kernel, **kwargs) -> Kernel:
    if method not in ('bilinear', 'nearest'):
      raise NotImplementedError(
          f'Only "bilinear" and "nearest" interpolation is '
          f'supported in `kernel_fn`, got {method}.')

    if shape[channel_axis] != -1:
      raise ValueError(f'Resizing the channel axis {channel_axis} is not '
                       f'well-defined in the infinite-width limit. ')

    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
    diagonal_spatial = k.diagonal_spatial

    def resize(kern, shape1, shape2, diagonal_batch):
      if kern is None or getattr(kern, "ndim", 0) == 0:
        return kern

      k_shape = (shape1[batch_axis],)
      if not diagonal_batch:
        k_shape += (shape2[batch_axis],)

      for i, (s1, s2) in enumerate(zip(shape1, shape2)):
        if i not in (batch_axis, channel_axis):
          k_shape += (s1,)
          if not diagonal_spatial:
            k_shape += (s2,)

      spatial_size = tuple(k_shape[i] for i in range(len(k_shape)) if i >= (1 if diagonal_batch else 2))
      if len(spatial_size) > 3:
          raise NotImplementedError('ImageResize kernel_fn only supports 1D, 2D, or 3D spatial dimensions in PyTorch.')
      
      dims = tuple(range(len(k_shape)))
      # Interpolate expects (N, C, D, H, W). Treat the batch dimensions as N and C?
      if diagonal_batch:
         kern_reshaped = kern.view((1, kern.shape[0]) + kern.shape[1:])
      else:
         kern_reshaped = kern.view((1, kern.shape[0]*kern.shape[1]) + kern.shape[2:])

      # Interpolation expects floating-point tensor
      original_dtype = kern_reshaped.dtype
      kern_reshaped = kern_reshaped.float()
      bmode = "bilinear" if method == "bilinear" and len(spatial_size) == 2 else ("trilinear" if method == "bilinear" and len(spatial_size) == 3 else ("linear" if method == "bilinear" and len(spatial_size) == 1 else method))

      kern_resized = torch.nn.functional.interpolate(kern_reshaped, size=spatial_size, mode=bmode, antialias=antialias if bmode in ['bilinear', 'bicubic'] else False)
      kern_resized = kern_resized.to(dtype=original_dtype)
      return kern_resized.reshape(k_shape)

    shape1 = _shape(k.shape1)
    shape2 = _shape(k.shape2)

    cov1_out = resize(cov1, shape1, shape1, k.diagonal_batch)
    cov2_out = None if cov2 is None else (cov1_out if (cov2 is cov1 or k.x1_is_x2) else resize(cov2, shape2, shape2, k.diagonal_batch))
    k_out = k.replace(cov1=cov1_out,
                  nngp=resize(nngp, shape1, shape2, False),
                  cov2=cov2_out,
                  ntk=resize(ntk, shape1, shape2, False))
    return k_out

  return init_fn, apply_fn, kernel_fn, mask_fn


class _Slice:
  def __getitem__(self, idx: Any) -> Any:
    return idx

Slice = _Slice()

@layer
@supports_masking(remask_kernel=False)
def Index(
    idx: Any,
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
) -> InternalLayerMasked:
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


@layer
@supports_masking(remask_kernel=True)
def Aggregate(
    aggregate_axis: Optional[Sequence[int]] = None,
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
    to_dense: Optional[Callable[[torch.Tensor], torch.Tensor]] = lambda p: p,
    implementation: str = AggregateImplementation.DENSE.value,
) -> InternalLayerMasked:
  implementation = AggregateImplementation(implementation)
  if implementation == AggregateImplementation.SPARSE:
    warnings.warn('Negative indices in `pattern` are considered as padding '
                  '(i.e. ignored), unlike typical numpy negative indexing.')

  init_fn = lambda rng, input_shape: (input_shape, ())

  def get_agg_axes(ndim: int) -> tuple[tuple[int, ...], int, int]:
    _batch_axis, _channel_axis = utils.mod((batch_axis, channel_axis), ndim)
    if aggregate_axis is None:
      agg_axes = tuple(i for i in range(ndim)
                       if i not in (_batch_axis, _channel_axis))
    else:
      agg_axes = tuple(utils.canonicalize_axis(aggregate_axis, ndim))
    return agg_axes, _batch_axis, _channel_axis

  def get_dimension_numbers(ndim: int):
    # returns JAX lax.DotDimensionNumbers format
    agg_axes, batch_axis_in, _ = get_agg_axes(ndim)
    agg_ndim = len(agg_axes)
    return (((agg_axes, tuple(range(1, agg_ndim + 1))), ((batch_axis_in,), (0,))))

  def make_indices(index_array, agg_shape):
    index_array = torch.movedim(index_array, -1, 0)
    strides = [1] * len(agg_shape)
    for i in range(len(agg_shape)-2, -1, -1):
        strides[i] = strides[i+1] * agg_shape[i+1]
    
    raveled = torch.zeros(index_array.shape[1:], dtype=torch.long, device=index_array.device)
    for i in range(len(agg_shape)):
        raveled = raveled + index_array[i] * strides[i]
        
    mask = torch.all(index_array >= 0, dim=0)
    return torch.where(mask, raveled, torch.tensor(-1, dtype=torch.long, device=index_array.device))

  def get_senders_receivers(pattern, batch_size: int, agg_ndim: int):
    if pattern.shape[-1] != 2:
      raise ValueError('`pattern` must have a trailing dimension of 2, got '
                       f'{pattern.shape[-1]}.')
    s, r = pattern[..., 0], pattern[..., 1]

    if agg_ndim == 1 and s.ndim == 2:
      s, r = s.unsqueeze(-1), r.unsqueeze(-1)

    if s.ndim != 3:
      raise ValueError(f'Tails and heads need to be 3-dimensional, '
                       f'got {s.ndim}.')

    if s.shape[2] != agg_ndim:
      raise ValueError(f'Trailing dimension of tails and heads need to have '
                       f'the same size as the number of aggregate axes of '
                       f'`aggregate_axis` ({agg_ndim}), got {s.shape[2]}.')

    if s.shape[0] != batch_size:
      raise ValueError(f'Tails and heads need to have leading dimension equal '
                       f'to batch size, got {s.shape[0]}.')

    return s, r

  def apply_fn(params, inputs: torch.Tensor, *, pattern: Optional[torch.Tensor] = None, **kwargs):
    if pattern is None:
      return inputs
      
    ndim = inputs.ndim
    agg_axes, batch_axis_in, channel_axis_in = get_agg_axes(ndim)
    agg_ndim = len(agg_axes)

    if implementation == AggregateImplementation.DENSE:
      pattern = to_dense(pattern)
      
      # torch.einsum instead of lax.dot_general
      alphabet = string.ascii_letters
      lhs_chars = list(alphabet[:ndim])
      rhs_chars = []
      
      rhs_chars.append(lhs_chars[batch_axis_in])
      for ax in agg_axes:
          rhs_chars.append(lhs_chars[ax])
          
      out_chars = list(lhs_chars)
      for ax in agg_axes:
          new_char = alphabet[len(alphabet) // 2 + ax]
          rhs_chars.append(new_char)
          out_chars[ax] = new_char
          
      lhs_str = "".join(lhs_chars)
      rhs_str = "".join(rhs_chars)
      out_str = "".join(out_chars)
      equation = f"{lhs_str},{rhs_str}->{out_str}"
      
      out = torch.einsum(equation, inputs, pattern.to(inputs.dtype))

    elif implementation == AggregateImplementation.SPARSE:
      s, r = get_senders_receivers(pattern, inputs.shape[batch_axis_in], agg_ndim)

      src_axes = (batch_axis_in,) + agg_axes + (channel_axis_in,)
      dst_axes = (0,) + tuple(range(1, agg_ndim + 1)) + (-1,)

      inputs = torch.movedim(inputs, src_axes, dst_axes)
      input_shape = inputs.shape
      inputs = inputs.reshape((inputs.shape[0],
                               functools.reduce(
                                   operator.mul, inputs.shape[1:agg_ndim + 1], 1))
                              + inputs.shape[agg_ndim + 1:])

      agg_shape = input_shape[1:agg_ndim + 1]
      s, r = make_indices(s, agg_shape), make_indices(r, agg_shape)

      n_nodes = inputs.shape[1]
      
      out_list = []
      for b in range(inputs.shape[0]):
          sender_in_b = inputs[b, s[b]]
          mask = r[b] >= 0
          sender_in_b = sender_in_b[mask]
          r_b = r[b][mask]
          
          out_b = torch.zeros((n_nodes,) + sender_in_b.shape[1:], dtype=inputs.dtype, device=inputs.device)
          if len(r_b) > 0:
              shape_for_index = [-1] + [1]*(sender_in_b.ndim - 1)
              out_b.scatter_add_(0, r_b.view(shape_for_index).expand_as(sender_in_b), sender_in_b)
          out_list.append(out_b)
          
      out = torch.stack(out_list, dim=0)
      out = out.reshape(input_shape)
      out = torch.movedim(out, dst_axes, src_axes)
    else:
      raise ValueError(f'Unrecognized `implementation == {implementation}.')

    return out

  @requires(batch_axis=batch_axis, channel_axis=channel_axis, diagonal_spatial=Diagonal(input=Bool.NO, output=Bool.NO))
  def kernel_fn(k: Kernel, *, pattern: tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None), **kwargs) -> Kernel:
    pattern1, pattern2 = pattern

    if pattern1 is None and pattern2 is None:
      return k

    if pattern1 is None or pattern2 is None:
      raise NotImplementedError('Having exactly one of two `pattern1/2=None` is not implemented. ')

    ndim = len(k.shape1)
    agg_axes, batch_axis_in, channel_axis_in = get_agg_axes(ndim)
    agg_ndim = len(agg_axes)
    agg_shape = tuple(k.shape1[a] for a in agg_axes)
    agg_size = functools.reduce(operator.mul, agg_shape, 1)

    def bucket_axes(ndim, start_axis):
      ndim_spatial = (ndim - start_axis) // 2
      agg_1 = tuple(a - int(batch_axis_in < a) - int(channel_axis_in < a) + start_axis for a in agg_axes)
      agg_2 = tuple(a + ndim_spatial for a in agg_1)
      non_agg_1 = tuple(a for a in range(start_axis, start_axis + ndim_spatial) if a not in agg_1)
      non_agg_2 = tuple(a for a in range(start_axis + ndim_spatial, ndim) if a not in agg_2)
      return tuple(range(start_axis)), agg_1, agg_2, non_agg_1, non_agg_2

    if implementation == AggregateImplementation.DENSE:
      pattern1 = None if pattern1 is None else to_dense(pattern1)
      pattern2 = None if pattern2 is None else to_dense(pattern2)

      k = k.dot_general(
          other1=pattern1,
          other2=pattern2,
          is_lhs=False,
          dimension_numbers=get_dimension_numbers(ndim)
      )

      def transpose(mat, diagonal_batch):
        if mat is None or mat.ndim == 0:
          return mat

        start_axis = 1 if diagonal_batch else 2
        mat = utils.unzip_axes(mat, start_axis)
        b, agg_1, agg_2, non_agg_1, non_agg_2 = bucket_axes(mat.ndim, start_axis)
        permutation = b + non_agg_1 + agg_1 + non_agg_2 + agg_2
        mat = torch.permute(mat, tuple(np.argsort(permutation).tolist()))
        return utils.zip_axes(mat, start_axis)

      cov1_out = transpose(k.cov1, k.diagonal_batch)
      cov2_out = None if k.cov2 is None else (cov1_out if (k.cov2 is k.cov1 or k.x1_is_x2) else transpose(k.cov2, k.diagonal_batch))
      k = k.replace(
          cov1=cov1_out,
          cov2=cov2_out,
          nngp=transpose(k.nngp, False),
          ntk=transpose(k.ntk, False),
          batch_axis=batch_axis_in % ndim,
          channel_axis=channel_axis_in % ndim
      )

    elif implementation == AggregateImplementation.SPARSE:
      raise NotImplementedError('Aggregate SPARSE kernel_fn logic not yet mapped from JAX vmap.')
    return k

  def mask_fn(mask, input_shape):
    raise NotImplementedError('Aggregate mask_fn in Torchtangents is yet to be fully ported.')

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def DotGeneral(
    *,
    lhs: Optional[Union[torch.Tensor, float]] = None,
    rhs: Optional[Union[torch.Tensor, float]] = None,
    dimension_numbers: tuple[tuple[Sequence[int], Sequence[int]], tuple[Sequence[int], Sequence[int]]] = (((), ()), ((), ())),
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
) -> InternalLayerMasked:
  if rhs is not None and lhs is not None:
    raise ValueError('At most one of constant `rhs` and `lhs` can be non-`None`'
                     ', since the other factor is considered to be the layer `inputs`.')
  is_lhs = rhs is None
  if is_lhs:
    other = torch.tensor(lhs) if not isinstance(lhs, torch.Tensor) else lhs
  else:
    other = torch.tensor(rhs) if not isinstance(rhs, torch.Tensor) else rhs

  def dot_fn(x):
    args = (x, other.to(x.dtype).to(x.device))[::(-1 if is_lhs else 1)]
    return utils.dot_general(*args, dimension_numbers)

  def init_fn(rng, input_shape):
    x_meta = torch.empty(input_shape, device='meta')
    out = dot_fn(x_meta)
    return tuple(out.shape), ()

  def apply_fn(params, inputs, **kwargs):
    return dot_fn(inputs)

  input_cs = dimension_numbers[0][1 if is_lhs else 0]
  diagonal_batch = (batch_axis not in input_cs) or (rhs is None and lhs is None)
  diagonal_spatial = Diagonal(
      input=Bool.YES
      if (input_cs in ((), (batch_axis,)) or (rhs is None and lhs is None))
      else Bool.NO)

  @requires(diagonal_batch=diagonal_batch,
            diagonal_spatial=diagonal_spatial,
            batch_axis=batch_axis,
            channel_axis=channel_axis)
  def kernel_fn(k: Kernel, **kwargs) -> Kernel:
    return k.dot_general(other, other, is_lhs, dimension_numbers)

  def mask_fn(mask, input_shape):
    mask_shape = list(input_shape)
    mask_shape[channel_axis] = mask.shape[channel_axis]
    br_mask = mask.expand(mask_shape)
    return ~dot_fn(~br_mask)

  return init_fn, apply_fn, kernel_fn, mask_fn

@layer
@supports_masking(remask_kernel=True)
def LayerNorm(
    axis: Axes = -1,
    eps: float = 1e-12,
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
) -> InternalLayer:
  def init_fn(rng, input_shape):
    return input_shape, ()

  def apply_fn(params, inputs, mask=None, **kwargs):
    _axis = utils.canonicalize_axis(axis, inputs)
    mean, var = req.mean_and_var(inputs, _axis, keepdims=True, mask=mask,
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
    
    # We need to make _axis a list so we can remove from it
    _axis = list(utils.canonicalize_axis(axis, k.shape1))

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
        m = torch.movedim(m, tuple(range(1, m.ndim)), tuple(range(m.ndim - 1, 0, -1)))

      return m

    prod11, prod12, prod22 = req.get_diagonal_outer_prods(
        eps + cov1,
        cov2 if cov2 is None else eps + cov2,
        k.diagonal_batch,
        k.diagonal_spatial,
        op.mul,
        axis=kernel_axis,
        mask1=prepare_mask(k.mask1),
        mask2=prepare_mask(k.mask2),
        )

    nngp = nngp / torch.sqrt(prod12)

    if ntk is not None:
      ntk = ntk / torch.sqrt(prod12)

    cov1 = cov1 / torch.sqrt(prod11)
    if cov2 is not None:
      cov2 = cov1 if (cov2 is cov1 or k.x1_is_x2) else cov2 / torch.sqrt(prod22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return init_fn, apply_fn, kernel_fn
