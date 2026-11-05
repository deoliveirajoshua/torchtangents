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

"""Requirement management for `torchtangents.stax` layers."""

import dataclasses
import enum
import math
from typing import Any, Callable, Optional, Sequence, Union
import warnings

import torch
import numpy as np

from ..utils import dataclasses as nt_dataclasses
from ..utils import utils
from ..utils.kernel import Kernel
from ..utils.typing import AnalyticKernelFn
from ..utils.typing import ApplyFn
from ..utils.typing import Axes
from ..utils.typing import Get
from ..utils.typing import InitFn
from ..utils.typing import InternalLayer
from ..utils.typing import KernelOrInput
from ..utils.typing import Layer
from ..utils.typing import LayerKernelFn
from ..utils.typing import NTTree
from ..utils.typing import PyTree


@dataclasses.dataclass(frozen=True)
class ShapedArray:
  shape: tuple[int, ...]
  dtype: torch.dtype


class StaxKernelModule(torch.nn.Module):
    """
    An nn.Module wrapper for torchtangents.stax kernel functions.
    Ensures that kernel functions can be used natively as PyTorch modules.
    """
    def __init__(self, kernel_fn: AnalyticKernelFn, name: Optional[str] = None):
        super().__init__()
        self.kernel_fn = kernel_fn
        self._name = name or getattr(kernel_fn, '__name__', 'KernelModule')

    def forward(self, x1: KernelOrInput, x2: Optional[NTTree[torch.Tensor]] = None, get: Get = None, **kwargs) -> Union[NTTree[Kernel], NTTree[torch.Tensor]]:
        """
        Computes the kernel. If x1 is a tt.Kernel, computes the next layer in the limit.
        If x1 is a Tensor, starts the kernel computation from the input.
        """
        return self.kernel_fn(x1, x2, get=get, **kwargs)
        
    def kernels(self, x1: KernelOrInput, x2: Optional[NTTree[torch.Tensor]] = None, get: Get = ('nngp', 'ntk'), **kwargs):
        """Unified interface for querying kernels."""
        return self.kernel_fn(x1, x2, get=get, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f"StaxKernelModule({self._name})"

    def __getattr__(self, name):
        # Delegate to the underlying kernel_fn for metadata like requirements
        if name == 'kernel_fn':
            return self.__dict__.get('kernel_fn')
        try:
            return super().__getattr__(name)
        except AttributeError:
            if 'kernel_fn' in self.__dict__:
                return getattr(self.kernel_fn, name)
            raise



def layer(layer_fn: Callable[..., Any]) -> Callable[..., Layer]:
  name = layer_fn.__name__

  @utils.wraps(layer_fn)
  def new_layer_fns(*args, **kwargs):
    res = layer_fn(*args, **kwargs)
    
    # Handle both old style (init, apply, kernel) and new style (kernel only)
    if isinstance(res, tuple) and len(res) >= 3:
        init_fn, apply_fn, kernel_fn = res[:3]
    else:
        # If it's already just a kernel_fn, we still need dummies for _preprocess
        kernel_fn = res
        init_fn = getattr(kernel_fn, '_init_fn', lambda rng, shape: (shape, ()))
        apply_fn = getattr(kernel_fn, '_apply_fn', lambda params, x, **kwargs: x)
        
    kernel_fn = _preprocess_kernel_fn(init_fn, apply_fn, kernel_fn)
    # Store these on the function itself for internal machinery (e.g. combinators)
    kernel_fn._init_fn = init_fn
    kernel_fn._apply_fn = apply_fn
    kernel_fn.__name__ = name
    
    return StaxKernelModule(kernel_fn, name)

  return new_layer_fns


def requires(**static_reqs):
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
                raise ValueError(f'{kernel_fn} requires `{key} == {v}`, but '
                                 f'input kernel has `{key} == True`, hence '
                                 f'does not contain sufficient information. '
                                 f'Please recompute the input kernel with '
                                 f'`{key} == {v}`.')

            elif key in ('batch_axis', 'channel_axis'):
              ndim = len(k.shape1)
              v_kernel = getattr(k, key)
              v_pos = v % ndim
              if v_kernel != v_pos:
                raise ValueError(f'{kernel_fn} requires `{key} == {v_pos}`, '
                                 f'but input kernel has `{key} == {v_kernel}`, '
                                 f'making the infinite limit ill-defined.')

      return kernel_fn(k, **kwargs)

    _set_req(new_kernel_fn, dict(static_reqs))
    return new_kernel_fn

  return req


def supports_masking(remask_kernel: bool):
  def supports_masking_wrapper(layer):

    @utils.wraps(layer)
    def layer_with_masking(*args, **kwargs) -> InternalLayer:
      layer_fns = layer(*args, **kwargs)
      
      if isinstance(layer_fns, StaxKernelModule):
          init_fn = getattr(layer_fns.kernel_fn, '_init_fn', None)
          apply_fn = getattr(layer_fns.kernel_fn, '_apply_fn', None)
          kernel_fn = layer_fns.kernel_fn
          mask_fn_orig = getattr(kernel_fn, '_mask_fn', lambda mask, input_shape: mask)
          is_triple = False
      else:
          init_fn, apply_fn, kernel_fn = layer_fns[:3]
          mask_fn_orig = layer_fns[3] if len(layer_fns) > 3 else lambda mask, input_shape: mask
          is_triple = True

      @utils.wraps(mask_fn_orig)
      def mask_fn(mask, input_shape):
        if mask is None:
          return None
        return mask_fn_orig(mask, input_shape)

      def apply_fn_with_masking(params, inputs, *, mask_constant=None, **kwargs):
        def _apply_get_masked(x):
            if isinstance(x, (torch.Tensor, MaskedArray)):
                return _get_masked_array(x, mask_constant)
            if isinstance(x, dict):
                return {k: _apply_get_masked(v) for k, v in x.items()}
            if isinstance(x, (tuple, list)):
                return type(x)(_apply_get_masked(v) for v in x)
            return x
            
        def _extract(x, get_mask):
            if isinstance(x, MaskedArray):
                return x.mask if get_mask else x.masked_value
            if isinstance(x, dict):
                return {k: _extract(v, get_mask) for k, v in x.items()}
            if isinstance(x, (tuple, list)):
                return type(x)(_extract(v, get_mask) for v in x)
            return x

        masked_inputs = _apply_get_masked(inputs)
        inputs_only = _extract(masked_inputs, False)
        mask_only = _extract(masked_inputs, True)

        kwargs.pop('mask', None)
        outputs = apply_fn(params, inputs_only, mask=mask_only, **kwargs)
        outputs_mask = mask_fn(mask_only,
                               inputs_only.shape if isinstance(inputs_only, torch.Tensor)
                               else [i.shape for i in inputs_only])
        if outputs_mask is None:
          return outputs
        return MaskedArray(outputs, outputs_mask)

      def kernel_fn_with_masking(k: NTTree[Kernel], **user_reqs):
        def _extract_k(obj, field):
            if isinstance(obj, Kernel):
                return getattr(obj, field)
            if isinstance(obj, dict):
                return {k: _extract_k(v, field) for k, v in obj.items()}
            if isinstance(obj, (tuple, list)):
                return type(obj)(_extract_k(v, field) for v in obj)
            return obj

        mask1 = _extract_k(k, 'mask1')
        shape1 = _extract_k(k, 'shape1')
        mask2 = _extract_k(k, 'mask2')
        shape2 = _extract_k(k, 'shape2')

        mask1, mask2 = mask_fn(mask1, shape1), mask_fn(mask2, shape2)

        k = kernel_fn(k, **user_reqs)

        if remask_kernel:
          remask_fn = lambda k, m1, m2: k.mask(m1, m2)
        else:
          remask_fn = lambda k, m1, m2: k.replace(mask1=m1, mask2=m2)

        def _remask_k(k_obj, m1_obj, m2_obj):
            if isinstance(k_obj, Kernel):
                return remask_fn(k_obj, m1_obj, m2_obj)
            if isinstance(k_obj, dict):
                return {key: _remask_k(k_obj[key], m1_obj[key], m2_obj[key]) for key in k_obj.keys()}
            if isinstance(k_obj, list):
                return [_remask_k(k_obj[i], m1_obj[i], m2_obj[i]) for i in range(len(k_obj))]
            if isinstance(k_obj, tuple):
                return tuple(_remask_k(k_obj[i], m1_obj[i], m2_obj[i]) for i in range(len(k_obj)))
            return k_obj

        k = _remask_k(k, mask1, mask2)
        return k

      if _has_req(kernel_fn):
        _set_req(kernel_fn_with_masking, get_req(kernel_fn))

      return init_fn, apply_fn_with_masking, kernel_fn_with_masking

    return layer_with_masking

  return supports_masking_wrapper


def unmask_fn(fn: ApplyFn) -> ApplyFn:
  def unmask(x: Union[MaskedArray, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, MaskedArray):
      x = utils.mask(x.masked_value, x.mask)
    return x

  @utils.wraps(fn)
  def fn_no_mask(*args, **kwargs):
    out = fn(*args, **kwargs)
    out = torch.utils._pytree.tree_map_only((torch.Tensor, MaskedArray), unmask, out)
    return out

  return fn_no_mask


from collections import namedtuple
MaskedArray = namedtuple('MaskedArray', ['masked_value', 'mask'])


def _get_masked_array(
    x: Union[None, torch.Tensor, ShapedArray, MaskedArray],
    mask_constant: Optional[float] = None
) -> MaskedArray:
  if x is None:
    mask_mat = None

  elif isinstance(x, MaskedArray):
    x, mask_mat = x.masked_value, x.mask

  elif isinstance(x, (np.ndarray, torch.Tensor, float, int)):
    if mask_constant is None:
      mask_mat = None
    else:
      if isinstance(x, torch.Tensor):
        if math.isnan(mask_constant):
            mask_mat = torch.isnan(x)
        else:
            mask_mat = x == mask_constant
      else:
        if math.isnan(mask_constant):
            mask_mat = np.isnan(x)
        else:
            mask_mat = x == mask_constant
  else:
    raise TypeError(x, type(x))

  x = utils.mask(x, mask_mat)
  return MaskedArray(x, mask_mat)


_INPUT_REQ = 'input_req'


def get_req(
    f: Callable,
    default: Optional[dict] = None
) -> dict:
  return getattr(f, _INPUT_REQ, default)


def _set_req(f: Callable, req: dict):
  setattr(f, _INPUT_REQ, req)


def _has_req(f: Callable) -> bool:
  return hasattr(f, _INPUT_REQ)


_DEFAULT_INPUT_REQ = {
    'diagonal_batch': True,
    'diagonal_spatial': False,
    'batch_axis': 0,
    'use_dropout': False,
    'channel_axis': 1, # This needs to be 1 because PyTorch convolutions expect the channel axis to be 1.
                       # Note: neural tangents uses -1 (last) for the channel axis.
    'mask_constant': None
}


class Bool(enum.IntEnum):
  NO = 0
  MAYBE = 1
  YES = 2

  def __and__(self, other: 'Bool') -> 'Bool':
    return min(self, other)

  __rand__ = __and__


@dataclasses.dataclass(frozen=True)
class Diagonal:
  input: Bool = Bool.YES
  output: Bool = Bool.NO

  def __rshift__(self, other: 'Diagonal') -> 'Diagonal':
    if self.output == Bool.YES:
      return self

    if self.output > Bool.NO and other.input > Bool.NO:
      input = self.input
    elif self.output == Bool.NO and other.input < Bool.YES:
      input = Bool.NO
    else:
      input = min(self.input, other.input)

    return Diagonal(input=input, output=other.output)

  def __and__(self, other: 'Diagonal') -> 'Diagonal':
    return Diagonal(input=self.input & other.input,
                    output=self.output & other.output)

  def __bool__(self) -> bool:
    return self.input == Bool.YES and self.output > Bool.NO

  def __lshift__(self, other: 'Diagonal') -> 'Diagonal':
    return other >> self

  __rand__ = __and__


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
  ndim = x.ndim
  batch_axis %= ndim
  channel_axis %= ndim

  letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  in_eq1 = list(letters[:ndim])
  in_eq2 = list(letters[:ndim])

  in_eq2[channel_axis] = in_eq1[channel_axis]
  
  c2 = ndim
  for i in range(ndim):
    if i != channel_axis and i != batch_axis:
      in_eq2[i] = letters[c2]
      c2 += 1

  out_eq = [in_eq1[batch_axis]]
  for i in range(ndim):
    if i != channel_axis and i != batch_axis:
      out_eq.append(in_eq1[i])
  for i in range(ndim):
    if i != channel_axis and i != batch_axis:
      out_eq.append(in_eq2[i])

  eq = f"{''.join(in_eq1)},{''.join(in_eq2)}->{''.join(out_eq)}"
  ret = torch.einsum(eq, x, x)
  ret = utils.zip_axes(ret, 1)
  return ret


def _cov_full_batch_full_spatial(
    x1: torch.Tensor,
    x2: torch.Tensor,
    batch_axis: int,
    channel_axis: int
) -> torch.Tensor:
  ret = torch.tensordot(x1, x2, dims=([channel_axis], [channel_axis]))
  new_batch_axis = batch_axis - (1 if batch_axis > channel_axis else 0)
  ret = torch.movedim(ret, (new_batch_axis, x1.ndim - 1 + new_batch_axis),
                      (0, 1))
  ret = utils.zip_axes(ret, 2)
  return ret


def _cov_full_batch_diag_spatial(
    x1: torch.Tensor,
    x2: torch.Tensor,
    batch_axis: int,
    channel_axis: int
) -> torch.Tensor:
  ndim = x1.ndim
  batch_axis %= ndim
  channel_axis %= ndim

  # This is an artifact of PyTorch's einsum implementation
  # TODO: make letters a global constant
  letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  in_eq1 = list(letters[:ndim])
  in_eq2 = list(letters[:ndim])

  in_eq2[batch_axis] = letters[ndim]

  out_eq = [in_eq1[batch_axis], in_eq2[batch_axis]]
  for i in range(ndim):
    if i != channel_axis and i != batch_axis:
      out_eq.append(in_eq1[i])

  eq = f"{''.join(in_eq1)},{''.join(in_eq2)}->{''.join(out_eq)}"
  ret = torch.einsum(eq, x1, x2)
  return ret


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
  if not (isinstance(x1, (np.ndarray, torch.Tensor)) and
          (x2 is None or isinstance(x2, (np.ndarray, torch.Tensor)))):
    raise TypeError(('Wrong input types given. Found `x1` of type '
                     f'{type(x1)} and `x2` of type {type(x2)}, need both to be'
                     f'`torch.Tensor`s (`x2` can be `None`).'))

  if isinstance(x1, np.ndarray):
    x1 = torch.from_numpy(x1)
  if isinstance(x2, np.ndarray):
    x2 = torch.from_numpy(x2)

  batch_axis %= x1.ndim
  diagonal_spatial = bool(diagonal_spatial)

  if batch_axis != 0:
    warnings.warn(f'!!! Non-leading (!= 0) batch dimension in the '
                  f'input layer is not supported for batching '
                  f'kernels, got batch_axis = {batch_axis}. !!!')

  if channel_axis is None:
    def flatten(x):
      if x is None:
        return x
      return torch.movedim(x, batch_axis, 0).reshape((x.shape[batch_axis], -1))

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

    x = _get_masked_array(x, mask_constant)
    x, mask = x.masked_value, x.mask

    if not x.is_floating_point():
      x = x.to(torch.get_default_dtype())

    if diagonal_batch:
      cov = _cov_diag_batch(x, diagonal_spatial, batch_axis, channel_axis)
    else:
      cov = _cov(x, x, diagonal_spatial, batch_axis, channel_axis)

    return x, cov, mask

  x1, cov1, mask1 = get_x_cov_mask(x1)
  if x2 is None:
    x2, cov2, mask2 = None, None, None
    x1_is_x2 = True
  elif utils.x1_is_x2(x1, x2, eps=eps):
    x2, cov2, mask2 = x1, cov1, mask1
    x1_is_x2 = True
  else:
    x2, cov2, mask2 = get_x_cov_mask(x2)
    x1_is_x2 = False
  nngp = _cov(x1, x2, diagonal_spatial, batch_axis, channel_axis)

  ntk = torch.zeros_like(nngp) if compute_ntk else None
  is_gaussian = False
  is_reversed = False
  pass
  is_input = False

  return Kernel(
      cov1=cov1,
      cov2=cov2,
      nngp=nngp,
      ntk=ntk,
      x1_is_x2=x1_is_x2,
      is_gaussian=is_gaussian,
      is_reversed=is_reversed,
      is_input=is_input,
      diagonal_batch=diagonal_batch,
      diagonal_spatial=diagonal_spatial,
      shape1=x1.shape,
      shape2=x1.shape if x2 is None else x2.shape,
      batch_axis=batch_axis,
      channel_axis=channel_axis,
      mask1=mask1,
      mask2=mask2,
  )


@torch._dynamo.disable
def _propagate_shape(
    init_fn: InitFn,
    apply_fn: ApplyFn,
    shaped: PyTree,
    **kwargs
) -> PyTree:
  try:
    # Use 'meta' device for shape evaluation to avoid actual memory
    # allocation and CPU/CUDA syncs.
    dev = 'meta'
    
    def get_shape(s):
        return s.shape if isinstance(s, ShapedArray) else s
        
    input_shape = torch.utils._pytree.tree_map(get_shape, shaped)

    # We mock randn to skip unnecessary generator logic entirely.
    # This avoids 'cpu device' warnings because we never look at torch.default_generator.
    old_randn = torch.randn
    @torch.no_grad()
    def _mock_randn(*args, **kwargs):
        kwargs.pop('generator', None)
        kwargs['device'] = dev
        # Filter sizes to ensure we handle SymInts/Tensors correctly
        new_args = []
        for a in args:
            if isinstance(a, torch.Tensor):
                new_args.append(a.shape[0] if a.ndim > 0 else int(a))
            else:
                new_args.append(a)
        return torch.empty(*new_args, **kwargs)

    try:
        torch.randn = _mock_randn
        # For analytic cases, rng can be None. If layers fail on None, 
        # we provide a dummy that is ignored by our mock.
        out_shape, params = init_fn(None, input_shape)
    finally:
        torch.randn = old_randn
    
    def build_shaped(s, dtype=torch.float32):
        if hasattr(s, 'shape'):
            return ShapedArray(tuple(s.shape), getattr(s, 'dtype', dtype))
        if isinstance(s, (list, tuple)):
            if all(isinstance(x, (int, torch.Tensor)) for x in s):
                try:
                    return ShapedArray(tuple(int(x) for x in s), dtype)
                except: pass
            return type(s)(build_shaped(v) for v in s)
        if isinstance(s, dict):
            return {k: build_shaped(v) for k, v in s.items()}
        return s

    shaped = build_shaped(out_shape)
  except NotImplementedError:
    pass

  if isinstance(shaped, MaskedArray):
    shaped = shaped.masked_value

  return shaped


# Global cache for shapes to speed up repeated calls
_SHAPE_CACHE = {}

@torch._dynamo.disable
def _set_shapes(
    init_fn: InitFn,
    apply_fn: ApplyFn,
    in_kernel: NTTree[Kernel],
    out_kernel: NTTree[Kernel],
    **kwargs
) -> NTTree[Kernel]:
  """Sets the shapes of `out_kernel` based on the input shapes and functions.
  Optimized with a cache to avoid graph breaks during torch.compile.
  """
  # Stable cache key for Kernel shapes
  def get_kernel_shapes(k):
      if isinstance(k, Kernel):
          return (k.shape1, k.shape2, k.nngp.dtype)
      if isinstance(k, dict):
          return tuple(sorted((key, get_kernel_shapes(v)) for key, v in k.items()))
      if isinstance(k, (list, tuple)):
          return tuple(get_kernel_shapes(v) for v in k)
      return None

  # Cache key based on stable names and structural shapes
  # Dynamo can change id(fn), so we prefer __module__ + __name__ if available.
  def fn_to_key(fn):
      if hasattr(fn, '__qualname__'):
          return (fn.__module__, fn.__qualname__)
      return id(fn)

  try:
      k_shapes = get_kernel_shapes(in_kernel)
      # We stringify the shapes if they contain SymInts, or use a tuple for static ints.
      cache_key = (fn_to_key(init_fn), fn_to_key(apply_fn), k_shapes)
  except Exception:
      cache_key = None 

  if cache_key is not None and cache_key in _SHAPE_CACHE:
      shape1, shape2 = _SHAPE_CACHE[cache_key]
  else:
      def tree_map_kernel(fn, tree):
          if isinstance(tree, Kernel):
              return fn(tree)
          if hasattr(tree, 'items'):
              return {k: tree_map_kernel(fn, v) for k, v in tree.items()}
          if isinstance(tree, (tuple, list)):
              return type(tree)(tree_map_kernel(fn, v) for v in tree)
          return tree

      s1 = tree_map_kernel(lambda k: ShapedArray(k.shape1, k.nngp.dtype), in_kernel)
      s2 = tree_map_kernel(lambda k: ShapedArray(k.shape2, k.nngp.dtype), in_kernel)

      k1, k2 = utils.split_kwargs(kwargs)
      shape1 = _propagate_shape(init_fn, unmask_fn(apply_fn), s1, **k1)
      shape2 = _propagate_shape(init_fn, unmask_fn(apply_fn), s2, **k2)
      
      _SHAPE_CACHE[cache_key] = (shape1, shape2)

  def _set_shape_map(k_obj, s1_obj, s2_obj):
      if isinstance(k_obj, Kernel):
          def get_actual_shape(s):
              return s.shape if hasattr(s, 'shape') else tuple(s)
          new_s1 = get_actual_shape(s1_obj)
          new_s2 = get_actual_shape(s2_obj)
          if k_obj.shape1 == new_s1 and k_obj.shape2 == new_s2:
              return k_obj
          return k_obj.replace(shape1=new_s1, shape2=new_s2)
      if hasattr(k_obj, 'items'):
          return {key: _set_shape_map(k_obj[key], s1_obj[key], s2_obj[key]) for key in k_obj.keys()}
      if isinstance(k_obj, (list, tuple)):
          return type(k_obj)(_set_shape_map(k_obj[i], s1_obj[i], s2_obj[i]) for i in range(len(k_obj)))
      return k_obj

  return _set_shape_map(out_kernel, shape1, shape2)


def _fuse_requirements(
    kernel_fn_reqs,
    default_reqs,
    **user_reqs
) -> dict:
  kernel_fn_reqs = dict(kernel_fn_reqs)
  for k, v_user in user_reqs.items():
    if v_user is not None:
      if k in kernel_fn_reqs:
        v_kernel = kernel_fn_reqs[k]
        if (v_user is True and
            (v_kernel is False or
             (isinstance(kernel_fn_reqs[k], Diagonal) and
              kernel_fn_reqs[k].input == Bool.NO))):
          raise ValueError(f'Asked to compute `kernel_fn` output with '
                           f'`{k} == {v_user}`, while `kernel_fn` '
                           f'requires `{k} == {kernel_fn_reqs[k]}`.')

      kernel_fn_reqs[k] = v_user

  for k, v_user in default_reqs.items():
    if k not in kernel_fn_reqs:
      kernel_fn_reqs[k] = v_user

  return kernel_fn_reqs


def _preprocess_kernel_fn(
    init_fn: InitFn,
    apply_fn: ApplyFn,
    kernel_fn: LayerKernelFn
) -> AnalyticKernelFn:
  if not _has_req(kernel_fn):
    kernel_fn = requires()(kernel_fn)

  def kernel_fn_kernel(kernel, **kwargs):
    out_kernel = kernel_fn(kernel, **kwargs)
    return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

  def kernel_fn_x1(x1, x2, get, **kwargs):
    kernel_fn_reqs = get_req(kernel_fn)
    reqs = _fuse_requirements(kernel_fn_reqs, _DEFAULT_INPUT_REQ, **kwargs)
    compute_ntk = (get is None) or ('ntk' in get)

    def _map_x1(obj):
        if isinstance(obj, (torch.Tensor, MaskedArray)):
            return None
        if hasattr(obj, 'items'):
            return {k: _map_x1(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return type(obj)(_map_x1(v) for v in obj)
        return None

    if x2 is None:
      x2 = _map_x1(x1)

    def input_fn(x1, x2):
      return _inputs_to_kernel(x1, x2, compute_ntk=compute_ntk, **reqs)
      
    def _tree_map2(t1, t2):
        if hasattr(t1, 'items'):
            return {k: _tree_map2(t1[k], t2[k]) for k in t1}
        elif isinstance(t1, (list, tuple)):
            return type(t1)(_tree_map2(op1, op2) for op1, op2 in zip(t1, t2))
        return input_fn(t1, t2)

    kernel = _tree_map2(x1, x2)

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
    def all_of(x, cls: type) -> bool:
      if isinstance(x, cls):
        return True
      if hasattr(x, 'items'):
        return all(all_of(v, cls) for v in x.values())
      if isinstance(x, (tuple, list)):
        return all(all_of(v, cls) for v in x)
      return False

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
    operation: Callable[[float, float], float],
    axis: Sequence[int] = (),
    mask1: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  axis = utils.canonicalize_axis(axis, cov1)

  cov1 = get_diagonal(cov1, diagonal_batch, diagonal_spatial)
  cov2 = cov1 if (cov2 is None or cov2 is cov1) else get_diagonal(cov2, diagonal_batch, diagonal_spatial)

  cov1, _ = mean_and_var(cov1, axis=axis, keepdims=True, mask=mask1)
  cov2, _ = (cov1, None) if (cov2 is cov1 and (mask2 is mask1 or mask2 is None)) else mean_and_var(cov2, axis=axis, keepdims=True, mask=mask2)

  if cov1 is None:
    raise ValueError('cov1 is None')
  end_axis = 1 if diagonal_spatial else cov1.ndim
  prod12 = utils.outer_prod(cov1, cov2, 0, end_axis, operation)

  start_axis = 1 if diagonal_batch else 0
  prod11 = utils.outer_prod(cov1, cov1, start_axis, end_axis, operation)
  prod22 = (utils.outer_prod(cov2, cov2, start_axis, end_axis, operation)
            if (cov2 is not None and cov2 is not cov1) else prod11)

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
  var_out = None
  if x is None:
    return x, var_out

  if mask is None:
    if axis is None:
        mean = torch.mean(x, dtype=dtype)
    elif len(axis) == 0:
        mean = x.to(dtype) if dtype is not None else x
    else:
        mean = torch.mean(x, dim=axis, keepdim=keepdims, dtype=dtype)

    if get_var:
      if axis is None:
          var_out = torch.var(x, correction=ddof)
      elif len(axis) == 0:
          var_out = torch.zeros_like(mean)
      else:
          var_out = torch.var(x, dim=axis, correction=ddof, keepdim=keepdims)

  else:
    axis = tuple(utils.canonicalize_axis(axis, x))
    if len(axis) == 0:
        return x, (torch.zeros_like(x) if get_var else None)

    size = utils.size_at(x, axis)
    mask = mask.expand_as(x)
    # torch.count_nonzero supports dims
    mask_size = torch.count_nonzero(mask, dim=axis)
    # expand the mask_size to the shape we need if keepdims is True
    # count_nonzero reduces dims, we'll restore if necessary implicitly
    # Wait, in JAX it did this. For PyTorch:
    for i in sorted(axis):
        mask_size = torch.unsqueeze(mask_size, i)
        
    size -= mask_size
    size = torch.clamp(size, min=1)

    mean = torch.sum(x, dim=axis, keepdim=True) / size
    if not keepdims:
      mean = torch.squeeze(mean, dim=axis)

    if get_var:
      var_out = torch.sum((x - mean) ** 2, dim=axis, keepdim=True) / (size - ddof)
      if not keepdims:
        var_out = torch.squeeze(var_out, dim=axis)

  return mean, var_out
