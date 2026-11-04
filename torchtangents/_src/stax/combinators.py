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

"""Layer combinators ported to PyTorch."""

import operator as op
from typing import Any, Callable
import warnings

import torch

from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer
from ..utils.typing import Layer
from ..utils.typing import LayerKernelFn
from ..utils.typing import NTTree
from ..utils.typing import NTTrees
from ..utils.typing import Shapes
from .requirements import Diagonal
from .requirements import get_req
from .requirements import layer
from .requirements import requires


@layer
def Sequential(*layers: Layer) -> InternalLayer:
  # layers are StaxKernelModule objects
  
  class SerialKernelModule(torch.nn.Module):
      def __init__(self, layers):
          super().__init__()
          self.layers = torch.nn.ModuleList(layers)
          
      def __call__(self, k, **kwargs):
          for layer in self.layers:
              k = layer(k, **kwargs)
          return k

  kernel_fns = [l.kernel_fn for l in layers]
  init_fns = [l._init_fn for l in layers]
  apply_fns = [l._apply_fn for l in layers]

  def init_fn(rng, input_shape):
    params = []
    for init in init_fns:
      input_shape, param = init(rng, input_shape)
      params.append(param)
    return input_shape, params

  def apply_fn(params, inputs, **kwargs):
    for fn, layer_params in zip(apply_fns, params):
      inputs = fn(layer_params, inputs, **kwargs)
    return inputs

  # We still need a function to wrap with @requires
  @requires(**_get_input_req_attr(kernel_fns, fold=op.rshift))
  def kernel_fn(k: NTTree[Kernel], **kwargs) -> NTTree[Kernel]:
    for f in kernel_fns:
      k = f(k, **kwargs)
    return k

  # Attach init/apply for the @layer decorator's StaxKernelModule to find
  kernel_fn._init_fn = init_fn
  kernel_fn._apply_fn = apply_fn
  
  return kernel_fn

serial = Sequential


@layer
def repeat(layer_tuple: Layer, n: int) -> InternalLayer:
  # layer_tuple is a StaxKernelModule
  kernel_fn_orig = layer_tuple.kernel_fn
  init_fn_orig = layer_tuple._init_fn
  apply_fn_orig = layer_tuple._apply_fn

  def init_fn_repeat(rng, input_shape):
    out_shape, _ = init_fn_orig(rng, input_shape)
    if out_shape != input_shape:
      raise ValueError(
          f'`init_fn` produces a different output shape {out_shape} than the '
          f'input shape {input_shape}. Please use the `serial(*([layer] * n)`) '
          f'construction in this setting.'
      )

    params_list = []
    for _ in range(n):
        _, param = init_fn_orig(rng, input_shape)
        params_list.append(param)

    if n == 0:
        return input_shape, ()

    # Stack tree elements across n iterations
    def stack_if_tensor(*xs):
        if len(xs) > 0 and isinstance(xs[0], torch.Tensor):
            return torch.stack(xs)
        if len(xs) > 0 and xs[0] is None:
            return None
        return tuple(xs)

    from torch.utils._pytree import tree_flatten, tree_unflatten
    flat_trees, specs = zip(*[tree_flatten(t) for t in params_list])
    spec = specs[0]
    stacked_leaves = [stack_if_tensor(*leaves) for leaves in zip(*flat_trees)]
    stacked_params = tree_unflatten(stacked_leaves, spec)
    return out_shape, stacked_params

  def apply_fn_repeat(params, inputs, **kwargs):
    if n == 0:
        return inputs
        
    for i in range(n):
        def get_slice(x):
            if isinstance(x, torch.Tensor):
                return x[i]
            if isinstance(x, tuple) and len(x) == n:
                return x[i]
            return x

        param_i = torch.utils._pytree.tree_map(get_slice, params)
        inputs = apply_fn_orig(param_i, inputs, **kwargs)
    return inputs

  @requires(**get_req(kernel_fn_orig))
  def kernel_fn_repeat(k: NTTree[Kernel], **kwargs) -> NTTree[Kernel]:
    for _ in range(n):
      k = kernel_fn_orig(k, **kwargs)
    return k

  kernel_fn_repeat._init_fn = init_fn_repeat
  kernel_fn_repeat._apply_fn = apply_fn_repeat
  return kernel_fn_repeat


@layer
def parallel(*layers: Layer) -> InternalLayer:
  kernel_fns = [l.kernel_fn for l in layers]
  init_fns = [l._init_fn for l in layers]
  apply_fns = [l._apply_fn for l in layers]

  def init_fn(rng: torch.Generator, input_shape: Shapes):
    out_shapes = []
    params = []
    for init, shape in zip(init_fns, input_shape):
      out_shape, param = init(rng, shape)
      out_shapes.append(out_shape)
      params.append(param)
    return type(input_shape)(out_shapes), type(input_shape)(params)

  def apply_fn(params, inputs, **kwargs):
    outputs = []
    for fn, layer_params, inp in zip(apply_fns, params, inputs):
      outputs.append(fn(layer_params, inp, **kwargs))
    return type(inputs)(outputs)

  @requires(**_get_input_req_attr(kernel_fns, fold=op.and_))
  def kernel_fn(ks: NTTrees[Kernel], **kwargs) -> NTTrees[Kernel]:
    return type(ks)(f(k, **kwargs) for k, f in zip(ks, kernel_fns))

  kernel_fn._init_fn = init_fn
  kernel_fn._apply_fn = apply_fn
  return kernel_fn


def _get_input_req_attr(
    kernel_fns: list[LayerKernelFn],
    fold: Callable[[Diagonal, Diagonal], Diagonal]
) -> dict:
  req = {}

  for f in kernel_fns:
    req_f = get_req(f, default={})

    for k, v in req_f.items():
      if k == 'use_dropout':
        if k in req and req[k] != v:
          raise ValueError('`use_dropout` is a single whole-network attribute '
                           'and cannot be set to different values.')
        req[k] = v

      elif k in ('batch_axis', 'channel_axis'):
        if k not in req:
          req[k] = v
        else:
          if fold is op.and_:
            if k in req and req[k] != v:
              if (req[k] >= 0 and v >= 0) or (req[k] < 0 and v < 0):
                warnings.warn(f'For `kernel_fn`, `{k}` parameters must match in'
                              f' all parallel branches, got {req[k]} and {v}. '
                              f'This WILL lead to [silent] errors if '
                              f'`kernel_fn` is called.')
              else:
                warnings.warn(f'Got potentially mismatching `{k}` values in '
                              f'parallel branches: {req[k]} and {v}.')

          elif fold is not op.rshift:
            raise ValueError(fold)

      elif k in ('diagonal_batch', 'diagonal_spatial'):
        if k in req:
          req[k] = fold(req[k], v)
        else:
          req[k] = v

      else:
        raise NotImplementedError(k)

  return req
