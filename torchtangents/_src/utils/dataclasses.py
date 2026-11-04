# Copyright 2020 Google LLC
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

"""Utilities for defining dataclasses that can be used with PyTorch transformations."""

import dataclasses
import functools
from typing import Any

import torch
from torch.utils import _pytree as tree


def dataclass(clz):
  """Create a class which can be passed to functional transformations."""
  data_clz = dataclasses.dataclass(frozen=True)(clz)
  meta_fields = []
  data_fields = []
  for name, field_info in data_clz.__dataclass_fields__.items():
    is_pytree_node = field_info.metadata.get('pytree_node', True)
    init = field_info.metadata.get('init', True)
    if init:
      if is_pytree_node:
        data_fields.append(name)
      else:
        meta_fields.append(name)

  # flatten
  def iterate_clz(x):
    meta = tuple(getattr(x, name) for name in meta_fields)
    data = tuple(getattr(x, name) for name in data_fields)
    return list(data), meta

  # unflatten
  def clz_from_iterable(data, meta):
    meta_args = tuple(zip(meta_fields, meta))
    data_args = tuple(zip(data_fields, data))
    kwargs = dict(meta_args + data_args)
    return data_clz(**kwargs)

  # Note:
  # In PyTorch, flatten returns (children, context), unflatten takes (children, context)
  # So iterate_clz returns (data, meta), which is exactly (children, context).
  # We intentionally DO NOT register this as a PyTree node in PyTorch here.
  # Neural Tangents often treats `Kernel` and `MaskedArray` as leaves during tree traversal,
  # but JAX allows this via `is_leaf=...`. PyTorch's tree_map lacks `is_leaf`, therefore
  # registering them as PyTrees causes PyTorch to over-flatten them.
  # By keeping them unregistered, they act as leaves and `tree_map_only` works correctly.
  # TODO: change this once torch changes pytree management

  @functools.wraps(
      dataclasses.replace,
      assigned=('__module__', '__name__', '__qualname__', '__annotations__'))
  def replace(self: data_clz, **kwargs) -> data_clz:
    return dataclasses.replace(self, **kwargs)

  @functools.wraps(
      dataclasses.asdict,
      assigned=('__module__', '__name__', '__qualname__', '__annotations__'))
  def asdict(self: data_clz) -> dict[str, Any]:
    return {
        f.name: getattr(self, f.name)
        for f in dataclasses.fields(self)
    }

  @functools.wraps(
      dataclasses.astuple,
      assigned=('__module__', '__name__', '__qualname__', '__annotations__'))
  def astuple(self: data_clz) -> tuple[Any, ...]:
    return tuple(
        getattr(self, f.name)
        for f in dataclasses.fields(self)
    )

  data_clz.replace = replace
  data_clz.asdict = asdict
  data_clz.astuple = astuple

  # Register with PyTorch's PyTree system to support torch.compile
  try:
    from torch.utils import _pytree as pytree
    
    def flatten_fn(obj):
      # Use the iterate_clz logic already defined
      data, meta = iterate_clz(obj)
      return data, meta
      
    def unflatten_fn(data, meta):
      return clz_from_iterable(data, meta)
      
    pytree.register_pytree_node(data_clz, flatten_fn, unflatten_fn)
  except (ImportError, AttributeError):
    pass

  return data_clz


def field(pytree_node: bool = True, **kwargs):
  metadata = {'pytree_node': pytree_node}
  if 'init' in kwargs:
    metadata['init'] = kwargs['init']
  return dataclasses.field(metadata=metadata, **kwargs)
