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

"""General-purpose internal utilities for PyTorch Fork."""

from collections import namedtuple
import functools
import inspect
import operator
import types
from typing import Any, Callable, Iterable, Optional, Sequence, Sized, TypeVar, Union
import warnings

import torch
import numpy as np
from torch.utils import _pytree as tree


PyTree = Any # Dummy edit to trigger pyre


Axes = Union[int, Sequence[int]]


def is_list_or_tuple(x) -> bool:
  return type(x) == list or type(x) == tuple


def is_nt_tree_of(x: Any, dtype: Any) -> bool:
  if isinstance(x, dtype):
    return True
  if not is_list_or_tuple(x):
    return False
  return all(is_nt_tree_of(_x, dtype) for _x in x)


def nt_tree_fn(
    nargs: Optional[int] = None,
    tree_structure_argnum: Optional[int] = None,
    reduce: Callable = lambda x: x
):
  def check_tree_structure(args):
    if any(is_list_or_tuple(x) for x in args):
      if not all(type(x) == type(args[0]) for x in args[1:]):
        raise TypeError(f'Inconsistent NTTree structure found. '
                        f'Node Types: {[type(x) for x in args]}.')
      for x in zip(*args):
        check_tree_structure(x)

  def tree_fn(fn):
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
      _nargs = len(args) if nargs is None else nargs
      recurse = tuple(args[i] for i in range(_nargs))
      norecurse = tuple(args[i] for i in range(_nargs, len(args)))

      structure_argnum = tree_structure_argnum
      if structure_argnum is None:
        check_tree_structure(recurse)
        structure_argnum = 0

      if is_list_or_tuple(args[structure_argnum]):
        list_or_tuple = type(args[structure_argnum])
        return reduce(list_or_tuple(
            wrapped_fn(*(xs + norecurse), **kwargs) for xs in zip(*recurse)))
      return fn(*args, **kwargs)
    return wrapped_fn
  return tree_fn


def all_none(x: Any, attr: Optional[str] = None) -> bool:
  if attr is None:
    get_fn = lambda x: x
  else:
    _attr: str = attr
    get_fn = lambda x: getattr(x, _attr)
  return tree.tree_all(tree.tree_map(lambda x: get_fn(x) is None, x))


def canonicalize_get(get):
  if get is None:
    return True, get

  if not get:
    raise ValueError('"get" must be non-empty.')

  get_is_not_tuple = isinstance(get, str)
  if get_is_not_tuple:
    get = (get,)

  get = tuple(s.lower() for s in get)
  if len(set(get)) < len(get):
    raise ValueError('All entries in "get" must be unique. Got {}'.format(get))
  return get_is_not_tuple, get


_KERNEL_NAMED_TUPLE_CACHE: dict[Any, Any] = {}


def _named_tuple_factory(name, get):
  key = (name, get)
  if key in _KERNEL_NAMED_TUPLE_CACHE:
    return _KERNEL_NAMED_TUPLE_CACHE[key]
  else:
    _KERNEL_NAMED_TUPLE_CACHE[key] = namedtuple(name, get)
    return _named_tuple_factory(name, get)


def _output_to_dict(output):
  if isinstance(output, dict):
    return output

  if hasattr(output, 'asdict'):
    return output.asdict()

  if hasattr(output, '_asdict'):
    return output._asdict()

  if isinstance(output, types.GeneratorType):
    return (_output_to_dict(out) for out in output)

  raise ValueError(type(output))


def wraps(f):
  def wrapper(g):
    @functools.wraps(f)
    def h(*args, **kwargs):
      return g(*args, **kwargs)

    if hasattr(f, "__signature__"):
      setattr(h, "__signature__", getattr(f, "__signature__"))
    return h
  return wrapper


def get_namedtuple(name):
  def getter_decorator(fn):
    try:
      argspec = inspect.getfullargspec(fn)
      get_index = argspec.args.index('get')
      defaults = argspec.defaults
    except:
      raise ValueError('`get_namedtuple` functions must have a `get` argument.')

    @wraps(fn)
    def getter_fn(*args, **kwargs):
      canonicalized_args = list(args)

      if 'get' in kwargs:
        get_is_not_tuple, get = canonicalize_get(kwargs['get'])
        kwargs['get'] = get
      elif get_index < len(args):
        get_is_not_tuple, get = canonicalize_get(args[get_index])
        canonicalized_args[get_index] = get
      elif defaults is None:
        raise ValueError(
            '`get_namedtuple` function must have a `get` argument provided or '
            'set by default.')
      else:
        get_is_not_tuple, get = canonicalize_get(defaults[get_index - len(args)])

      fn_out = fn(*canonicalized_args, **kwargs)

      @nt_tree_fn()
      def canonicalize_output(out):
        if get is None:
          if isinstance(out, dict):
            ReturnType = _named_tuple_factory(name, tuple(out.keys()))
            out = ReturnType(*out.values())
          return out

        out = _output_to_dict(out)

        if get_is_not_tuple:
          if isinstance(out, types.GeneratorType):
            return (output[get[0]] for output in out)
          else:
            return out[get[0]]

        ReturnType = _named_tuple_factory(name, get)
        if isinstance(out, types.GeneratorType):
          return (ReturnType(*tuple(output[g] for g in get)) for output in out)
        else:
          return ReturnType(*tuple(out[g] for g in get))

      return canonicalize_output(fn_out)

    return getter_fn

  return getter_decorator


@nt_tree_fn(nargs=2, reduce=lambda x: all(x) if isinstance(x, list) else x)
def x1_is_x2(
    x1: torch.Tensor,
    x2: Optional[torch.Tensor] = None,
    eps: float = 1e-12
) -> Union[bool, torch.Tensor]:
  if not isinstance(x1, (np.ndarray, torch.Tensor)):
    raise TypeError('`x1` must be a tensor. A {} is found.'.format(type(x1)))

  if x2 is None:
    return True

  if x1 is x2:
    return True

  if x1.shape != x2.shape:
    return False

  try:
    diff = x1 - x2
  except TypeError:
    if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
      return torch.all(x1 == x2)
    return np.all(x1 == x2)
  else:
    if isinstance(diff, torch.Tensor):
      return torch.all(torch.abs(diff) < eps)
    return np.all(np.abs(diff) < eps)


def _get_ndim(x: Any) -> int:
  if hasattr(x, 'ndim'):
    n = getattr(x, 'ndim')
  elif hasattr(x, '__len__'):
    n = len(x)  # type: ignore
  elif isinstance(x, int):
    n = x
  else:
    raise TypeError(x, type(x))
  return n


def mod(axis: Axes, x: Any) -> list[int]:
  n = _get_ndim(x)
  if isinstance(axis, int):
    axis = [axis]
  return [(i % n) if n > 0 else i for i in axis]


def canonicalize_axis(
    axis: Axes,
    x: Any
) -> list[int]:
  axis = [axis] if isinstance(axis, int) else list(axis)
  n = _get_ndim(x)
  return list(set(np.arange(n)[axis]))


def zip_axes(
    x: torch.Tensor,
    start_axis: int = 0,
    end_axis: Optional[int] = None
) -> torch.Tensor:
  return _zip_axes(x, start_axis, end_axis, unzip=False)


def unzip_axes(x: torch.Tensor,
               start_axis: int = 0,
               end_axis: Optional[int] = None) -> torch.Tensor:
  return _zip_axes(x, start_axis, end_axis, unzip=True)


def _zip_axes(
    x: torch.Tensor,
    start_axis: int = 0,
    end_axis: Optional[int] = None,
    unzip: bool = False
) -> torch.Tensor:
  if end_axis is None:
    end_axis = x.ndim

  half_ndim, ragged = divmod(end_axis - start_axis, 2)
  if ragged:
    raise ValueError(
        f'Need even number of axes to zip, got {end_axis - start_axis}.')

  odd_axes = range(start_axis + 1, end_axis, 2)
  last_axes = range(end_axis - half_ndim, end_axis)

  if unzip:
    x = torch.movedim(x, tuple(odd_axes), tuple(last_axes))
  else:
    x = torch.movedim(x, tuple(last_axes), tuple(odd_axes))
  return x


def diagonal_between(
    x: torch.Tensor,
    start_axis: int = 0,
    end_axis: Optional[int] = None
) -> torch.Tensor:
  if end_axis is None:
    end_axis = x.ndim

  half_ndim, ragged = divmod(end_axis - start_axis, 2)
  if ragged:
    raise ValueError(
        f'Need even number of axes to flatten, got {end_axis - start_axis}.')
  if half_ndim == 0:
    return x

  side_shape = x.shape[start_axis:start_axis + half_ndim]
  side_size = size_at(side_shape)

  shape_2d = x.shape[:start_axis] + (side_size, side_size) + x.shape[end_axis:]
  shape_result = x.shape[:start_axis] + side_shape + x.shape[end_axis:]

  x = torch.diagonal(x.reshape(shape_2d), offset=0, dim1=start_axis, dim2=start_axis + 1)
  x = torch.movedim(x, -1, start_axis)
  return x.reshape(shape_result)


def zip_flat(x, y):
  return tuple(c for xy in zip(x, y) for c in xy)


def interleave_ones(x, start_axis, end_axis, x_first):
  x_axes = x.shape[start_axis:end_axis]
  ones = (1,) * (end_axis - start_axis)
  shape = x.shape[:start_axis]
  if x_first:
    shape += zip_flat(x_axes, ones)
  else:
    shape += zip_flat(ones, x_axes)
  shape += x.shape[end_axis:]
  return x.reshape(shape)


def outer_prod(x, y, start_axis, end_axis, prod_op):
  if y is None:
    y = x
  x = interleave_ones(x, start_axis, end_axis, True)
  y = interleave_ones(y, start_axis, end_axis, False)
  return prod_op(x, y)


_ArrayOrShape = TypeVar(
    '_ArrayOrShape',
    np.ndarray,
    torch.Tensor,
    list,
    tuple,
)


def reverse_zipped(
    x: _ArrayOrShape,
    start_axis: int = 0
) -> _ArrayOrShape:
  if x is not None:
    ndim = _get_ndim(x)
    source_axes = tuple(j
                        for i in range(ndim - 2, start_axis - 1, -2)
                        for j in (i, i + 1))

    if isinstance(x, (np.ndarray, torch.Tensor)):
      target_axes = tuple(range(start_axis, ndim))
      if isinstance(x, torch.Tensor):
          x = torch.movedim(x, source_axes, target_axes)  # type: ignore
      else:
          x = np.moveaxis(x, source_axes, target_axes)   # type: ignore
    elif isinstance(x, tuple):
      x = x[:start_axis] + tuple(x[i] for i in source_axes)  # type: ignore
    elif isinstance(x, list):
      x = x[:start_axis] + [x[i] for i in source_axes]   # type: ignore
  return x


def mask(
    x: Optional[torch.Tensor],
    mask_mat: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
  if x is None or mask_mat is None:
    return x
  
  assert x is not None and mask_mat is not None
  return torch.where(mask_mat, torch.zeros((), dtype=x.dtype, device=x.device), x)


def size_at(
    x: Any,
    axes: Optional[Iterable[int]] = None
) -> int:
  if hasattr(x, 'shape'):
    x = getattr(x, 'shape')

  if axes is None:
    axes = range(len(x))  # type: ignore

  return functools.reduce(operator.mul, [x[a] for a in axes], 1)


def axis_after_dot(
    axis: int,
    contracting_dims: Sequence[int],
    batch_dims: Sequence[int],
    lhs_ndim: Optional[int] = None
) -> int:
  if axis in batch_dims:
    return list(batch_dims).index(axis)

  return (
      axis -
      sum(1 for i in contracting_dims if i < axis) +
      sum(1 for i in batch_dims if i > axis) +
      (0 if lhs_ndim is None
       else lhs_ndim - len(batch_dims) - len(contracting_dims))
  )


def make_2d(
    x: Optional[torch.Tensor],
    start_axis: int = 0,
    end_axis: Optional[int] = None
) -> Optional[torch.Tensor]:
  if x is None:
    return x

  if end_axis is None:
    end_axis = x.ndim

  x = unzip_axes(x, start_axis, end_axis)

  half_ndim = (end_axis - start_axis) // 2
  x = x.reshape(x.shape[:start_axis] +
                (size_at(x.shape[start_axis:start_axis + half_ndim]),
                 size_at(x.shape[start_axis + half_ndim:end_axis])) +
                x.shape[end_axis:])
  return x


def _read_keys(key, x1, x2):
  if key is None or all_none(x2):
    key1 = key2 = key
  elif isinstance(key, tuple) and len(key) == 2:
    key1, key2 = key
    is_same = x1_is_x2(x1, x2)
    key2 = key1 if is_same else None
  return key1, key2


def split_kwargs(kwargs, x1=None, x2=None):
  kwargs1 = {}
  kwargs2 = {}
  for k, v in kwargs.items():
    if x2 is not None and k == 'rng':
      key1, key2 = _read_keys(v, x1, x2)
      kwargs1[k] = key1
      kwargs2[k] = key2
    elif isinstance(v, tuple) and len(v) == 2:
      kwargs1[k] = v[0]
      kwargs2[k] = v[1]
    else:
      kwargs1[k] = kwargs2[k] = v

  return kwargs1, kwargs2


_SingleSlice = Any

SliceType = Any


def canonicalize_idx(
    idx: Any,
    ndim: int
) -> tuple[Any, ...]:
  if idx is Ellipsis or isinstance(idx, (int, slice)):
    idx = (idx,) + (slice(None),) * (ndim - 1)

  for i, s in enumerate(idx):
    if s is Ellipsis:
      idx = idx[:i] + (slice(None),) * (ndim - len(idx) + 1) + idx[i + 1:]

  idx += (slice(None),) * (ndim - len(idx))
  return idx


def slice_shape(shape: tuple[int, ...], idx: SliceType) -> tuple[int, ...]:
  canonical_idx = canonicalize_idx(idx, len(shape))

  np_shape = list(shape)
  unknown_axes = {}
  n_ints = 0

  for a, (i, s) in enumerate(zip(canonical_idx, shape)):
    if s < 0 or s is None:
      if i == slice(None):
        np_shape[a] = 0
        unknown_axes[a - n_ints] = s
      else:
        raise ValueError(f'Unknown array size at {a}.')

    if isinstance(i, int):
      n_ints += 1

  out_shape = list(np.empty(np_shape)[idx].shape) # type: ignore
  for a, v in unknown_axes.items():
    out_shape[a] = v # type: ignore

  return tuple(out_shape) # type: ignore


_T = TypeVar('_T')


def double_tuple(x: Iterable[_T]) -> tuple[_T, ...]:
  return tuple(v for v in x for _ in range(2))


def dot_general(lhs: torch.Tensor, rhs: torch.Tensor, dimension_numbers: tuple[tuple[Sequence[int], Sequence[int]], tuple[Sequence[int], Sequence[int]]]) -> torch.Tensor:
  if lhs.ndim == 0 or rhs.ndim == 0:
    return lhs * rhs

  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers

  if not lhs_batch and not rhs_batch:
    if not lhs_contracting and not rhs_contracting:
      letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
      lhs_letters = letters[:lhs.ndim]
      rhs_letters = letters[lhs.ndim:lhs.ndim+rhs.ndim]
      return torch.einsum(f'{lhs_letters},{rhs_letters}->{lhs_letters}{rhs_letters}', lhs, rhs)
    else:
      return torch.tensordot(lhs, rhs, dims=(list(lhs_contracting), list(rhs_contracting)))

  # Note: torch.einsum still doesn't support ellipsis in the script dimensions?
  # TODO: There is probably a better way to do this. Change this once torch.einsum supports ellipsis?
  letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
  idx = 0

  batch_letters = {}
  for l, r in zip(lhs_batch, rhs_batch):
    batch_letters[(l, r)] = letters[idx]
    idx += 1

  contracting_letters = {}
  for l, r in zip(lhs_contracting, rhs_contracting):
    contracting_letters[(l, r)] = letters[idx]
    idx += 1

  lhs_str = ''
  rhs_str = ''
  out_str1 = ''
  out_str2 = ''

  for i in range(lhs.ndim):
    char = None
    for b_l, b_r in batch_letters.keys():
      if i == b_l:
        char = batch_letters[(b_l, b_r)]
        break
    if char is None:
      for c_l, c_r in contracting_letters.keys():
        if i == c_l:
          char = contracting_letters[(c_l, c_r)]
          break
    if char is None:
      char = letters[idx]
      idx += 1
      out_str1 += char
    lhs_str += char

  for i in range(rhs.ndim):
    char = None
    for b_l, b_r in batch_letters.keys():
      if i == b_r:
        char = batch_letters[(b_l, b_r)]
        break
    if char is None:
      for c_l, c_r in contracting_letters.keys():
        if i == c_r:
          char = contracting_letters[(c_l, c_r)]
          break
    if char is None:
      char = letters[idx]
      idx += 1
      out_str2 += char
    rhs_str += char

  batch_str = ''.join([batch_letters[k] for k in batch_letters])
  out_str = f'{batch_str}{out_str1}{out_str2}'

  einsum_str = f'{lhs_str},{rhs_str}->{out_str}'
  return torch.einsum(einsum_str, lhs, rhs)
