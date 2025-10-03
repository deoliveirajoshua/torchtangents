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

"""General-purpose internal utilities (PyTorch version).

If a function or class is used in multiple modules, put it here.
"""

from collections import namedtuple
import functools
import inspect
import operator
import types
from typing import Any, Callable, Iterable, Optional, Sequence, Sized, TypeVar, Union
import warnings

import numpy as np
import torch

PyTree = Any
Axes = Union[int, Sequence[int]]

# -------------------------
# Tree helpers (NTTree only)
# -------------------------

def is_list_or_tuple(x) -> bool:
    # We do not want to return True if x is a subclass of list or tuple since
    # otherwise this will return true for namedtuples.
    return type(x) == list or type(x) == tuple


def is_nt_tree_of(x, dtype: Union[type, tuple[type, ...]]) -> bool:
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
    """Convert a function that acts on single inputs to one that acts on trees.

    Mirrors the original JAX-based behavior but specialized to NTTree
    (lists/tuples as internal nodes).
    """

    def check_tree_structure(args):
        """Ensure the structure of the trees in each of the `nargs` is the same."""
        if any(is_list_or_tuple(x) for x in args):
            if not all(type(x) == type(args[0]) for x in args[1:]):
                raise TypeError(f'Inconsistent NTTree structure found. '
                                f'Node Types: {[type(x) for x in args]}.')
            for x in zip(*args):
                check_tree_structure(x)

    def tree_fn(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            _nargs = len(args) if nargs is None else nargs
            recurse, norecurse = args[:_nargs], args[_nargs:]

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


def _tree_all_bool(x) -> bool:
    """all() over a (potential) NTTree of booleans."""
    if is_list_or_tuple(x):
        return all(_tree_all_bool(e) for e in x)
    # accept Python bools, numpy.bool_, torch.bool tensor scalars
    if isinstance(x, torch.Tensor):
        return bool(x)
    if isinstance(x, (np.ndarray,)):
        return bool(x.all())
    return bool(x)


def _tree_map(fn: Callable[[Any], Any], x):
    if is_list_or_tuple(x):
        t = type(x)
        return t(_tree_map(fn, e) for e in x)
    return fn(x)


def all_none(x, attr: Optional[str] = None) -> bool:
    get_fn = (lambda x: x) if attr is None else lambda x: getattr(x, attr)
    return _tree_all_bool(_tree_map(lambda x: get_fn(x) is None, x))


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

        h.__signature__ = inspect.signature(f)
        return h
    return wrapper


def get_namedtuple(name):
    def getter_decorator(fn):
        try:
            argspec = inspect.getfullargspec(fn)
            get_index = argspec.args.index('get')
            defaults = argspec.defaults
        except Exception:
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
                get_is_not_tuple, get = canonicalize_get(defaults[get_index -
                                                                  len(args)])

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

# -------------------------
# Array helpers
# -------------------------

@nt_tree_fn(nargs=2, reduce=lambda x: all(bool(v) for v in x))
def x1_is_x2(
    x1: Union[np.ndarray, torch.Tensor],
    x2: Optional[Union[np.ndarray, torch.Tensor]] = None,
    eps: float = 1e-12
) -> Union[bool, torch.Tensor]:
    if not isinstance(x1, (np.ndarray, torch.Tensor)):
        raise TypeError('`x1` must be an ndarray/Tensor. A {} is found.'.format(type(x1)))

    if x2 is None:
        return True

    if x1 is x2:
        return True

    if tuple(x1.shape) != tuple(x2.shape):
        return False

    # Looser tolerance for low-precision dtypes:
    if isinstance(x1, torch.Tensor):
        if x1.dtype in (torch.float16, torch.bfloat16) or \
           (isinstance(x2, torch.Tensor) and x2.dtype in (torch.float16, torch.bfloat16)):
            eps = max(eps, 1e-4)

    try:
        diff = x1 - x2
    except TypeError:
        # inputs are e.g. custom types which don't define subtraction.
        if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            return torch.all(x1 == x2)
        if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
            return np.all(x1 == x2).item()
        return x1 == x2
    else:
        if isinstance(diff, torch.Tensor):
            return torch.all(torch.abs(diff) < eps)
        else:
            return np.all(np.abs(diff) < eps).item()


def _get_ndim(x: Union[int, Sized, np.ndarray, torch.Tensor]) -> int:
    """Get number of dimensions given number of dimensions / shape / array."""
    if hasattr(x, 'ndim'):
        n = int(x.ndim)
    elif hasattr(x, '__len__'):
        n = len(x)
    elif isinstance(x, int):
        n = x
    else:
        raise TypeError(x, type(x))
    return n


def mod(axis: Axes, x: Union[int, Sized, np.ndarray, torch.Tensor]) -> list[int]:
    """Makes `axis` non-negative given number of dimensions / shape / array."""
    n = _get_ndim(x)
    if isinstance(axis, int):
        axis = [axis]
    return [(i % n) if n > 0 else i for i in axis]


def canonicalize_axis(
    axis: Axes,
    x: Union[int, Sized, np.ndarray, torch.Tensor]
) -> list[int]:
    """Converts axis into a sorted non-negative list."""
    axis = [axis] if isinstance(axis, int) else list(axis)
    n = _get_ndim(x)
    return list(set(np.arange(n)[axis]))


def zip_axes(
    x: torch.Tensor,
    start_axis: int = 0,
    end_axis: Optional[int] = None
) -> torch.Tensor:
    """Zip (interleave) axes starting from `start_axis`.

    `[..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ...]`
    """
    return _zip_axes(x, start_axis, end_axis, unzip=False)


def unzip_axes(x: torch.Tensor,
               start_axis: int = 0,
               end_axis: Optional[int] = None) -> torch.Tensor:
    """Unzip (de-interleave) axes starting from `start_axis`.

    `[..., X, X, ..., Y, Y, ..., Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ...]`
    """
    return _zip_axes(x, start_axis, end_axis, unzip=True)


def _zip_axes(
    x: torch.Tensor,
    start_axis: int = 0,
    end_axis: Optional[int] = None,
    unzip: bool = False
) -> torch.Tensor:
    """Zip/unzip (interleave/de-interleave) axes starting from `start_axis`."""
    if end_axis is None:
        end_axis = x.ndim

    half_ndim, ragged = divmod(end_axis - start_axis, 2)
    if ragged:
        raise ValueError(
            f'Need even number of axes to zip, got {end_axis - start_axis}.')

    odd_axes = range(start_axis + 1, end_axis, 2)
    last_axes = range(end_axis - half_ndim, end_axis)

    if unzip:
        x = torch.movedim(x, odd_axes, last_axes)
    else:
        x = torch.movedim(x, last_axes, odd_axes)
    return x


def diagonal_between(
    x: torch.Tensor,
    start_axis: int = 0,
    end_axis: Optional[int] = None
) -> torch.Tensor:
    """Returns the diagonal along all dimensions between start and end axes."""
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

    x2d = x.reshape(shape_2d)
    xdiag = torch.diagonal(x2d, offset=0, dim1=start_axis, dim2=start_axis + 1)
    xdiag = torch.movedim(xdiag, -1, start_axis)
    return xdiag.reshape(shape_result)


def zip_flat(x, y):
    return tuple(c for xy in zip(x, y) for c in xy)


def interleave_ones(x: torch.Tensor, start_axis: int, end_axis: int, x_first: bool):
    x_axes = x.shape[start_axis:end_axis]
    ones = (1,) * (end_axis - start_axis)
    shape = x.shape[:start_axis]
    if x_first:
        shape += zip_flat(x_axes, ones)
    else:
        shape += zip_flat(ones, x_axes)
    shape += x.shape[end_axis:]
    return x.reshape(shape)


def outer_prod(
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    start_axis: int,
    end_axis: int,
    prod_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    if y is None:
        y = x
    x_ = interleave_ones(x, start_axis, end_axis, True)
    y_ = interleave_ones(y, start_axis, end_axis, False)
    return prod_op(x_, y_)


_ArrayOrShape = TypeVar(
    '_ArrayOrShape',
    np.ndarray,
    torch.Tensor,
    list[int],
    tuple[int, ...],
    torch.Size,
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
            target_axes = range(start_axis, ndim)
            x = torch.movedim(torch.as_tensor(x), source_axes, target_axes) \
                if not isinstance(x, torch.Tensor) else torch.movedim(x, source_axes, target_axes)
        else:
            x = x[:start_axis] + type(x)(x[i] for i in source_axes)
    return x


def mask(
    x: Optional[torch.Tensor],
    mask_mat: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if x is None or mask_mat is None:
        return x
    # True = masked (blocked). Replace with zeros of same dtype.
    return torch.where(mask_mat, torch.zeros((), dtype=x.dtype, device=x.device), x)


def size_at(
    x: Union[_ArrayOrShape],
    axes: Optional[Iterable[int]] = None
) -> int:
    if hasattr(x, 'shape'):
        x = tuple(int(d) for d in x.shape)  # torch.Size -> tuple

    if axes is None:
        axes = range(len(x))

    out = 1
    for a in axes:
        out *= int(x[a])
    return out

# -------------------------
# Axis bookkeeping
# -------------------------

def axis_after_dot(
    axis: int,
    contracting_dims: Sequence[int],
    batch_dims: Sequence[int],
    lhs_ndim: Optional[int] = None
) -> int:
    if axis in batch_dims:
        return batch_dims.index(axis)

    return (
        axis -
        sum(1 for i in contracting_dims if i < axis) +
        sum(1 for i in batch_dims if i > axis) +
        (0 if lhs_ndim is None
         else lhs_ndim - len(batch_dims) - len(contracting_dims))
    )

# -------------------------
# Reshaping helpers
# -------------------------

def make_2d(
    x: Optional[torch.Tensor],
    start_axis: int = 0,
    end_axis: Optional[int] = None
) -> Optional[torch.Tensor]:
    """Makes `x` 2D from `start_axis` to `end_axis`, preserving other axes.

    `x` is assumed to follow the (`X, X, Y, Y, Z, Z`) axes layout.
    """
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

# -------------------------
# RNG / kwargs splitting (PyTorch)
# -------------------------

def _fold_in(gen: torch.Generator, i: int) -> torch.Generator:
    """Derive a new torch.Generator from an existing one and an integer."""
    # Use 64-bit hash mixing on the base seed.
    base = int(gen.initial_seed())
    mixed = (base ^ (0x9E3779B97F4A7C15 + i + ((base << 6) & ((1<<63)-1)) + (base >> 2))) & ((1<<63)-1)
    device = getattr(gen, 'device', torch.device('cpu'))
    new_gen = torch.Generator(device=device)
    new_gen.manual_seed(mixed)
    return new_gen


def _read_keys(key, x1, x2):
    """Read dropout key.

    `key` might be a tuple of two RNGs or a single RNG or None. In
    either case, `key` will be mapped into two RNGs `key1` and `key2` to
    make sure `(x1==x2) == (key1==key2)`.
    """
    if key is None or all_none(x2):
        key1 = key2 = key
    elif isinstance(key, tuple) and len(key) == 2:
        key1, key2 = key
        # If key1 == key2 but x1 != x2, fold-in to get a different key for x2.
        # If key1 != key2 but x1 == x2, force key2 = key1.
        same_keys = (key1 is key2)
        if same_keys:
            new_key = _fold_in(key2, 1)
        else:
            new_key = key2
        key2 = key1 if x1_is_x2(x1, x2) else new_key
        warnings.warn('The value of `key[1]` might be replaced by a new value if '
                      'key[0] == key[1] and x1 != x2 or key[0] != key[1] and '
                      'x1 == x2.')
    elif isinstance(key, torch.Generator):
        key1 = key
        key2 = key1 if x1_is_x2(x1, x2) else _fold_in(key1, 1)
    else:
        raise TypeError(type(key))
    return key1, key2


def split_kwargs(kwargs, x1=None, x2=None):
    """Splitting `kwargs`.

       Specifically,
         1. if kwarg is an RNG (`torch.Generator`), it will be split into two RNGs.
         2. else if it is a tuple of length two, the tuple will be split into two
            parts, one for kwargs1 and the other for kwargs2.
         3. else it is copied to kwargs1 and kwargs2.
    """
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

# -------------------------
# Slicing helpers
# -------------------------

_SingleSlice = Union[int, slice, type(Ellipsis)]
SliceType = Union[_SingleSlice, tuple[_SingleSlice, ...]]
"""Slice specification type (matches numpy/PyTorch indexing semantics)."""


def canonicalize_idx(
    idx: SliceType,
    ndim: int
) -> tuple[Union[int, slice], ...]:
    if idx is Ellipsis or isinstance(idx, (int, slice)):
        idx = (idx,) + (slice(None),) * (ndim - 1)

    for i, s in enumerate(idx):
        if s is Ellipsis:
            idx = idx[:i] + (slice(None),) * (ndim - len(idx) + 1) + idx[i + 1:]

    idx += (slice(None),) * (ndim - len(idx))
    return idx


def slice_shape(shape: tuple[int, ...], idx: SliceType) -> tuple[int, ...]:
    # Keep `None` or negative-sized axes if they aren't indexed into.
    canonical_idx = canonicalize_idx(idx, len(shape))

    np_shape = list(shape)
    unknown_axes = {}
    n_ints = 0  # Keep track of vanishing axes due to integer indexing.

    for a, (i, s) in enumerate(zip(canonical_idx, shape)):
        if s < 0 or s is None:
            if i == slice(None):
                np_shape[a] = 0
                unknown_axes[a - n_ints] = s
            else:
                raise ValueError(
                    f'Trying to index with {i} axis {a} of unknown size {s}. '
                    f'Please provide input shape {shape} with non-negative integer '
                    f'size at axis {a}.')

        if isinstance(i, int):
            n_ints += 1

    out_shape = list(np.empty(np_shape)[idx].shape)
    for a, v in unknown_axes.items():
        out_shape[a] = v

    return tuple(out_shape)


_T = TypeVar('_T')


def double_tuple(x: Iterable[_T]) -> tuple[_T, ...]:
    return tuple(v for v in x for _ in range(2))
