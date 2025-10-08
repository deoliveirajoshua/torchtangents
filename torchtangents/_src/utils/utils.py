# Copyright 2025 torchtangents contributors
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

"""Utility functions for torchtangents."""

import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import torch


def canonicalize_get(get: Optional[Union[str, Tuple[str, ...]]], 
                    x2_is_none: bool) -> Tuple[bool, bool]:
    """Canonicalizes the `get` argument for kernel computation.
    
    Args:
        get: either `None`, a string, or a tuple of strings specifying which
            covariance matrices to compute. Can be 'nngp', 'ntk', or both.
        x2_is_none: boolean specifying whether `x2` is `None`.
        
    Returns:
        A tuple `(get_nngp, get_ntk)` of booleans specifying whether to compute
        NNGP and NTK respectively.
    """
    if get is None:
        get_nngp = True
        get_ntk = True
    elif isinstance(get, str):
        get_nngp = 'nngp' in get.lower()
        get_ntk = 'ntk' in get.lower()
    else:
        get_nngp = any('nngp' in g.lower() for g in get)
        get_ntk = any('ntk' in g.lower() for g in get)
        
    return get_nngp, get_ntk


def get_device(x: Optional[torch.Tensor]) -> torch.device:
    """Get the device of a tensor, defaulting to CPU if tensor is None."""
    if x is None:
        return torch.device('cpu')
    return x.device


def get_dtype(x: Optional[torch.Tensor]) -> torch.dtype:
    """Get the dtype of a tensor, defaulting to float32 if tensor is None.""" 
    if x is None:
        return torch.float32
    return x.dtype


def canonicalize_axis(axis: Union[int, Sequence[int]], 
                     ndim: int) -> Tuple[int, ...]:
    """Convert axis specification to canonical form.
    
    Args:
        axis: int or sequence of ints specifying axes.
        ndim: number of dimensions in the array.
        
    Returns:
        Tuple of positive axis indices.
    """
    if isinstance(axis, int):
        axis = (axis,)
    
    canonical = []
    for ax in axis:
        if ax < 0:
            ax = ndim + ax
        if ax < 0 or ax >= ndim:
            raise ValueError(f"Axis {ax} is out of bounds for array of dimension {ndim}")
        canonical.append(ax)
    
    return tuple(sorted(canonical))


def batch_dim_is_leading(batch_axis: Union[int, Sequence[int]], 
                        ndim: int) -> bool:
    """Check if batch dimensions are leading dimensions.
    
    Args:
        batch_axis: int or sequence of ints specifying batch axes.
        ndim: number of dimensions in the array.
        
    Returns:
        True if batch dimensions are leading (0, 1, ..., len(batch_axis)-1).
    """
    if isinstance(batch_axis, int):
        return batch_axis == 0
    
    canonical = canonicalize_axis(batch_axis, ndim)
    return canonical == tuple(range(len(canonical)))


def make_2d(x: torch.Tensor) -> torch.Tensor:
    """Reshape tensor to 2D, flattening all dimensions except the first.
    
    Args:
        x: input tensor of shape (batch_size, ...).
        
    Returns:
        2D tensor of shape (batch_size, -1).
    """
    return x.view(x.shape[0], -1)


def outer_prod(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute outer product of vectors or matrices.
    
    Args:
        x: first tensor.
        y: second tensor. If None, computes x ⊗ x.
        
    Returns:
        Outer product tensor.
    """
    if y is None:
        y = x
    
    # Handle different dimensionalities
    if x.ndim == 1 and y.ndim == 1:
        return torch.outer(x, y)
    elif x.ndim == 2 and y.ndim == 2:
        # Matrix outer product: (i,j) ⊗ (k,l) -> (i,k,j,l)
        return torch.einsum('ij,kl->ikjl', x, y)
    else:
        raise NotImplementedError(f"Outer product not implemented for shapes {x.shape}, {y.shape}")


def trace(x: torch.Tensor, axis1: int = -2, axis2: int = -1) -> torch.Tensor:
    """Compute trace along specified axes.
    
    Args:
        x: input tensor.
        axis1: first axis for trace.
        axis2: second axis for trace.
        
    Returns:
        Tensor with trace computed.
    """
    return torch.diagonal(x, dim1=axis1, dim2=axis2).sum(dim=-1)


def unflatten_axes(x: torch.Tensor, 
                  axis: int, 
                  new_shape: Sequence[int]) -> torch.Tensor:
    """Unflatten a single axis into multiple axes.
    
    Args:
        x: input tensor.
        axis: axis to unflatten.
        new_shape: shape to unflatten into.
        
    Returns:
        Tensor with axis unflattened.
    """
    if axis < 0:
        axis = x.ndim + axis
    
    shape = list(x.shape)
    old_size = shape.pop(axis)
    
    if old_size != torch.prod(torch.tensor(new_shape)).item():
        raise ValueError(f"Cannot unflatten axis of size {old_size} into shape {new_shape}")
    
    new_full_shape = shape[:axis] + list(new_shape) + shape[axis:]
    return x.view(new_full_shape)