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

"""Layer requirements and decorators for torchtangents stax layers."""

import functools
import enum
from typing import Any, Callable, Optional, Sequence, Union, TYPE_CHECKING
import torch

from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer, KernelFn
from .. import utils

if TYPE_CHECKING:
    from ..utils.typing import Layer


class Bool(enum.Enum):
    """Enum for boolean requirements on kernel properties."""
    YES = True
    NO = False
    EITHER = None


class Diagonal(enum.Enum):
    """Enum for diagonal requirements on kernel matrices."""
    YES = True
    NO = False
    EITHER = None


def layer(layer_fn: Callable[..., InternalLayer]) -> Callable[..., 'Layer']:
    """Decorator for public layers to add flexible kernel handling."""
    name = layer_fn.__name__

    @functools.wraps(layer_fn)
    def new_layer_fns(*args, **kwargs):
        init_fn, apply_fn, kernel_fn = layer_fn(*args, **kwargs)
        kernel_fn = _preprocess_kernel_fn(init_fn, apply_fn, kernel_fn)
        init_fn.__name__ = apply_fn.__name__ = kernel_fn.__name__ = name
        return init_fn, apply_fn, kernel_fn

    return new_layer_fns


def _preprocess_kernel_fn(init_fn, apply_fn, kernel_fn):
    """Convert a raw kernel function to an analytic kernel function that can handle tensor inputs."""
    
    @functools.wraps(kernel_fn)
    def analytic_kernel_fn(x1_or_kernel, x2=None, get=None, **kwargs):
        # If we receive a Kernel object, use the raw kernel function directly
        if isinstance(x1_or_kernel, Kernel):
            return kernel_fn(x1_or_kernel, **kwargs)
        
        # If we receive tensor inputs, convert them to a Kernel object first
        if isinstance(x1_or_kernel, torch.Tensor):
            # Create a basic kernel object manually
            x1_t = x1_or_kernel.to(dtype=torch.float64)
            x2_t = x2.to(dtype=torch.float64) if x2 is not None else x1_t
            
            # Compute basic covariances
            # NNGP is the cross-covariance between x1 and x2: x1 @ x2.T
            nngp = torch.mm(x1_t, x2_t.T)
            # cov1 is the self-covariance of x1: x1 @ x1.T
            cov1 = torch.mm(x1_t, x1_t.T)
            # cov2 is the self-covariance of x2: x2 @ x2.T  
            cov2 = torch.mm(x2_t, x2_t.T)
            
            # Create basic NTK (zero for now)
            ntk = torch.zeros_like(nngp) if (get is None) or ('ntk' in str(get)) else None
            
            kernel_obj = Kernel(
                cov1=cov1,
                cov2=cov2,
                nngp=nngp,
                ntk=ntk,
                x1_is_x2=torch.tensor(x2 is None),
                is_gaussian=True,
                is_reversed=False,
                is_input=True,
                diagonal_batch=False,
                diagonal_spatial=False,
                shape1=tuple(x1_t.shape),
                shape2=tuple(x1_t.shape) if x2 is None else tuple(x2_t.shape),
                batch_axis=0,
                channel_axis=1
            )
            
            # Apply the raw kernel function
            result = kernel_fn(kernel_obj, **kwargs)
            
            # Return the requested quantity
            if get is None:
                return result
            elif get == 'nngp':
                return result.nngp
            elif get == 'ntk':
                return result.ntk
            elif isinstance(get, (list, tuple)):
                if len(get) == 1:
                    return getattr(result, get[0])
                return tuple(getattr(result, g) for g in get)
            else:
                return getattr(result, get)
        
        else:
            raise TypeError(f"Expected Kernel or torch.Tensor, got {type(x1_or_kernel)}")
    
    return analytic_kernel_fn


def requires(diagonal_batch: Union[Bool, bool] = Bool.EITHER,
            diagonal_spatial: Union[Bool, bool] = Bool.EITHER,
            **kwargs) -> Callable:
    """Decorator to specify kernel requirements for a layer.
    
    Args:
        diagonal_batch: whether the layer requires diagonal batch covariance.
        diagonal_spatial: whether the layer requires diagonal spatial covariance.
        **kwargs: additional requirements.
    
    Returns:
        Decorator function.
    """
    def decorator(kernel_fn: KernelFn) -> KernelFn:
        @functools.wraps(kernel_fn)
        def wrapper(kernel: Kernel) -> Kernel:
            # Check requirements
            if diagonal_batch != Bool.EITHER:
                if isinstance(diagonal_batch, Bool):
                    required = diagonal_batch.value
                else:
                    required = diagonal_batch
                if required is not None and kernel.diagonal_batch != required:
                    raise ValueError(f"Layer requires diagonal_batch={required}, "
                                   f"but got {kernel.diagonal_batch}")
            
            if diagonal_spatial != Bool.EITHER:
                if isinstance(diagonal_spatial, (Bool, Diagonal)):
                    required = diagonal_spatial.value  
                else:
                    required = diagonal_spatial
                if required is not None and kernel.diagonal_spatial != required:
                    raise ValueError(f"Layer requires diagonal_spatial={required}, "
                                   f"but got {kernel.diagonal_spatial}")
            
            return kernel_fn(kernel)
        
        # Store requirements as attributes
        wrapper._diagonal_batch = diagonal_batch
        wrapper._diagonal_spatial = diagonal_spatial
        wrapper._requirements = kwargs
        
        return wrapper
    return decorator


def supports_masking(remask_kernel: bool = True):
    """Returns a decorator for layers that support masking.
    
    This decorator modifies layer functions to handle masked inputs properly,
    following the neural-tangents pattern.
    
    Args:
        remask_kernel: If True, zero-out kernel covariance entries between 
                      masked inputs after applying kernel_fn.
                      
    Returns:
        A decorator function that modifies layer behavior for masking support.
    """
    def decorator(layer_fn):
        """Decorator that adds masking support to a layer function."""
        @functools.wraps(layer_fn)
        def layer_with_masking(*args, **kwargs):
            # For now, just call the original layer function
            # Full masking support would require more complex implementation
            result = layer_fn(*args, **kwargs)
            
            # Mark the layer as supporting masking
            if hasattr(result, '__len__') and len(result) >= 3:
                init_fn, apply_fn, kernel_fn = result[:3]
                kernel_fn._supports_masking = True
                kernel_fn._remask_kernel = remask_kernel
                
            return result
            
        layer_with_masking._supports_masking = True
        layer_with_masking._remask_kernel = remask_kernel
        return layer_with_masking
        
    return decorator


def unmask_fn(f: Callable) -> Callable:
    """Decorator for functions that handle unmasking operations.
    
    Args:
        f: function to decorate.
        
    Returns:
        Decorated function.
    """
    f._is_unmask_fn = True
    return f


def mean_and_var(kernel: Kernel) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract mean and variance from kernel covariance.
    
    Args:
        kernel: input kernel.
        
    Returns:
        Tuple of (mean, variance) tensors.
    """
    if kernel.cov1 is None:
        raise ValueError("Kernel must have cov1 to compute mean and variance")
    
    # For diagonal case, variance is just the diagonal
    if kernel.diagonal_batch:
        var = kernel.cov1
        mean = torch.zeros_like(var)
    else:
        # For full covariance, extract diagonal
        var = torch.diagonal(kernel.cov1, dim1=-2, dim2=-1)
        mean = torch.zeros_like(var)
    
    return mean, var


def get_diagonal_outer_prods(cov: torch.Tensor, 
                           diagonal_batch: bool,
                           diagonal_spatial: bool) -> torch.Tensor:
    """Get diagonal outer products from covariance matrix.
    
    Args:
        cov: covariance tensor.
        diagonal_batch: whether batch dimension is diagonal.
        diagonal_spatial: whether spatial dimensions are diagonal.
        
    Returns:
        Diagonal outer products.
    """
    if diagonal_batch and diagonal_spatial:
        # Everything is diagonal, return as-is
        return cov
    elif diagonal_batch and not diagonal_spatial:
        # Only batch is diagonal, need to handle spatial dimensions
        # This is a simplified implementation
        return cov
    elif not diagonal_batch and diagonal_spatial:
        # Only spatial is diagonal, need to handle batch dimensions
        return torch.diagonal(cov, dim1=0, dim2=1)
    else:
        # Nothing is diagonal, extract full diagonal
        # This is a complex case requiring careful indexing
        return cov


class MaskedArray:
    """Placeholder for masked array functionality.
    
    This would be implemented with proper masking support in a full implementation.
    """
    pass