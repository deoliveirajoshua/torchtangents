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

'''
Batch kernel computations serially or in parallel.

This module contains a decorator `batch` that can be applied to any `kernel_fn`
of signature `kernel_fn(x1, x2, *args, **kwargs)`. The decorated function
performs the same computation by batching over `x1` and `x2` and concatenating
the result, allowing to both use multiple accelerators and stay within memory
limits.
'''

# This file is adapted from the neural-tangents library:


import functools
import math
from typing import Any, Callable, Iterable, Optional, TypeVar, Union
import torch

from .utils.kernel import Kernel
from .utils.typing import Axes
from .utils.typing import KernelFn

# A type variable to indicate that `nt.batch` and other functions here do not
# change the function signature.
_KernelFn = TypeVar('_KernelFn', bound=KernelFn)

def batch(
        kernel_fn: Callable,
         batch_size: int = 0,
         device_count: int = -1,
         store_on_device: bool = True
) -> Callable:
    """Returns a function that computes a kernel in batches over all devices.

  Note that you typically should not apply the `jax.jit` decorator to the
  resulting `batched_kernel_fn`, as its purpose is explicitly serial execution
  in order to save memory. Further, you do not need to apply `jax.jit` to the
  input `kernel_fn` function, as it is JITted internally.

  Args:
    kernel_fn:
      A function that computes a kernel on two batches,
      `kernel_fn(x1, x2, *args, **kwargs)`. Here `x1` and `x2` are
      `jnp.ndarray`s of shapes `(n1,) + input_shape` and `(n2,) + input_shape`.
      The kernel function should return a `PyTree`.

    batch_size:
      specifies the size of each batch that gets processed per physical device.
      Because we parallelize the computation over columns it should be the case
      that `x1.shape[0]` is divisible by `device_count * batch_size` and
      `x2.shape[0]` is divisible by `batch_size`.

    device_count:
      specifies the number of physical devices to be used. If
      `device_count == -1` all devices are used. If `device_count == 0`, no
      device parallelism is used (a single default device is used).

    store_on_device:
      specifies whether the output should be kept on device or brought back to
      CPU RAM as it is computed. Defaults to `True`. Set to `False` to store
      and concatenate results using CPU RAM, allowing to compute larger kernels.

  Returns:
    A new function with the same signature as `kernel_fn` that computes the
    kernel by batching over the dataset in parallel with the specified
    `batch_size` using `device_count` devices.
  """
    if batch_size <= 0:
        return kernel_fn
    
    @functools.wraps(kernel_fn)
    def batched_kernel_fn(x1: torch.Tensor,
                         x2: Optional[torch.Tensor] = None,
                         get: Optional[Union[str, tuple]] = None,
                         **kwargs) -> Union[torch.Tensor, Kernel]:
        """Compute kernel in batches."""
        
        if x2 is None:
            x2 = x1
            
        n1, n2 = x1.shape[0], x2.shape[0]
        
        # If small enough, compute directly
        if n1 <= batch_size and n2 <= batch_size:
            try:
                return kernel_fn(x1, x2, get, **kwargs)
            except TypeError as e:
                if "positional argument" in str(e) and get is None and not kwargs:
                    # Fallback to simple 2-argument call
                    return kernel_fn(x1, x2)
                else:
                    raise
        
        # Determine batch sizes
        n_batches_1 = math.ceil(n1 / batch_size)
        n_batches_2 = math.ceil(n2 / batch_size)
        
        # Collect results
        results = []
        
        for i in range(n_batches_1):
            start_1 = i * batch_size
            end_1 = min((i + 1) * batch_size, n1)
            x1_batch = x1[start_1:end_1]
            
            row_results = []
            
            for j in range(n_batches_2):
                start_2 = j * batch_size
                end_2 = min((j + 1) * batch_size, n2)
                x2_batch = x2[start_2:end_2]
                
                # Compute kernel for this batch
                # Try neural-tangents style (3-arg) first, fallback to simple (2-arg)
                try:
                    result_batch = kernel_fn(x1_batch, x2_batch, get, **kwargs)
                except TypeError as e:
                    if "positional argument" in str(e) and get is None and not kwargs:
                        # Fallback to simple 2-argument call
                        result_batch = kernel_fn(x1_batch, x2_batch)
                    else:
                        raise
                row_results.append(result_batch)
            
            # Concatenate along second dimension
            if isinstance(row_results[0], torch.Tensor):
                row_result = torch.cat(row_results, dim=1)
            else:
                # Handle Kernel objects
                row_result = _concat_kernels(row_results, dim=1)
            
            results.append(row_result)
        
        # Concatenate along first dimension
        if isinstance(results[0], torch.Tensor):
            return torch.cat(results, dim=0)
        else:
            return _concat_kernels(results, dim=0)
    
    return batched_kernel_fn


def _concat_kernels(kernels: list, dim: int) -> Kernel:
    """Concatenate a list of kernels along specified dimension."""
    if not kernels:
        return None
    
    first_kernel = kernels[0]
    if not isinstance(first_kernel, Kernel):
        # If not Kernel objects, just concatenate tensors
        return torch.cat(kernels, dim=dim)
    
    def concat_field(field_name):
        field_values = [getattr(k, field_name) for k in kernels]
        if field_values[0] is None:
            return None
        return torch.cat(field_values, dim=dim)
    
    return Kernel(
        nngp=concat_field('nngp'),
        ntk=concat_field('ntk'),
        cov1=concat_field('cov1') if dim == 0 else first_kernel.cov1,
        cov2=concat_field('cov2') if dim == 1 else first_kernel.cov2,
        x1_is_x2=first_kernel.x1_is_x2,
        is_gaussian=first_kernel.is_gaussian,
        is_reversed=first_kernel.is_reversed,
        is_input=first_kernel.is_input,
        diagonal_batch=first_kernel.diagonal_batch,
        diagonal_spatial=first_kernel.diagonal_spatial,
        shape1=first_kernel.shape1,
        shape2=first_kernel.shape2
    )


def serial(*functions):
    """Compose functions serially (for compatibility)."""
    def composed_fn(*args, **kwargs):
        result = args
        for fn in functions:
            if callable(fn):
                result = fn(*result, **kwargs)
        return result
    return composed_fn