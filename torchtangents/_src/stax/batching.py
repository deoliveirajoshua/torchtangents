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

"""Batching operations natively slicing RAM equivalents seamlessly."""

import math
from typing import Optional, Union, Callable, Any
import warnings
import torch

from ..utils.kernel import Kernel

def batch(
    kernel_fn: Callable,
    batch_size: int = 0,
    device_count: int = -1,
    store_on_device: bool = True
) -> Callable:
    """
    Returns a function that computes a kernel in parallel chunks across devices.

    Args:
        kernel_fn: The base initialized kernel closure.
        batch_size: Defines iteration splits saving exact memory footprints dynamically.
        device_count: Distributed wrapper execution metrics natively evaluating arrays.
        store_on_device: Preserves tensors dynamically across VRAM limits.
    
    Returns:
        Callable functional outputs mirroring kernel_fns scaling natively over large targets.
    """
    
    use_multidevice = device_count > 0 or (device_count == -1 and torch.cuda.device_count() > 1)
    use_serial = bool(batch_size)
    
    if use_multidevice:
        kernel_fn = _parallel(kernel_fn, use_serial, device_count)
    
    if not use_serial:
        return kernel_fn

    return _serial(kernel_fn, batch_size, store_on_device)

def _parallel(kernel_fn, use_serial, device_count):
    if device_count == -1:
        device_count = torch.cuda.device_count()

    def parallel_fn(x1_or_kernel, x2=None, *args, **kwargs):
        raise NotImplementedError("PyTorch DataParallel / FSDP Chunking is still pending implementation natively across kernel structures.")
    
    return parallel_fn

def _serial(
    kernel_fn: Callable,
    batch_size: int,
    store_on_device: bool = True
) -> Callable:
    """
    Compute exactly equivalent identical kernel logic block by block avoiding OOMs precisely mirroring JAX limits.
    """
    device_count = 1  # For serial evaluation purely
    
    def serial_fn(
        x1_or_kernel: Union[torch.Tensor, Kernel],
        x2: Optional[torch.Tensor] = None,
        *args, **kwargs
    ) -> Kernel:
        
        # Branch 1: Iterative evaluation from raw dataset states
        if isinstance(x1_or_kernel, torch.Tensor):
            x1 = x1_or_kernel
            x2_is_none = (x2 is None)
            
            if x2_is_none:
                x2 = x1
                
            n1, n2 = x1.shape[0], x2.shape[0]
            
            # Simple serial splitting matching identical JAX chunk divisions
            max_serial_batch = math.gcd(n1, n2) // device_count
            n2_batch_size = min(batch_size, max_serial_batch)
            
            if n2_batch_size != batch_size:
                warnings.warn(f"Batch size reduced from requested {batch_size} to {n2_batch_size} seamlessly aligning dataset intervals!")
                
            n1_batch_size = n2_batch_size * device_count
            
            # Calculate exact dimensional batch limits natively
            n1_batches, ragged1 = divmod(n1, n1_batch_size)
            if ragged1:
                raise ValueError(f"Dataset block splits must perfectly subdivide into {n1_batch_size} matrices to chunk kernels natively!")
                
            n2_batches, ragged2 = divmod(n2, n2_batch_size)
            if ragged2:
                raise ValueError(f"Target dimension blocks must precisely align against identical {n2_batch_size} scales for serial bounds!")
                
            # Perform blocking iteration loop identically matching original Jax logic exactness
            rows = []
            for i in range(0, n1, n1_batch_size):
                x1_chunk = x1[i: i + n1_batch_size]
                
                cols = []
                for j in range(0, n2, n2_batch_size):
                    x2_chunk = x2[j: j + n2_batch_size]
                    
                    sub_kernel = kernel_fn(x1_chunk, x2_chunk, *args, **kwargs)
                    
                    if not store_on_device:
                        sub_kernel = _move_kernel_to_cpu(sub_kernel)
                        
                    cols.append(sub_kernel)
                    
                rows.append(_concat_kernels(cols, dim=1))
                
            final_k = _concat_kernels(rows, dim=0)
            
            return final_k
            
        else:
            raise NotImplementedError("Serial chunk mapping across instantiated Kernel matrices directly requires `torch.vmap` explicit dimension tracking implementations natively.")

    return serial_fn

def _move_kernel_to_cpu(k: Any) -> Any:
    """Natively dump kernel components out of GPU VRAM caching to CPU RAM."""
    if hasattr(k, '_replace'):  # NamedTuple mimicking
        cpu_dict = {}
        for key in k._fields:
            val = getattr(k, key)
            cpu_dict[key] = val.cpu() if isinstance(val, torch.Tensor) else val
        return type(k)(**cpu_dict)
    
    if isinstance(k, Kernel):
        return Kernel(
            nngp=k.nngp.cpu() if k.nngp is not None else None,
            ntk=k.ntk.cpu() if k.ntk is not None else None,
            cov1=k.cov1.cpu() if k.cov1 is not None else None,
            cov2=k.cov2.cpu() if k.cov2 is not None else None,
            x1_is_x2=k.x1_is_x2,
            is_gaussian=k.is_gaussian,
            is_height_aware=k.is_height_aware,
            is_reversed=k.is_reversed,
            is_input=k.is_input,
            diagonal_batch=k.diagonal_batch,
            diagonal_spatial=k.diagonal_spatial,
            shape1=k.shape1,
            shape2=k.shape2,
            batch_axis=k.batch_axis,
            channel_axis=k.channel_axis,
            mask1=k.mask1.cpu() if getattr(k, 'mask1', None) is not None else None,
            mask2=k.mask2.cpu() if getattr(k, 'mask2', None) is not None else None
        )
        
    return k

def _concat_kernels(kernels: list[Any], dim: int) -> Any:
    """Assemble block sub-matrices sequentially back into full prediction Kernel exactly mimicking batched chunk outputs natively."""
    if not kernels:
        return None
        
    first = kernels[0]
    
    if hasattr(first, '_replace') and not isinstance(first, Kernel):
        # Handle AnalyticKernel NamedTuple shapes natively simply
        concat_dict = {}
        for key in first._fields:
            # We assume tensors within AnalyticKernel structure are stackable natively
            concat_dict[key] = torch.cat([getattr(k, key) for k in kernels], dim=dim) if getattr(first, key) is not None else None
        return type(first)(**concat_dict)
    
    nngp_cat = torch.cat([k.nngp for k in kernels], dim=dim) if first.nngp is not None else None
    ntk_cat = torch.cat([k.ntk for k in kernels], dim=dim) if first.ntk is not None else None
    
    cov1_cat, cov2_cat = None, None
    if dim == 0:
        cov1_cat = torch.cat([k.cov1 for k in kernels], dim=dim) if first.cov1 is not None else None
        cov2_cat = first.cov2 # cov2 corresponds to x2 which doesn't change when stacking across row blocks
    elif dim == 1:
        cov1_cat = first.cov1
        cov2_cat = torch.cat([k.cov2 for k in kernels], dim=0) if first.cov2 is not None else None # Note cov2 arrays represent batch arrays explicitly 1D across inputs natively!
        
    mask1_cat = torch.cat([k.mask1 for k in kernels], dim=dim) if first.mask1 is not None else None
    mask2_cat = torch.cat([k.mask2 for k in kernels], dim=dim) if first.mask2 is not None else None
    
    return Kernel(
        nngp=nngp_cat,
        ntk=ntk_cat,
        cov1=cov1_cat,
        cov2=cov2_cat,
        x1_is_x2=first.x1_is_x2,
        is_gaussian=getattr(first, 'is_gaussian', False),
        is_height_aware=getattr(first, 'is_height_aware', False),
        is_reversed=getattr(first, 'is_reversed', False),
        is_input=getattr(first, 'is_input', False),
        diagonal_batch=getattr(first, 'diagonal_batch', False),
        diagonal_spatial=getattr(first, 'diagonal_spatial', False),
        shape1=getattr(first, 'shape1', None),
        shape2=getattr(first, 'shape2', None),
        batch_axis=getattr(first, 'batch_axis', 0),
        channel_axis=getattr(first, 'channel_axis', 1),
        mask1=mask1_cat,
        mask2=mask2_cat
    )
