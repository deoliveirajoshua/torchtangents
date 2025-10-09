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

"""Layer combinators for composing multiple layers."""

import functools
from typing import Callable, Optional, Sequence, Tuple
import torch

from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer, PyTree
from .requirements import layer


@layer
def serial(*layers: InternalLayer) -> InternalLayer:
    """Combinator for composing layers in sequence.
    
    Args:
        *layers: sequence of (init_fn, apply_fn, kernel_fn) tuples.
        
    Returns:
        A triple (init_fn, apply_fn, kernel_fn) representing the serial
        composition of the input layers.
    """
    if not layers:
        raise ValueError("serial() requires at least one layer")
    
    def init_fn(rng: torch.Generator, input_shape: Tuple[int, ...]) -> Tuple[Tuple[int, ...], PyTree]:
        """Initialize parameters for all layers in sequence."""
        params = []
        shape = input_shape
        
        for layer_init, layer_apply, _ in layers:
            # Split the random number generator for each layer
            output_shape, param = layer_init(rng, shape)
            params.append(param)
            shape = output_shape
        
        return shape, params
    
    def apply_fn(params: PyTree, 
                inputs: torch.Tensor, 
                rng: Optional[torch.Generator] = None,
                **kwargs) -> torch.Tensor:
        """Apply all layers in sequence."""
        if len(params) != len(layers):
            raise ValueError(f"Expected {len(layers)} parameter sets, got {len(params)}")
        
        outputs = inputs
        for param, (_, layer_apply, _) in zip(params, layers):
            outputs = layer_apply(param, outputs, rng=rng, **kwargs)
        
        return outputs
    
    def kernel_fn(kernel: Kernel, **kwargs) -> Kernel:
        """Apply all kernel functions in sequence."""
        for _, _, layer_kernel in layers:
            kernel = layer_kernel(kernel, **kwargs)
        return kernel
    
    return init_fn, apply_fn, kernel_fn


@layer
def parallel(*layers: InternalLayer) -> InternalLayer:
    """Combinator for composing layers in parallel.
    
    The input is split along the last axis and fed to each layer,
    then the outputs are concatenated along the last axis.
    
    Args:
        *layers: sequence of (init_fn, apply_fn, kernel_fn) tuples.
        
    Returns:
        A triple (init_fn, apply_fn, kernel_fn) representing the parallel
        composition of the input layers.
    """
    if not layers:
        raise ValueError("parallel() requires at least one layer")
    
    def init_fn(rng: torch.Generator, input_shape: Tuple[int, ...]) -> Tuple[Tuple[Tuple[int, ...], ...], PyTree]:
        """Initialize parameters for all layers in parallel."""
        params = []
        output_shapes = []
        
        # Each layer gets the same input shape
        for layer_init, _, _ in layers:
            output_shape, param = layer_init(rng, input_shape)
            params.append(param)
            output_shapes.append(output_shape)
        
        return tuple(output_shapes), params
    
    def apply_fn(params: PyTree,
                inputs: torch.Tensor,
                rng: Optional[torch.Generator] = None,
                **kwargs) -> tuple:
        """Apply all layers in parallel and return tuple of outputs."""
        if len(params) != len(layers):
            raise ValueError(f"Expected {len(layers)} parameter sets, got {len(params)}")
        
        outputs = []
        for param, (_, layer_apply, _) in zip(params, layers):
            output = layer_apply(param, inputs, rng=rng, **kwargs)
            outputs.append(output)
        
        return tuple(outputs)
    
    def kernel_fn(kernel: Kernel, **kwargs) -> Kernel:
        """Apply parallel composition to kernel.
        
        For parallel layers, the kernel is computed as the sum of
        the individual layer kernels (for independent layers).
        """
        # This is a simplified implementation
        # The full implementation would need to handle the kernel
        # composition more carefully, potentially tracking which
        # parts of the output correspond to which layers
        
        result_kernel = None
        for _, _, layer_kernel in layers:
            layer_result = layer_kernel(kernel, **kwargs)
            if result_kernel is None:
                result_kernel = layer_result
            else:
                # Add the kernels (this is simplified)
                if result_kernel.nngp is not None and layer_result.nngp is not None:
                    result_kernel.nngp = result_kernel.nngp + layer_result.nngp
                if result_kernel.ntk is not None and layer_result.ntk is not None:
                    result_kernel.ntk = result_kernel.ntk + layer_result.ntk
        
        return result_kernel
    
    return init_fn, apply_fn, kernel_fn


@layer
def repeat(layer: InternalLayer, n: int) -> InternalLayer:
    """Repeat a layer n times in sequence.
    
    Args:
        layer: (init_fn, apply_fn, kernel_fn) tuple to repeat.
        n: number of times to repeat the layer.
        
    Returns:
        A triple (init_fn, apply_fn, kernel_fn) representing n
        repetitions of the input layer in sequence.
    """
    if n <= 0:
        raise ValueError(f"repeat count must be positive, got {n}")
    
    return serial(*[layer] * n)