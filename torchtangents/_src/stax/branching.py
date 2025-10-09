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

"""Branching layers for torchtangents."""

from typing import Optional, Tuple, List, Union
import torch

from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer, PyTree
from .requirements import layer


def FanOut(num_outputs: int) -> InternalLayer:
    """Fan out layer that duplicates input to multiple outputs.
    
    Args:
        num_outputs: number of output branches.
        
    Returns:
        (init_fn, apply_fn, kernel_fn) triple.
    """
    def init_fn(rng: torch.Generator, input_shape: Tuple[int, ...]) -> Tuple[Tuple[Tuple[int, ...], ...], PyTree]:
        # FanOut doesn't change the shape, just duplicates it
        output_shapes = tuple([input_shape] * num_outputs)
        return output_shapes, {}
    
    def apply_fn(params: PyTree,
                inputs: torch.Tensor,
                rng: Optional[torch.Generator] = None, **kwargs) -> List[torch.Tensor]:
        """Fan out the input to multiple identical outputs."""
        return [inputs] * num_outputs
    
    @layer
    def kernel_fn(kernel: Kernel) -> List[Kernel]:
        """Fan out kernel to multiple identical kernels."""
        return [kernel] * num_outputs
    
    return init_fn, apply_fn, kernel_fn


def FanInSum(*ignore_axis: int) -> InternalLayer:
    """Fan in layer that sums multiple inputs.
    
    Args:
        *ignore_axis: axes to ignore during summation.
        
    Returns:
        (init_fn, apply_fn, kernel_fn) triple.
    """
    def init_fn(rng: torch.Generator, input_shape: Tuple[int, ...]) -> Tuple[Tuple[int, ...], PyTree]:
        # For summation, output shape is the same as input shapes (all must be equal)
        if isinstance(input_shape[0], (tuple, list)):
            # input_shape is ((shape1), (shape2), ...) - all should be equal for sum
            output_shape = input_shape[0]  # Use first shape as output shape
        else:
            # Single input case
            output_shape = input_shape
        
        return output_shape, {}
    
    def apply_fn(params: PyTree,
                inputs: List[torch.Tensor],
                rng: Optional[torch.Generator] = None, **kwargs) -> torch.Tensor:
        """Sum multiple inputs."""
        if len(inputs) == 1:
            return inputs[0]
        
        result = inputs[0]
        for inp in inputs[1:]:
            result = result + inp
        return result
    
    @layer
    def kernel_fn(kernels: List[Kernel]) -> Kernel:
        """Combine kernels through summation."""
        if len(kernels) == 1:
            return kernels[0]
        
        # Sum the kernels
        combined_nngp = None
        combined_ntk = None
        combined_cov1 = None
        combined_cov2 = None
        
        for i, kernel in enumerate(kernels):
            if kernel.nngp is not None:
                if combined_nngp is None:
                    combined_nngp = kernel.nngp
                else:
                    combined_nngp = combined_nngp + kernel.nngp
            
            if kernel.ntk is not None:
                if combined_ntk is None:
                    combined_ntk = kernel.ntk
                else:
                    combined_ntk = combined_ntk + kernel.ntk
            
            if kernel.cov1 is not None:
                if combined_cov1 is None:
                    combined_cov1 = kernel.cov1
                else:
                    combined_cov1 = combined_cov1 + kernel.cov1
            
            if kernel.cov2 is not None:
                if combined_cov2 is None:
                    combined_cov2 = kernel.cov2
                else:
                    combined_cov2 = combined_cov2 + kernel.cov2
        
        # Use properties from the first kernel
        base_kernel = kernels[0]
        return base_kernel.replace(
            nngp=combined_nngp,
            ntk=combined_ntk,
            cov1=combined_cov1,
            cov2=combined_cov2
        )
    
    return init_fn, apply_fn, kernel_fn


def FanInProd(*ignore_axis: int) -> InternalLayer:
    """Fan in layer that takes element-wise product of multiple inputs.
    
    Args:
        *ignore_axis: axes to ignore during product.
        
    Returns:
        (init_fn, apply_fn, kernel_fn) triple.
    """
    def init_fn(rng: torch.Generator, input_shape: Tuple[int, ...]) -> PyTree:
        return {}
    
    def apply_fn(params: PyTree,
                inputs: List[torch.Tensor],
                rng: Optional[torch.Generator] = None, **kwargs) -> torch.Tensor:
        """Take element-wise product of multiple inputs."""
        if len(inputs) == 1:
            return inputs[0]
        
        result = inputs[0]
        for inp in inputs[1:]:
            result = result * inp
        return result
    
    @layer
    def kernel_fn(kernels: List[Kernel]) -> Kernel:
        """Combine kernels through element-wise product."""
        if len(kernels) == 1:
            return kernels[0]
        
        # Product of kernels is more complex
        # This is a simplified implementation
        combined_nngp = None
        combined_ntk = None
        combined_cov1 = None
        combined_cov2 = None
        
        for i, kernel in enumerate(kernels):
            if kernel.nngp is not None:
                if combined_nngp is None:
                    combined_nngp = kernel.nngp
                else:
                    combined_nngp = combined_nngp * kernel.nngp
            
            if kernel.ntk is not None:
                if combined_ntk is None:
                    combined_ntk = kernel.ntk
                else:
                    combined_ntk = combined_ntk * kernel.ntk
            
            if kernel.cov1 is not None:
                if combined_cov1 is None:
                    combined_cov1 = kernel.cov1
                else:
                    combined_cov1 = combined_cov1 * kernel.cov1
            
            if kernel.cov2 is not None:
                if combined_cov2 is None:
                    combined_cov2 = kernel.cov2
                else:
                    combined_cov2 = combined_cov2 * kernel.cov2
        
        # Use properties from the first kernel
        base_kernel = kernels[0]
        return base_kernel.replace(
            nngp=combined_nngp,
            ntk=combined_ntk,
            cov1=combined_cov1,
            cov2=combined_cov2,
            is_gaussian=False
        )
    
    return init_fn, apply_fn, kernel_fn


def FanInConcat(axis: int = -1) -> InternalLayer:
    """Fan in layer that concatenates multiple inputs along an axis.
    
    Args:
        axis: axis along which to concatenate.
        
    Returns:
        (init_fn, apply_fn, kernel_fn) triple.
    """
    def init_fn(rng: torch.Generator, input_shape: Tuple[int, ...]) -> Tuple[Tuple[int, ...], PyTree]:
        # input_shape should be a tuple of shapes from parallel branches
        # For concatenation, we need to compute the output shape
        if isinstance(input_shape[0], (tuple, list)):
            # input_shape is ((shape1), (shape2), ...)
            shapes = input_shape
            # Compute concatenated shape along the specified axis
            base_shape = list(shapes[0])
            concat_dim = base_shape[axis]
            for shape in shapes[1:]:
                concat_dim += shape[axis]
            base_shape[axis] = concat_dim
            output_shape = tuple(base_shape)
        else:
            # Single input case
            output_shape = input_shape
        
        return output_shape, {}
    
    def apply_fn(params: PyTree,
                inputs: List[torch.Tensor],
                rng: Optional[torch.Generator] = None, **kwargs) -> torch.Tensor:
        """Concatenate multiple inputs along specified axis."""
        if len(inputs) == 1:
            return inputs[0]
        
        return torch.cat(inputs, dim=axis)
    
    @layer
    def kernel_fn(kernels: List[Kernel]) -> Kernel:
        """Combine kernels through concatenation."""
        if len(kernels) == 1:
            return kernels[0]
        
        # Concatenation of kernels requires careful handling of dimensions
        # This is a simplified implementation
        combined_nngp = None
        combined_ntk = None
        combined_cov1 = None
        combined_cov2 = None
        
        # For concatenation, we need to block-concatenate the kernels
        nngp_blocks = []
        ntk_blocks = []
        cov1_blocks = []
        cov2_blocks = []
        
        for kernel in kernels:
            if kernel.nngp is not None:
                nngp_blocks.append(kernel.nngp)
            if kernel.ntk is not None:
                ntk_blocks.append(kernel.ntk)
            if kernel.cov1 is not None:
                cov1_blocks.append(kernel.cov1)
            if kernel.cov2 is not None:
                cov2_blocks.append(kernel.cov2)
        
        if nngp_blocks:
            combined_nngp = torch.block_diag(*nngp_blocks)
        if ntk_blocks:
            combined_ntk = torch.block_diag(*ntk_blocks)
        if cov1_blocks:
            combined_cov1 = torch.block_diag(*cov1_blocks)
        if cov2_blocks:
            combined_cov2 = torch.block_diag(*cov2_blocks)
        
        # Use properties from the first kernel
        base_kernel = kernels[0]
        return base_kernel.replace(
            nngp=combined_nngp,
            ntk=combined_ntk,
            cov1=combined_cov1,
            cov2=combined_cov2
        )
    
    return init_fn, apply_fn, kernel_fn