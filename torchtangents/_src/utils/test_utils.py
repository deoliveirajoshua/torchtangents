"""Testing utilities for torchtangents."""

import torch
from typing import Optional, Union
from ..utils.kernel import Kernel


def create_base_kernel(x1: torch.Tensor, 
                      x2: Optional[torch.Tensor] = None,
                      kernel_type: str = 'nngp') -> Kernel:
    """Create a base kernel from input tensors for testing.
    
    This creates a simple dot-product kernel as a starting point
    for layer kernel transformations.
    
    Args:
        x1: First input tensor of shape (n1, ...)
        x2: Second input tensor of shape (n2, ...) or None for x1=x2
        kernel_type: Type of kernel to create ('nngp', 'ntk', or 'both')
        
    Returns:
        Kernel object with the requested kernel type(s)
    """
    if x2 is None:
        x2 = x1
    
    # Flatten inputs for dot product computation
    x1_flat = x1.view(x1.shape[0], -1)
    x2_flat = x2.view(x2.shape[0], -1)
    
    # Compute dot product kernel (simple NNGP approximation)
    dot_product = torch.matmul(x1_flat, x2_flat.T)
    
    # Normalize by input dimension for stability
    input_dim = x1_flat.shape[1]
    normalized_kernel = dot_product / input_dim
    
    # Create kernel components based on requested type
    if kernel_type.lower() == 'nngp':
        return Kernel(
            nngp=normalized_kernel,
            cov1=None,
            cov2=None,
            ntk=None
        )
    elif kernel_type.lower() == 'ntk':
        # For NTK, use a simple approximation
        return Kernel(
            nngp=None,
            cov1=None,
            cov2=None,
            ntk=normalized_kernel
        )
    elif kernel_type.lower() == 'both':
        return Kernel(
            nngp=normalized_kernel,
            cov1=normalized_kernel,  # Simplified: same as NNGP
            cov2=normalized_kernel if torch.equal(x1, x2) else normalized_kernel,
            ntk=normalized_kernel
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def apply_layer_kernel(layer_kernel_fn, 
                      x1: torch.Tensor,
                      x2: Optional[torch.Tensor] = None,
                      kernel_type: str = 'nngp') -> Kernel:
    """Apply a layer kernel function to input tensors.
    
    This is a testing utility that creates a base kernel from input tensors
    and then applies the layer's kernel transformation.
    
    Args:
        layer_kernel_fn: The kernel function from a stax layer
        x1: First input tensor
        x2: Second input tensor or None
        kernel_type: Type of kernel to compute
        
    Returns:
        Transformed kernel
    """
    # Create base kernel
    base_kernel = create_base_kernel(x1, x2, 'both')  # Always create both for flexibility
    
    # Apply layer transformation
    transformed_kernel = layer_kernel_fn(base_kernel)
    
    return transformed_kernel