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

"""Elementwise activation functions for torchtangents."""

import operator as op
from typing import Optional, Callable, Tuple
import torch
import torch.nn.functional as F

from ..utils.typing import PyTree, InternalLayer
from .requirements import layer, supports_masking


def get_diagonal(cov, diagonal_batch: bool, diagonal_spatial: bool):
    """Extract diagonal of cov over all (sample, spatial) dimensions."""
    if cov is None:
        return cov
    
    batch_ndim = 1 if diagonal_batch else 2
    start_axis = 2 - batch_ndim
    end_axis = batch_ndim if diagonal_spatial else cov.ndim
    
    # For PyTorch, we need to implement unzip_axes and diagonal_between utilities
    # For now, return as-is for diagonal cases
    if diagonal_batch and diagonal_spatial:
        return cov
    
    # For non-diagonal cases, extract diagonal
    if not diagonal_batch and not diagonal_spatial:
        # Extract diagonal from full covariance matrix
        ndim = cov.ndim
        mid = ndim // 2
        return torch.diagonal(cov, dim1=start_axis, dim2=mid)
    
    return cov


def mean_and_var(x, axis=None, keepdims=False, mask=None):
    """PyTorch implementation of mean and var with masking."""
    var = None
    if x is None:
        return x, var
    
    if mask is None:
        if axis is None:
            mean = torch.mean(x)
        else:
            mean = torch.mean(x, dim=axis, keepdim=keepdims)
    else:
        # Apply mask
        masked_x = torch.where(mask, torch.zeros_like(x), x)
        if axis is None:
            count = torch.sum(~mask).float()
            mean = torch.sum(masked_x) / torch.max(count, torch.tensor(1.0))
        else:
            count = torch.sum(~mask, dim=axis, keepdim=True).float()
            mean = torch.sum(masked_x, dim=axis, keepdim=keepdims) / torch.max(count, torch.tensor(1.0))
            if not keepdims and count.ndim > mean.ndim:
                count = torch.squeeze(count, dim=axis)
    
    return mean, var


def outer_prod(x1, x2, start_axis, end_axis, operation):
    """Compute outer product between x1 and x2."""
    if x2 is None:
        x2 = x1
    
    # Simple broadcasting for cross-batch operations
    # Handle different batch sizes by broadcasting
    try:
        return operation(x1, x2)
    except RuntimeError as e:
        if "size mismatch" in str(e) or "shape" in str(e):
            # Try to broadcast by expanding dimensions
            x1_expanded = x1.unsqueeze(-1)  # Add dim for x2 batch
            x2_expanded = x2.unsqueeze(-2)  # Add dim for x1 batch
            return operation(x1_expanded, x2_expanded)
        else:
            raise


def _ensure_cov_values(kernels):
    """Ensure cov1 and cov2 are not None for elementwise kernel computation.
    
    Args:
        kernels: Kernel object that might have None cov1/cov2
        
    Returns:
        tuple: (cov1, cov2) with proper fallback values
    """
    cov1 = kernels.cov1  
    cov2 = kernels.cov2
    nngp = kernels.nngp
    
    # Handle case where cov1/cov2 might be None (e.g., in tests)
    # For elementwise functions, if we only have nngp, we can still compute the kernel
    if cov1 is None and cov2 is None:
        # Use nngp diagonal as a fallback for cov computation
        if nngp.shape[0] == nngp.shape[1]:  # Square matrix
            cov1 = cov2 = torch.diag(nngp)
        else:
            # For non-square, create basic covariance based on nngp
            cov1 = torch.diag(torch.diag(nngp @ nngp.T))
            cov2 = torch.diag(torch.diag(nngp.T @ nngp))
    elif cov1 is None:
        cov1 = torch.diag(torch.diag(nngp @ nngp.T)) if nngp.shape[0] == nngp.shape[1] else torch.ones(nngp.shape[0])
    elif cov2 is None:
        cov2 = torch.diag(torch.diag(nngp.T @ nngp)) if nngp.shape[0] == nngp.shape[1] else torch.ones(nngp.shape[1])
        
    return cov1, cov2


def get_diagonal_outer_prods(cov1, 
                             cov2, 
                             diagonal_batch: bool,
                             diagonal_spatial: bool,
                             operation=torch.mul,
                             axis=(),
                             mask1=None,
                             mask2=None):
    """Get diagonal outer products for kernel computation."""
    
    cov1_diag = get_diagonal(cov1, diagonal_batch, diagonal_spatial)
    cov2_diag = get_diagonal(cov2, diagonal_batch, diagonal_spatial)
    
    # Compute means
    cov1_mean, _ = mean_and_var(cov1_diag, axis=axis, keepdims=True, mask=mask1)
    cov2_mean, _ = mean_and_var(cov2_diag, axis=axis, keepdims=True, mask=mask2)
    
    if cov1_mean is None:
        raise ValueError('cov1 is None')
    
    end_axis = 1 if diagonal_spatial else (cov1_mean.ndim if cov1_mean is not None else 0)
    
    # Handle case where cov2 is None
    if cov2_mean is None:
        cov2_mean = cov1_mean
    
    # Compute outer products with proper broadcasting
    prod12 = outer_prod(cov1_mean, cov2_mean, 0, end_axis, operation)
    
    start_axis = 1 if diagonal_batch else 0
    prod11 = outer_prod(cov1_mean, cov1_mean, start_axis, end_axis, operation)
    
    if cov2_mean is not cov1_mean:
        prod22 = outer_prod(cov2_mean, cov2_mean, start_axis, end_axis, operation)
    else:
        prod22 = prod11
    
    return prod11, prod12, prod22


def _elementwise(fn: Callable, name: str, kernel_fn: Optional[Callable] = None) -> InternalLayer:
    """Helper function to create elementwise activation functions.
    
    Args:
        fn: The activation function to apply element-wise
        name: Name of the activation function
        kernel_fn: Optional custom kernel function. If None, creates identity kernel.
        
    Returns:
        (init_fn, apply_fn, kernel_fn) triple
    """
    def init_fn(rng: Optional[torch.Generator], 
               input_shape: Tuple[int, ...]) -> Tuple[Tuple[int, ...], PyTree]:
        """Initialize parameters (elementwise functions have no parameters)."""
        output_shape = input_shape
        return output_shape, {}
    
    def apply_fn(params: PyTree,
                inputs: torch.Tensor,
                rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Apply the activation function."""
        return fn(inputs)
    
    def default_kernel_fn(kernels, **kwargs):
        """Default kernel function that passes through unchanged."""
        return kernels
    
    actual_kernel_fn = kernel_fn if kernel_fn is not None else default_kernel_fn
    actual_kernel_fn.__name__ = f"{name}_kernel"
    
    return init_fn, apply_fn, actual_kernel_fn


@layer
@supports_masking(remask_kernel=True)
def ABRelu(a: float, b: float, do_stabilize: bool = False) -> InternalLayer:
    """ABReLU nonlinearity: a * |x| + b * max(x, 0).
    
    Args:
        a: coefficient for the absolute value term
        b: coefficient for the ReLU term
        do_stabilize: whether to use stabilized computation
        
    Returns:
        (init_fn, apply_fn, kernel_fn) triple
    """
    
    def fn(x):
        return a * torch.abs(x) + b * torch.relu(x)
    
    def kernel_fn(kernels, **kwargs):
        """Kernel function for ABReLU."""
        cov1, cov2 = _ensure_cov_values(kernels)
        nngp = kernels.nngp
        ntk = kernels.ntk
        
        # Get diagonal outer products  
        prod11, prod12, prod22 = get_diagonal_outer_prods(
            cov1, cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.mul)
        
        # Apply ABReLU kernel computation
        # This is a simplified version - full implementation would need proper integrals
        
        def nngp_ntk_fn(nngp_val, prod, ntk_val=None):
            # Simplified AB-ReLU kernel computation
            if prod is None:
                prod = torch.ones_like(nngp_val)
                
            # Basic AB-ReLU formula (simplified)
            sqrt_prod = torch.sqrt(torch.clamp(prod, min=1e-15))
            angle = torch.atan2(torch.sqrt(torch.clamp(prod - nngp_val**2, min=0)), 
                               torch.abs(nngp_val))
            
            new_nngp = (a**2 + b**2) * nngp_val / (2 * torch.pi) + \
                      (a**2 - b**2) * sqrt_prod * torch.cos(angle) / torch.pi
            
            if ntk_val is not None:
                # NTK derivative computation (simplified)
                new_ntk = ntk_val * ((a**2 + b**2) / (2 * torch.pi))
                return new_nngp, new_ntk
            return new_nngp, None
        
        # Apply to cross terms
        new_nngp, new_ntk = nngp_ntk_fn(nngp, prod12, ntk)
        
        # Apply to diagonal terms
        if kernels.diagonal_batch and kernels.diagonal_spatial:
            new_cov1 = nngp_ntk_fn(cov1, prod11)[0] if cov1 is not None else None
            new_cov2 = nngp_ntk_fn(cov2, prod22)[0] if cov2 is not None else None
        else:
            new_cov1, _ = nngp_ntk_fn(cov1, prod11, None) if cov1 is not None else (None, None)
            new_cov2, _ = nngp_ntk_fn(cov2, prod22, None) if cov2 is not None else (None, None)
        
        return kernels.replace(
            nngp=new_nngp,
            cov1=new_cov1, 
            cov2=new_cov2,
            ntk=new_ntk
        )
    
    return _elementwise(fn, 'ABRelu', kernel_fn)


def Relu() -> InternalLayer:
    """ReLU nonlinearity."""
    return ABRelu(0.0, 1.0)


def LeakyRelu(alpha: float = 0.01) -> InternalLayer:
    """Leaky ReLU nonlinearity."""
    return ABRelu(alpha, 1.0)


@layer
@supports_masking(remask_kernel=True)
def Erf(a: float = 1., b: float = 1., c: float = 0.) -> InternalLayer:
    """Error function nonlinearity: erf(a * x).
    
    For compatibility, the full form is: b * erf(a * x) + c
    But the common usage is just Erf(a=scale) which gives erf(scale * x).
    
    Args:
        a: input scale (most commonly used parameter)
        b: output scale  
        c: output shift
    """
    
    def fn(x):
        return b * torch.erf(a * x) + c
    
    def kernel_fn(kernels, **kwargs):
        """Kernel function for Erf."""
        cov1, cov2 = _ensure_cov_values(kernels)
        nngp = kernels.nngp
        ntk = kernels.ntk
        
        # Scale by input factor (a is now the input scale)
        scaled_cov1 = cov1 * (a**2)
        scaled_cov2 = cov2 * (a**2)
        scaled_nngp = nngp * (a**2)
        scaled_ntk = ntk * (a**2) if ntk is not None else None
        
        # Get diagonal outer products
        cov1_denom = 1 + 2 * scaled_cov1
        cov2_denom = 1 + 2 * scaled_cov2
        
        prod11, prod12, prod22 = get_diagonal_outer_prods(
            cov1_denom, cov2_denom, kernels.diagonal_batch, kernels.diagonal_spatial, op.mul)
        
        def nngp_ntk_fn(nngp_val, prod, ntk_val=None):
            factor = 2 / torch.pi
            sqrt_term = torch.sqrt(torch.clamp(prod - 4 * nngp_val**2, min=1e-15))
            new_nngp = factor * torch.atan2(2 * nngp_val, sqrt_term)
            
            if ntk_val is not None:
                dot_sigma = 2 * factor / sqrt_term
                new_ntk = ntk_val * dot_sigma
                return new_nngp, new_ntk
            return new_nngp, None
        
        # Apply transformations
        new_nngp, new_ntk = nngp_ntk_fn(scaled_nngp, prod12, scaled_ntk)
        
        # Scale by output factor (b) and add shift (c)
        new_nngp = b * new_nngp + c
        if new_ntk is not None:
            new_ntk = b * new_ntk
            
        # Handle diagonal terms
        def nngp_fn_diag(nngp_val):
            factor = 2 / torch.pi  
            return factor * torch.atan2(nngp_val, torch.sqrt(nngp_val + 0.25))
        
        if kernels.diagonal_batch and kernels.diagonal_spatial:
            new_cov1 = b * nngp_fn_diag(scaled_cov1) + c
            new_cov2 = b * nngp_fn_diag(scaled_cov2) + c
        else:
            new_cov1, _ = nngp_ntk_fn(scaled_cov1, prod11)
            new_cov2, _ = nngp_ntk_fn(scaled_cov2, prod22)
            if new_cov1 is not None:
                new_cov1 = b * new_cov1 + c
            if new_cov2 is not None:
                new_cov2 = b * new_cov2 + c
        
        return kernels.replace(
            nngp=new_nngp,
            cov1=new_cov1,
            cov2=new_cov2,
            ntk=new_ntk
        )
    
    return _elementwise(fn, 'Erf', kernel_fn)


@layer  
@supports_masking(remask_kernel=False)
def Sign() -> InternalLayer:
    """Sign activation function."""
    
    def fn(x):
        return torch.sign(x)
    
    def kernel_fn(kernels, **kwargs):
        """Kernel function for Sign."""
        cov1, cov2 = _ensure_cov_values(kernels)
        nngp = kernels.nngp
        ntk = kernels.ntk
        
        # NTK becomes zero for sign function
        if ntk is not None:
            ntk = torch.zeros_like(ntk)
        
        # Get diagonal outer products
        _, prod12, _ = get_diagonal_outer_prods(
            cov1, cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.mul)
        
        # Sign kernel computation
        sqrt_term = torch.sqrt(torch.clamp(prod12 - nngp**2, min=1e-15))
        angles = torch.atan2(sqrt_term, nngp)
        new_nngp = 1 - angles * 2 / torch.pi
        
        # Covariance becomes 1 where non-zero, 0 where zero
        new_cov1 = torch.where(cov1 == 0., 0., 1.) if cov1 is not None else None
        new_cov2 = torch.where(cov2 == 0., 0., 1.) if cov2 is not None else None
        
        return kernels.replace(
            nngp=new_nngp,
            cov1=new_cov1,
            cov2=new_cov2,
            ntk=ntk
        )
    
    return _elementwise(fn, 'Sign', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Sin(a: float = 1., b: float = 1., c: float = 0.) -> InternalLayer:
    """Affine transform of Sine nonlinearity: a * sin(b * x + c).
    
    Args:
        a: output scale
        b: input scale
        c: phase shift
    """
    
    def fn(x):
        return a * torch.sin(b * x + c)
    
    def kernel_fn(kernels, **kwargs):
        """Kernel function for Sin."""
        cov1, cov2 = _ensure_cov_values(kernels)
        nngp = kernels.nngp
        ntk = kernels.ntk
        
        # Get diagonal outer products (scaled by b^2)
        scaled_cov1 = cov1 * (b**2) if cov1 is not None else None
        scaled_cov2 = cov2 * (b**2) if cov2 is not None else None
        scaled_nngp = nngp * (b**2)
        
        sum11, sum12, sum22 = get_diagonal_outer_prods(
            scaled_cov1, scaled_cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.add)
        
        def nngp_ntk_fn(nngp_val, sum_val, ntk_val=None):
            # Sin kernel computation with phase shift
            exp_term = torch.exp(-sum_val / 2)
            new_nngp = exp_term * torch.sin(nngp_val + c)
            
            if ntk_val is not None:
                # NTK computation for sin
                new_ntk = ntk_val * exp_term * torch.cos(nngp_val + c) * b
                return new_nngp, new_ntk
            return new_nngp, None
        
        new_nngp, new_ntk = nngp_ntk_fn(scaled_nngp, sum12, ntk * (b**2) if ntk is not None else None)
        
        # Scale by output factor
        new_nngp = a * new_nngp
        if new_ntk is not None:
            new_ntk = a * new_ntk
        
        # Handle diagonal terms
        if kernels.diagonal_batch and kernels.diagonal_spatial:
            new_cov1 = a * nngp_ntk_fn(scaled_cov1, sum11)[0] if scaled_cov1 is not None else None
            new_cov2 = a * nngp_ntk_fn(scaled_cov2, sum22)[0] if scaled_cov2 is not None else None
        else:
            new_cov1, _ = nngp_ntk_fn(scaled_cov1, sum11) if scaled_cov1 is not None else (None, None)
            new_cov2, _ = nngp_ntk_fn(scaled_cov2, sum22) if scaled_cov2 is not None else (None, None)
            if new_cov1 is not None:
                new_cov1 = a * new_cov1
            if new_cov2 is not None:
                new_cov2 = a * new_cov2
        
        return kernels.replace(
            nngp=new_nngp,
            cov1=new_cov1,
            cov2=new_cov2,
            ntk=new_ntk
        )
    
    return _elementwise(fn, 'Sin', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Cos(a: float = 1., b: float = 1., c: float = 0.) -> InternalLayer:
    """Affine transform of Cosine nonlinearity: a * cos(b * x + c)."""
    
    def fn(x):
        return a * torch.cos(b * x + c)
    
    def kernel_fn(kernels, **kwargs):
        """Kernel function for Cosine - similar to Sin but with cosine."""
        # Use the same computation as Sin but with cosine
        sin_kernels = Sin(a, b, c + torch.pi/2)  # cos(x) = sin(x + π/2)
        return sin_kernels[2](kernels, **kwargs)  # Return the kernel_fn result
    
    return _elementwise(fn, 'Cos', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Tanh(a: float = 1., b: float = 1., c: float = 0.) -> InternalLayer:
    """Affine transform of Tanh nonlinearity: a * tanh(b * x) + c."""
    
    def fn(x):
        return a * torch.tanh(b * x) + c
    
    def kernel_fn(kernels, **kwargs):
        """Kernel function for Tanh."""
        # Simplified tanh kernel - in practice this would need more complex computation
        cov1, cov2 = _ensure_cov_values(kernels)
        nngp = kernels.nngp
        ntk = kernels.ntk
        
        # Apply tanh scaling (simplified)
        new_nngp = a * torch.tanh(b * nngp) + c
        new_cov1 = a * torch.tanh(b * cov1) + c if cov1 is not None else None
        new_cov2 = a * torch.tanh(b * cov2) + c if cov2 is not None else None
        
        # Scale NTK
        if ntk is not None:
            sech_sq = 1 - torch.tanh(b * nngp)**2  # sech^2 = 1 - tanh^2
            new_ntk = a * b * sech_sq * ntk
        else:
            new_ntk = None
        
        return kernels.replace(
            nngp=new_nngp,
            cov1=new_cov1,
            cov2=new_cov2,
            ntk=new_ntk
        )
    
    return _elementwise(fn, 'Tanh', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Sigmoid(a: float = 1., b: float = 1., c: float = 0.) -> InternalLayer:
    """Affine transform of Sigmoid nonlinearity: a * sigmoid(b * x) + c."""
    
    def fn(x):
        return a * torch.sigmoid(b * x) + c
    
    def kernel_fn(kernels, **kwargs):
        """Kernel function for Sigmoid."""
        # Simplified sigmoid kernel
        cov1, cov2 = _ensure_cov_values(kernels)
        nngp = kernels.nngp
        ntk = kernels.ntk
        
        # Apply sigmoid (simplified)
        new_nngp = a * torch.sigmoid(b * nngp) + c
        new_cov1 = a * torch.sigmoid(b * cov1) + c if cov1 is not None else None
        new_cov2 = a * torch.sigmoid(b * cov2) + c if cov2 is not None else None
        
        # Scale NTK  
        if ntk is not None:
            sigmoid_val = torch.sigmoid(b * nngp)
            sigmoid_deriv = sigmoid_val * (1 - sigmoid_val)  # sigmoid derivative
            new_ntk = a * b * sigmoid_deriv * ntk
        else:
            new_ntk = None
        
        return kernels.replace(
            nngp=new_nngp,
            cov1=new_cov1,
            cov2=new_cov2,
            ntk=new_ntk
        )
    
    return _elementwise(fn, 'Sigmoid', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Exp(a: float = 1., b: float = 1.) -> InternalLayer:
    """Exponential nonlinearity: a * exp(b * x)."""
    
    def fn(x):
        return a * torch.exp(b * x)
    
    def kernel_fn(kernels, **kwargs):
        """Kernel function for Exponential."""
        cov1, cov2 = _ensure_cov_values(kernels)
        nngp = kernels.nngp
        ntk = kernels.ntk
        
        # Get diagonal outer products
        sum11, sum12, sum22 = get_diagonal_outer_prods(
            cov1, cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.add)
        
        def nngp_ntk_fn(nngp_val, sum_val, ntk_val=None):
            # Exponential kernel computation
            new_nngp = torch.exp(b**2 * (sum_val / 2 + nngp_val))
            
            if ntk_val is not None:
                new_ntk = new_nngp * ntk_val  # Exponential amplifies NTK
                return new_nngp, new_ntk
            return new_nngp, None
        
        new_nngp, new_ntk = nngp_ntk_fn(nngp, sum12, ntk)
        
        # Scale by output factor
        new_nngp = a * new_nngp
        if new_ntk is not None:
            new_ntk = a * new_ntk
        
        # Handle diagonal terms
        if kernels.diagonal_batch and kernels.diagonal_spatial:
            new_cov1 = a * nngp_ntk_fn(cov1, sum11)[0] if cov1 is not None else None
            new_cov2 = a * nngp_ntk_fn(cov2, sum22)[0] if cov2 is not None else None
        else:
            new_cov1, _ = nngp_ntk_fn(cov1, sum11) if cov1 is not None else (None, None)
            new_cov2, _ = nngp_ntk_fn(cov2, sum22) if cov2 is not None else (None, None)
            if new_cov1 is not None:
                new_cov1 = a * new_cov1
            if new_cov2 is not None:
                new_cov2 = a * new_cov2
        
        return kernels.replace(
            nngp=new_nngp,
            cov1=new_cov1,
            cov2=new_cov2,
            ntk=new_ntk
        )
    
    return _elementwise(fn, 'Exp', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Abs(do_stabilize: bool = False) -> InternalLayer:
    """Absolute value nonlinearity."""
    return ABRelu(-1, 1, do_stabilize)


@layer
@supports_masking(remask_kernel=True)
def Gelu(approximate: bool = False) -> InternalLayer:
    """Gaussian Error Linear Unit (GELU) activation.
    
    Args:
        approximate: If True, use tanh approximation for GELU.
    """
    
    def fn(x):
        return F.gelu(x, approximate='tanh' if approximate else 'none')
    
    def kernel_fn(kernels, **kwargs):
        """Kernel function for GELU - simplified implementation."""
        # For now, use a simplified approximation
        # Full GELU kernel would require complex integral computation
        cov1, cov2 = _ensure_cov_values(kernels)
        nngp = kernels.nngp
        ntk = kernels.ntk
        
        # Approximate GELU with scaled sigmoid-like behavior
        scale = 0.5
        pi_tensor = torch.tensor(torch.pi)
        sqrt_2_over_pi = torch.sqrt(2.0 / pi_tensor)
        new_nngp = nngp * (0.5 + 0.5 * torch.tanh(sqrt_2_over_pi * (nngp + 0.044715 * nngp**3)))
        new_cov1 = cov1 * scale if cov1 is not None else None
        new_cov2 = cov2 * scale if cov2 is not None else None
        
        # Scale NTK appropriately
        if ntk is not None:
            new_ntk = ntk * scale
        else:
            new_ntk = None
        
        return kernels.replace(
            nngp=new_nngp,
            cov1=new_cov1,
            cov2=new_cov2,
            ntk=new_ntk
        )
    
    return _elementwise(fn, 'Gelu', kernel_fn)


# Placeholder implementations for more complex functions that need specialized math
def Elementwise(*args, **kwargs): 
    """Placeholder for general Elementwise."""
    raise NotImplementedError("General Elementwise not yet implemented")

def ElementwiseNumerical(*args, **kwargs):
    """Placeholder for ElementwiseNumerical.""" 
    raise NotImplementedError("ElementwiseNumerical not yet implemented")

def ExpNormalized(*args, **kwargs):
    """Placeholder for ExpNormalized."""
    raise NotImplementedError("ExpNormalized not yet implemented")

def Gabor(*args, **kwargs):
    """Placeholder for Gabor."""
    raise NotImplementedError("Gabor not yet implemented")

def Gaussian(*args, **kwargs):
    """Placeholder for Gaussian."""
    raise NotImplementedError("Gaussian not yet implemented")

def Hermite(*args, **kwargs):
    """Placeholder for Hermite."""
    raise NotImplementedError("Hermite not yet implemented")

def Monomial(*args, **kwargs):
    """Placeholder for Monomial."""
    raise NotImplementedError("Monomial not yet implemented")

def Polynomial(*args, **kwargs):
    """Placeholder for Polynomial."""
    raise NotImplementedError("Polynomial not yet implemented")

def Rbf(*args, **kwargs):
    """Placeholder for Rbf."""
    raise NotImplementedError("Rbf not yet implemented")

def RectifiedMonomial(*args, **kwargs):
    """Placeholder for RectifiedMonomial."""
    raise NotImplementedError("RectifiedMonomial not yet implemented")

def Sigmoid_like(*args, **kwargs):
    """Placeholder for Sigmoid_like."""
    raise NotImplementedError("Sigmoid_like not yet implemented")