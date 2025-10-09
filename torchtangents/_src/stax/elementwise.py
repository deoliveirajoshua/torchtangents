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
from typing import Optional, Callable, Tuple, Sequence
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
                rng: Optional[torch.Generator] = None,
                mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
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


@layer
@supports_masking(remask_kernel=True)
def Elementwise(
    fn: Optional[Callable[[float], float]] = None,
    nngp_fn: Optional[Callable[[float, float, float], float]] = None,
    d_nngp_fn: Optional[Callable[[float, float, float], float]] = None
) -> InternalLayer:
    """Elementwise application of `fn` using provided `nngp_fn`.

    Constructs a layer given only scalar-valued nonlinearity / activation
    `fn` and the 2D integral `nngp_fn`. NTK function is derived automatically in
    closed form from `nngp_fn`.

    If you cannot provide the `nngp_fn`, see :obj:`ElementwiseNumerical` to use
    numerical integration.

    If your function is implemented separately (e.g. `Relu` etc) it's best
    to use the custom implementation, since it uses symbolically simplified
    expressions that are more precise and numerically stable.

    Args:
        fn: a scalar-input/valued function `fn : R -> R`, the activation /
            nonlinearity. If `None`, invoking the finite width `apply_fn` will 
            raise an exception.

        nngp_fn: a scalar-valued function
            `nngp_fn : (cov12, var1, var2) |-> E[fn(x_1) * fn(x_2)]`, where the
            expectation is over bivariate normal `x1, x2` with variances `var1`,
            `var2` and covariance `cov12`. Needed for both NNGP and NTK calculation.
            If `None`, invoking infinite width `kernel_fn` will raise an exception.

        d_nngp_fn: an optional scalar-valued function
            `d_nngp_fn : (cov12, var1, var2) |-> E[fn'(x_1) * fn'(x_2)]` with the same
            `x1, x2` distribution as in `nngp_fn`. If `None`, will be computed using
            automatic differentiation as `d_nngp_fn = d(nngp_fn)/d(cov12)`.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.

    Raises:
        NotImplementedError: if a `fn`/`nngp_fn` is not provided, but `apply_fn`/
          `kernel_fn` is called respectively.
    """
    if fn is not None:
        name = fn.__name__
    elif nngp_fn is not None:
        name = nngp_fn.__name__
    else:
        raise ValueError('No finite (`fn`) or infinite (`nngp_fn`) functions '
                         'provided, the layer will not do anything.')

    if nngp_fn is None:
        kernel_fn = None

    else:
        if d_nngp_fn is None:
            import warnings
            warnings.warn(
                'Using PyTorch autodiff to compute the `fn` derivative for NTK. '
                'This may lead to numerical issues.')
            
            def d_nngp_fn_auto(cov12, var1, var2):
                # Create tensors with requires_grad=True
                cov12_t = torch.tensor(cov12, requires_grad=True, dtype=torch.float64)
                var1_t = torch.tensor(var1, dtype=torch.float64)
                var2_t = torch.tensor(var2, dtype=torch.float64)
                
                # Compute nngp_fn and get gradient
                result = nngp_fn(cov12_t, var1_t, var2_t)
                if result.requires_grad:
                    grad_output = torch.autograd.grad(result, cov12_t, 
                                                    create_graph=False, 
                                                    retain_graph=False)[0]
                    return grad_output.item()
                else:
                    return 0.0
            
            d_nngp_fn = d_nngp_fn_auto

        def kernel_fn(kernels, **kwargs):
            cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

            var1 = get_diagonal(cov1, kernels.diagonal_batch, kernels.diagonal_spatial)
            var2 = get_diagonal(cov2, kernels.diagonal_batch, kernels.diagonal_spatial)

            if ntk is not None:
                ntk_new = torch.zeros_like(ntk)
                # Apply d_nngp_fn element-wise
                for i in range(nngp.shape[0]):
                    for j in range(nngp.shape[1]):
                        v1 = var1[i] if var1.ndim > 0 else var1.item()
                        v2 = var2[j] if var2.ndim > 0 else var2.item()
                        ntk_new[i, j] = d_nngp_fn(nngp[i, j].item(), v1, v2)
                ntk = ntk * ntk_new

            # Apply nngp_fn element-wise
            nngp_new = torch.zeros_like(nngp)
            for i in range(nngp.shape[0]):
                for j in range(nngp.shape[1]):
                    v1 = var1[i] if var1.ndim > 0 else var1.item()
                    v2 = var2[j] if var2.ndim > 0 else var2.item()
                    nngp_new[i, j] = nngp_fn(nngp[i, j].item(), v1, v2)
            
            # Apply to diagonal covariances
            if cov1 is not None:
                cov1_new = torch.zeros_like(cov1)
                for i in range(cov1.shape[0]):
                    for j in range(cov1.shape[1]):
                        v1 = var1[i] if var1.ndim > 0 else var1.item()
                        cov1_new[i, j] = nngp_fn(cov1[i, j].item(), v1, v1)
                cov1 = cov1_new
            
            if cov2 is not None:
                cov2_new = torch.zeros_like(cov2)
                for i in range(cov2.shape[0]):
                    for j in range(cov2.shape[1]):
                        v2 = var2[i] if var2.ndim > 0 else var2.item()
                        cov2_new[i, j] = nngp_fn(cov2[i, j].item(), v2, v2)
                cov2 = cov2_new
            
            return kernels.replace(cov1=cov1, nngp=nngp_new, cov2=cov2, ntk=ntk)

    return _elementwise(fn, name, kernel_fn)

@layer
@supports_masking(remask_kernel=True)
def ElementwiseNumerical(
    fn: Callable[[float], float],
    deg: int,
    df: Optional[Callable[[float], float]] = None
) -> InternalLayer:
    """Activation function using numerical integration.

    Supports general activation functions using Gauss-Hermite quadrature.

    Args:
        fn: activation function.
        deg: number of sample points and weights for quadrature. It must be >= 1.
             We observe for smooth activations `deg=25` is a good place to start.
             For non-smooth activation functions (e.g. ReLU, Abs) quadrature is not
             recommended. Due to bivariate integration, compute time and memory 
             scale as O(deg**2) for more precision.
        df: optional, derivative of the activation function (`fn`). If not provided,
            it is computed by torch.autograd. Providing analytic derivative can 
            speed up the NTK computations.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.
    """
    import warnings
    warnings.warn(
        f'Numerical Activation Layer with fn={fn}, deg={deg} used! '
        'Note that numerical error is controlled by `deg` and for a given '
        'tolerance level, required `deg` will highly be dependent on the choice '
        'of `fn`.')

    # Use scipy for Gauss-Hermite quadrature points and weights
    try:
        import scipy.special
        xs, ws = scipy.special.roots_hermite(deg)
        xs = torch.tensor(xs, dtype=torch.float64)
        ws = torch.tensor(ws, dtype=torch.float64)
    except ImportError:
        raise ImportError("scipy is required for ElementwiseNumerical. Please install scipy.")

    if df is None:
        warnings.warn(
            'Using PyTorch autodiff to compute the `fn` derivative for NTK.')
        
        def df_auto(x):
            x_t = torch.tensor(x, requires_grad=True, dtype=torch.float64)
            y = fn(x_t)
            if y.requires_grad:
                grad_output = torch.autograd.grad(y, x_t, create_graph=False)[0]
                return grad_output.item()
            else:
                return 0.0
        df = df_auto

    def kernel_fn(kernels, **kwargs):
        """Kernel transformation of activation function using quadrature."""
        cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

        d1 = get_diagonal(cov1, kernels.diagonal_batch, kernels.diagonal_spatial)
        d2 = get_diagonal(cov2, kernels.diagonal_batch, kernels.diagonal_spatial)

        def nngp_ntk_fn(nngp_val, q11, q22, ntk_val=None):
            """Gauss-Hermite quadrature routine."""
            # Expand dimensions for broadcasting
            grid = torch.outer(ws, ws)  # (deg, deg)
            
            # Compute integration points
            sqrt_2q11 = torch.sqrt(2 * q11.clamp(min=1e-30))
            sqrt_2q22_minus_nngp2_over_q11 = torch.sqrt(2 * (q22 - nngp_val**2 / q11.clamp(min=1e-30)).clamp(min=0))
            nngp_over_sqrt_q11_2 = nngp_val / torch.sqrt(q11.clamp(min=1e-30) / 2)
            
            def integrate(f):
                # Vectorized integration over quadrature points
                total = 0.0
                for i in range(deg):
                    for j in range(deg):
                        x_val = sqrt_2q11 * xs[i]
                        y_val = nngp_over_sqrt_q11_2 * xs[i] + sqrt_2q22_minus_nngp2_over_q11 * xs[j]
                        
                        f_x = f(x_val.item())
                        f_y = f(y_val.item())
                        
                        total += grid[i, j] * f_x * f_y
                
                return total / torch.pi

            if ntk_val is not None:
                ntk_result = integrate(df)
                ntk_val = ntk_val * ntk_result
            
            nngp_result = integrate(fn)
            return nngp_result, ntk_val

        def nngp_fn_diag(nngp_val):
            """Diagonal quadrature for variance computation."""
            sqrt_2nngp = torch.sqrt(2 * nngp_val.clamp(min=1e-30))
            
            total = 0.0
            for i in range(deg):
                x_val = sqrt_2nngp * xs[i]
                f_val = fn(x_val.item())
                total += ws[i] * f_val**2
            
            return total / torch.sqrt(torch.pi)

        # Apply to cross-covariance
        nngp_new = torch.zeros_like(nngp)
        ntk_new = torch.zeros_like(ntk) if ntk is not None else None
        
        for i in range(nngp.shape[0]):
            for j in range(nngp.shape[1]):
                q11 = d1[i] if d1.ndim > 0 else d1
                q22 = d2[j] if d2.ndim > 0 else d2
                ntk_ij = ntk[i, j] if ntk is not None else None
                
                nngp_result, ntk_result = nngp_ntk_fn(nngp[i, j], q11, q22, ntk_ij)
                nngp_new[i, j] = nngp_result
                if ntk is not None:
                    ntk_new[i, j] = ntk_result

        # Apply to diagonal covariances
        if kernels.diagonal_batch and kernels.diagonal_spatial:
            cov1_new = torch.zeros_like(cov1) if cov1 is not None else None
            cov2_new = torch.zeros_like(cov2) if cov2 is not None else None
            
            if cov1 is not None:
                for i in range(cov1.shape[0]):
                    cov1_new[i] = nngp_fn_diag(cov1[i])
            
            if cov2 is not None:
                for i in range(cov2.shape[0]):
                    cov2_new[i] = nngp_fn_diag(cov2[i])
        else:
            # For non-diagonal cases, apply the same transformation
            cov1_new = torch.zeros_like(cov1) if cov1 is not None else None
            cov2_new = torch.zeros_like(cov2) if cov2 is not None else None
            
            if cov1 is not None:
                for i in range(cov1.shape[0]):
                    for j in range(cov1.shape[1]):
                        q11 = d1[i] if d1.ndim > 0 else d1
                        q22 = d1[j] if d1.ndim > 0 else d1
                        cov1_new[i, j], _ = nngp_ntk_fn(cov1[i, j], q11, q22, None)
            
            if cov2 is not None:
                for i in range(cov2.shape[0]):
                    for j in range(cov2.shape[1]):
                        q11 = d2[i] if d2.ndim > 0 else d2
                        q22 = d2[j] if d2.ndim > 0 else d2
                        cov2_new[i, j], _ = nngp_ntk_fn(cov2[i, j], q11, q22, None)

        return kernels.replace(cov1=cov1_new, nngp=nngp_new, cov2=cov2_new, ntk=ntk_new)

    return _elementwise(fn, f'ElementwiseNumerical({fn},deg={deg})', kernel_fn)

@layer
@supports_masking(remask_kernel=True)
def ExpNormalized(
    gamma: float = 1,
    shift: float = -1,
    do_clip: bool = False
) -> InternalLayer:
    """Simulates the "Gaussian normalized kernel".

    Args:
        gamma: exponent scalar coefficient.
        shift: shift exponentiated normalized covariance by this much.
        do_clip: True to clip normalized covariance, potentially improving accuracy.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.

    Raises:
        NotImplementedError: if finite width `apply_fn` is called.
    """

    def kernel_fn(kernels, **kwargs):
        cov1, cov2, nngp, ntk = kernels.cov1, kernels.cov2, kernels.nngp, kernels.ntk
        
        # Get diagonal outer products
        prod11, prod12, prod22 = get_diagonal_outer_prods(
            cov1, cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.mul)
        
        tol = 1e-30
        prod11 = torch.sqrt(torch.clamp(prod11, min=tol))
        prod12 = torch.sqrt(torch.clamp(prod12, min=tol))
        prod22 = torch.sqrt(torch.clamp(prod22, min=tol)) if prod22 is not None else None

        def exp_fn(cov, prod):
            if cov is not None:
                cov_normalized = cov / prod
                if do_clip:
                    cov_normalized = torch.clamp(cov_normalized, -1, 1)
                cov_exp = torch.exp(gamma * (cov_normalized + shift))
                return cov_exp
            return cov

        exp12 = exp_fn(nngp, prod12)

        return kernels.replace(
            nngp=prod12 * exp12,
            cov1=prod11 * exp_fn(cov1, prod11) if cov1 is not None else None,
            cov2=prod22 * exp_fn(cov2, prod22) if cov2 is not None and prod22 is not None else None,
            ntk=ntk * gamma * exp12 if ntk is not None else None)

    return _elementwise(None, 'ExpNormalized', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def Gabor() -> InternalLayer:
    """Gabor function `exp(-x^2) * sin(x)`.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.
    """

    def fn(x):
        return torch.exp(-x ** 2) * torch.sin(x)

    def kernel_fn(kernels, **kwargs):
        """Kernel function for Gabor."""
        cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

        prod11, prod12, prod22 = get_diagonal_outer_prods(
            cov1, cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.mul)

        sum11, sum12, sum22 = get_diagonal_outer_prods(
            cov1, cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.add)

        def nngp_ntk_fn(nngp_val, prod, sum_, ntk_val=None):
            diff = 4 * (prod - nngp_val**2)
            denom = 2 * sum_ + diff + 1
            num = sum_ + diff + 2 * nngp_val
            exp_left = torch.exp(-num / (2 * denom))
            exp_right = torch.exp(2 * nngp_val / denom)

            if ntk_val is not None:
                shared_term = 1 + 2 * sum_ + 4 * (nngp_val**2 + prod)
                diff_term = 4 * nngp_val * (diff + 3 * sum_ + 2)
                lhs = shared_term - diff_term
                rhs = shared_term + diff_term
                t_dot = exp_left * (lhs + exp_right * rhs) / (denom**(5. / 2))
                ntk_val = ntk_val * t_dot / 2

            nngp_new = exp_left * (exp_right - 1) / (2 * torch.sqrt(torch.clamp(denom, min=1e-15)))
            return nngp_new, ntk_val

        def nngp_fn_diag(nngp_val):
            denom = 1 + 4 * nngp_val
            return (1 - torch.exp(-2 * nngp_val / denom)) / (2 * torch.sqrt(torch.clamp(denom, min=1e-15)))

        nngp, ntk = nngp_ntk_fn(nngp, prod12, sum12, ntk)

        if kernels.diagonal_batch and kernels.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1) if cov1 is not None else None
            cov2 = nngp_fn_diag(cov2) if cov2 is not None else None
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11, sum11) if cov1 is not None else (None, None)
            cov2, _ = nngp_ntk_fn(cov2, prod22, sum22) if cov2 is not None else (None, None)

        return kernels.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, 'Gabor', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Gaussian(a: float = 1, b: float = -1) -> InternalLayer:
    """Elementwise Gaussian function `a * exp(b * x**2)`.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.
    """
    def fn(x):
        return a * torch.exp(b * x ** 2)

    def kernel_fn(kernels, **kwargs):
        cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

        cov1_denom = 1 - 2 * b * cov1 if cov1 is not None else None
        cov2_denom = 1 - 2 * b * cov2 if cov2 is not None else None

        prod11, prod12, prod22 = get_diagonal_outer_prods(
            cov1_denom, cov2_denom, kernels.diagonal_batch, kernels.diagonal_spatial, op.mul)

        factor = 4 * b**2

        def nngp_ntk_fn(nngp_val, prod, ntk_val=None):
            det = torch.sqrt(torch.clamp(prod - factor * nngp_val**2, min=1e-15))

            if ntk_val is not None:
                ntk_val = ntk_val * factor * nngp_val / det**3

            nngp_new = 1 / det
            return nngp_new, ntk_val

        def nngp_fn_diag(nngp_val):
            return 1 / torch.sqrt(torch.clamp(1 - 4 * b * nngp_val, min=1e-15))

        nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

        if kernels.diagonal_batch and kernels.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1) if cov1 is not None else None
            cov2 = nngp_fn_diag(cov2) if cov2 is not None else None
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11) if cov1 is not None else (None, None)
            cov2, _ = nngp_ntk_fn(cov2, prod22) if cov2 is not None else (None, None)

        return kernels.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    # Apply the output scaling
    if a != 1:
        original_kernel_fn = kernel_fn
        def scaled_kernel_fn(kernels, **kwargs):
            result = original_kernel_fn(kernels, **kwargs)
            return result.replace(
                nngp=a * result.nngp,
                cov1=a * result.cov1 if result.cov1 is not None else None,
                cov2=a * result.cov2 if result.cov2 is not None else None,
                ntk=a * result.ntk if result.ntk is not None else None
            )
        kernel_fn = scaled_kernel_fn

    return _elementwise(fn, f'Gaussian({a}, {b})', kernel_fn)


def Sigmoid_like() -> InternalLayer:
    """A sigmoid like function `f(x) = .5 * erf(x / 2.4020563531719796) + .5`.

    The constant `2.4020563531719796` is chosen so that the squared loss between
    this function and the ground truth sigmoid is minimized on the interval
    `[-5, 5]`.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.
    """
    return Erf(a=0.5, b=1/2.4020563531719796, c=0.5)


@layer
@supports_masking(remask_kernel=True)
def Rbf(gamma: float = 1.0) -> InternalLayer:
    """Dual activation function for normalized RBF or squared exponential kernel.

    Dual activation function is `f(x) = sqrt(2)*sin(sqrt(2*gamma) x + pi/4)`.
    NNGP kernel transformation correspond to (with input dimension `d`)
    `k = exp(- gamma / d * ||x - x'||^2) = exp(- gamma*(q11 + q22 - 2 * q12))`.

    Args:
        gamma: related to characteristic length-scale (l) that controls width of the
               kernel, where `gamma = 1 / (2 l^2)`.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.
    """
    def fn(x):
        return torch.sqrt(torch.tensor(2.0)) * torch.sin(torch.sqrt(torch.tensor(2.0 * gamma)) * x + torch.pi / 4)

    def kernel_fn(kernels, **kwargs):
        """Compute new kernels after an `Rbf` layer."""
        cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

        sum11, sum12, sum22 = get_diagonal_outer_prods(
            cov1, cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.add)

        def nngp_ntk_fn(nngp_val, sum_, ntk_val):
            nngp_new = torch.exp(gamma * (-sum_ + 2 * nngp_val))
            if ntk_val is not None:
                ntk_val = ntk_val * 2 * gamma * nngp_new
            return nngp_new, ntk_val

        def nngp_fn_diag(nngp_val):
            return torch.ones_like(nngp_val)

        nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

        if kernels.diagonal_batch and kernels.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1) if cov1 is not None else None
            cov2 = nngp_fn_diag(cov2) if cov2 is not None else None
        else:
            cov1, _ = nngp_ntk_fn(cov1, sum11, None) if cov1 is not None else (None, None)
            cov2, _ = nngp_ntk_fn(cov2, sum22, None) if cov2 is not None else (None, None)

        return kernels.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'Rbf({gamma})', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Hermite(degree: int) -> InternalLayer:
    """Hermite polynomials.

    Inputs to this layer are assumed to have unit norm, i.e.
    `torch.std(x, dim=channel_axis) == 1`. The Hermite polynomials are normalized
    so that the L2 norm w.r.t. standard Gaussian is 1.

    Args:
        degree: a non-negative integer.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.
    """
    if degree < 0:
        raise NotImplementedError('`degree` must be a non-negative integer.')

    # Convert Hermite polynomial coefficients to regular polynomial
    # For Hermite polynomials, we use a simplified approximation
    try:
        import numpy as np
        import math
        p = np.polynomial.hermite_e.herme2poly([0] * degree + [1])[::-1]
        coeff = (math.factorial(degree))**0.5
    except ImportError:
        # Fallback implementation for low-degree Hermite polynomials
        import math
        if degree == 0:
            p = [1.0]
            coeff = 1.0
        elif degree == 1:
            p = [0.0, 1.0]
            coeff = 1.0
        elif degree == 2:
            p = [-1.0, 0.0, 1.0]
            coeff = math.sqrt(2.0)
        else:
            raise ImportError("numpy is required for high-degree Hermite polynomials")

    def fn(x):
        # Evaluate polynomial using Horner's method
        result = torch.zeros_like(x)
        for coef in p:
            result = result * x + coef
        return result / coeff

    def kernel_fn(kernels, **kwargs):
        import warnings
        warnings.warn(
            'Inputs to this layer are assumed to have unit norm across '
            ' channels/features, i.e. torch.std(x, dim=channel_axis) == 1.')

        cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

        if ntk is not None:
            if degree == 0:
                ntk = torch.zeros_like(ntk)
            else:
                ntk = degree * (nngp**(degree - 1)) * ntk

        def _power(mat):
            return mat**degree if mat is not None else None

        nngp_new, cov1_new, cov2_new = map(_power, (nngp, cov1, cov2))
        return kernels.replace(cov1=cov1_new, nngp=nngp_new, cov2=cov2_new, ntk=ntk)

    return _elementwise(fn, f'{degree}-Hermite polynomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def Monomial(degree: int) -> InternalLayer:
    """Monomials, i.e. `x^degree`.

    Args:
        degree: an integer between 0 and 5.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.
    """
    if degree not in [0, 1, 2, 3, 4, 5]:
        raise NotImplementedError('The `degree` must be an integer between '
                                  '`0` and `5`.')

    def fn(x):
        return x**degree

    def kernel_fn(kernels, **kwargs):
        cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

        prod11, prod12, prod22 = get_diagonal_outer_prods(
            cov1, cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.mul)

        def nngp_ntk_fn(nngp_val, prod, ntk_val=None):

            def nngp_fn(nngp_val, degree):
                if degree == -1:
                    nngp_new = torch.zeros_like(nngp_val)
                elif degree == 0:
                    nngp_new = torch.ones_like(nngp_val)
                elif degree == 1:
                    nngp_new = nngp_val
                elif degree == 2:
                    nngp_new = 2 * nngp_val ** 2 + prod
                elif degree == 3:
                    nngp_new = 6 * nngp_val ** 3 + 9 * nngp_val * prod
                elif degree == 4:
                    nngp_new = 3 * (8 * nngp_val ** 4 + 3 * prod * (8 * nngp_val ** 2 + prod))
                elif degree == 5:
                    nngp_new = 15 * nngp_val * (
                        8 * nngp_val ** 4 + 5 * prod * (8 * nngp_val ** 2 + 3 * prod))
                else:
                    raise NotImplementedError(degree)

                return nngp_new

            if ntk_val is not None:
                ntk_val = ntk_val * (degree**2) * nngp_fn(nngp_val, degree - 1)

            nngp_new = nngp_fn(nngp_val, degree)
            return nngp_new, ntk_val

        def nngp_fn_diag(nngp_val):
            return _double_factorial(2 * degree - 1) * (nngp_val**degree)

        nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

        if kernels.diagonal_batch and kernels.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1) if cov1 is not None else None
            cov2 = nngp_fn_diag(cov2) if cov2 is not None else None
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11) if cov1 is not None else (None, None)
            cov2, _ = nngp_ntk_fn(cov2, prod22) if cov2 is not None else (None, None)

        return kernels.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'{degree}-monomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def RectifiedMonomial(degree: int) -> InternalLayer:
    """Rectified monomials, i.e. `(x >= 0) * x^degree`.

    Args:
        degree: a non-negative integer power.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.
    """
    if degree < 0:
        raise NotImplementedError('`degree` must be a non-negative integer.')

    def fn(x):
        return torch.where(x >= 0, x**degree, torch.zeros_like(x))

    def kernel_fn(kernels, **kwargs):
        cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

        prod11, prod12, prod22 = get_diagonal_outer_prods(
            cov1, cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.mul)

        def j_fn(nngp_val, sqrt_prod):
            # Simplified approximation for rectified moments
            # This is a complex integral - using approximation
            theta = torch.acos(torch.clamp(nngp_val / sqrt_prod, -1, 1))
            
            if degree == 0:
                return (torch.pi - theta) / torch.pi
            elif degree == 1:
                return sqrt_prod * torch.sin(theta) / torch.pi + nngp_val * (torch.pi - theta) / torch.pi
            else:
                # For higher degrees, use a simplified approximation
                base_val = sqrt_prod * torch.sin(theta) / torch.pi + nngp_val * (torch.pi - theta) / torch.pi
                return base_val * (sqrt_prod ** (degree - 1))

        def nngp_ntk_fn(nngp_val, prod, ntk_val=None):
            sqrt_prod = torch.sqrt(torch.clamp(prod, min=1e-15))
            coeff = (sqrt_prod**degree) / (2 * torch.pi)

            if ntk_val is not None:
                if degree == 0:
                    ntk_val = torch.zeros_like(ntk_val)
                else:
                    # Simplified derivative approximation
                    j_val = j_fn(nngp_val, sqrt_prod)
                    ntk_val = ntk_val * coeff * j_val * degree

            nngp_new = coeff * j_fn(nngp_val, sqrt_prod)
            return nngp_new, ntk_val

        def nngp_fn_diag(nngp_val):
            return _double_factorial(2 * degree - 1) * (nngp_val**degree) / 2

        nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

        if kernels.diagonal_batch and kernels.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1) if cov1 is not None else None
            cov2 = nngp_fn_diag(cov2) if cov2 is not None else None
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11) if cov1 is not None else (None, None)
            cov2, _ = nngp_ntk_fn(cov2, prod22) if cov2 is not None else (None, None)

        return kernels.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'{degree}-rectified-monomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def Polynomial(coef: Sequence[float]) -> InternalLayer:
    """Polynomials, i.e. `coef[0] + coef[1] * x + … + coef[n] * x**n`.

    Args:
        coef: a sequence of coefficients. Follows numpy.polynomial.polynomial.Polynomial API.

    Returns:
        `(init_fn, apply_fn, kernel_fn)`.
    """
    try:
        import numpy as np
        coef = np.array(coef)
    except ImportError:
        coef = list(coef)

    def fn(x):
        # Evaluate polynomial using coefficients (reverse order for torch.polyval)
        if isinstance(coef, list):
            coef_tensor = torch.tensor(coef[::-1], dtype=x.dtype, device=x.device)
        else:
            coef_tensor = torch.tensor(coef[::-1], dtype=x.dtype, device=x.device)
        return torch.polyval(coef_tensor, x)

    degree = len(coef)

    def _factorial(n):
        """Compute factorial."""
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def kernel_fn(kernels, **kwargs):
        cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

        def r_fn(n, l):
            """Compute r_l(n) for polynomial kernel."""
            if n is None:
                return None

            coef_dict = {}
            for i in range(0, (degree - 1 - l) // 2 + 1):
                idx = 2 * i + l
                if idx < degree:
                    coef_val = coef[idx] * _factorial(idx) / (
                        (2**i) * _factorial(i) * (_factorial(l)**0.5))
                    coef_dict[idx] = coef_val

            # Evaluate polynomial at sqrt(n)
            coef_l = [coef_dict.get(i, 0) for i in range(degree)]
            if isinstance(coef_l, list):
                coef_tensor = torch.tensor(coef_l[::-1], dtype=n.dtype, device=n.device)
            else:
                coef_tensor = torch.tensor(coef_l[::-1], dtype=n.dtype, device=n.device)
            
            sqrt_n = torch.sqrt(torch.clamp(n, min=0))
            return torch.polyval(coef_tensor, sqrt_n)

        if degree == 0:
            rs11, rs12, rs22 = [], [], []
        else:
            rs11, rs12, rs22 = [], [], []
            for l in range(degree):
                r11_l, r12_l, r22_l = get_diagonal_outer_prods(
                    r_fn(cov1, l), r_fn(cov2, l),
                    kernels.diagonal_batch, kernels.diagonal_spatial, op.mul)
                rs11.append(r11_l)
                rs12.append(r12_l)
                rs22.append(r22_l)

        prod11, prod12, prod22 = get_diagonal_outer_prods(
            cov1, cov2, kernels.diagonal_batch, kernels.diagonal_spatial, op.mul)

        def nngp_ntk_fn(nngp_val, prod, r_prods, ntk_val=None):
            ratio = nngp_val / torch.sqrt(torch.clamp(prod, min=1e-15))

            if ntk_val is not None:
                t_dot = torch.zeros_like(ntk_val)
                for l in range(1, degree):
                    if l < len(r_prods):
                        t_dot += l * r_prods[l] * (ratio**(l - 1))
                ntk_val = ntk_val * t_dot / torch.sqrt(torch.clamp(prod, min=1e-15))

            nngp_new = torch.zeros_like(nngp_val)
            for l in range(degree):
                if l < len(r_prods):
                    nngp_new += r_prods[l] * (ratio ** l)

            return nngp_new, ntk_val

        def nngp_fn_diag(nngp_val, r_prods):
            out = torch.zeros_like(nngp_val)
            for l in range(min(degree, len(r_prods))):
                out += r_prods[l]
            return out

        nngp, ntk = nngp_ntk_fn(nngp, prod12, rs12, ntk)

        if kernels.diagonal_batch and kernels.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1, rs11) if cov1 is not None else None
            cov2 = nngp_fn_diag(cov2, rs22) if cov2 is not None else None
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11, rs11) if cov1 is not None else (None, None)
            cov2, _ = nngp_ntk_fn(cov2, prod22, rs22) if cov2 is not None else (None, None)

        return kernels.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'{coef}-polynomial', kernel_fn)


def _double_factorial(n):
    """Compute double factorial n!!."""
    if n <= 0:
        return 1
    result = 1
    for i in range(n, 0, -2):
        result *= i
    return result