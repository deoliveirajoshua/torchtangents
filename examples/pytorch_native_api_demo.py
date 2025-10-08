#!/usr/bin/env python3
"""
Example implementations of the proposed PyTorch-native API for TorchTangents.

This file demonstrates how the current stax-based API could be extended
with more PyTorch-native interfaces that work directly with nn.Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable, Tuple
import warnings

# Import current torchtangents functionality
import torchtangents as tt
from torchtangents._src.empirical import empirical_ntk_fn, empirical_nngp_fn
from torchtangents._src.utils.kernel import Kernel


def ntk(model: nn.Module, 
        x1: torch.Tensor, 
        x2: Optional[torch.Tensor] = None,
        infinite_width: bool = False,
        device: Optional[torch.device] = None,
        batch_size: Optional[Union[int, str]] = None) -> torch.Tensor:
    """
    Compute Neural Tangent Kernel directly from a PyTorch model.
    
    Args:
        model: PyTorch model (nn.Module)
        x1: First set of inputs [N1, ...]
        x2: Second set of inputs [N2, ...]. If None, uses x1
        infinite_width: Whether to compute infinite-width limit
        device: Device to compute on
        batch_size: Batch size for computation ('auto' for automatic)
        
    Returns:
        NTK matrix of shape [N1, N2]
    """
    if device is None:
        device = next(model.parameters()).device
    
    x1 = x1.to(device)
    if x2 is not None:
        x2 = x2.to(device)
    else:
        x2 = x1
    
    model = model.to(device)
    
    if infinite_width:
        # For infinite width, we'd need to implement the infinite-width kernel
        # This is a placeholder - would require converting model to stax representation
        warnings.warn("Infinite width computation not yet implemented for arbitrary models")
        return _compute_finite_ntk(model, x1, x2)
    else:
        return _compute_finite_ntk(model, x1, x2)


def _compute_finite_ntk(model: nn.Module, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute finite-width NTK using empirical computation."""
    
    def model_fn(params, x):
        # Helper to apply model with given parameters
        return torch.func.functional_call(model, params, x)
    
    # Get model parameters
    params = dict(model.named_parameters())
    
    # Create empirical NTK function
    ntk_fn = empirical_ntk_fn(model_fn, trace_axes=(), vmap_axes=0)
    
    # Compute NTK
    ntk_matrix = ntk_fn(x1, x2, params)
    
    return ntk_matrix


def nngp(model: nn.Module,
         x1: torch.Tensor,
         x2: Optional[torch.Tensor] = None,
         device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Compute Neural Network Gaussian Process kernel from a PyTorch model.
    
    Args:
        model: PyTorch model (nn.Module) 
        x1: First set of inputs [N1, ...]
        x2: Second set of inputs [N2, ...]. If None, uses x1
        device: Device to compute on
        
    Returns:
        NNGP matrix of shape [N1, N2]
    """
    if device is None:
        device = next(model.parameters()).device
        
    x1 = x1.to(device)
    if x2 is not None:
        x2 = x2.to(device)
    else:
        x2 = x1
        
    model = model.to(device)
    
    def model_fn(params, x):
        return torch.func.functional_call(model, params, x)
    
    params = dict(model.named_parameters())
    nngp_fn = empirical_nngp_fn(model_fn)
    
    return nngp_fn(x1, x2, params)


def make_ntk_fn(model: nn.Module) -> Callable:
    """
    Create a reusable NTK function from a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Function that computes NTK given inputs
    """
    def ntk_fn(x1, x2=None):
        return ntk(model, x1, x2)
    
    return ntk_fn


def make_nngp_fn(model: nn.Module) -> Callable:
    """
    Create a reusable NNGP function from a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Function that computes NNGP given inputs
    """
    def nngp_fn(x1, x2=None):
        return nngp(model, x1, x2)
    
    return nngp_fn


class NTKTrainer:
    """
    Trainer that uses NTK dynamics for infinite-width training.
    
    This implements gradient descent in function space using the NTK.
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.initial_params = {name: param.clone() 
                              for name, param in model.named_parameters()}
    
    def step(self, x: torch.Tensor, y: torch.Tensor):
        """
        Perform one step of NTK-based training.
        
        Args:
            x: Input batch
            y: Target batch
        """
        # Compute current predictions
        current_pred = self.model(x)
        
        # Compute NTK at current point
        K = ntk(self.model, x)
        
        # Compute gradient in function space
        residual = y - current_pred
        
        # Update in function space: f_new = f_old + lr * K^(-1) * residual
        # This is a simplified version - full implementation would need regularization
        try:
            K_inv = torch.linalg.inv(K + 1e-6 * torch.eye(K.shape[0], device=K.device))
            function_update = self.learning_rate * K_inv @ residual
            
            # Convert function space update back to parameter space
            # This is non-trivial and would require proper NTK theory implementation
            self._apply_function_update(x, function_update)
            
        except torch.linalg.LinAlgError:
            print("Warning: Singular NTK matrix, skipping update")
    
    def _apply_function_update(self, x: torch.Tensor, function_update: torch.Tensor):
        """Apply function space update to model parameters."""
        # This is a placeholder - proper implementation would use NTK theory
        # to convert function updates back to parameter updates
        
        # For now, just do standard gradient descent
        with torch.no_grad():
            current_pred = self.model(x)
            loss = F.mse_loss(current_pred, current_pred + function_update)
            
        loss.backward()
        
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= self.learning_rate * param.grad
                    param.grad.zero_()


def kernel_predict(K_train: torch.Tensor, 
                  y_train: torch.Tensor,
                  K_test: torch.Tensor,
                  regularization: float = 1e-6) -> torch.Tensor:
    """
    Make predictions using kernel regression.
    
    Args:
        K_train: Training kernel matrix [N_train, N_train]
        y_train: Training targets [N_train, output_dim]
        K_test: Test kernel matrix [N_test, N_train]  
        regularization: Ridge regularization parameter
        
    Returns:
        Predictions [N_test, output_dim]
    """
    # Solve: (K + λI) α = y
    I = torch.eye(K_train.shape[0], device=K_train.device)
    alpha = torch.linalg.solve(K_train + regularization * I, y_train)
    
    # Predict: y_test = K_test @ α
    return K_test @ alpha


def analyze_kernel(K: torch.Tensor) -> dict:
    """
    Analyze properties of a kernel matrix.
    
    Args:
        K: Kernel matrix [N, N]
        
    Returns:
        Dictionary with kernel statistics
    """
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(K).real
    eigenvalues = torch.sort(eigenvalues, descending=True)[0]
    
    # Compute various statistics
    condition_number = eigenvalues[0] / eigenvalues[-1]
    trace = torch.trace(K)
    frobenius_norm = torch.norm(K, 'fro')
    nuclear_norm = torch.sum(eigenvalues)
    
    # Effective rank (number of eigenvalues > 1% of max)
    threshold = 0.01 * eigenvalues[0]
    effective_rank = torch.sum(eigenvalues > threshold).item()
    
    return {
        'eigenvalues': eigenvalues,
        'condition_number': condition_number.item(),
        'trace': trace.item(),
        'frobenius_norm': frobenius_norm.item(),
        'nuclear_norm': nuclear_norm.item(),
        'effective_rank': effective_rank,
        'rank': torch.linalg.matrix_rank(K).item()
    }


def kernel_similarity(K1: torch.Tensor, K2: torch.Tensor) -> float:
    """
    Compute similarity between two kernel matrices.
    
    Uses centered kernel alignment (CKA).
    
    Args:
        K1: First kernel matrix
        K2: Second kernel matrix
        
    Returns:
        Similarity score between 0 and 1
    """
    # Center the kernels
    n = K1.shape[0]
    H = torch.eye(n, device=K1.device) - torch.ones(n, n, device=K1.device) / n
    
    K1_centered = H @ K1 @ H
    K2_centered = H @ K2 @ H
    
    # Compute CKA
    numerator = torch.trace(K1_centered @ K2_centered)
    denominator = torch.sqrt(torch.trace(K1_centered @ K1_centered) * 
                           torch.trace(K2_centered @ K2_centered))
    
    return (numerator / denominator).item()


# Decorator for making models NTK-compatible
def ntk_compatible(cls):
    """
    Decorator to add NTK computation methods to a model class.
    
    Usage:
        @ntk_compatible
        class MyModel(nn.Module):
            ...
            
        model = MyModel()
        ntk_matrix = model.compute_ntk(x_train, x_test)
    """
    
    def compute_ntk(self, x1, x2=None, **kwargs):
        return ntk(self, x1, x2, **kwargs)
    
    def compute_nngp(self, x1, x2=None, **kwargs):
        return nngp(self, x1, x2, **kwargs)
    
    def analyze_kernels(self, x):
        """Analyze both NTK and NNGP properties."""
        ntk_matrix = self.compute_ntk(x)
        nngp_matrix = self.compute_nngp(x)
        
        return {
            'ntk': analyze_kernel(ntk_matrix),
            'nngp': analyze_kernel(nngp_matrix),
            'similarity': kernel_similarity(ntk_matrix, nngp_matrix)
        }
    
    # Add methods to class
    cls.compute_ntk = compute_ntk
    cls.compute_nngp = compute_nngp  
    cls.analyze_kernels = analyze_kernels
    
    return cls


# Example usage
if __name__ == "__main__":
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(), 
        nn.Linear(32, 1)
    )
    
    # Generate some data
    x_train = torch.randn(20, 10)
    x_test = torch.randn(10, 10)
    y_train = torch.randn(20, 1)
    
    print("=== PyTorch-Native NTK API Demo ===")
    
    # Compute NTK using new API
    print("Computing NTK...")
    K_train = ntk(model, x_train)
    K_test = ntk(model, x_test, x_train)
    
    print(f"Training kernel shape: {K_train.shape}")
    print(f"Test kernel shape: {K_test.shape}")
    
    # Analyze kernel properties
    print("\nAnalyzing kernel...")
    stats = analyze_kernel(K_train)
    print(f"Condition number: {stats['condition_number']:.2f}")
    print(f"Effective rank: {stats['effective_rank']}/{K_train.shape[0]}")
    
    # Make predictions
    print("\nMaking kernel predictions...")
    y_pred = kernel_predict(K_train, y_train, K_test)
    print(f"Predictions shape: {y_pred.shape}")
    
    # Demonstrate decorator usage
    @ntk_compatible
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.net(x)
    
    print("\n=== Decorator Demo ===")
    custom_model = CustomModel()
    
    # Use added methods
    ntk_matrix = custom_model.compute_ntk(x_train[:5])
    analysis = custom_model.analyze_kernels(x_train[:5])
    
    print(f"Custom model NTK shape: {ntk_matrix.shape}")
    print(f"NTK-NNGP similarity: {analysis['similarity']:.3f}")
    
    print("\nDemo completed successfully!")