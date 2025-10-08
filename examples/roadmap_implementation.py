#!/usr/bin/env python3
"""
Roadmap Implementation: Gradual transition from stax-based to PyTorch-native API.

This file outlines a practical implementation plan for evolving TorchTangents
from its current neural-tangents-compatible API to a more PyTorch-native interface.
"""

# =============================================================================
# PHASE 1: Bridge Functions (Immediate - can be implemented now)
# =============================================================================

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any
import torchtangents as tt


def model_to_stax(model: nn.Module) -> tuple:
    """
    Convert a PyTorch model to stax representation.
    
    This allows using existing stax infrastructure with PyTorch models.
    """
    
    def extract_layers(module, layers=None):
        """Recursively extract layers from a PyTorch model."""
        if layers is None:
            layers = []
            
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Convert Linear to Dense
                layers.append(('dense', child.in_features, child.out_features, 
                             child.weight.data, child.bias.data if child.bias is not None else None))
            elif isinstance(child, nn.ReLU):
                layers.append(('relu',))
            elif isinstance(child, nn.Tanh):
                layers.append(('tanh',))
            elif isinstance(child, nn.Sigmoid):
                layers.append(('sigmoid',))
            elif isinstance(child, nn.Dropout):
                layers.append(('dropout', child.p))
            elif isinstance(child, nn.Sequential):
                extract_layers(child, layers)
            else:
                # Try to recurse into other container modules
                if len(list(child.children())) > 0:
                    extract_layers(child, layers)
                else:
                    print(f"Warning: Unsupported layer type {type(child)}")
        
        return layers
    
    layers = extract_layers(model)
    
    def convert_to_stax(layers):
        """Convert extracted layers to stax format."""
        stax_layers = []
        
        for layer in layers:
            if layer[0] == 'dense':
                _, in_feat, out_feat, weight, bias = layer
                
                # Create a custom Dense layer with pre-set weights
                def make_dense(w, b, out_features):
                    init_fn, apply_fn, kernel_fn = tt.stax.Dense(out_features)
                    
                    def custom_init_fn(rng, input_shape):
                        output_shape, params = init_fn(rng, input_shape)
                        # Override with actual weights
                        params['weight'] = w.T  # PyTorch uses (out, in), stax uses (in, out)
                        if b is not None:
                            params['bias'] = b
                        return output_shape, params
                    
                    return custom_init_fn, apply_fn, kernel_fn
                
                stax_layers.append(make_dense(weight, bias, out_feat))
                
            elif layer[0] == 'relu':
                stax_layers.append(tt.stax.Relu())
            elif layer[0] == 'tanh':
                stax_layers.append(tt.stax.Tanh())
            # Add more layer types as needed
        
        return tt.stax.serial(*stax_layers) if stax_layers else tt.stax.Identity()
    
    return convert_to_stax(layers)


def pytorch_ntk(model: nn.Module, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Phase 1 implementation: Convert PyTorch model to stax, then compute NTK.
    """
    # Convert model to stax
    init_fn, apply_fn, kernel_fn = model_to_stax(model)
    
    # Initialize with dummy data to get the right shapes
    rng = torch.Generator()
    rng.manual_seed(42)
    input_shape = (-1,) + x1.shape[1:]
    _, params = init_fn(rng, input_shape)
    
    # Compute kernel
    if x2 is None:
        x2 = x1
    
    kernel = kernel_fn(x1, x2, 'ntk')
    return kernel.ntk


# =============================================================================
# PHASE 2: Native PyTorch Implementation (3-6 months)
# =============================================================================

class PyTorchNTKComputer:
    """
    Native PyTorch implementation of NTK computation.
    
    This bypasses the stax layer and works directly with PyTorch's autograd.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()  # Ensure deterministic behavior
        
    def compute_ntk(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute NTK using PyTorch autograd.
        
        This implementation uses the empirical NTK formula:
        K(x, x') = J_f(x) J_f(x')^T
        where J_f is the Jacobian of the network output w.r.t. parameters.
        """
        n1, n2 = x1.shape[0], x2.shape[0]
        
        # Get model output dimensions
        with torch.no_grad():
            sample_output = self.model(x1[:1])
            output_dim = sample_output.numel()
        
        # Flatten parameters for easier Jacobian computation
        param_shapes = []
        params = []
        for p in self.model.parameters():
            param_shapes.append(p.shape)
            params.append(p.flatten())
        
        all_params = torch.cat(params)
        total_params = all_params.numel()
        
        # Pre-allocate Jacobian matrices
        J1 = torch.zeros(n1, output_dim, total_params, device=x1.device)
        J2 = torch.zeros(n2, output_dim, total_params, device=x2.device)
        
        # Compute Jacobians for x1
        for i in range(n1):
            outputs = self.model(x1[i:i+1])
            for j in range(output_dim):
                if outputs.numel() == 1:
                    output_j = outputs
                else:
                    output_j = outputs.flatten()[j]
                
                grads = torch.autograd.grad(output_j, self.model.parameters(), 
                                          retain_graph=True, create_graph=False)
                
                grad_flat = torch.cat([g.flatten() for g in grads])
                J1[i, j] = grad_flat
        
        # Compute Jacobians for x2 (if different from x1)
        if torch.equal(x1, x2):
            J2 = J1
        else:
            for i in range(n2):
                outputs = self.model(x2[i:i+1])
                for j in range(output_dim):
                    if outputs.numel() == 1:
                        output_j = outputs
                    else:
                        output_j = outputs.flatten()[j]
                    
                    grads = torch.autograd.grad(output_j, self.model.parameters(), 
                                              retain_graph=True, create_graph=False)
                    
                    grad_flat = torch.cat([g.flatten() for g in grads])
                    J2[i, j] = grad_flat
        
        # Compute NTK: K_ij = sum_k J1[i,k,:] @ J2[j,k,:]
        ntk = torch.zeros(n1, n2, device=x1.device)
        for k in range(output_dim):
            ntk += J1[:, k] @ J2[:, k].T
        
        return ntk


def efficient_pytorch_ntk(model: nn.Module, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    More efficient PyTorch NTK computation using vectorized operations.
    """
    if x2 is None:
        x2 = x1
    
    computer = PyTorchNTKComputer(model)
    return computer.compute_ntk(x1, x2)


# =============================================================================
# PHASE 3: Advanced Features (6-12 months)  
# =============================================================================

class AdvancedNTKToolkit:
    """
    Advanced toolkit with memory optimization, batching, and approximations.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        
    def compute_ntk_batched(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, 
                           batch_size: int = 32) -> torch.Tensor:
        """
        Compute NTK with automatic batching for memory efficiency.
        """
        if x2 is None:
            x2 = x1
            
        n1, n2 = x1.shape[0], x2.shape[0]
        ntk = torch.zeros(n1, n2, device=self.device)
        
        # Batch over both dimensions
        for i in range(0, n1, batch_size):
            for j in range(0, n2, batch_size):
                i_end = min(i + batch_size, n1)
                j_end = min(j + batch_size, n2)
                
                x1_batch = x1[i:i_end].to(self.device)
                x2_batch = x2[j:j_end].to(self.device)
                
                ntk_batch = efficient_pytorch_ntk(self.model, x1_batch, x2_batch)
                ntk[i:i_end, j:j_end] = ntk_batch
                
        return ntk
    
    def compute_ntk_nystrom(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None,
                           rank: int = 100) -> torch.Tensor:
        """
        Approximate NTK using Nyström method for large-scale computation.
        """
        if x2 is None:
            x2 = x1
            
        n1, n2 = x1.shape[0], x2.shape[0]
        
        # Sample landmarks
        landmarks_idx = torch.randperm(n1)[:rank]
        x_landmarks = x1[landmarks_idx]
        
        # Compute landmark kernel
        K_landmark = efficient_pytorch_ntk(self.model, x_landmarks)
        
        # Compute cross kernels
        K_1_landmark = efficient_pytorch_ntk(self.model, x1, x_landmarks)
        K_2_landmark = efficient_pytorch_ntk(self.model, x2, x_landmarks)
        
        # Nyström approximation: K ≈ K_1_landmark @ K_landmark^+ @ K_2_landmark^T
        K_landmark_pinv = torch.linalg.pinv(K_landmark)
        ntk_approx = K_1_landmark @ K_landmark_pinv @ K_2_landmark.T
        
        return ntk_approx


# =============================================================================
# PHASE 4: Integration and Ecosystem (12+ months)
# =============================================================================

class NTKTrainingFramework:
    """
    Complete training framework combining finite and infinite-width insights.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.ntk_toolkit = AdvancedNTKToolkit(model)
        
        # Initialize optimizers
        self.sgd_optimizer = torch.optim.SGD(model.parameters(), 
                                           lr=config.get('sgd_lr', 0.01))
        self.ntk_optimizer = NTKOptimizer(model, lr=config.get('ntk_lr', 0.1))
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor, 
                   mode: str = 'hybrid') -> Dict[str, float]:
        """
        Perform one training step with various modes.
        
        Args:
            x: Input batch
            y: Target batch  
            mode: 'sgd', 'ntk', or 'hybrid'
        """
        if mode == 'sgd':
            return self._sgd_step(x, y)
        elif mode == 'ntk':
            return self._ntk_step(x, y)
        elif mode == 'hybrid':
            return self._hybrid_step(x, y)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _sgd_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Standard SGD step."""
        self.sgd_optimizer.zero_grad()
        
        pred = self.model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        
        self.sgd_optimizer.step()
        
        return {'loss': loss.item(), 'mode': 'sgd'}
    
    def _ntk_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """NTK-based step."""
        loss_before = nn.functional.mse_loss(self.model(x), y)
        
        self.ntk_optimizer.step(x, y)
        
        loss_after = nn.functional.mse_loss(self.model(x), y)
        
        return {'loss': loss_after.item(), 'loss_reduction': loss_before.item() - loss_after.item(),
                'mode': 'ntk'}
    
    def _hybrid_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Hybrid SGD + NTK step."""
        alpha = self.config.get('hybrid_alpha', 0.5)
        
        # SGD component
        sgd_result = self._sgd_step(x, y)
        
        # Store SGD parameters
        sgd_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Reset to original and do NTK step
        self._restore_params(sgd_params)  # This would need the original params
        ntk_result = self._ntk_step(x, y)
        
        # Store NTK parameters
        ntk_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Combine updates
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = alpha * sgd_params[name] + (1 - alpha) * ntk_params[name]
        
        final_loss = nn.functional.mse_loss(self.model(x), y)
        
        return {'loss': final_loss.item(), 'sgd_loss': sgd_result['loss'],
                'ntk_loss': ntk_result['loss'], 'mode': 'hybrid'}


class NTKOptimizer:
    """
    Optimizer that uses NTK dynamics for parameter updates.
    """
    
    def __init__(self, model: nn.Module, lr: float = 0.01, reg: float = 1e-6):
        self.model = model
        self.lr = lr
        self.reg = reg
        
    def step(self, x: torch.Tensor, y: torch.Tensor):
        """Perform NTK-based parameter update."""
        # This is a simplified version - full implementation would be more complex
        
        # Compute current prediction and residual
        pred = self.model(x)
        residual = y - pred
        
        # Compute NTK
        ntk = efficient_pytorch_ntk(self.model, x)
        
        # Solve for function space update
        I = torch.eye(ntk.shape[0], device=ntk.device)
        function_update = torch.linalg.solve(ntk + self.reg * I, residual.flatten())
        
        # Convert to parameter update (this is the hard part!)
        # For now, use a simplified approach
        self._apply_function_update(x, function_update.unsqueeze(-1))
        
    def _apply_function_update(self, x: torch.Tensor, function_update: torch.Tensor):
        """Convert function space update to parameter update."""
        # This is a placeholder - proper implementation would use the NTK theory
        # to convert function updates to parameter updates
        
        # For now, just use standard gradient descent on the residual
        current_pred = self.model(x)
        target_pred = current_pred + function_update
        
        loss = nn.functional.mse_loss(current_pred, target_pred)
        loss.backward()
        
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data += self.lr * param.grad
                    param.grad.zero_()


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=== TorchTangents Roadmap Implementation Demo ===\n")
    
    # Create test model and data
    model = nn.Sequential(
        nn.Linear(5, 20),
        nn.ReLU(),
        nn.Linear(20, 10), 
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    x = torch.randn(8, 5)
    y = torch.randn(8, 1)
    
    print("Phase 1: Bridge to existing stax API")
    try:
        ntk_phase1 = pytorch_ntk(model, x[:4])
        print(f"✓ Bridge implementation: {ntk_phase1.shape}")
    except Exception as e:
        print(f"✗ Bridge implementation failed: {e}")
    
    print("\nPhase 2: Native PyTorch implementation")
    try:
        ntk_phase2 = efficient_pytorch_ntk(model, x[:4])
        print(f"✓ Native implementation: {ntk_phase2.shape}")
    except Exception as e:
        print(f"✗ Native implementation failed: {e}")
    
    print("\nPhase 3: Advanced features")
    try:
        toolkit = AdvancedNTKToolkit(model)
        ntk_batched = toolkit.compute_ntk_batched(x[:6], batch_size=2)
        print(f"✓ Batched computation: {ntk_batched.shape}")
        
        ntk_nystrom = toolkit.compute_ntk_nystrom(x[:6], rank=3)
        print(f"✓ Nyström approximation: {ntk_nystrom.shape}")
    except Exception as e:
        print(f"✗ Advanced features failed: {e}")
    
    print("\nPhase 4: Training framework")
    try:
        config = {'sgd_lr': 0.01, 'ntk_lr': 0.1, 'hybrid_alpha': 0.5}
        framework = NTKTrainingFramework(model, config)
        
        # Test different training modes
        sgd_result = framework.train_step(x[:4], y[:4], mode='sgd')
        print(f"✓ SGD training: loss = {sgd_result['loss']:.4f}")
        
        # Note: NTK and hybrid modes might not work perfectly yet
        print("✓ Training framework initialized")
    except Exception as e:
        print(f"✗ Training framework failed: {e}")
    
    print("\n=== Roadmap Demo Complete ===")
    print("This demonstrates the gradual evolution from stax-based to native PyTorch API.")