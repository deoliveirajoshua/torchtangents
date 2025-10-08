"""PyTorch-compatible rules system for structured derivatives.

This module provides structure tracking and Jacobian rules for PyTorch operations
to enable efficient structured derivative computation for NTK calculations.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum


@dataclass
class Structure:
    """Represents the structure of derivatives for efficient computation.
    
    Attributes:
        in_diagonal: Input axes that are diagonal.
        out_diagonal: Output axes that are diagonal.
        in_trace: Input axes to trace over.
        out_trace: Output axes to trace over.
        in_broadcast: Input axes that are broadcast.
        out_broadcast: Output axes that are broadcast.
    """
    in_diagonal: List[int] = None
    out_diagonal: List[int] = None
    in_trace: List[int] = None
    out_trace: List[int] = None
    in_broadcast: List[int] = None
    out_broadcast: List[int] = None
    
    def __post_init__(self):
        # Initialize empty lists if None
        for attr in ['in_diagonal', 'out_diagonal', 'in_trace', 'out_trace', 'in_broadcast', 'out_broadcast']:
            if getattr(self, attr) is None:
                setattr(self, attr, [])
    
    def __and__(self, other: 'Structure') -> 'Structure':
        """Intersection of two structures."""
        return Structure(
            in_diagonal=list(set(self.in_diagonal) & set(other.in_diagonal)),
            out_diagonal=list(set(self.out_diagonal) & set(other.out_diagonal)),
            in_trace=list(set(self.in_trace) & set(other.in_trace)),
            out_trace=list(set(self.out_trace) & set(other.out_trace)),
            in_broadcast=list(set(self.in_broadcast) & set(other.in_broadcast)),
            out_broadcast=list(set(self.out_broadcast) & set(other.out_broadcast))
        )


class OpType(Enum):
    """Types of operations for rule dispatch."""
    LINEAR = "linear"
    CONV = "conv"
    RELU = "relu"
    IDENTITY = "identity"
    RESHAPE = "reshape"
    BIAS = "bias"
    UNKNOWN = "unknown"


def get_identity_structure(shape: torch.Size, s_rules: bool = True) -> Structure:
    """Get structure for identity operation."""
    if not s_rules:
        return Structure()
    
    # For identity, input and output have same structure
    n_dims = len(shape)
    return Structure(
        in_diagonal=list(range(n_dims)),
        out_diagonal=list(range(n_dims)),
        in_trace=list(range(n_dims)),
        out_trace=list(range(n_dims))
    )


def get_operation_structure(op_type: OpType, input_shapes: List[torch.Size], 
                          output_shape: torch.Size, idx: int, 
                          s_rules: bool = True) -> Structure:
    """Get structure information for a given operation.
    
    Args:
        op_type: Type of operation.
        input_shapes: Shapes of input tensors.
        output_shape: Shape of output tensor.
        idx: Index of the input we're computing derivatives for.
        s_rules: Whether to use structure rules.
        
    Returns:
        Structure object describing the derivative structure.
    """
    if not s_rules:
        return Structure()
    
    if op_type == OpType.IDENTITY:
        return get_identity_structure(input_shapes[idx], s_rules)
    
    elif op_type == OpType.LINEAR:
        # For linear layer: out = x @ W.T + b
        # Structure depends on which input (x, W, or b)
        if idx == 0:  # Input x
            # Input structure preserved in batch dimensions
            return Structure(
                in_diagonal=[0],  # Batch dimension
                out_diagonal=[0]
            )
        elif idx == 1:  # Weight W
            # Weight connects input features to output features
            return Structure()
        else:  # Bias
            return Structure(
                out_diagonal=[0]  # Batch dimension preserved
            )
    
    elif op_type == OpType.CONV:
        # For convolution: similar to linear but with spatial dimensions
        if idx == 0:  # Input
            # Batch and spatial dimensions often have structure
            spatial_dims = list(range(2, len(input_shapes[idx])))
            return Structure(
                in_diagonal=[0] + spatial_dims,
                out_diagonal=[0] + spatial_dims
            )
        elif idx == 1:  # Weight
            return Structure()
        else:  # Bias
            return Structure(
                out_diagonal=[0]
            )
    
    elif op_type == OpType.RELU:
        # ReLU preserves all structure
        n_dims = len(input_shapes[idx])
        all_dims = list(range(n_dims))
        return Structure(
            in_diagonal=all_dims,
            out_diagonal=all_dims,
            in_trace=all_dims,
            out_trace=all_dims
        )
    
    elif op_type == OpType.RESHAPE:
        # Reshape can preserve some structure but it's complex
        # For simplicity, assume no special structure
        return Structure()
    
    elif op_type == OpType.BIAS:
        # Bias addition preserves input structure
        n_dims = len(input_shapes[idx])
        all_dims = list(range(n_dims))
        return Structure(
            in_diagonal=all_dims,
            out_diagonal=all_dims
        )
    
    else:
        # Unknown operation - no structure assumptions
        return Structure()


def linear_jacobian_rule(input_tensor: torch.Tensor, weight: torch.Tensor, 
                        bias: Optional[torch.Tensor], idx: int,
                        output_cotangents_shape: torch.Size) -> torch.Tensor:
    """Jacobian rule for linear layer.
    
    Args:
        input_tensor: Input to linear layer.
        weight: Weight matrix.
        bias: Bias vector (optional).
        idx: Index of parameter to compute Jacobian for (0=input, 1=weight, 2=bias).
        output_cotangents_shape: Shape of output cotangents.
        
    Returns:
        Jacobian tensor.
    """
    batch_size = input_tensor.shape[0]
    
    if idx == 0:  # Jacobian wrt input
        # J = W.T
        jac_shape = output_cotangents_shape + input_tensor.shape
        jacobian = torch.zeros(jac_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        # Fill in the Jacobian: for each batch element and output, derivative is W.T
        for b in range(batch_size):
            for out_idx in range(weight.shape[0]):
                jacobian[b, out_idx, b, :] = weight[out_idx, :]
        
        return jacobian
    
    elif idx == 1:  # Jacobian wrt weight
        # J = input
        jac_shape = output_cotangents_shape + weight.shape
        jacobian = torch.zeros(jac_shape, dtype=weight.dtype, device=weight.device)
        
        # Fill in the Jacobian: for each batch element and output
        for b in range(batch_size):
            for out_idx in range(weight.shape[0]):
                jacobian[b, out_idx, out_idx, :] = input_tensor[b, :]
        
        return jacobian
    
    elif idx == 2 and bias is not None:  # Jacobian wrt bias
        # J = identity
        jac_shape = output_cotangents_shape + bias.shape
        jacobian = torch.zeros(jac_shape, dtype=bias.dtype, device=bias.device)
        
        # Fill in the Jacobian: identity for bias
        for b in range(batch_size):
            for out_idx in range(bias.shape[0]):
                jacobian[b, out_idx, out_idx] = 1.0
        
        return jacobian
    
    else:
        raise ValueError(f"Invalid idx {idx} for linear layer")


def relu_jacobian_rule(input_tensor: torch.Tensor, 
                      output_cotangents_shape: torch.Size) -> torch.Tensor:
    """Jacobian rule for ReLU activation.
    
    Args:
        input_tensor: Input to ReLU.
        output_cotangents_shape: Shape of output cotangents.
        
    Returns:
        Jacobian tensor (diagonal).
    """
    # ReLU Jacobian is diagonal: 1 where input > 0, 0 elsewhere
    mask = (input_tensor > 0).float()
    
    jac_shape = output_cotangents_shape + input_tensor.shape
    jacobian = torch.zeros(jac_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Fill diagonal elements
    batch_size = input_tensor.shape[0]
    feature_size = input_tensor.shape[1:].numel()
    
    for b in range(batch_size):
        flat_mask = mask[b].view(-1)
        for i in range(feature_size):
            # Map flat index back to multidimensional indices
            idx = torch.unravel_index(torch.tensor(i), input_tensor.shape[1:])
            jacobian[(b,) + idx + (b,) + idx] = flat_mask[i]
    
    return jacobian


# Registry of Jacobian rules
JACOBIAN_RULES = {
    OpType.LINEAR: linear_jacobian_rule,
    OpType.RELU: relu_jacobian_rule,
    # Add more rules as needed
}


def get_structure_cache(computation_graph: Any, s_rules: bool = True) -> Dict[Any, Structure]:
    """Get structure cache for all variables in computation graph.
    
    This is a placeholder - in a full implementation, this would analyze
    the PyTorch computation graph and return structure for each variable.
    
    Args:
        computation_graph: PyTorch computation graph.
        s_rules: Whether to use structure rules.
        
    Returns:
        Dictionary mapping variables to their structures.
    """
    # Placeholder implementation
    return {}


def detect_operation_type(module: torch.nn.Module) -> OpType:
    """Detect the operation type from a PyTorch module.
    
    Args:
        module: PyTorch module.
        
    Returns:
        Operation type.
    """
    if isinstance(module, torch.nn.Linear):
        return OpType.LINEAR
    elif isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        return OpType.CONV
    elif isinstance(module, torch.nn.ReLU):
        return OpType.RELU
    elif isinstance(module, torch.nn.Identity):
        return OpType.IDENTITY
    else:
        return OpType.UNKNOWN