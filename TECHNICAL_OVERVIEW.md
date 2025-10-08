# TorchTangents Technical Overview

## Architecture and Design Philosophy

TorchTangents is a PyTorch implementation of Neural Tangent Kernel (NTK) theory, designed to provide efficient computation of infinite-width neural network kernels. The package follows a functional programming paradigm inspired by JAX's neural-tangents library while leveraging PyTorch's automatic differentiation and tensor operations.

## Core Concepts

### 1. The Stax Architecture

TorchTangents uses a **functional layer composition system** called "stax" (short for "stack"), where each layer is represented as a triple of functions:

```python
(init_fn, apply_fn, kernel_fn) = layer_definition()
```

- **`init_fn`**: Initializes layer parameters and computes output shape
- **`apply_fn`**: Forward pass computation (standard neural network)
- **`kernel_fn`**: Infinite-width kernel computation (the key innovation)

This design enables **composability** - layers can be easily combined through combinators like `serial()` and `parallel()`.

### 2. The Kernel Data Structure

The `Kernel` dataclass is the heart of TorchTangents' computation:

```python
@dataclass
class Kernel:
    nngp: Optional[torch.Tensor] = None      # Neural Network Gaussian Process
    ntk: Optional[torch.Tensor] = None       # Neural Tangent Kernel  
    cov1: Optional[torch.Tensor] = None      # Input batch 1 covariance
    cov2: Optional[torch.Tensor] = None      # Input batch 2 covariance
    x1_is_x2: Optional[bool] = None          # Whether inputs are identical
    is_gaussian: bool = True                 # Whether outputs are Gaussian
    diagonal_spatial: bool = False           # Spatial dimension handling
    diagonal_batch: bool = True              # Batch dimension handling
    # ... additional metadata fields
```

**Why this structure matters:**

- **NNGP**: Represents the infinite-width limit behavior (Bayesian prior)
- **NTK**: Represents the training dynamics (gradient descent behavior)
- **Covariances**: Track input statistics needed for kernel propagation
- **Flags**: Control memory efficiency and computation strategies

### 3. PyTrees: Flexible Parameter Management

TorchTangents uses **PyTrees** (Python tree structures) for parameter management:

```python
PyTree = Union[torch.Tensor, Dict[str, Any], Tuple[Any, ...], Any]
```

**Benefits of PyTrees:**

1. **Hierarchical organization**: Natural representation of nested network structures
2. **Automatic batching**: Operations apply recursively across tree structures  
3. **Functional programming**: Immutable parameter updates and transformations
4. **Compatibility**: Easy integration with PyTorch's existing ecosystem

**Example PyTree structure:**
```python
params = {
    'layer_0': {'weights': torch.tensor(...), 'bias': torch.tensor(...)},
    'layer_1': {'weights': torch.tensor(...), 'bias': torch.tensor(...)},
    'layer_2': {'weights': torch.tensor(...)}
}
```

## Key Decorators and Their Purpose

### 1. `@layer` Decorator

The `@layer` decorator is fundamental to the stax system:

```python
@layer
def Dense(out_dim: int) -> InternalLayer:
    def init_fn(rng, input_shape):
        # Initialize parameters
    def apply_fn(params, inputs, rng=None):
        # Forward pass
    def kernel_fn(kernel):
        # Kernel computation  
    return init_fn, apply_fn, kernel_fn
```

**What `@layer` does:**

1. **Kernel preprocessing**: Wraps `kernel_fn` to handle both tensor inputs and Kernel objects
2. **Input validation**: Ensures proper shapes and types
3. **Automatic conversion**: Converts tensor inputs to Kernel objects when needed
4. **API consistency**: Provides uniform interface across all layers

**Under the hood:**
```python
def layer(layer_fn):
    def new_layer_fns(*args, **kwargs):
        init_fn, apply_fn, kernel_fn = layer_fn(*args, **kwargs)
        kernel_fn = _preprocess_kernel_fn(init_fn, apply_fn, kernel_fn)
        return init_fn, apply_fn, kernel_fn
    return new_layer_fns
```

### 2. `@requires` Decorator

The `@requires` decorator enforces kernel computation requirements:

```python
@requires(diagonal_spatial=Diagonal.EITHER, diagonal_batch=Bool.YES)
def kernel_fn(kernel: Kernel) -> Kernel:
    # Layer-specific kernel computation
```

**Purpose:**

1. **Constraint validation**: Ensures kernel properties match layer requirements
2. **Optimization hints**: Guides efficient computation strategies
3. **Error prevention**: Catches incompatible layer combinations at runtime
4. **Documentation**: Makes layer requirements explicit

**Example requirements:**
- `diagonal_spatial=Diagonal.YES`: Layer requires diagonal spatial covariance (e.g., after flattening)
- `diagonal_batch=Bool.NO`: Layer needs full batch covariance (rare, memory intensive)

### 3. `@supports_masking` Decorator

Handles masked input patterns (sparse data, attention masks):

```python
@supports_masking(remask_kernel=True)
def SomeLayer(...):
    # Layer definition with masking support
```

**Functionality:**

1. **Mask propagation**: Tracks which inputs are masked through the network
2. **Kernel masking**: Zeros out kernel entries between masked inputs
3. **Attention support**: Enables transformer-style attention mechanisms
4. **Sparse computation**: Optimizes computation for sparse input patterns

## Kernel Computation Under the Hood

### 1. Infinite-Width Theory

TorchTangents implements the mathematical theory that neural networks converge to Gaussian Processes in the infinite width limit:

**NNGP (Bayesian Prior):**
```
K_NNGP(x1, x2) = σ²_w * Σ(x1, x2) + σ²_b
```

**NTK (Training Dynamics):**
```  
K_NTK(x1, x2) = Σ ∂f(x1)/∂θ * ∂f(x2)/∂θ
```

### 2. Layer-by-Layer Propagation

Each layer transforms the kernel according to its mathematical properties:

**Dense Layer:**
```python
def kernel_fn(kernel: Kernel) -> Kernel:
    # Linear transformation: K_out = σ²_w * K_in + σ²_b
    nngp = σ_w**2 * kernel.nngp + σ_b**2
    ntk = σ_w**2 * (kernel.nngp * kernel.ntk + kernel.nngp)
    return kernel.replace(nngp=nngp, ntk=ntk)
```

**ReLU Activation:**
```python
def kernel_fn(kernel: Kernel) -> Kernel:
    # Nonlinear transformation using arc-cosine kernels
    nngp = arc_cosine_kernel(kernel.nngp, degree=1)
    ntk = nngp * kernel.ntk
    return kernel.replace(nngp=nngp, ntk=ntk)
```

### 3. Compositional Computation

**Serial Composition** (feedforward):
```python
def serial(*layers):
    def kernel_fn(kernel):
        for _, _, layer_kernel in layers:
            kernel = layer_kernel(kernel)
        return kernel
```

**Parallel Composition** (branching):
```python
def parallel(*layers):
    def kernel_fn(kernel):
        results = [layer_kernel(kernel) for _, _, layer_kernel in layers]
        # Combine results (sum, concatenate, etc.)
        return combine_kernels(results)
```

## Memory and Computational Efficiency

### 1. Diagonal Approximations

TorchTangents uses strategic approximations to manage memory:

**Spatial Diagonalization:**
- `diagonal_spatial=True`: Store only spatial diagonal, O(HW) instead of O(H²W²)
- Used after pooling/flattening layers where spatial correlations are broken

**Batch Diagonalization:**
- `diagonal_batch=True`: Store only batch diagonal, O(N) instead of O(N²)
- Default mode for most computations

### 2. Empirical vs Infinite-Width

**Infinite-Width Kernels** (analytic):
- Computed via mathematical formulas
- Exact in the infinite width limit
- Memory efficient: O(N²) for N inputs

**Empirical Kernels** (finite networks):
- Computed via automatic differentiation
- Exact for finite networks of any width
- More memory intensive but supports any architecture

### 3. Implementation Strategies

**NTK Computation Methods:**

1. **Jacobian Contraction**: `J @ J.T` approach
   - Memory: O(N² * P) where P = parameters
   - Best for: Wide networks, small datasets

2. **NTK-Vector Products**: Implicit NTK computation
   - Memory: O(N * sqrt(P))
   - Best for: Very large parameter counts

3. **Structured Derivatives**: Custom autodiff
   - Memory: O(N²) 
   - Best for: Production use, balanced efficiency

## Advanced Features

### 1. Batching System

Automatic computation batching for large datasets:

```python
kernel_fn = tt.batch(kernel_fn, batch_size=32)
```

**Benefits:**
- Constant memory usage regardless of dataset size
- Automatic chunking and reassembly
- Transparent to user code

### 2. Device Management

Seamless GPU/CPU computation:

```python
# Automatic device detection and optimization
kernel = kernel_fn(x.cuda(), y.cuda())  # GPU computation
kernel = kernel_fn(x.cpu(), y.cpu())    # CPU computation
```

### 3. Mixed Precision

Support for efficient computation:

```python
# Automatic mixed precision for memory efficiency
with torch.autocast('cuda'):
    kernel = kernel_fn(x, y)
```

## Integration with PyTorch Ecosystem

### 1. Autodiff Compatibility

TorchTangents layers are fully compatible with PyTorch's autograd:

```python
# Standard PyTorch training
loss = criterion(apply_fn(params, x), y)
grads = torch.autograd.grad(loss, params)

# NTK prediction (no gradients needed)
kernel = kernel_fn(x_train, x_test)
predictions = tt.predict.gp_inference(kernel, y_train)
```

### 2. Model Conversion

Easy conversion between PyTorch models and stax:

```python
# PyTorch to stax
pytorch_model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
init_fn, apply_fn, kernel_fn = tt.convert_to_stax(pytorch_model)

# Stax to PyTorch
stax_model = tt.stax.serial(tt.stax.Dense(50), tt.stax.Relu(), tt.stax.Dense(1))
pytorch_model = tt.convert_to_pytorch(stax_model)
```

## Why This Design Matters

### 1. **Composability**: 
Layers compose naturally through mathematical operations, enabling complex architectures from simple building blocks.

### 2. **Efficiency**: 
Strategic approximations and specialized algorithms make infinite-width computation tractable for practical datasets.

### 3. **Theoretical Grounding**: 
Direct implementation of mathematical theory ensures correctness and enables research into infinite-width phenomena.

### 4. **Practical Utility**: 
Despite theoretical foundations, the library provides practical tools for uncertainty quantification, architecture design, and transfer learning.

### 5. **Extensibility**: 
The functional design makes it easy to add new layers, architectures, and computation strategies while maintaining compatibility.

This architecture enables TorchTangents to bridge theoretical understanding of infinite-width neural networks with practical machine learning applications, providing both research insights and production-ready tools for the PyTorch ecosystem.