# NTK Gradient Computation Examples

This directory contains comprehensive examples showing how to compute gradients of Neural Tangent Kernels (NTK) with respect to inputs using TorchTangents.

## Examples Overview

### 1. `simple_ntk_gradients.py` 
**Best starting point for beginners**

- **Infinite-width NTK gradients**: Using `kernel_fn` from stax
- **Empirical NTK gradients**: Using `empirical_kernel_fn`
- **Input optimization**: Maximize NTK values using gradients
- **Sensitivity analysis**: Find which input dimensions matter most

```bash
python examples/simple_ntk_gradients.py
```

### 2. `advanced_ntk_gradients.py`
**Advanced applications and research use cases**

- **Adversarial generation**: Create inputs that maximally change NTK
- **Higher-order gradients**: Hessian analysis for curvature information
- **Method comparison**: Infinite-width vs empirical gradient analysis
- **Landscape analysis**: Understanding local NTK topology

```bash
python examples/advanced_ntk_gradients.py
```

### 3. `ntk_gradients.py`
**Comprehensive tutorial with detailed explanations**

- **Three computation methods**: Infinite-width, empirical, manual Jacobian
- **Educational examples**: Step-by-step gradient computation
- **Practical applications**: Real-world use cases
- **Method comparison**: Performance and accuracy analysis

```bash
python examples/ntk_gradients.py
```

## Key Capabilities

### ✅ What Works Perfectly

1. **Infinite-width NTK gradients w.r.t. inputs**
   ```python
   # Create infinite-width network
   init_fn, apply_fn, kernel_fn = stax.serial(
       stax.Dense(512), stax.Relu(), stax.Dense(1)
   )
   
   # Compute gradients
   infinite_kernel = kernel_fn(base_kernel)
   grad_x = torch.autograd.grad(infinite_kernel.ntk.sum(), x)[0]
   ```

2. **Empirical NTK gradients w.r.t. inputs**
   ```python
   # Create empirical kernel function
   empirical_kernel_fn = tt.empirical_kernel_fn(apply_fn)
   ntk = empirical_kernel_fn(x1, x2, 'ntk', params).ntk
   
   # Compute gradients
   grad_x = torch.autograd.grad(ntk.sum(), x)[0]
   ```

## Applications

### 🎯 Input Optimization
- Optimize inputs to maximize/minimize kernel values
- Design inputs for optimal network behavior
- Find critical points in input space

### 🔍 Sensitivity Analysis
- Identify which input dimensions most affect the NTK
- Understand input-output relationships in infinite-width limit
- Guide feature selection and input preprocessing

### ⚔️ Adversarial Analysis
- Generate adversarial inputs that change NTK properties
- Study robustness in the infinite-width regime
- Understand worst-case input perturbations

### 📈 Theoretical Analysis
- Analyze optimization landscapes using NTK theory
- Study convergence properties through gradient analysis
- Understand infinite-width vs finite-width differences

## Technical Notes

### Gradient Flow Preservation
Both infinite-width and empirical methods preserve PyTorch's automatic differentiation:
- ✅ Input gradients work perfectly
- ✅ Higher-order gradients supported
- ✅ Compatible with PyTorch optimizers

### Method Comparison
| Method | Speed | Memory | Accuracy | Use Case |
|--------|-------|--------|----------|----------|
| Infinite-width | Fast | Low | Theoretical limit | Research, theory |
| Empirical | Slower | Higher | Finite approximation | Practical applications |
| Manual Jacobian | Variable | High | Educational | Understanding math |

### Performance Tips
- Use infinite-width for theoretical analysis (faster)
- Use empirical for practical applications (more realistic)
- Start with smaller networks and scale up
- Enable `create_graph=True` for higher-order gradients

## Getting Started

1. **Start with `simple_ntk_gradients.py`** to learn the basics
2. **Try input optimization** examples to see practical applications
3. **Explore `advanced_ntk_gradients.py`** for research use cases
4. **Read `ntk_gradients.py`** for comprehensive understanding

## Research Applications

These gradient computation capabilities enable:
- **Neural architecture search** using NTK properties
- **Input design** for optimal network training
- **Theoretical analysis** of infinite-width networks
- **Optimization landscape** studies
- **Robustness analysis** in the infinite-width limit

## Citation

If you use these gradient computation methods in your research, please cite:
- TorchTangents: [repository link]
- Neural Tangents: [original paper]
- Infinite-width theory: [relevant papers]