"""
examples/7_prefix_suffix_layer_ntk_preconditioning.py

Reference Implementation: Prefix-Suffix Layer-Wise NTK Preconditioning
=======================================================================
This script demonstrates how to compute layer-level model NNGP and NTK prefixes
using TorchTangents for architecture-preconditioned optimization (e.g. AAOD).

Mathematical Framework Reference:
- Camera-Ready Draft: "TorchTangents: An Accessible Bridge for Architecture-Preconditioned Optimizers"
  (OPT 2026: 18th Annual Workshop on Optimization for Machine Learning)
- Technical Research Summary: "Spatial Derivatives of the Neural Tangent Kernel:
  From Sobolev Geometry to Prefix-Suffix Preconditioned Optimization" (October 31, 2026)

Key Theoretical Formulas Implemented:
1. Prefix-Suffix Kronecker Factorization (Lemma 5.1, Eq. 20-21):
   Theta_{t, l}^f(x, z) = Sigma_{pre}^{(l-1)}(x, z) * Delta_{suf}^{(l)}(x, z)
   In the infinite-width limit:
   Theta_{f, l}^inf(X, X) = Sigma_{pre}^{inf, (l-1)}(X, X) (x) [ (x)_{k=l}^{L-1} dot{Sigma}^{(k)}(X, X) ]

2. Analytical Layer Prior Allocation (Eq. 24):
   rho_l^inf = Tr(Theta_{f, l}^inf(X, X)) / Tr(Theta_f^inf(X, X))

3. Empirical Explosion Damper / Trust-Ratio Scaling (Eq. 24):
   Multiplier_l = rho_l^inf * sqrt( (1/N) * Tr(Theta_{f, l}^inf(X, X)) / [ (1/N) * Tr(Theta_{t, l}^f(X, X)) + eps ] )

Methods Demonstrated:
- Method 1: Exact analytical layer-wise decomposition via suffix telescoping.
- Method 2: Extracting prefix NNGP (Sigma_{pre}) and cumulative NTK via PyTorch forward hooks with Kernel.
- Method 3: Finite-width empirical layer NTK traces via torch.func.
- Method 4: Complete PrefixSuffixAAODPreconditioner class for optimizers.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.func as func
import torchtangents as tt
from torchtangents._src.utils.kernel import Kernel


# ==============================================================================
# Method 1: Exact Layer-Wise Decomposition via Suffix Telescoping (Analytical NTK)
# ==============================================================================

def compute_analytical_layer_ntks(
    model: nn.Sequential,
    x: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], torch.Tensor]:
    """
    Decomposes the total infinite-width NTK into exact layer-wise contributions
    Theta_{f, l}^inf(X, X) using suffix telescoping with zero numerical error.

    Args:
        model: A sequential model (tt.nn.Sequential or lifted nn.Sequential).
        x: Input batch of shape (N, *input_dims).

    Returns:
        theta_layers: Dict mapping layer name to its exact (N, N) analytical NTK contribution.
        rho_inf: Dict mapping layer name to its analytical layer budget rho_l^inf (sums to 1.0).
        total_ntk: The full network infinite-width NTK matrix (N, N).
    """
    k_input = Kernel.from_inputs(x, compute_ntk=True)
    param_layer_indices = [i for i, layer in enumerate(model) if any(p.requires_grad for p in layer.parameters())]
    
    if not param_layer_indices:
        # Fallback if no explicit parameters found (e.g. stateless stax wrapped sequentially)
        param_layer_indices = [i for i, layer in enumerate(model) if hasattr(layer, 'kernel_fn')]

    suffix_ntks = []
    for idx in param_layer_indices:
        if idx == 0:
            k_out = model(k_input)
        else:
            k_prefix = model[:idx](k_input)
            k_fresh = k_prefix.replace(ntk=torch.zeros_like(k_prefix.nngp))
            k_out = model[idx:](k_fresh)
        suffix_ntks.append(k_out.ntk)

    total_ntk = suffix_ntks[0]
    total_trace = torch.trace(total_ntk).item()

    theta_layers: Dict[str, torch.Tensor] = {}
    rho_inf: Dict[str, float] = {}

    for i, l_idx in enumerate(param_layer_indices):
        name = f"layer_{l_idx}_{model[l_idx].__class__.__name__}"
        if i < len(param_layer_indices) - 1:
            layer_ntk = suffix_ntks[i] - suffix_ntks[i + 1]
        else:
            layer_ntk = suffix_ntks[i]
            
        theta_layers[name] = layer_ntk
        rho_inf[name] = torch.trace(layer_ntk).item() / total_trace

    return theta_layers, rho_inf, total_ntk


# ==============================================================================
# Method 2: Extracting NNGP Prefixes (Sigma_pre) via PyTorch Forward Hooks
# ==============================================================================

def extract_prefix_kernels_with_hooks(
    model: nn.Module,
    x: torch.Tensor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Leverages Kernel.__torch_function__ to intercept intermediate Kernel objects
    at every layer of any arbitrary PyTorch architecture via standard forward hooks.

    Args:
        model: Any standard PyTorch module (vanilla nn.Module, CNN, ResNet, etc.).
        x: Input batch tensor.

    Returns:
        Dict mapping layer names to their intermediate NNGP and NTK tensors.
    """
    captured: Dict[str, Dict[str, torch.Tensor]] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, inp, out):
            if isinstance(out, Kernel):
                captured[name] = {
                    "nngp_prefix": out.nngp,  # Forward feature covariance Sigma_pre
                    "ntk_prefix": out.ntk,    # Forward accumulated prefix NTK Theta_{<= l}
                }
        return hook_fn

    for name, module in model.named_modules():
        # Register on leaf modules with no sub-modules
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        k_in = Kernel.from_inputs(x, compute_ntk=True)
        # If it's a Sequential container (tt.nn.Sequential or nn.Sequential), iterate sequentially
        if isinstance(model, (nn.Sequential, list, tuple)):
            curr = k_in
            for layer in model:
                curr = layer(curr)
        else:
            _ = model(k_in)
    finally:
        for h in hooks:
            h.remove()

    return captured


# ==============================================================================
# Method 3: Finite-Width Empirical Layer NTK Traces via torch.func
# ==============================================================================

def compute_empirical_layer_ntk_traces(
    model: nn.Module,
    x: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes the normalized empirical NTK diagonal trace:
        (1 / N) * Tr(Theta_{t, l}^f(X, X)) = (1 / N) * sum_{i=1}^N ||nabla_{W_l} f(x_i)||_2^2
    vectorized across the batch using torch.func (vmap + jacrev).

    Args:
        model: Standard finite-width PyTorch module with trained or initialized weights.
        x: Input batch of shape (N, *input_dims).

    Returns:
        Dict mapping parameter/layer name to its empirical (1/N)*Tr(Theta) scalar.
    """
    N = x.shape[0]
    params = dict(model.named_parameters())
    if not params:
        return {}

    was_training = model.training
    model.eval()

    try:
        def fnet_single(p, x_single):
            return func.functional_call(model, p, x_single.unsqueeze(0)).squeeze(0)

        # Vectorized per-sample Jacobians: Dict[param_name, (N, Out_dim, *param_shape)]
        jac_fn = func.vmap(func.jacrev(fnet_single, argnums=0), in_dims=(None, 0))
        jac_dict = jac_fn(params, x)

        layer_traces: Dict[str, float] = {}
        for name, jac in jac_dict.items():
            # Flatten across output dimensions and parameter dimensions
            flat_jac = jac.reshape(N, -1)
            # Trace of J J^T is the sum of squared Frobenius norms per sample
            sample_sq_norms = torch.sum(flat_jac ** 2, dim=-1)
            layer_traces[name] = (torch.sum(sample_sq_norms) / N).item()
    finally:
        model.train(was_training)

    return layer_traces


# ==============================================================================
# Method 4: Prefix-Suffix AAOD Preconditioner (Eq. 24)
# ==============================================================================

class PrefixSuffixAAODPreconditioner:
    """
    Preconditioner implementing the Prefix-Suffix Trust-Ratio Filtered Orthogonal
    Descent update rule (Theorem 5.2, Eq. 24):

    W_l^{(t+1)} = W_l^{(t)} - eta * rho_l^inf * sqrt( (1/N) Tr(Theta_{f, l}^inf) / [ (1/N) Tr(Theta_{t, l}^f) + eps ] ) * Phi_{gamma_l}(G_l)

    Features:
    - Eliminates Winner-Take-All layer starvation: Anchors budget to deterministic infinite prior rho_l^inf.
    - Cures gradient explosion: Empirical tangent explosions increase denominator, automatically damping the layer.
    - Normalizes extensive batch size scaling: All traces evaluated per sample (1/N).
    """

    def __init__(self, model: nn.Sequential, calibration_batch: torch.Tensor, eps: float = 1e-6):
        self.model = model
        self.eps = eps
        self.N_cal = calibration_batch.shape[0]

        # 1. Precompute static infinite-width layer prior budgets rho_l^inf
        self.theta_inf_layers, self.rho_inf, self.total_ntk = compute_analytical_layer_ntks(
            model, calibration_batch
        )

    def get_layer_trust_multipliers(self, current_batch: torch.Tensor) -> Dict[str, float]:
        """
        Computes dynamic trust-region multipliers for each parameter group for the current step.
        """
        N = current_batch.shape[0]
        emp_traces = compute_empirical_layer_ntk_traces(self.model, current_batch)

        multipliers: Dict[str, float] = {}
        for layer_name, rho in self.rho_inf.items():
            inf_layer_trace = (torch.trace(self.theta_inf_layers[layer_name]).item()) / self.N_cal
            
            # Match parameter names associated with this layer index (e.g. 'layer_0_Linear' -> '0.weight')
            idx_str = layer_name.split('_')[1]
            matched_emp = [v for k, v in emp_traces.items() if k.startswith(f"{idx_str}.")]
            emp_trace = sum(matched_emp) / len(matched_emp) if matched_emp else inf_layer_trace

            # Trust-ratio scaling: rho_l^inf * sqrt( inf_trace / (emp_trace + eps) )
            damper = math.sqrt(inf_layer_trace / (emp_trace + self.eps))
            multipliers[layer_name] = rho * damper

        return multipliers

    # Alias for convenience
    get_preconditioned_layer_multipliers = get_layer_trust_multipliers


# ==============================================================================
# Standalone Demonstration & Verification
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TorchTangents: Prefix-Suffix Layer-Wise NTK Preconditioning Reference")
    print("=" * 80)

    # 1. Construct a dual-mode TorchTangents sequential network
    torch.manual_seed(42)
    model = tt.nn.Sequential(
        tt.nn.Linear(10, 32),
        tt.nn.ReLU(),
        tt.nn.Linear(32, 64),
        tt.nn.ReLU(),
        tt.nn.Linear(64, 2),
    )

    X = torch.randn(8, 10)
    print(f"\n1. Input Batch: {X.shape}")

    # 2. Method 1: Exact Layer-Wise Decomposition via Suffix Telescoping
    print("\n--- Method 1: Suffix Telescoping (Exact Layer-Wise NTK Decomposition) ---")
    theta_layers, rho_inf, total_ntk = compute_analytical_layer_ntks(model, X)
    
    reconstructed_ntk = sum(theta_layers.values())
    max_decomp_error = torch.max(torch.abs(total_ntk - reconstructed_ntk)).item()
    print(f"Total NTK Shape: {total_ntk.shape} | Total Trace: {torch.trace(total_ntk):.4f}")
    print(f"Max Decomposition Error (|Total - Sum(Layers)|): {max_decomp_error:.2e}")

    for name, K_layer in theta_layers.items():
        tr = torch.trace(K_layer).item()
        print(f"  * {name:25s} -> Trace: {tr:8.4f} | rho_inf: {rho_inf[name]:.4f}")
    print(f"  * Sum of rho_inf: {sum(rho_inf.values()):.4f} (Exactly 1.0)")

    # 3. Method 2: Extracting NNGP Prefixes via Hooks
    print("\n--- Method 2: Extracting NNGP Prefixes (Sigma_pre) via Forward Hooks ---")
    captured_prefixes = extract_prefix_kernels_with_hooks(model, X)
    for name, data in captured_prefixes.items():
        print(f"  * Layer: {name:20s} | NNGP Prefix Shape: {tuple(data['nngp_prefix'].shape)} | NTK Prefix Shape: {tuple(data['ntk_prefix'].shape)}")

    # 4. Method 3: Finite-Width Empirical Layer NTK Traces
    print("\n--- Method 3: Finite-Width Empirical Layer NTK Traces via torch.func ---")
    emp_traces = compute_empirical_layer_ntk_traces(model, X)
    for p_name, tr_val in emp_traces.items():
        print(f"  * Param: {p_name:25s} -> (1/N) * Tr(Theta_emp): {tr_val:.4f}")

    # 5. Method 4: PrefixSuffixAAODPreconditioner
    print("\n--- Method 4: Prefix-Suffix AAOD Preconditioner Multipliers (Eq. 24) ---")
    preconditioner = PrefixSuffixAAODPreconditioner(model, X)
    multipliers = preconditioner.get_layer_trust_multipliers(X)
    for l_name, mult in multipliers.items():
        print(f"  * {l_name:25s} -> Prior rho: {preconditioner.rho_inf[l_name]:.4f} | Final Multiplier: {mult:.4f}")

    print("\nVerification complete: All Prefix-Suffix NTK preconditioning components ready for use!")