"""
Infinite-width FC inference on CIFAR-10 (subset) using analytic NNGP/NTK
for a 1-hidden-layer ReLU MLP with Dense(·, W_std=2.0, b_std=0.05) on both
layers, matching the JAX example structure:
    Dense -> ReLU -> Dense

We compute both kernels:
- NNGP for the 2-layer net
- NTK for the 2-layer net

Then do kernel ridge regression (MSE) with diag_reg=1e-3.
"""

from __future__ import annotations
import math
import time
import numpy as np
import torch
from torchvision import datasets, transforms

# -------------------------
# Config (match the example)
# -------------------------
TRAIN_SIZE = 1000
TEST_SIZE = 1000
BATCH_SIZE = 0              # not used here; we build kernels directly
W_STD = 2.0                 # stax.Dense(..., W_std=2.0)
B_STD = 0.05                # stax.Dense(..., b_std=0.05)
DIAG_REG = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# -------------------------
# Data
# -------------------------
def load_cifar10_subset(train_n=1000, test_n=1000):
    tfm = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
        # keep it simple; can normalize if you want:
        # transforms.Normalize((0.4914,0.4822,0.4465),
        #                      (0.2023,0.1994,0.2010)),
    ])
    tr = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    te = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)

    # shuffle deterministically then take subset
    g = torch.Generator().manual_seed(SEED)
    tr_idx = torch.randperm(len(tr), generator=g)[:train_n]
    te_idx = torch.randperm(len(te), generator=g)[:test_n]

    def tensorize(ds, idxs):
        xs, ys = [], []
        for i in idxs:
            x, y = ds[i]
            xs.append(x.view(-1))   # flatten
            ys.append(y)
        X = torch.stack(xs, 0)     # [N, 3072]
        y = torch.tensor(ys)       # [N]
        return X, y

    Xtr, ytr = tensorize(tr, tr_idx)
    Xte, yte = tensorize(te, te_idx)

    # one-hot labels for MSE ridge
    C = 10
    Ytr = torch.zeros(len(ytr), C)
    Ytr[torch.arange(len(ytr)), ytr] = 1.0
    Yte = torch.zeros(len(yte), C)
    Yte[torch.arange(len(yte)), yte] = 1.0

    return Xtr, Ytr, ytr, Xte, Yte, yte

# -------------------------
# Kernel utilities (ReLU)
# -------------------------
def _safe_clip(x: np.ndarray, eps: float = 1e-7):
    return np.clip(x, -1.0 + eps, 1.0 - eps)

def relu_nngp_from_corr(c: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    ReLU NNGP transform (arc-cosine kernel, order-1).
    Inputs:
      c  : correlation matrix (N1 x N2), entries in [-1,1]
      v1 : var vector for set 1 (N1,)
      v2 : var vector for set 2 (N2,)
    Output:
      NNGP after ReLU (N1 x N2)
    Formula:
      J(c) = (1/(2π)) * ( sqrt(1-c^2) + (π - arccos c) * c )
      Σ_out = σ_w^2 * sqrt(v1 v2) * J(c) + σ_b^2
    """
    c = _safe_clip(c)
    theta = np.arccos(c)
    J = (np.sqrt(1.0 - c*c) + (np.pi - theta) * c) / (2.0 * np.pi)
    scale = np.sqrt(v1)[:, None] * np.sqrt(v2)[None, :]
    return (W_STD**2) * scale * J + (B_STD**2)

def relu_ntk_factor_from_corr(c: np.ndarray) -> np.ndarray:
    """
    For ReLU derivative kernel factor:
      dotJ(c) = (1/(2π)) * (π - arccos c)
    """
    c = _safe_clip(c)
    theta = np.arccos(c)
    dotJ = (np.pi - theta) / (2.0 * np.pi)
    return dotJ

def cov_and_var(X1: np.ndarray, X2: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Σ^0 and variances v1,v2 for the input layer (pre-activation),
    including the affine (Dense) with (W_STD, B_STD) scaling:
      Σ^1_affine = σ_w^2 * (X X^T) + σ_b^2
    """
    if X2 is None:
        X2 = X1
    S = X1 @ X2.T
    S = (W_STD**2) * S + (B_STD**2)  # affine covariance
    v1 = np.diag((W_STD**2) * (X1 @ X1.T) + (B_STD**2))
    v2 = np.diag((W_STD**2) * (X2 @ X2.T) + (B_STD**2))
    return S, v1, v2

# -------------------------
# Build 2-layer NNGP and NTK
#   Layer1: Dense(W_STD,B_STD) -> ReLU
#   Layer2: Dense(W_STD,B_STD) (linear readout)
# -------------------------
def two_layer_nngp(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      Kxx : train-train NNGP (Ntr x Ntr)
      Ktx : test-train NNGP (Nte x Ntr)
    """
    # After first affine:
    S_tt, v_t, _ = cov_and_var(Xtr, None)
    S_ut, v_u, v_t2 = cov_and_var(Xte, Xtr)  # v_t2==v_t

    # ReLU nonlinearity:
    c_tt = S_tt / np.sqrt(v_t[:, None] * v_t[None, :])
    c_ut = S_ut / np.sqrt(v_u[:, None] * v_t[None, :])
    Sigma1_tt = relu_nngp_from_corr(c_tt, v_t, v_t)
    Sigma1_ut = relu_nngp_from_corr(c_ut, v_u, v_t)

    # Final linear readout (another Dense layer)
    # For a *linear* top layer, the NNGP simply applies the affine again:
    Kxx = (W_STD**2) * Sigma1_tt + (B_STD**2)
    Ktx = (W_STD**2) * Sigma1_ut + (B_STD**2)
    return Kxx, Ktx

def two_layer_ntk(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      Θxx : train-train NTK (Ntr x Ntr)
      Θtx : test-train NTK (Nte x Ntr)

    Recurrence for 2-layer ReLU (Dense -> ReLU -> Dense):
      - Compute NNGP after first affine+ReLU: Σ1
      - dotΣ1 factor = W_STD^2 * dotJ(c)
      - NTK = Σ2 (NNGP at output) + (W_STD^2) * (X·X^T + B_STD^2/ W_STD^2 * 1?) propagated with dot terms

    A compact correct relation (for 2-layer, linear readout):
      Θ = Σ_out  +  (W_STD^2) * Σ1_dot
    where:
      Σ_out = (W_STD^2)*Σ1 + (B_STD^2)              (final linear readout NNGP)
      Σ1_dot = elementwise(dotJ(c)) * (W_STD**2 * X X^T + B_STD**2)
    We implement this directly for train/train and test/train blocks.
    """
    # Input-layer affine covariances:
    S_tt, v_t, _ = cov_and_var(Xtr, None)
    S_ut, v_u, v_t2 = cov_and_var(Xte, Xtr)

    # ReLU NNGP part (same as above):
    c_tt = S_tt / np.sqrt(v_t[:, None] * v_t[None, :])
    c_ut = S_ut / np.sqrt(v_u[:, None] * v_t[None, :])
    Sigma1_tt = relu_nngp_from_corr(c_tt, v_t, v_t)
    Sigma1_ut = relu_nngp_from_corr(c_ut, v_u, v_t)

    # Final layer NNGP (linear readout):
    Sigma_out_tt = (W_STD**2) * Sigma1_tt + (B_STD**2)
    Sigma_out_ut = (W_STD**2) * Sigma1_ut + (B_STD**2)

    # Derivative factor dotJ(c):
    dot_tt = relu_ntk_factor_from_corr(c_tt)
    dot_ut = relu_ntk_factor_from_corr(c_ut)

    # “Pre-activation” covs entering the nonlinearity (i.e., first affine output):
    # That’s S_tt and S_ut already computed above.

    # NTK = Σ_out + W_STD^2 * (dotJ ⊙ S_affine)
    Theta_tt = Sigma_out_tt + (W_STD**2) * (dot_tt * S_tt)
    Theta_ut = Sigma_out_ut + (W_STD**2) * (dot_ut * S_ut)
    return Theta_tt, Theta_ut

# -------------------------
# Kernel ridge regression
# -------------------------
def krr_predict(Kxx: np.ndarray, Ktx: np.ndarray, Ytr: np.ndarray, reg: float) -> np.ndarray:
    """
    Solve (Kxx + reg I) A = Ytr, then f_test = Ktx A
    """
    n = Kxx.shape[0]
    A = np.linalg.solve(Kxx + reg * np.eye(n), Ytr)
    return Ktx @ A

# -------------------------
# Metrics
# -------------------------
def mse(fx: np.ndarray, Y: np.ndarray) -> float:
    return 0.5 * float(np.mean((fx - Y) ** 2))

def top1_acc(fx: np.ndarray, y: np.ndarray) -> float:
    pred = fx.argmax(axis=1)
    return float((pred == y).mean())

# -------------------------
# Main
# -------------------------
def main():
    torch.manual_seed(SEED)
    print("Loading data.")
    Xtr_t, Ytr_t, ytr_t, Xte_t, Yte_t, yte_t = load_cifar10_subset(TRAIN_SIZE, TEST_SIZE)

    # Move to CPU numpy for kernel math
    Xtr = Xtr_t.numpy().astype(np.float64)
    Xte = Xte_t.numpy().astype(np.float64)
    Ytr = Ytr_t.numpy().astype(np.float64)
    Yte = Yte_t.numpy().astype(np.float64)
    ytr = ytr_t.numpy()
    yte = yte_t.numpy()

    start = time.time()

    # ----- NNGP -----
    Kxx_nngp, Ktx_nngp = two_layer_nngp(Xtr, Xte)
    fx_te_nngp = krr_predict(Kxx_nngp, Ktx_nngp, Ytr, DIAG_REG)

    # ----- NTK -----
    Kxx_ntk, Ktx_ntk = two_layer_ntk(Xtr, Xte)
    fx_te_ntk = krr_predict(Kxx_ntk, Ktx_ntk, Ytr, DIAG_REG)

    # (Optionally block; here it’s all NumPy)
    duration = time.time() - start
    print(f"Kernel construction and inference done in {duration:.2f} seconds.")

    # Summaries (match the JAX example style)
    print("\n[NNGP test]")
    print(f"loss={mse(fx_te_nngp, Yte):.4f}  acc={100*top1_acc(fx_te_nngp, yte):.2f}%")

    print("\n[NTK test]")
    print(f"loss={mse(fx_te_ntk, Yte):.4f}  acc={100*top1_acc(fx_te_ntk, yte):.2f}%")

if __name__ == "__main__":
    main()
