"""4. CNN Kernels & Forced Lifting for Unsupported Layers (e.g. BatchNorm)."""
import torch
import torch.nn as nn
import torchtangents as tt

x1 = torch.randn(2, 3, 16, 16)
x2 = torch.randn(3, 3, 16, 16)

# 1. Native TorchTangents CNN (supports both infinite-width and empirical NTK)
cnn = tt.nn.Sequential(
    tt.nn.Conv2d(3, 16, kernel_size=3, padding="SAME"),
    tt.nn.ReLU(),
    tt.nn.AdaptiveAvgPool2d(),
    tt.nn.Linear(16, 4),
)
print(f"CNN Infinite NTK:  {cnn.inf_kernel(x1, x2, get='ntk').shape}")
print(f"CNN Empirical NTK: {cnn.empirical_ntk(x1, x2).shape}")

# 2. Vanilla CNN with BatchNorm (no analytical infinite-width limit; use force=True for empirical NTK)
bn_cnn = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(8 * 16 * 16, 4),
)
lifted_bn = tt.convert.lift(bn_cnn, force=True)
print(f"Forced-Lift Empirical NTK: {lifted_bn.empirical_ntk(x1, x2).shape}")
