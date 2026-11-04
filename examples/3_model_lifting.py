"""3. Model Lifting: Convert any vanilla PyTorch nn.Module into TorchTangents."""
import torch
import torch.nn as nn
import torchtangents as tt


class ResMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 16)
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h)) + h  # Residual skip connection
        return self.head(h)


x1, x2 = torch.randn(4, 10), torch.randn(3, 10)

# Lift a standard PyTorch module via torch.fx tracing
lifted = tt.convert.lift(ResMLP())
print(f"Lifted forward shape: {lifted(x1).shape}")
print(f"Lifted NTK shape:     {lifted.inf_kernel(x1, x2, get='ntk').shape}")

# Or get a lightweight stateless kernel function directly
kernel_fn = tt.convert.infinite_analog(ResMLP())
print(f"Analog NTK shape:     {kernel_fn(x1, x2, get='ntk').shape}")
