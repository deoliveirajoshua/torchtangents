import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtangents as tt


def test_tt_kernel_mlp_consistency():
    """Verifies tt.kernel on unmodified nn.Module matches tt.lift, tt.nn, and tt.stax."""
    torch.manual_seed(0)
    x1 = torch.randn(5, 10)
    x2 = torch.randn(3, 10)

    # 1. Standard unmodified PyTorch model
    pt_mlp = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )

    # 2. Equivalent tt.nn model
    tt_mlp = tt.nn.Sequential(
        tt.nn.Linear(10, 32, W_std=1.0, b_std=1.0),
        tt.nn.ReLU(),
        tt.nn.Linear(32, 16, W_std=1.0, b_std=1.0),
        tt.nn.GELU(),
        tt.nn.Linear(16, 4, W_std=1.0, b_std=1.0),
    )

    # 3. Equivalent tt.stax model
    stax_mlp = tt.stax.Sequential(
        tt.stax.Linear(32, W_std=1.0, b_std=1.0),
        tt.stax.ReLU(),
        tt.stax.Linear(16, W_std=1.0, b_std=1.0),
        tt.stax.GeLU(),
        tt.stax.Linear(4, W_std=1.0, b_std=1.0),
    )

    # 4. Lifted model
    lifted_mlp = tt.lift(pt_mlp)

    k_direct = tt.kernel(pt_mlp, x1, x2, get=("nngp", "ntk"))
    k_lifted = tt.kernel(lifted_mlp, x1, x2, get=("nngp", "ntk"))
    k_tt_nn = tt.kernel(tt_mlp, x1, x2, get=("nngp", "ntk"))
    k_stax = tt.kernel(stax_mlp, x1, x2, get=("nngp", "ntk"))

    for other in (k_lifted, k_tt_nn, k_stax):
        assert torch.allclose(k_direct.nngp, other.nngp, atol=1e-6)
        assert torch.allclose(k_direct.ntk, other.ntk, atol=1e-6)


def test_tt_kernel_cnn_and_residual_consistency():
    """Verifies tt.kernel on unmodified PyTorch CNN with residual skip connections matches tt.lift and tt.stax."""
    torch.manual_seed(1)
    x1 = torch.randn(4, 3, 8, 8)
    x2 = torch.randn(3, 3, 8, 8)

    class ResidualCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(8, 2)

        def forward(self, x):
            h = self.conv1(x)
            h = h + self.conv2(F.relu(h))
            h = self.pool(F.relu(h))
            h = torch.flatten(h, 1)
            return self.fc(h)

    pt_cnn = ResidualCNN()
    lifted_cnn = tt.lift(pt_cnn)

    stax_cnn = tt.stax.Sequential(
        tt.stax.Conv(8, (3, 3), padding="SAME", W_std=1.0, b_std=1.0),
        tt.stax.FanOut(2),
        tt.stax.parallel(
            tt.stax.Identity(),
            tt.stax.Sequential(tt.stax.ReLU(), tt.stax.Conv(8, (3, 3), padding="SAME", W_std=1.0, b_std=1.0)),
        ),
        tt.stax.FanInSum(),
        tt.stax.ReLU(),
        tt.stax.GlobalAvgPool(),
        tt.stax.Flatten(),
        tt.stax.Linear(2, W_std=1.0, b_std=1.0),
    )

    k_direct = tt.kernel(pt_cnn, x1, x2, get=("nngp", "ntk"))
    k_lifted = tt.kernel(lifted_cnn, x1, x2, get=("nngp", "ntk"))
    k_stax = tt.kernel(stax_cnn, x1, x2, get=("nngp", "ntk"))

    assert torch.allclose(k_direct.nngp, k_lifted.nngp, atol=1e-6)
    assert torch.allclose(k_direct.ntk, k_lifted.ntk, atol=1e-6)
    assert torch.allclose(k_direct.nngp, k_stax.nngp, atol=1e-6)
    assert torch.allclose(k_direct.ntk, k_stax.ntk, atol=1e-6)


def test_pretrained_weight_preservation_on_lift():
    """Verifies that lifting a standard PyTorch model preserves its exact pretrained forward outputs."""
    torch.manual_seed(2)
    x = torch.randn(4, 3, 8, 8)

    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 3),
    )
    lifted = tt.lift(model)

    assert torch.allclose(model(x), lifted(x), atol=1e-5)


def test_dynamic_control_flow_kernel_and_lift_consistency():
    """Verifies tt.kernel and tt.lift work on models with dynamic Python control flow that break torch.fx."""
    torch.manual_seed(3)
    x1 = torch.randn(4, 12)
    x2 = torch.randn(2, 12)

    class DynamicBlockNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(12, 12) for _ in range(3)])
            self.head = nn.Linear(12, 3)

        def forward(self, x):
            # Data-dependent check on batch size that breaks symbolic_trace
            if x.size(0) > 1:
                for i, layer in enumerate(self.layers):
                    x = x + F.relu(layer(x)) if i % 2 == 0 else F.leaky_relu(layer(x), 0.1)
            return self.head(x)

    dyn_model = DynamicBlockNet()
    lifted_dyn = tt.lift(dyn_model)

    k_direct = tt.kernel(dyn_model, x1, x2, get=("nngp", "ntk"))
    k_lifted = lifted_dyn.kernels(x1, x2, get=("nngp", "ntk"))

    assert torch.allclose(dyn_model(x1), lifted_dyn(x1), atol=1e-6)
    assert torch.allclose(k_direct.nngp, k_lifted.nngp, atol=1e-6)
    assert torch.allclose(k_direct.ntk, k_lifted.ntk, atol=1e-6)


def test_empirical_ntk_consistency():
    """Verifies tt.empirical_ntk works directly on unmodified nn.Module and matches symmetry properties."""
    torch.manual_seed(4)
    x1 = torch.randn(4, 8)
    x2 = torch.randn(3, 8)

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    emp_12 = tt.empirical_ntk(model, x1, x2)
    emp_21 = tt.empirical_ntk(model, x2, x1)

    assert emp_12.shape == (4, 3, 2, 2)
    assert emp_21.shape == (3, 4, 2, 2)
    # K(x1, x2)_{i, j, o1, o2} == K(x2, x1)_{j, i, o2, o1}
    assert torch.allclose(emp_12, emp_21.permute(1, 0, 3, 2), atol=1e-5)

