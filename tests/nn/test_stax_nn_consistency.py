import pytest
import torch
import torchtangents as tt


@pytest.mark.parametrize(
    "stax_layer_fn, nn_module_fn, input_shape, requires_gaussian",
    [
        (lambda: tt.stax.Linear(16, W_std=1.2, b_std=0.5), lambda: tt.nn.Linear(8, 16, W_std=1.2, b_std=0.5), (4, 8), False),
        (lambda: tt.stax.Linear(16, W_std=1.0, b_std=None), lambda: tt.nn.Linear(8, 16, W_std=1.0, bias=False), (4, 8), False),
        (lambda: tt.stax.Conv(8, (3,), strides=(1,), padding="SAME", b_std=0.1), lambda: tt.nn.Conv1d(3, 8, 3, stride=1, padding="SAME", b_std=0.1), (4, 3, 10), False),
        (lambda: tt.stax.Conv(8, (3, 3), strides=(1, 1), padding="SAME", b_std=0.1), lambda: tt.nn.Conv2d(3, 8, 3, stride=1, padding="SAME", b_std=0.1), (4, 3, 8, 8), False),
        (lambda: tt.stax.Conv(6, (3, 3), strides=(2, 2), padding="VALID", b_std=None), lambda: tt.nn.Conv2d(3, 6, 3, stride=2, padding="VALID", bias=False), (4, 3, 8, 8), False),
        (lambda: tt.stax.ConvTranspose(6, (3, 3), strides=(1, 1), padding="SAME", b_std=0.1), lambda: tt.nn.ConvTranspose2d(3, 6, 3, stride=1, padding="SAME", b_std=0.1), (4, 3, 6, 6), False),
        (lambda: tt.stax.ReLU(), lambda: tt.nn.ReLU(), (4, 8), True),
        (lambda: tt.stax.LeakyReLU(0.1), lambda: tt.nn.LeakyReLU(0.1), (4, 8), True),
        (lambda: tt.stax.ABRelu(-0.2, 1.5), lambda: tt.nn.ABReLU(-0.2, 1.5), (4, 8), True),
        (lambda: tt.stax.GeLU(approximate=False), lambda: tt.nn.GELU(approximate="none"), (4, 8), True),
        (lambda: tt.stax.GeLU(approximate=True), lambda: tt.nn.GELU(approximate="tanh"), (4, 8), True),
        (lambda: tt.stax.Erf(1.2, 0.8, 0.1), lambda: tt.nn.Erf(1.2, 0.8, 0.1), (4, 8), True),
        (lambda: tt.stax.Sin(1.1, 0.9, 0.2), lambda: tt.nn.Sin(1.1, 0.9, 0.2), (4, 8), True),
        (lambda: tt.stax.Cos(1.1, 0.9, 0.2), lambda: tt.nn.Cos(1.1, 0.9, 0.2), (4, 8), True),
        (lambda: tt.stax.Exp(0.5, 0.2), lambda: tt.nn.Exp(0.5, 0.2), (4, 8), True),
        (lambda: tt.stax.Rbf(1.5), lambda: tt.nn.RBF(1.5), (4, 8), True),
        (lambda: tt.stax.Abs(), lambda: tt.nn.Abs(), (4, 8), True),
        (lambda: tt.stax.Sigmoid_like(), lambda: tt.nn.SigmoidLike(), (4, 8), True),
        (lambda: tt.stax.Sign(), lambda: tt.nn.Sign(), (4, 8), True),
        (lambda: tt.stax.Gaussian(1.0, -0.5), lambda: tt.nn.Gaussian(1.0, -0.5), (4, 8), True),
        (lambda: tt.stax.Gabor(), lambda: tt.nn.Gabor(), (4, 8), True),
        (lambda: tt.stax.Flatten(), lambda: tt.nn.Flatten(), (4, 3, 6, 6), False),
        (lambda: tt.stax.Identity(), lambda: tt.nn.Identity(), (4, 8), False),
        (lambda: tt.stax.GlobalAvgPool(), lambda: tt.nn.AdaptiveAvgPool2d(), (4, 3, 6, 6), False),
        (lambda: tt.stax.GlobalSumPool(), lambda: tt.nn.GlobalSumPool2d(), (4, 3, 6, 6), False),
        (lambda: tt.stax.AvgPool((2, 2), strides=(2, 2), padding="VALID"), lambda: tt.nn.AvgPool2d(2, stride=2, padding="VALID"), (4, 3, 6, 6), False),
        (lambda: tt.stax.SumPool((2, 2), strides=(2, 2), padding="VALID"), lambda: tt.nn.SumPool2d(2, stride=2, padding="VALID"), (4, 3, 6, 6), False),
    ],
)
def test_single_layer_stax_module_vs_nn_module_consistency(stax_layer_fn, nn_module_fn, input_shape, requires_gaussian):
    """Verifies that StaxModule(stax_layer) and stateful tt.nn modules produce identical NNGP & NTK kernels."""
    torch.manual_seed(42)
    x1 = torch.randn(*input_shape)
    x2 = torch.randn(input_shape[0] - 1, *input_shape[1:])

    if requires_gaussian:
        stax_mod = tt.nn.StaxModule(tt.stax.Sequential(tt.stax.Linear(8), stax_layer_fn()), input_shape=input_shape)
        nn_mod = tt.nn.Sequential(tt.nn.Linear(input_shape[1], 8, b_std=None), nn_module_fn())
    else:
        stax_mod = tt.nn.StaxModule(stax_layer_fn(), input_shape=input_shape)
        nn_mod = nn_module_fn()

    k_stax = stax_mod.kernels(x1, x2, get=("nngp", "ntk"))
    k_nn = nn_mod.kernels(x1, x2, get=("nngp", "ntk"))

    assert torch.allclose(k_stax.nngp, k_nn.nngp, atol=1e-6), "NNGP mismatch between StaxModule and tt.nn module"
    assert torch.allclose(k_stax.ntk, k_nn.ntk, atol=1e-6), "NTK mismatch between StaxModule and tt.nn module"

    # Also verify forward pass shapes match
    y_stax = stax_mod(x1)
    y_nn = nn_mod(x1)
    assert y_stax.shape == y_nn.shape


def test_sequential_mlp_stax_module_vs_nn_sequential():
    """Verifies deep MLP consistency between StaxModule(stax.Sequential) and tt.nn.Sequential."""
    torch.manual_seed(0)
    x1 = torch.randn(5, 12)
    x2 = torch.randn(3, 12)

    stax_seq = tt.stax.Sequential(
        tt.stax.Linear(24, W_std=1.1, b_std=0.2),
        tt.stax.LayerNorm(),
        tt.stax.ReLU(),
        tt.stax.Dropout(0.8),
        tt.stax.Linear(6, W_std=0.9, b_std=0.1),
    )
    stax_mod = tt.nn.StaxModule(stax_seq, input_shape=(5, 12))

    nn_seq = tt.nn.Sequential(
        tt.nn.Linear(12, 24, W_std=1.1, b_std=0.2),
        tt.nn.LayerNorm(24),
        tt.nn.ReLU(),
        tt.nn.Dropout(0.8),
        tt.nn.Linear(24, 6, W_std=0.9, b_std=0.1),
    )

    k_stax = stax_mod.kernels(x1, x2, get=("nngp", "ntk"))
    k_nn = nn_seq.kernels(x1, x2, get=("nngp", "ntk"))

    assert torch.allclose(k_stax.nngp, k_nn.nngp, atol=1e-6)
    assert torch.allclose(k_stax.ntk, k_nn.ntk, atol=1e-6)


def test_cnn_and_residual_branching_consistency():
    """Verifies CNN + residual FanOut/Parallel/FanInSum consistency between StaxModule and tt.nn."""
    torch.manual_seed(7)
    x1 = torch.randn(4, 3, 8, 8)
    x2 = torch.randn(2, 3, 8, 8)

    stax_net = tt.stax.Sequential(
        tt.stax.Conv(8, (3, 3), padding="SAME", b_std=0.1),
        tt.stax.FanOut(2),
        tt.stax.parallel(
            tt.stax.Sequential(tt.stax.ReLU(), tt.stax.Conv(8, (3, 3), padding="SAME", b_std=0.1)),
            tt.stax.Identity(),
        ),
        tt.stax.FanInSum(),
        tt.stax.GlobalAvgPool(),
        tt.stax.Linear(4, b_std=0.1),
    )
    stax_mod = tt.nn.StaxModule(stax_net, input_shape=(4, 3, 8, 8))

    nn_net = tt.nn.Sequential(
        tt.nn.Conv2d(3, 8, 3, padding="SAME", b_std=0.1),
        tt.nn.FanOut(2),
        tt.nn.Parallel(
            tt.nn.Sequential(tt.nn.ReLU(), tt.nn.Conv2d(8, 8, 3, padding="SAME", b_std=0.1)),
            tt.nn.Identity(),
        ),
        tt.nn.FanInSum(),
        tt.nn.AdaptiveAvgPool2d(),
        tt.nn.Linear(8, 4, b_std=0.1),
    )

    k_stax = stax_mod.kernels(x1, x2, get=("nngp", "ntk"))
    k_nn = nn_net.kernels(x1, x2, get=("nngp", "ntk"))

    assert torch.allclose(k_stax.nngp, k_nn.nngp, atol=1e-6)
    assert torch.allclose(k_stax.ntk, k_nn.ntk, atol=1e-6)
    assert stax_mod(x1).shape == nn_net(x1).shape == (4, 4)


def test_repeat_and_input_derivatives_consistency():
    """Verifies Repeat combinator as well as ntk_gradient and ntk_hessian consistency."""
    torch.manual_seed(11)
    x1 = torch.randn(3, 6)
    x2 = torch.randn(2, 6)

    stax_rep = tt.stax.Sequential(
        tt.stax.Linear(6, b_std=0.1),
        tt.stax.repeat(tt.stax.Sequential(tt.stax.ReLU(), tt.stax.Linear(6, b_std=0.1)), 2),
    )
    stax_mod = tt.nn.StaxModule(stax_rep, input_shape=(3, 6))

    nn_rep = tt.nn.Sequential(
        tt.nn.Linear(6, 6, b_std=0.1),
        tt.nn.Repeat(tt.nn.Sequential(tt.nn.ReLU(), tt.nn.Linear(6, 6, b_std=0.1)), 2),
    )

    # Check kernels
    k1 = stax_mod.kernels(x1, x2)
    k2 = nn_rep.kernels(x1, x2)
    assert torch.allclose(k1.nngp, k2.nngp, atol=1e-6)
    assert torch.allclose(k1.ntk, k2.ntk, atol=1e-6)

    # Check ntk_gradient and ntk_hessian
    for mode in ("gram", "all_pairs"):
        g1 = stax_mod.ntk_gradient(x1, x2, mode=mode)
        g2 = nn_rep.ntk_gradient(x1, x2, mode=mode)
        assert torch.allclose(g1, g2, atol=1e-6)

        h1 = stax_mod.ntk_hessian(x1, x2, mode=mode)
        h2 = nn_rep.ntk_hessian(x1, x2, mode=mode)
        assert torch.allclose(h1, h2, atol=1e-6)
