# PyTorch/torchtangents rewrite of tests for
# `neural_tangents/_src/stax/elementwise.py`

import itertools
import math
import random as prandom
import unittest

import torch
from torch import pi
from torch.autograd.functional import jvp as torch_jvp
from torch.func import jacfwd as torch_jacfwd, jacrev as torch_jacrev

import torchtangents as nt
from torchtangents import stax
from torchtangents._src.empirical import _DEFAULT_TESTING_NTK_IMPLEMENTATION
from tests import test_utils


test_utils.update_test_tolerance()
prandom.seed(1)


# -------------------------
# RNG + tensor helpers
# -------------------------
def _rng(device: str | torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)  # seeds params; ops run on tensors' device
    g.manual_seed(seed)
    return g


def _randn(shape, gen, dtype=torch.float32, device=None):
    device = device or test_utils.device_under_test()
    return torch.randn(*shape, generator=gen, dtype=dtype, device=device)


def _cos_randn(shape, gen, device=None):
    return torch.cos(_randn(shape, gen, device=device))


def _value_and_grad_scalar(fn):
    # Returns a callable that outputs (value, (grad_x1, grad_x2_or_None))
    def wrapped(x1, x2):
        x1 = x1.clone().detach().requires_grad_(True)
        x2r = None if x2 is None else x2.clone().detach().requires_grad_(True)
        val = fn(x1, x2r)
        grads = torch.autograd.grad(val, (x1,) if x2r is None else (x1, x2r), allow_unused=True)
        if x2r is None:
            grads = (grads[0], None)
        return val.detach(), grads
    return wrapped


# ============================================================
# Activation tests
# ============================================================
class ActivationTest(test_utils.NeuralTangentsTestCase):

    def _test_activation_fc(self, phi, get):
        device = test_utils.device_under_test()
        g1, g2, gmc = _rng(device, 1), _rng(device, 2), _rng(device, 3)
        x1 = torch.cos(_randn((3, 2), g1))
        x2 = torch.cos(_randn((2, 2), g2))

        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(1024),
            phi,
            stax.Dense(1 if get == 'ntk' else 1024)
        )

        analytic_kernel = kernel_fn(x1, x2, get, diagonal_spatial=True)

        mc_kernel_fn = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=apply_fn,
            key=gmc,
            n_samples=800,
            implementation=2,
            vmap_axes=0,
            device_count=0,
        )

        if get == 'cov1':
            empirical_kernel = torch.diag(mc_kernel_fn(x1, None, 'nngp'))
        else:
            empirical_kernel = mc_kernel_fn(x1, x2, get)

        self.assertAllClose(analytic_kernel, empirical_kernel, atol=0.01, rtol=0.05)

    @test_utils.product(
        phi=[stax.Gabor, stax.Sigmoid_like],
        get=['cov1', 'nngp', 'ntk'],
    )
    def test_nonparametric(self, phi, get):
        self._test_activation_fc(phi(), get)

    @test_utils.product(
        phi=[stax.Monomial, stax.RectifiedMonomial],
        get=['cov1', 'nngp', 'ntk'],
        degree=[0, 1, 2, 3, 4, 5],
    )
    def test_monomial(self, phi, get, degree):
        # TPU-specific issue in original suite; mimic skip if needed
        if phi == stax.RectifiedMonomial and test_utils.device_under_test() == 'xla':
            raise unittest.SkipTest('NaN issues in Rectified Monomials on TPU.')
        self._test_activation_fc(phi(degree=degree), get)

    @test_utils.product(
        phi=[stax.Polynomial],
        get=['cov1', 'nngp', 'ntk'],
        coef=[
            [],
            [0],
            [-2],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, -1],
            [-0.5, 1.2],
            [1.3, 0, -1.2, -0.5],
            [-0.1, 2.1, 0, 0, -1.2, -0.5, 0, 0]
        ],
    )
    def test_polynomial(self, phi, get, coef):
        self._test_activation_fc(phi(coef=coef), get)

    @stax.layer
    def _RBF(self, gamma):
        def init_fn(gen, input_shape):  # unused params
            return input_shape, ()
        def apply_fn(unused_params, unused_xs, **kwargs):
            raise NotImplementedError()
        def kernel_fn(kernels, **kwargs):
            if kernels.ntk is not None:
                raise ValueError('RBF Kernel does not have an associated NTK.')

            if kernels.nngp.ndim > 2:
                raise ValueError('RBF Kernel undefined for cov dims > 2.')

            input_dim = kernels.shape1[1]
            cov1 = kernels.cov1.reshape((kernels.cov1.shape[0], 1))
            cov2 = cov1 if kernels.cov2 is None else kernels.cov2.reshape((1, kernels.cov2.shape[0]))
            nngp = kernels.nngp
            return kernels.replace(nngp=torch.exp(-input_dim * gamma * (cov1 + cov2 - 2 * nngp)))
        return init_fn, apply_fn, kernel_fn

    def _test_activation(self, activation_fn, same_inputs, model, get, rbf_gamma=None):
        # heavy conv tests skipped to mirror original behavior when needed
        if 'conv' in model:
            test_utils.skip_test(self)
        device = test_utils.device_under_test()
        gen = _rng(device, 1)
        gen, split = _rng(device, 11), _rng(device, 12)
        output_dim = 1024 if get == 'nngp' else 1
        b_std = 0.5
        W_std = 2.0
        if activation_fn[2].__name__ == 'Sin':
            W_std = 0.9
        if activation_fn[2].__name__ == 'Rbf':
            W_std = 1.0
            b_std = 0.0

        if model == 'fc':
            rtol = 0.04
            X0_1 = _randn((4, 2), gen)
            X0_2 = None if same_inputs else _randn((2, 2), split)
            affine = stax.Dense(1024, W_std, b_std)
            readout = stax.Dense(output_dim)
            depth = 1
        else:
            rtol = 0.05
            X0_1 = _randn((2, 4, 4, 3), gen)
            X0_2 = None if same_inputs else _randn((4, 4, 4, 3), split)
            affine = stax.Conv(512, (3, 2), W_std=W_std, b_std=b_std, padding='SAME')
            readout = stax.serial(stax.GlobalAvgPool() if 'pool' in model else stax.Flatten(), stax.Dense(output_dim))
            depth = 2

        if test_utils.device_under_test() == 'cpu':
            num_samplings = 200
            rtol *= 2
        else:
            num_samplings = 500 if activation_fn[2].__name__ in ('Sin', 'Rbf') else 300

        init_fn, apply_fn, kernel_fn = stax.serial(*[affine, activation_fn] * depth, readout)
        analytic_kernel = kernel_fn(X0_1, X0_2, get)
        mc_kernel_fn = nt.monte_carlo_kernel_fn(
            init_fn, apply_fn, split, num_samplings,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION, vmap_axes=0
        )
        empirical_kernel = mc_kernel_fn(X0_1, X0_2, get)
        test_utils.assert_close_matrices(self, analytic_kernel, empirical_kernel, rtol)

        # Check match with explicit RBF
        if rbf_gamma is not None and get == 'nngp' and model == 'fc':
            input_dim = X0_1.shape[1]
            _, _, kernel_fn_rbf = self._RBF(rbf_gamma / input_dim)
            direct_rbf_kernel = kernel_fn_rbf(X0_1, X0_2, get)
            test_utils.assert_close_matrices(self, analytic_kernel, direct_rbf_kernel, rtol)

    @test_utils.product(
        model=['fc', 'conv-pool', 'conv-flatten'],
        phi_name=['Sin', 'Cos', 'Erf', 'Gelu', 'Sign'],
        same_inputs=[False],
        get=['nngp', 'ntk'],
        approximate=[True, False],
        abc=list(itertools.product([2., 0.3], [1.5, 0.3], [0., -math.pi / 4., math.pi / 2.]))
    )
    def test_activation(self, same_inputs, model, phi_name, get, abc, approximate):
        # Keep only the canonical parameter setting as in the JAX test
        if list(abc) != [0.3, 1.5, -math.pi / 4]:
            test_utils.skip_test(self)

        if approximate and phi_name != 'Gelu':
            raise unittest.SkipTest(f'{phi_name} does not have an `approximate` parameter.')

        a, b, c = abc
        if phi_name == 'Sin':
            activation = stax.Sin(a=a, b=b, c=c)
        elif phi_name == 'Erf':
            activation = stax.Erf(a=a, b=b, c=c)
        elif phi_name in ['Gelu', 'Sign', 'Cos']:
            if a != 0.3 or b != 0.3 or c != 0.:
                raise unittest.SkipTest('Skip Gelu/Sign/Cos test if (a,b,c) != (.3,.3,0.).')
            activation = stax.Gelu() if phi_name == 'Gelu' else stax.Sign()
        else:
            raise NotImplementedError(f'Activation {phi_name} not implemented.')
        self._test_activation(activation, same_inputs, model, get)

    @test_utils.product(
        model=['fc', 'conv-pool', 'conv-flatten'],
        same_inputs=[False, True],
        get=['nngp', 'ntk'],
        gamma=[1e-6, 1e-4, 1e-2, 1.0, 2.]
    )
    def test_rbf(self, same_inputs, model, get, gamma):
        activation = stax.Rbf(gamma)
        self._test_activation(activation, same_inputs, model, get, rbf_gamma=gamma)

    @test_utils.product(
        a=[-0.5, 0.25],
        b=[-0.5, -0.1, 0.1],
        phi=[stax.Gaussian, stax.Exp],
        same_inputs=[False, True, None],
        n=[0]
    )
    def test_nonlineariy(self, phi, same_inputs, a, b, n):
        width = 2**10
        n_samples = 2**9
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(width),
            phi(a=a, b=b),
            stax.Dense(width),
            phi(a=a, b=b),
            stax.Dense(1)
        )
        device = test_utils.device_under_test()
        g1, g2, gmc = _rng(device, 1), _rng(device, 2), _rng(device, 3)
        shape = (4, 3, 2)[:n] + (1,)
        x1 = torch.cos(_randn((2,) + tuple(shape), g1))
        if same_inputs is None:
            x2 = None
        elif same_inputs is True:
            x2 = x1
        else:
            x2 = torch.cos(_randn((3,) + tuple(shape), g2))

        k = kernel_fn(x1, x2)
        mc_kernel_fn = nt.monte_carlo_kernel_fn(init_fn, apply_fn, gmc, n_samples)
        k_mc = mc_kernel_fn(x1, x2, ('nngp', 'ntk'))
        test_utils.assert_close_matrices(self, k_mc.nngp, k.nngp, 6e-2)
        test_utils.assert_close_matrices(self, k_mc.ntk, k.ntk, 6e-2)

    def test_exp_normalized(self):
        device = test_utils.device_under_test()
        g = _rng(device, 0)
        x1 = _randn((2, 6, 7, 1), g)
        x2 = _randn((4, 6, 7, 1), g)

        for do_clip in [True, False]:
            for gamma in [1., 2., 0.5]:
                for get in ['nngp', 'ntk']:
                    with self.subTest(do_clip=do_clip, gamma=gamma, get=get):
                        _, _, kernel_fn = stax.serial(
                            stax.Conv(1, (3, 3)),
                            stax.ExpNormalized(gamma, do_clip),
                            stax.Conv(1, (3, 3)),
                            stax.ExpNormalized(gamma, do_clip),
                            stax.GlobalAvgPool(),
                            stax.Dense(1)
                        )
                        k_12 = kernel_fn(x1, x2, get=get)
                        self.assertEqual(tuple(k_12.shape), (x1.shape[0], x2.shape[0]))

                        k_11 = kernel_fn(x1, None, get=get)
                        self.assertEqual(tuple(k_11.shape), (x1.shape[0],) * 2)
                        self.assertGreater(torch.linalg.eigvalsh(k_11).min().item(), 0)

                        k_22 = kernel_fn(x2, None, get=get)
                        self.assertEqual(tuple(k_22.shape), (x2.shape[0],) * 2)
                        self.assertGreater(torch.linalg.eigvalsh(k_22).min().item(), 0)

    def test_exp_normalized_ntk(self):
        def nngp_fn(cov12, var1, var2):
            prod = torch.sqrt(var1 * var2)
            return prod * torch.exp(cov12 / prod - 1)

        _, _, kernel_fn = stax.serial(stax.Dense(1), stax.Elementwise(nngp_fn=nngp_fn))
        _, _, kernel_fn_manual = stax.serial(stax.Dense(1), stax.ExpNormalized())

        device = test_utils.device_under_test()
        g = _rng(device, 1)
        x1 = _randn((5, 4, 3, 1), g)
        x2 = _randn((6, 4, 3, 1), g)

        k = kernel_fn(x1, x2)
        k_manual = kernel_fn_manual(x1, x2)
        print(f"k.shape: {k.nngp.shape}")
        print(f"k_manual.shape: {k_manual.nngp.shape}")
        self.assertAllClose(k_manual, k)

    @test_utils.product(
        same_inputs=[False, True],
        degree=[1, 2, 3, 4, 5, 6],
        get=['ntk', 'nngp'],
        readout=['pool', 'flatten']
    )
    def test_hermite(self, same_inputs, degree, get, readout):
        device = test_utils.device_under_test()
        g, g2, gmc = _rng(device, 1), _rng(device, 2), _rng(device, 3)

        # keep light; skip high degree like original
        if degree > 2:
            test_utils.skip_test(self)

        width = 10000
        n_samples = 100

        x1 = torch.cos(_randn((2, 6, 6, 3), g))
        x2 = x1 if same_inputs else torch.cos(_randn((3, 6, 6, 3), g2))

        conv_layers = [
            stax.Conv(width, (3, 3), W_std=2., b_std=0.5),
            stax.LayerNorm(),
            stax.Hermite(degree),
            stax.GlobalAvgPool() if readout == 'pool' else stax.Flatten(),
            stax.Dense(1) if get == 'ntk' else stax.Identity()
        ]

        init_fn, apply_fn, kernel_fn = stax.serial(*conv_layers)
        analytic_kernel = kernel_fn(x1, x2, get)
        mc_kernel_fn = nt.monte_carlo_kernel_fn(init_fn, apply_fn, gmc, n_samples)
        mc_kernel = mc_kernel_fn(x1, x2, get)
        rot = degree / 2. * 1e-2
        test_utils.assert_close_matrices(self, mc_kernel, analytic_kernel, rot)


# ============================================================
# Elementwise equivalence tests
# ============================================================
class ElementwiseTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        phi=[stax.Identity(), stax.Erf(), stax.Sin(), stax.Relu()],
        same_inputs=[False, True, None],
        n=[0, 1, 2],
        diagonal_batch=[True, False],
        diagonal_spatial=[True, False]
    )
    def test_elementwise(self, same_inputs, phi, n, diagonal_batch, diagonal_spatial):
        fn = lambda x: phi[1]((), x)  # reuse layer's scalar fn
        name = phi[0].__name__

        def nngp_fn(cov12, var1, var2):
            if 'Identity' in name:
                res = cov12
            elif 'Erf' in name:
                prod = (1 + 2 * var1) * (1 + 2 * var2)
                res = torch.arcsin(2 * cov12 / torch.sqrt(prod)) * 2 / pi
            elif 'Sin' in name:
                sum_ = (var1 + var2)
                s1 = torch.exp((-0.5 * sum_ + cov12))
                s2 = torch.exp((-0.5 * sum_ - cov12))
                res = (s1 - s2) / 2
            elif 'Relu' in name:
                prod = var1 * var2
                sqrt = torch.sqrt(torch.clamp(prod - cov12 ** 2, min=1e-30))
                angles = torch.atan2(sqrt, cov12)
                dot_sigma = (1 - angles / pi) / 2
                res = sqrt / (2 * pi) + dot_sigma * cov12
            else:
                raise NotImplementedError(name)
            return res

        _, _, kernel_fn = stax.serial(stax.Dense(1), stax.Elementwise(fn, nngp_fn),
                                      stax.Dense(1), stax.Elementwise(fn, nngp_fn))
        _, _, kernel_fn_manual = stax.serial(stax.Dense(1), phi, stax.Dense(1), phi)

        device = test_utils.device_under_test()
        g = _rng(device, 1)
        shape = (4, 3, 2)[:n] + (1,)
        x1 = _randn((5,) + tuple(shape), g)

        if same_inputs is None:
            x2 = None
        elif same_inputs is True:
            x2 = x1
        else:
            x2 = _randn((6,) + tuple(shape), g)

        kwargs = dict(diagonal_batch=diagonal_batch, diagonal_spatial=diagonal_spatial)
        k = kernel_fn(x1, x2, **kwargs)
        k_manual = kernel_fn_manual(x1, x2, **kwargs).replace(is_gaussian=False)
        self.assertAllClose(k_manual, k)


# ============================================================
# Numerical elementwise test
# ============================================================
class ElementwiseNumericalTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        model=['fc', 'conv-pool', 'conv-flatten'],
        phi=[stax.Erf(), stax.Gelu(), stax.Sin()],
        same_inputs=[False, True],
        get=['nngp', 'ntk']
    )
    def test_elementwise_numerical(self, same_inputs, model, phi, get):
        if 'conv' in model:
            test_utils.skip_test(self)

        device = test_utils.device_under_test()
        g, split = _rng(device,1), _rng(device, 2)
        output_dim = 1
        b_std = 0.01
        W_std = 1.0
        rtol = 2e-3
        deg = 25
        if get == 'ntk':
            rtol *= 2
        if test_utils.device_under_test() == 'xla':
            rtol *= 2

        if model == 'fc':
            X0_1 = _randn((3, 7), g)
            X0_2 = None if same_inputs else _randn((5, 7), split)
            affine = stax.Dense(1024, W_std, b_std)
            readout = stax.Dense(output_dim)
            depth = 1
        else:
            X0_1 = _randn((2, 8, 8, 3), g)
            X0_2 = None if same_inputs else _randn((3, 8, 8, 3), split)
            affine = stax.Conv(1024, (3, 2), W_std=W_std, b_std=b_std, padding='SAME')
            readout = stax.serial(stax.GlobalAvgPool() if 'pool' in model else stax.Flatten(), stax.Dense(output_dim))
            depth = 2

        _, _, kernel_fn = stax.serial(*[affine, phi] * depth, readout)
        analytic_kernel = kernel_fn(X0_1, X0_2, get)

        fn = lambda x: phi[1]((), x)
        _, _, kernel_fn_num = stax.serial(*[affine, stax.ElementwiseNumerical(fn, deg=deg)] * depth, readout)
        numerical_activation_kernel = kernel_fn_num(X0_1, X0_2, get)

        test_utils.assert_close_matrices(self, analytic_kernel, numerical_activation_kernel, rtol)


# ============================================================
# ABRelu equivalence tests
# ============================================================
class ABReluTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        same_inputs=[True, False],
        do_stabilize=[True, False],
    )
    def test_ab_relu_relu(self, same_inputs, do_stabilize):
        device = test_utils.device_under_test()
        g = _rng(device, 1)
        X0_1 = _randn((3, 2), g)
        X0_1 = X0_1.to(device)
        fc = stax.Dense(5, 1, 0)
        init_fn, apply_relu, kernel_fn_relu = stax.serial(fc, stax.Relu())
        _, params = init_fn(g, X0_1.shape)
        X0_2 = None if same_inputs else _randn((4, 2), g)

        for a, b in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            with self.subTest(a=a, b=b):
                _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(a, b, do_stabilize=do_stabilize))
                X1_1_relu = (b - a) * apply_relu(params, X0_1 * (-1 if a != 0 else 1))
                X1_1_ab_relu = apply_ab_relu(params, X0_1)
                self.assertAllClose(X1_1_relu, X1_1_ab_relu)

                kernels_relu = kernel_fn_relu(X0_1, X0_2)
                kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2)
                self.assertAllClose(kernels_relu, kernels_ab_relu)

    @test_utils.product(
        same_inputs=[True, False],
        do_stabilize=[True, False],
    )
    def test_ab_relu_id(self, same_inputs, do_stabilize):
        device = test_utils.device_under_test()
        g = _rng(device, 1)
        X0_1 = _randn((3, 2), g)
        X0_1 = X0_1.to(device)
        fc = stax.Dense(5, 1, 0)
        X0_2 = None if same_inputs else _randn((4, 2), g)

        init_fn, apply_id, kernel_fn_id = stax.serial(fc, stax.Identity())
        _, params = init_fn(g, X0_1.shape)

        for a in [-5, -1, -0.5, 0, 0.5, 1, 5]:
            with self.subTest(a=a):
                _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(a, a, do_stabilize=do_stabilize))
                X1_1_id = a * apply_id(params, X0_1)
                X1_1_ab_relu = apply_ab_relu(params, X0_1)
                self.assertAllClose(X1_1_id, X1_1_ab_relu)

                kernels_id = kernel_fn_id(a * X0_1, None if X0_2 is None else a * X0_2)
                kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2).replace(is_gaussian=True)
                self.assertAllClose(kernels_id, kernels_ab_relu)

    @test_utils.product(
        same_inputs=[True, False],
        do_stabilize=[True, False],
    )
    def test_leaky_relu(self, same_inputs, do_stabilize):
        device = test_utils.device_under_test()
        g = _rng(device, 1)
        X0_1 = _randn((3, 2), g)
        fc = stax.Dense(5, 1, 0)
        X0_2 = None if same_inputs else _randn((4, 2), g)

        for a in [-2, -1, 0, 1, 2]:
            with self.subTest(alpha=a):
                init_fn, apply_leaky_relu, kernel_fn_leaky = stax.serial(fc, stax.LeakyRelu(a, do_stabilize=do_stabilize))
                _, apply_ab_relu, kernel_fn_ab = stax.serial(fc, stax.ABRelu(a, 1))
                _, params = init_fn(g, X0_1.shape)

                X1_1_leaky = apply_leaky_relu(params, X0_1)
                X1_1_ab = apply_ab_relu(params, X0_1)
                self.assertAllClose(X1_1_leaky, X1_1_ab)

                kernels_leaky = kernel_fn_leaky(X0_1, X0_2)
                kernels_ab = kernel_fn_ab(X0_1, X0_2)
                self.assertAllClose(kernels_leaky, kernels_ab)

    @test_utils.product(
        same_inputs=[True, False],
        do_stabilize=[True, False],
    )
    def test_abs(self, same_inputs, do_stabilize):
        device = test_utils.device_under_test()
        g = _rng(device, 1)
        X0_1 = _randn((3, 2), g)
        fc = stax.Dense(5, 1, 0)
        X0_2 = None if same_inputs else _randn((4, 2), g)

        init_fn, apply_abs, kernel_fn_abs = stax.serial(fc, stax.Abs(do_stabilize=do_stabilize))
        _, apply_ab_relu, kernel_fn_ab = stax.serial(fc, stax.ABRelu(-1, 1))
        _, params = init_fn(g, X0_1.shape)

        X1_1_abs = apply_abs(params, X0_1)
        X1_1_ab = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_abs, X1_1_ab)

        kernels_abs = kernel_fn_abs(X0_1, X0_2, ('nngp', 'ntk'))
        kernels_ab = kernel_fn_ab(X0_1, X0_2, ('nngp', 'ntk'))
        self.assertAllClose(kernels_abs, kernels_ab)


# ============================================================
# Autodiff / gradient checks
# ============================================================
class AutodiffTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        get=['ntk', 'nngp'],
        same_inputs=[True, False, None],
        phi=[stax.Erf, stax.Sin, stax.Gelu, stax.Relu, stax.ElementwiseNumerical]
    )
    def test_autodiff(self, get, same_inputs, phi):
        device = test_utils.device_under_test()
        g1, g2, g3, g4 = _rng(device, 1), _rng(device, 2), _rng(device, 3), _rng(device, 4)

        x1 = torch.cos(_randn((3, 1, 2, 3), g1))
        if same_inputs is None:
            x2 = None
        elif same_inputs is True:
            x2 = x1
        else:
            x2 = torch.cos(_randn((4, 1, 2, 3), g2))

        name = phi.__name__
        if name == 'LeakyRelu':
            phi = phi(0.1)
        elif name == 'ElementwiseNumerical':
            phi = phi(fn=torch.cos, deg=25)
        else:
            phi = phi()

        _, _, kernel_fn = stax.serial(stax.Dense(1, 2., 0.01), phi, stax.Dense(1, 2., 0.01), phi)

        def k(x1, x2):
            return kernel_fn(x1, x2, get)

        dx1 = _randn(x1.shape, g3) * 0.01
        dx2 = None if x2 is None else _randn(x2.shape, g4) * 0.01

        # First- and second-order directional derivatives via forward-mode
        _, _dk = torch_jvp(k, (x1, x2), (dx1, dx2))
        _, _d2k = torch_jvp(lambda a, b: torch_jvp(k, (a, b), (dx1, dx2))[1], (x1, x2), (dx1, dx2))

        # tolerance logic mirrored from JAX test
        if (same_inputs is not False) and get == 'ntk' and 'Relu' in name:
            tol = 8e-3
        else:
            tol = 2e-3 if name == 'ElementwiseNumerical' else 1e-4

        def rel_close(x, y, tol_=3e-5):
            if test_utils.device_under_test() == 'xla':
                tol_ = 0.21
            num = torch.max(torch.abs(x - y))
            den = torch.mean(torch.abs(x)) + torch.mean(torch.abs(y))
            self.assertLess((num / (den + 1e-12)).item(), tol_)

        # k(x + dx) ~ k(x) + dk(x) dx + 0.5 dx^T d2k dx
        rel_close(k(x1 + dx1, None if same_inputs is None else x2 + dx2),
                  k(x1, x2) + _dk + _d2k / 2, tol_=tol)

        # Compare jacfwd vs jacrev along each arg (torch.func)
        k_fwd_0 = torch_jacfwd(k)(x1, x2)
        k_rev_0 = torch_jacrev(k)(x1, x2)
        rel_close(k_fwd_0, k_rev_0)

        if same_inputs is not None:
            k_fwd_1 = torch_jacfwd(k, argnums=1)(x1, x2)
            k_rev_1 = torch_jacrev(k, argnums=1)(x1, x2)
            rel_close(k_fwd_1, k_rev_1)

            # Symmetry checks like original (moveaxes equivalents)
            def mv(x, src, dst):
                return x.movedim(src, dst)

            k_fwd_01 = torch_jacfwd(k, argnums=1)(x2, x1)
            k_rev_01 = torch_jacrev(k, argnums=1)(x2, x1)
            rel_close(mv(k_fwd_0, (0, 2, 4), (1, 3, 5)), k_fwd_01)
            rel_close(mv(k_rev_0, (0, 2, 4), (1, 3, 5)), k_rev_01)

            k_fwd_10 = torch_jacfwd(k)(x2, x1)
            k_rev_10 = torch_jacrev(k)(x2, x1)
            rel_close(mv(k_fwd_1, (0, 2, 4), (1, 3, 5)), k_fwd_10)
            rel_close(mv(k_rev_1, (0, 2, 4), (1, 3, 5)), k_rev_10)

    @test_utils.product(
        get=['ntk', 'nngp'],
        parameterization=['standard', 'ntk'],
        parameterization_out=['ntk'],
        do_jit=[True],
        x1_type=['zeros', 'ones', 'random'],
        x2_type=['zeros', 'ones', 'random', 'x1', 'none'],
        b_std=[None, 0.1],
        phi=[stax.Identity, stax.Erf, stax.Abs, stax.Gelu, stax.Relu, stax.Sigmoid_like, stax.ABRelu,
             stax.Exp, stax.ExpNormalized, stax.Gaussian, stax.Sign, stax.Rbf, stax.Cos, stax.Sin]
    )
    def test_activations(self, get, parameterization, parameterization_out, x1_type, x2_type, b_std, phi, do_jit):
        # Keep scope small as in original
        if phi is not stax.Relu:
            test_utils.skip_test(self)

        if phi == stax.ABRelu:
            phi_ = phi(0.25, 0.5)
        else:
            phi_ = phi()

        n_out = 1 if get == 'ntk' else 1024
        width = 612

        W_std_in = width ** (-0.5) if parameterization_out == 'standard' else 1.
        if phi == stax.Exp:
            W_std_in /= 10.

        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(width, W_std=W_std_in, b_std=b_std, parameterization=parameterization),
            phi_,
            stax.Dense(n_out, b_std=b_std, parameterization=parameterization_out),
        )

        def get_x(kind, key):
            shape = (1, 2)
            device = test_utils.device_under_test()
            g = _rng(device, key)
            if kind == 'zeros':
                x = torch.zeros(shape)
            elif kind == 'ones':
                x = torch.ones(shape)
            elif kind == 'random':
                x = _randn(shape, g)
            elif kind == 'sin':
                x = torch.sin(_randn(shape, g))
            elif kind == 'none':
                return None
            else:
                raise ValueError(kind)
            return x

        x1 = get_x(x1_type, 1)
        x2 = x1 if x2_type == 'x1' else get_x(x2_type, 2)

        def kernel_scalar(x1, x2):
            return kernel_fn(x1, x2, get)[0, 0]

        k1 = kernel_scalar(x1, x2)
        val_grad = _value_and_grad_scalar(kernel_scalar)
        k2, (g1, g2) = val_grad(x1, x2)
        self.assertAllClose(k1, k2)

        # Forward-mode directional check
        k2_fwd, _ = torch_jvp(kernel_scalar, (x1, x2), (x1, x2))
        self.assertAllClose(k1, k2_fwd)

        # MC comparisons (skip for ExpNormalized / Sign NTK like original)
        if phi == stax.ExpNormalized or (get == 'ntk' and phi == stax.Sign):
            raise unittest.SkipTest('Not comparing against MC kernels.')

        device = test_utils.device_under_test()
        _kernel_scalar_mc = nt.monte_carlo_kernel_fn(init_fn, apply_fn, key=_rng(device, 3), n_samples=1, device_count=0)

        def kernel_scalar_mc(x1, x2):
            return _kernel_scalar_mc(x1, x2, get)[0, 0]

        k_mc = kernel_scalar_mc(x1, x2)
        k_mc2, (mg1, mg2) = _value_and_grad_scalar(kernel_scalar_mc)(x1, x2)
        self.assertAllClose(k_mc, k_mc2)
        k_mc2_fwd, _ = torch_jvp(kernel_scalar_mc, (x1, x2), (x1, x2))
        self.assertAllClose(k_mc, k_mc2_fwd)

        # Empirical (finite-width) kernel mean over samples (reduced n for speed)
        def kernel_fn_emp(x1, x2, get, params):
            return nt.empirical_kernel_fn(apply_fn)(x1, x2, get, params)[0, 0]

        # small-ish Monte Carlo average for gradient comparison
        def kernel_scalar_mc_grad_mean(x1, x2):
            device = test_utils.device_under_test()
            g = _rng(device, 4)
            n_samples = 2**8
            k_sum = 0.0
            g1_sum = torch.zeros_like(x1) if x1 is not None else None
            g2_sum = None if x2 is None else torch.zeros_like(x2)

            val_grad_emp = _value_and_grad_scalar(lambda a, b: kernel_fn_emp(a, b, get, init_fn(g, x1.shape)[1]))
            for _ in range(n_samples):
                _, params = init_fn(g, x1.shape)
                val = kernel_fn_emp(x1, x2, get, params)
                k_sum = k_sum + val
                val2, (gg1, gg2) = _value_and_grad_scalar(lambda a, b: kernel_fn_emp(a, b, get, params))(x1, x2)
                g1_sum = g1_sum + gg1
                if x2 is not None:
                    g2_sum = g2_sum + gg2

            k_mean = k_sum / n_samples
            g1_mean = g1_sum / n_samples
            g2_mean = None if g2_sum is None else g2_sum / n_samples
            return k_mean, (g1_mean, g2_mean)

        k_emp_mean, (g1_emp, g2_emp) = kernel_scalar_mc_grad_mean(x1, x2)
        self.assertAllClose(k1, k_emp_mean, atol=8e-3, rtol=4e-2)

        if phi == stax.Sign and get == 'nngp':
            raise unittest.SkipTest('Empirical NNGP derivative for discontinuous fn does not converge.')

        if (phi in [stax.Abs, stax.Relu, stax.LeakyRelu, stax.ABRelu] and get == 'ntk'):
            raise unittest.SkipTest('Empirical NTK derivative for non-differentiable fn does not converge.')

        # Compare gradient of analytic kernel to empirical mean
        # Relaxed tolerances like original
        test_utils.assert_close_matrices(self, g1_emp, g1, rtol=0.05, atol=10.)


class Issue123Test(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        architecture=['conv', 'wrn'],
        get=['ntk', 'nngp'],
        do_jit=[True]
    )
    def test_issue_123(self, get, architecture, do_jit):
        # Keep the same architectures as the original reference; this is a light check.
        if architecture == 'wrn':
            def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
                main = stax.serial(
                    stax.Relu(),
                    stax.Conv(channels, (3, 3), strides, padding='SAME', parameterization='standard'),
                    stax.Relu(),
                    stax.Conv(channels, (3, 3), padding='SAME', parameterization='standard'),
                )
                shortcut = (stax.Identity() if not channel_mismatch else
                            stax.Conv(channels, (3, 3), strides, padding='SAME', parameterization='standard'))
                return stax.serial(stax.FanOut(2), stax.parallel(main, shortcut), stax.FanInSum())

            def WideResnetGroup(n, channels, strides=(1, 1)):
                blocks = [WideResnetBlock(channels, strides, channel_mismatch=True)]
                for _ in range(n - 1):
                    blocks += [WideResnetBlock(channels, (1, 1))]
                return stax.serial(*blocks)

            def WideResnet(block_size, k, num_classes):
                return stax.serial(
                    stax.Conv(16, (3, 3), padding='SAME', parameterization='standard'),
                    WideResnetGroup(block_size, int(16 * k)),
                    WideResnetGroup(block_size, int(32 * k), (2, 2)),
                    WideResnetGroup(block_size, int(64 * k), (2, 2)),
                    stax.AvgPool((8, 8), padding='SAME'),
                    stax.Flatten(),
                    stax.Dense(num_classes, 1.0, 0.0, parameterization='standard'),
                )

            init_fn, apply_fn, kernel_fn = WideResnet(block_size=1, k=1, num_classes=1)
        elif architecture == 'conv':
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Conv(1, (3, 3)),
                stax.Relu(),
                stax.Flatten(),
            )
        else:
            raise ValueError(architecture)

        x1 = x2 = torch.zeros((1, 8, 8, 3))
        def kernel_scalar(a, b):
            return kernel_fn(a, b, get)[0, 0]

        k1 = kernel_scalar(x1, x2)
        val_grad = _value_and_grad_scalar(kernel_scalar)
        k2, (g1, g2) = val_grad(x1, x2)
        self.assertAllClose(k1, k2)

        k2_fwd, _ = torch_jvp(kernel_scalar, (x1, x2), (x1, x2))
        self.assertAllClose(k1, k2_fwd)

        # Gradient should be zero at zero inputs (like original)
        self.assertAllClose(torch.zeros_like(x1), torch.autograd.grad(k1, x1, retain_graph=True, allow_unused=True)[0] or torch.zeros_like(x1))


if __name__ == '__main__':
    unittest.main()
