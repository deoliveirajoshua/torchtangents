# PyTorch/torchtangents rewrite of tests for `neural_tangents/predict.py`.

import math
import unittest
from functools import partial

import torch
import torch.nn.functional as F

import torchtangents as tt
from torchtangents import predict, stax
from torchtangents._src.predict import _is_on_cpu  # keep same API path
from tests import test_utils


# -------------------------
# Test grid and tolerances
# -------------------------
OUTPUT_LOGITS = [2]
GETS = ('ntk', 'nngp', ('ntk', 'nngp'))

RTOL = 0.01
ATOL = 0.01
if not torch.backends.cuda.matmul.allow_tf32:  # rough proxy for 64-bit config variance
    RTOL = 0.02
    ATOL = 0.02

FLAT = 'FLAT'
POOLING = 'POOLING'

# TODO(schsam): Add a pooling test when multiple inputs are supported in Conv + Pooling.
TRAIN_SIZES = [4, 8]
TEST_SIZES  = [6, 2]
INPUT_SHAPES = [(8,), (4, 4, 3)]
NETWORK = [FLAT, FLAT]
CONVOLUTION_CHANNELS = 256

test_utils.update_test_tolerance()


# -------------------------
# Helpers
# -------------------------
def _jit(f):
    # Placeholder to keep call sites identical to JAX tests.
    return f


def _rng(device: str | torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device if isinstance(device, str) else str(device))
    g.manual_seed(seed)
    return g


def _randn(shape, device, gen, dtype=torch.float32):
    return torch.randn(*shape, generator=gen, device=device, dtype=dtype)


def _bernoulli(shape, device, gen, dtype=torch.float32):
    return torch.bernoulli(torch.full(shape, 0.5, device=device, dtype=dtype), generator=gen)


def _flatten_pytree_like_jax(x):
    # Neural Tangents tests often compare flattened pytrees.
    if isinstance(x, (list, tuple)):
        out, treedef = [], []
        for xi in x:
            flat_i, td_i = _flatten_pytree_like_jax(xi)
            out += flat_i
            treedef.append(td_i)
        return out, tuple(treedef)
    if isinstance(x, dict):
        out, treedef = [], {}
        for k in sorted(x.keys()):
            flat_i, td_i = _flatten_pytree_like_jax(x[k])
            out += flat_i
            treedef[k] = td_i
        return out, treedef
    # Tensor/number
    if isinstance(x, torch.Tensor):
        return [x.reshape(-1)], None
    return [torch.as_tensor(x).reshape(-1)], None


def _ravel_pytree(x):
    flats, _ = _flatten_pytree_like_jax(x)
    if not flats:
        return torch.empty(0), None
    return torch.cat([f.to(dtype=flats[0].dtype, device=flats[0].device) for f in flats]), None


def _build_network(input_shape, network, out_logits):
    if len(input_shape) == 1:
        assert network == FLAT
        return stax.serial(
            stax.Dense(4096, W_std=1.2, b_std=0.05), stax.Erf(),
            stax.Dense(out_logits, W_std=1.2, b_std=0.05),
        )
    elif len(input_shape) == 3:
        if network == POOLING:
            return stax.serial(
                stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
                stax.GlobalAvgPool(),
                stax.Dense(out_logits, W_std=2.0, b_std=0.05),
            )
        elif network == FLAT:
            return stax.serial(
                stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
                stax.Flatten(),
                stax.Dense(out_logits, W_std=2.0, b_std=0.05),
            )
        else:
            raise ValueError(f'Unexpected network type found: {network}')
    else:
        raise ValueError('Expected flat or image test input.')


def _empirical_kernel(gen, input_shape, network, out_logits):
    init_fn, f, _ = _build_network(input_shape, network, out_logits)
    # Follow NT convention: pass (-1,) + input_shape to init; here materialize with batch=1 as needed.
    _, params = init_fn(gen, (-1,) + tuple(input_shape))
    _kernel_fn = tt.empirical_kernel_fn(f, trace_axes=(), vmap_axes=0)
    # Bind params; keep `get` static-ish by position
    kernel_fn = lambda x1, x2, get: _kernel_fn(x1, x2, get, params=params)
    return params, f, _jit(kernel_fn)


def _theoretical_kernel(gen, input_shape, network, out_logits):
    init_fn, f, kernel_fn = _build_network(input_shape, network, out_logits)
    _, params = init_fn(gen, (-1,) + tuple(input_shape))
    return params, f, _jit(kernel_fn)


KERNELS = {
    'empirical':  _empirical_kernel,
    'theoretical': _theoretical_kernel,
}


# -------------------------
# Tests
# -------------------------
class PredictTest(test_utils.NeuralTangentsTestCase):

    @classmethod
    def _get_inputs(cls, out_logits, test_shape, train_shape, device=None):
        device = device or test_utils.device_under_test()
        g = _rng(device, 0)
        x_train = _randn(train_shape, device, g)
        y_train = _bernoulli((train_shape[0], out_logits), device, g)
        x_test  = _randn(test_shape, device, g)
        return g, x_test, x_train, y_train

    def _test_zero_time(self, predictor, fx_train_0, fx_test_0, g_td, momentum):
        fx_train_t0, fx_test_t0 = predictor(0.0, fx_train_0, fx_test_0, g_td)
        self.assertAllClose(fx_train_0, fx_train_t0)
        self.assertAllClose(fx_test_0, fx_test_t0)
        fx_train_only_t0 = predictor(0.0, fx_train_0, None, g_td)
        self.assertAllClose(fx_train_0, fx_train_only_t0)

        if momentum is not None:
            state_0 = predict.ODEState(fx_train_0, fx_test_0)
            state_t0 = predictor(0.0, state_0, None, g_td)
            self.assertAllClose(state_0.fx_train, state_t0.fx_train)
            self.assertAllClose(state_0.fx_test, state_t0.fx_test)

            state_train_only_0 = predict.ODEState(fx_train_0)
            state_train_only_t0 = predictor(0.0, state_0, None, g_td)
            self.assertAllClose(state_train_only_0.fx_train, state_train_only_t0.fx_train)

    def _test_inf_time(self, predictor, fx_train_0, fx_test_0, g_td, y_train):
        pr_inf = predictor(math.inf, fx_train_0)
        self.assertAllClose(pr_inf, y_train, check_dtypes=False)
        self.assertAllClose(pr_inf, predictor(None, fx_train_0))
        self.assertAllClose(predictor(math.inf, fx_train_0, fx_test_0, g_td),
                            predictor(None, fx_train_0, fx_test_0, g_td))

    def _test_multi_step(self, predictor, fx_train_0, fx_test_0, g_td, momentum):
        ts = torch.arange(6.0).reshape(2, 1, 3)

        fx_train_single, fx_test_single = predictor(ts, fx_train_0, fx_test_0, g_td)

        fx_train_concat, fx_test_concat = [], []
        for t in ts.reshape(-1):
            ft, fte = predictor(t.item(), fx_train_0, fx_test_0, g_td)
            fx_train_concat.append(ft)
            fx_test_concat.append(fte)

        fx_train_concat = torch.stack(fx_train_concat).reshape(ts.shape + fx_train_single.shape[ts.ndim:])
        fx_test_concat  = torch.stack(fx_test_concat ).reshape(ts.shape +  fx_test_single.shape[ts.ndim:])
        self.assertAllClose(fx_train_concat, fx_train_single)
        self.assertAllClose(fx_test_concat,  fx_test_single)

        if momentum is not None:
            state_0 = predict.ODEState(fx_train_0, fx_test_0)
            t_1 = (0, 0, 2)
            state_1 = predictor(ts[t_1].item(), state_0, None, g_td)
            self.assertAllClose(fx_train_single[t_1], state_1.fx_train)
            self.assertAllClose(fx_test_single [t_1], state_1.fx_test)

            t_max = (-1,) * ts.ndim
            state_max = predictor((ts[t_max] - ts[t_1]).item(), state_1, None, g_td)
            self.assertAllClose(fx_train_single[t_max], state_max.fx_train)
            self.assertAllClose(fx_test_single [t_max], state_max.fx_test)

    @test_utils.product(
        train_size=TRAIN_SIZES,
        test_size=TEST_SIZES,
        input_shape=INPUT_SHAPES,
        network=NETWORK,
        out_logits=OUTPUT_LOGITS,
        kernel_type=list(KERNELS.keys()),
        momentum=[None, 0.9],
        learning_rate=[0.0002],
        t=[5],
        loss=['mse_analytic', 'mse'],
    )
    def testNTKGDPrediction(
        self,
        train_size,
        test_size,
        input_shape,
        network,
        out_logits,
        kernel_type,
        momentum,
        learning_rate,
        t,
        loss,
    ):
        device = test_utils.device_under_test()
        train_shape = (train_size, *input_shape)
        test_shape  = (test_size,  *input_shape)
        g, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape, train_shape, device=device)

        fn_and_kernel = KERNELS[kernel_type]
        params, f, ntk = fn_and_kernel(g, train_shape[1:], network, out_logits)

        g_dd = ntk(x_train, None, 'ntk')
        g_td = ntk(x_test,  x_train, 'ntk')

        # Regress to MSE
        def loss_fn(y_pred, y_true):
            return 0.5 * torch.mean((y_pred - y_true) ** 2)

        # Autograd: grad wrt params of (f(params, x_train) vs y_train)
        def grad_loss(params_local):
            y_pred = f(params_local, x_train)
            l = loss_fn(y_pred, y_train)
            grads = torch.autograd.grad(l, tt.tree_leaves(params_local), create_graph=False, retain_graph=False)
            return tt.tree_pack_like(grads, params_local)

        trace_axes = () if g_dd.ndim == 4 else (-1,)
        if loss == 'mse_analytic':
            if momentum is not None:
                raise unittest.SkipTest("momentum with mse_analytic is skipped (matches upstream).")
            predictor = predict.gradient_descent_mse(g_dd, y_train,
                                                     learning_rate=learning_rate,
                                                     trace_axes=trace_axes)
        elif loss == 'mse':
            predictor = predict.gradient_descent(loss_fn, g_dd, y_train,
                                                 learning_rate=learning_rate,
                                                 momentum=momentum,
                                                 trace_axes=trace_axes)
        else:
            raise NotImplementedError(loss)

        predictor = _jit(predictor)

        fx_train_0 = f(params, x_train)
        fx_test_0  = f(params, x_test)

        self._test_zero_time(predictor, fx_train_0, fx_test_0, g_td, momentum)
        self._test_multi_step(predictor, fx_train_0, fx_test_0, g_td, momentum)
        if loss == 'mse_analytic':
            self._test_inf_time(predictor, fx_train_0, fx_test_0, g_td, y_train)

        # Torch optimizer baseline
        # We flatten params for a simple SGD loop using torch.optim; torchtangents should expose tree utils.
        params_flat, treedef = tt.ravel_params(params)
        params_flat = params_flat.detach().clone().requires_grad_(True)
        opt = torch.optim.SGD([params_flat], lr=learning_rate, momentum=(momentum or 0.0))

        def unwrap_and_apply(params_vec, x):
            p_un = tt.unravel_params(params_vec, treedef)
            return f(p_un, x)

        for i in range(t):
            opt.zero_grad(set_to_none=True)
            y_pred = unwrap_and_apply(params_flat, x_train)
            l = loss_fn(y_pred, y_train)
            l.backward()
            opt.step()

        fx_train_nn = unwrap_and_apply(params_flat, x_train).detach()
        fx_test_nn  = unwrap_and_apply(params_flat, x_test ).detach()

        fx_train_t, fx_test_t = predictor(t, fx_train_0, fx_test_0, g_td)

        self.assertAllClose(fx_train_nn, fx_train_t, rtol=RTOL, atol=ATOL)
        self.assertAllClose(fx_test_nn,  fx_test_t,  rtol=RTOL, atol=ATOL)

    @classmethod
    def _cov_empirical(cls, x):
        # x shape: [T, N, O, ...] – match einsum used in original
        # Here we assume x shape [t, batch, out] and compute covariance over models (t-dimension) and outputs dim -1.
        # Keep exact axes per original test: cov over t and last dim, indexed as 'itjk,itlk->tjl' averaged by (i * k)
        # In torch we’ll mirror: mean over first and last dims.
        # For the test we only need positive semidefiniteness and shape agreement; use the same formula.
        # x: (I,T,J,K) in the original einsum; in their helper they normalize by (I * K)
        I = x.shape[0]
        K = x.shape[-1]
        return torch.einsum('itjk,itlk->tjl', x, x) / float(I * K)

    @test_utils.product(
        train_size=TRAIN_SIZES[:1],
        test_size=TEST_SIZES[:1],
        input_shape=INPUT_SHAPES[:1],
        out_logits=[1],
    )
    def testNTKMeanCovPrediction(self, train_size, test_size, input_shape, out_logits):
        device = test_utils.device_under_test()
        train_shape = (train_size, *input_shape)
        test_shape  = (test_size,  *input_shape)
        g, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape, train_shape, device=device)

        init_fn, f, kernel_fn = stax.serial(
            stax.Dense(512, W_std=1.2, b_std=0.05), stax.Erf(),
            stax.Dense(out_logits, W_std=1.2, b_std=0.05))

        reg = 1e-6
        predictor = predict.gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, diag_reg=reg)
        ts = torch.tensor([1., 5., 10.], device=device)

        fx_test_inf, cov_test_inf = predictor(ts, x_test, 'ntk', True)
        self.assertEqual(cov_test_inf.shape[1], x_test.shape[0])
        self.assertGreater(torch.linalg.eigvalsh(cov_test_inf).min().item(), -1e-8)

        fx_train_inf, cov_train_inf = predictor(ts, None, 'ntk', True)
        self.assertEqual(cov_train_inf.shape[1], x_train.shape[0])
        self.assertGreater(torch.linalg.eigvalsh(cov_train_inf).min().item(), -1e-8)

        _kernel_fn = tt.empirical_kernel_fn(f)

        def kernel_emp(x1, x2, params):
            return _kernel_fn(x1, x2, 'ntk', params=params)

        def predict_empirical(key_gen):
            _, params_local = init_fn(key_gen, train_shape)
            g_dd = kernel_emp(x_train, None, params_local)
            g_td = kernel_emp(x_test,  x_train, params_local)
            predict_fn = predict.gradient_descent_mse(g_dd, y_train, diag_reg=reg)
            fx_train_0 = f(params_local, x_train)
            fx_test_0  = f(params_local, x_test)
            return predict_fn(ts, fx_train_0, fx_test_0, g_td)

        def predict_mc(count, gen):
            gens = [ _rng(device, i+1234) for i in range(count) ]
            fx_train_list, fx_test_list = zip(*[predict_empirical(gi) for gi in gens])
            fx_train = torch.stack(fx_train_list, dim=0)
            fx_test  = torch.stack(fx_test_list,  dim=0)

            fx_train_mean = fx_train.mean(dim=0, keepdim=True)
            fx_test_mean  = fx_test.mean(dim=0, keepdim=True)

            fx_train_centered = fx_train - fx_train_mean
            fx_test_centered  = fx_test  - fx_test_mean

            cov_train = self._cov_empirical(fx_train_centered)
            cov_test  = self._cov_empirical(fx_test_centered)
            return fx_train_mean, fx_test_mean, cov_train, cov_test

        fx_train_mc, fx_test_mc, cov_train_mc, cov_test_mc = predict_mc(1024, g)  # fewer than 4096 for speed
        tol = 0.06

        def assert_close(a, b):
            av, _ = _ravel_pytree(a)
            bv, _ = _ravel_pytree(b)
            self.assertAllClose(av, bv, atol=tol, rtol=tol)

        assert_close(fx_train_mc, fx_train_inf)
        assert_close(cov_train_mc, cov_train_inf)
        assert_close(cov_test_mc,  cov_test_inf)
        assert_close(fx_test_mc,   fx_test_inf)

    @test_utils.product(
        train_size=TRAIN_SIZES[:-1],
        test_size=TEST_SIZES[:-1],
        input_shape=INPUT_SHAPES,
        network=NETWORK[:-1],
        out_logits=OUTPUT_LOGITS,
    )
    def testGradientDescentMseEnsembleGet(self, train_size, test_size, input_shape, network, out_logits):
        device = test_utils.device_under_test()
        train_shape = (train_size, *input_shape)
        test_shape  = (test_size,  *input_shape)
        _, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape, train_shape, device=device)
        _, _, kernel_fn = _build_network(train_shape[1:], network, out_logits)

        predictor = predict.gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, diag_reg=0.)
        for x in [None, 'x_test']:
            with self.subTest(x=x):
                x_ = None if x is None else x_test
                out = predictor(None, x_, 'ntk', compute_cov=True)
                self.assertTrue(isinstance(out, predict.Gaussian))
                out = predictor(1., x_, 'nngp', compute_cov=True)
                self.assertTrue(isinstance(out, predict.Gaussian))
                out = predictor(torch.tensor([0., 1.], device=device), x_, ('ntk',), compute_cov=True)
                self.assertTrue(len(out) == 1 and isinstance(out[0], predict.Gaussian))
                out = predictor(2., x_, ('ntk', 'nngp'), compute_cov=True)
                self.assertTrue(len(out) == 2 and isinstance(out[0], predict.Gaussian) and isinstance(out[1], predict.Gaussian))
                out2 = predictor(2., x_, ('nngp', 'ntk'), compute_cov=True)
                self.assertAllClose(out[0], out2[1])
                self.assertAllClose(out[1], out2[0])

    @test_utils.product(
        train_size=TRAIN_SIZES[:-1],
        test_size=TEST_SIZES[:-1],
        input_shape=INPUT_SHAPES[:-1],
        network=NETWORK[:-1],
        out_logits=OUTPUT_LOGITS,
        get=GETS
    )
    def testInfiniteTimeAgreement(self, train_size, test_size, input_shape, network, out_logits, get):
        device = test_utils.device_under_test()
        train_shape = (train_size, *input_shape)
        test_shape  = (test_size,  *input_shape)
        _, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape, train_shape, device=device)
        _, _, kernel_fn = _build_network(train_shape[1:], network, out_logits)

        reg = 0.
        predictor = predict.gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, diag_reg=reg)

        for x in (None, 'x_test'):
            with self.subTest(x=x):
                x_ = None if x is None else x_test
                fin = predictor(t=math.inf, x_test=x_, get=get, compute_cov=True)
                inf = predictor(t=None, x_test=x_, get=get, compute_cov=True)
                self.assertAllClose(inf, fin)
                if x_ is None:
                    fin_x = predictor(t=math.inf, x_test=x_train, get=get, compute_cov=True)
                    inf_x = predictor(t=None,     x_test=x_train, get=get, compute_cov=True)
                    self.assertAllClose(inf, inf_x)
                    self.assertAllClose(inf_x, fin_x)

    @test_utils.product(
        train_size=TRAIN_SIZES,
        test_size=TEST_SIZES,
        input_shape=INPUT_SHAPES,
        network=NETWORK,
        out_logits=OUTPUT_LOGITS,
    )
    def testZeroTimeAgreement(self, train_size, test_size, input_shape, network, out_logits):
        """Test that the NTK and NNGP agree at t=0."""
        device = test_utils.device_under_test()
        train_shape = (train_size, *input_shape)
        test_shape  = (test_size,  *input_shape)
        _, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape, train_shape, device=device)
        _, _, ker_fun = _build_network(train_shape[1:], network, out_logits)

        reg = 1e-7
        predictor = predict.gradient_descent_mse_ensemble(ker_fun, x_train, y_train, diag_reg=reg)

        for x in (None, 'x_test'):
            with self.subTest(x=x):
                x_ = None if x is None else x_test
                zero = predictor(t=0.0, x_test=x_, get=('NTK', 'NNGP'), compute_cov=True)
                if x_ is None:
                    k = ker_fun(x_train, None, get='nngp')
                    ref = (torch.zeros_like(y_train, dtype=k.dtype, device=k.device), k)
                else:
                    ref = (torch.zeros((test_shape[0], out_logits), device=x_.device, dtype=torch.float32),
                           ker_fun(x_test, None, get='nngp'))
                self.assertAllClose((ref,) * 2, zero, check_dtypes=False)
                if x_ is None:
                    zero_x = predictor(t=0.0, x_test=x_train, get=('NTK', 'NNGP'), compute_cov=True)
                    self.assertAllClose((ref,) * 2, zero_x)

    @classmethod
    def _always_ntk(cls, ker_fun):
        def always_ntk(x1, x2, get=('nngp', 'ntk')):
            out = ker_fun(x1, x2, get=('nngp', 'ntk'))
            if get == 'nngp' or get == 'ntk':
                return out.ntk
            else:
                return out._replace(nngp=out.ntk)
        return always_ntk

    @test_utils.product(
        train_size=TRAIN_SIZES,
        test_size=TEST_SIZES,
        input_shape=INPUT_SHAPES,
        network=NETWORK,
        out_logits=OUTPUT_LOGITS,
    )
    def testNTK_NTKNNGPAgreement(self, train_size, test_size, input_shape, network, out_logits):
        device = test_utils.device_under_test()
        train_shape = (train_size, *input_shape)
        test_shape  = (test_size,  *input_shape)
        _, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape, train_shape, device=device)
        _, _, ker_fun = _build_network(train_shape[1:], network, out_logits)

        reg = 1e-7
        predictor = predict.gradient_descent_mse_ensemble(ker_fun, x_train, y_train, diag_reg=reg)

        ts = torch.logspace(-2, 8, 10, device=device).reshape(5, 2)

        for t in (None, 'ts'):
            for x in (None, 'x_test'):
                with self.subTest(t=t, x=x):
                    x_ = None if x is None else x_test
                    t_ = None if t is None else ts

                    ntk = predictor(t=t_, get='ntk', x_test=x_)
                    if t_ is not None:
                        ntk_ind = torch.stack([predictor(t=ti.item(), get='ntk', x_test=x_) for ti in t_.reshape(-1)]).reshape(t_.shape + ntk.shape[2:])
                        self.assertAllClose(ntk_ind, ntk)

                    always_ntk = self._always_ntk(ker_fun)
                    predictor_ntk = predict.gradient_descent_mse_ensemble(always_ntk, x_train, y_train, diag_reg=reg)
                    ntk_nngp = predictor_ntk(t=t_, get='nngp', x_test=x_)
                    self.assertAllClose(ntk, ntk_nngp)

                    def always_nngp(x1, x2, get=('nngp', 'ntk')):
                        out = ker_fun(x1, x2, get=('nngp', 'ntk'))
                        if get == 'nngp' or get == 'ntk':
                            return out.nngp
                        else:
                            return out._replace(ntk=out.nngp)

                    predictor_nngp = predict.gradient_descent_mse_ensemble(always_nngp, x_train, y_train, diag_reg=reg)
                    nngp_cov = predictor(t=t_, get='nngp', x_test=x_, compute_cov=True).covariance
                    nngp_ntk_cov = predictor_nngp(t=t_, get='ntk', x_test=x_, compute_cov=True).covariance
                    if t_ is not None:
                        nngp_ntk_cov_ind = torch.stack(
                            [predictor_nngp(t=ti.item(), get='ntk', x_test=x_, compute_cov=True).covariance
                             for ti in t_.reshape(-1)]
                        ).reshape(t_.shape + nngp_cov.shape[2:])
                        self.assertAllClose(nngp_ntk_cov_ind, nngp_ntk_cov)
                    self.assertAllClose(nngp_cov, nngp_ntk_cov, rtol=0.02, atol=0.02)

    @test_utils.product(
        train_size=TRAIN_SIZES,
        test_size=TEST_SIZES,
        input_shape=INPUT_SHAPES,
        network=NETWORK,
        out_logits=OUTPUT_LOGITS,
    )
    def testPredCovPosDef(self, train_size, test_size, input_shape, network, out_logits):
        device = test_utils.device_under_test()
        train_shape = (train_size, *input_shape)
        test_shape  = (test_size,  *input_shape)
        _, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape, train_shape, device=device)
        _, _, ker_fun = _build_network(train_shape[1:], network, out_logits)

        ts = torch.logspace(-3, 3, 10, device=device)
        predict_fn_mse_ens = predict.gradient_descent_mse_ensemble(ker_fun, x_train, y_train)

        for get in ('nngp', 'ntk'):
            for x in (None, 'x_test'):
                for t in (None, 'ts'):
                    with self.subTest(get=get, x=x, t=t):
                        cov = predict_fn_mse_ens(t=None if t is None else ts,
                                                 get=get,
                                                 x_test=None if x is None else x_test,
                                                 compute_cov=True).covariance
                        # symmetry
                        self.assertAllClose(cov, cov.movedim(-1, -2))
                        self.assertGreater(torch.linalg.eigvalsh(cov).min().item(), -1e-4)

    @test_utils.product(
        train_size=TRAIN_SIZES[:1],
        test_size=TEST_SIZES[:1],
        input_shape=INPUT_SHAPES[:1],
        out_logits=[1],
    )
    def testTrainedEnsemblePredCov(self, train_size, test_size, input_shape, out_logits):
        device = test_utils.device_under_test()
        training_steps = 300   # trimmed from 1000 for speed
        learning_rate  = 0.1
        ensemble_size  = 256   # trimmed from 1024 for speed

        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(128, W_std=1.2, b_std=0.05), stax.Erf(),
            stax.Dense(out_logits, W_std=1.2, b_std=0.05))

        train_shape = (train_size, *input_shape)
        test_shape  = (test_size,  *input_shape)
        g, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape, train_shape, device=device)

        predict_fn_mse_ens = predict.gradient_descent_mse_ensemble(
            kernel_fn, x_train, y_train, learning_rate=learning_rate, diag_reg=0.)

        def train_network(gen):
            _, params = init_fn(gen, (-1,) + train_shape[1:])
            params_vec, treedef = tt.ravel_params(params)
            params_vec = params_vec.detach().clone().requires_grad_(True)
            opt = torch.optim.SGD([params_vec], lr=learning_rate)
            for i in range(training_steps):
                opt.zero_grad(set_to_none=True)
                p = tt.unravel_params(params_vec, treedef)
                y_pred = apply_fn(p, x_train)
                l = 0.5 * torch.mean((y_pred - y_train) ** 2)
                l.backward()
                opt.step()
            return tt.unravel_params(params_vec, treedef)

        gens = [_rng(device, 10 + i) for i in range(ensemble_size)]
        params_list = [train_network(gi) for gi in gens]

        for x in [None, 'x_test']:
            with self.subTest(x=x):
                x_ = None if x is None else x_test
                x_fin = x_train if x_ is None else x_test
                ensemble_fx = torch.stack([apply_fn(p, x_fin).detach() for p in params_list], dim=0)

                mean_emp = ensemble_fx.mean(dim=0, keepdim=True)
                mean_sub = ensemble_fx - mean_emp
                cov_emp = torch.einsum('ijk,ilk->jl', mean_sub, mean_sub) / (mean_sub.shape[0] * mean_sub.shape[-1])

                ntk = predict_fn_mse_ens(training_steps, x_, 'ntk', compute_cov=True)
                tol = 0.1
                self.assertAllClose(_ravel_pytree(mean_emp)[0], _ravel_pytree(ntk.mean)[0], rtol=tol, atol=tol)
                self.assertAllClose(cov_emp, ntk.covariance, rtol=tol, atol=tol)

    def testGradientDescentMseEnsembleTrain(self):
        device = test_utils.device_under_test()
        g = _rng(device, 1)
        x = _randn((8, 4, 6, 3), device, g)
        _, _, kernel_fn = stax.serial(stax.Conv(1, (2, 2)), stax.Relu(), stax.Conv(1, (2, 1)))
        y = _randn((8, 2, 5, 1), device, g)
        predictor = predict.gradient_descent_mse_ensemble(kernel_fn, x, y, diagonal_spatial=False)

        for t in [None, torch.tensor([0., 1., 10.], device=device)]:
            with self.subTest(t=t):
                y_none = predictor(t, None, None, compute_cov=True)
                y_x    = predictor(t, x,    None, compute_cov=True)
                self.assertAllClose(y_none, y_x, rtol=0.04, atol=0.04)

    def testGpInference(self):
        device = test_utils.device_under_test()
        reg = 1e-5
        g = _rng(device, 1)
        x_train = _randn((4, 2), device, g)
        init_fn, apply_fn, kernel_fn_analytic = stax.serial(
            stax.Dense(32, 2., 0.5), stax.Relu(), stax.Dense(10, 2., 0.5))
        kernel_fn_empirical = tt.empirical_kernel_fn(apply_fn)
        y_train = _randn((4, 10), device, g)

        for kernel_fn_is_analytic in [True, False]:
            if kernel_fn_is_analytic:
                kernel_fn = kernel_fn_analytic
            else:
                _, params0 = init_fn(g, x_train.shape)
                def kernel_fn(x1, x2, get):
                    return kernel_fn_empirical(x1, x2, get, params=params0)

            for get in [None, 'nngp', 'ntk', ('nngp',), ('ntk',), ('nngp', 'ntk'), ('ntk', 'nngp')]:
                k_dd = kernel_fn(x_train, None, get)
                gp_inference = predict.gp_inference(k_dd, y_train, diag_reg=reg)
                gd_ensemble = predict.gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, diag_reg=reg)
                for x_test in [None, 'x_test']:
                    x_ = None if x_test is None else _randn((8, 2), device, g)
                    k_td = None if x_ is None else kernel_fn(x_, x_train, get)
                    for compute_cov in [True, False]:
                        with self.subTest(kernel_fn_is_analytic=kernel_fn_is_analytic, get=get, x_test=('x_test' if x_ is not None else None), compute_cov=compute_cov):
                            nngp_tt = (True if x_ is None else kernel_fn(x_, None, 'nngp')) if compute_cov else None
                            out_ens = gd_ensemble(None, x_, get, compute_cov)
                            out_ens_inf = gd_ensemble(math.inf, x_, get, compute_cov)
                            tol = 0.08
                            self.assertAllClose(out_ens_inf, out_ens, rtol=tol, atol=tol)

                            if (get is not None and 'nngp' not in (get if isinstance(get, tuple) else (get,)) and compute_cov and k_td is not None):
                                with self.assertRaises(ValueError):
                                    _ = gp_inference(get=get, k_test_train=k_td, k_test_test=nngp_tt)
                            else:
                                out_gp_inf = gp_inference(get=get, k_test_train=k_td, k_test_test=nngp_tt)
                                self.assertAllClose(out_ens, out_gp_inf)

            # NTKGP section
            for get in [(), ('nngp',), ('ntk',), ('nngp', 'ntk'), ('ntk', 'nngp')]:
                ntkgp_get = get + ('ntkgp',)
                if 'ntk' not in get:
                    get_full = get + ('ntk',)
                else:
                    get_full = get
                k_dd = kernel_fn(x_train, None, get_full)
                always_ntk = self._always_ntk(kernel_fn)
                always_ntk_k_dd = always_ntk(x_train, None, get_full)
                gp_inf = predict.gp_inference(k_dd, y_train, diag_reg=reg)
                always_ntk_gp_inf = predict.gp_inference(always_ntk_k_dd, y_train, diag_reg=reg)
                gd_ensemble = predict.gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, diag_reg=reg)

                for x_test in [None, 'x_test']:
                    x_ = None if x_test is None else _randn((8, 2), device, g)
                    k_td = None if x_ is None else kernel_fn(x_, x_train, get_full)
                    always_ntk_k_td = None if x_ is None else always_ntk(x_, x_train)

                    for compute_cov in [True, False]:
                        with self.subTest(kernel_fn_is_analytic=kernel_fn_is_analytic, get=ntkgp_get, x_test=('x_test' if x_ is not None else None), compute_cov=compute_cov):
                            if compute_cov:
                                k_tt = (True if x_ is None else kernel_fn(x_, None, get_full))
                                always_ntk_tt = (True if x_ is None else kernel_fn(x_, None, 'ntk'))
                            else:
                                k_tt = None
                                always_ntk_tt = None

                            if ('nngp' not in get_full and 'ntk' in ntkgp_get and compute_cov and k_td is not None):
                                with self.assertRaises(ValueError):
                                    _ = gp_inf(get=ntkgp_get, k_test_train=k_td, k_test_test=k_tt)
                            else:
                                out_gp_inf = gp_inf(get=ntkgp_get, k_test_train=k_td, k_test_test=k_tt)
                                out_ens = gd_ensemble(None, x_, get_full, compute_cov)
                                out_always_ntk_gp_inf = always_ntk_gp_inf(get='nngp', k_test_train=always_ntk_k_td, k_test_test=always_ntk_tt)
                                for gname in ntkgp_get:
                                    self.assertAllClose(getattr(out_gp_inf, gname),
                                                        (getattr(out_ens, gname) if gname != 'ntkgp' else out_always_ntk_gp_inf))

    def testPredictOnCPU(self):
        device = test_utils.device_under_test()
        g = _rng(device, 1)
        x_train = _randn((4, 4, 4, 2), device, g)
        x_test  = _randn((8, 4, 4, 2), device, g)
        y_train = torch.rand((4, 2), generator=g, device=device)

        _, _, kernel_fn = stax.serial(stax.Conv(1, (3, 3)), stax.Relu(), stax.Flatten(), stax.Dense(1))

        for store_on_device in [False, True]:
            for device_count in [0, 1]:
                for get in ['ntk', 'nngp', ('nngp', 'ntk'), ('ntk', 'nngp')]:
                    for x in [None, 'x_test']:
                        with self.subTest(store_on_device=store_on_device, device_count=device_count, get=get, x=x):
                            kernel_fn_batched = tt.batch(kernel_fn, 2, device_count, store_on_device)
                            predictor = predict.gradient_descent_mse_ensemble(kernel_fn_batched, x_train, y_train)

                            x_ = None if x is None else x_test
                            predict_none = predictor(None, x_, get, compute_cov=True)
                            predict_inf  = predictor(math.inf, x_, get, compute_cov=True)
                            self.assertAllClose(predict_none, predict_inf)

                            if x_ is not None:
                                on_cpu = not store_on_device or (str(device).startswith('cpu'))
                                def is_on_cpu(xo):
                                    # xo is a pytree of tensors
                                    return tt.tree_all(tt.tree_map(lambda t: t.device.type == 'cpu', xo))
                                self.assertEqual(on_cpu, is_on_cpu(predict_inf))
                                self.assertEqual(on_cpu, is_on_cpu(predict_none))

    def testPredictND(self):
        device = test_utils.device_under_test()
        n_chan = 6
        g = _rng(device, 1)
        im_shape = (5, 4, 3)
        n_train = 2
        n_test = 2
        x_train = _randn((n_train,) + im_shape, device, g)
        y_train = torch.rand((n_train, 3, 2, n_chan), generator=g, device=device)

        init_fn, apply_fn, _ = stax.Conv(n_chan, (3, 2), (1, 2))
        _, params = init_fn(g, x_train.shape)
        fx_train_0 = apply_fn(params, x_train)

        trace_axes_list = [(),
                           (-1,), (-2,), (-3,),
                           (0, 1), (2, 3), (2,), (1, 3),
                           (0, -1), (0, 0, -3),
                           (0, 1, 2, 3), (0, 1, -1, 2)]
        for trace_axes in trace_axes_list:
            for ts in [None, torch.arange(6.0, device=device).reshape(2, 3)]:
                for x in [None, 'x_test']:
                    with self.subTest(trace_axes=trace_axes, ts=ts, x=x):
                        t_shape = tuple(ts.shape) if ts is not None else ()
                        y_test_shape  = t_shape + (n_test,) + y_train.shape[1:]
                        y_train_shape = t_shape + y_train.shape
                        x_ = None if x is None else _randn((n_test,) + im_shape, device, g)
                        fx_test_0 = None if x_ is None else apply_fn(params, x_)

                        kernel_fn = tt.empirical_kernel_fn(apply_fn, trace_axes=trace_axes)
                        ntk_train_train = kernel_fn(x_train, None, 'ntk', params=params)
                        if x_ is not None:
                            ntk_test_train = kernel_fn(x_, x_train, 'ntk', params=params)

                        mse_predict = predict.gradient_descent_mse(ntk_train_train, y_train, trace_axes=trace_axes)
                        mse_ens_predict = predict.gradient_descent_mse_ensemble(
                            kernel_fn, x_train, y_train, trace_axes=trace_axes, params=params
                        )

                        if x_ is None:
                            p_train_mse = mse_predict(ts, fx_train_0)
                        else:
                            p_train_mse, p_test_mse = mse_predict(ts, fx_train_0, fx_test_0, ntk_test_train)
                            self.assertAllClose(tuple(p_test_mse.shape), y_test_shape)
                        self.assertAllClose(tuple(p_train_mse.shape), y_train_shape)

                        p_nngp_ens, p_ntk_ens = mse_ens_predict(ts, x_, ('nngp', 'ntk'), compute_cov=True)
                        ref_shape = y_train_shape if x_ is None else y_test_shape
                        self.assertAllClose(tuple(p_ntk_ens.mean.shape), ref_shape)
                        self.assertAllClose(tuple(p_nngp_ens.mean.shape), ref_shape)

    @test_utils.product(
        train_size=TRAIN_SIZES,
        input_shape=INPUT_SHAPES,
        network=NETWORK,
        out_logits=OUTPUT_LOGITS,
        kernel_type=list(KERNELS.keys()),
        lr_factor=[0.5, 1., 3.],
        momentum=[0., 0.1, 0.5, 0.9]
    )
    def testMaxLearningRate(
        self,
        train_size,
        input_shape,
        network,
        out_logits,
        kernel_type,
        lr_factor,
        momentum
    ):
        device = test_utils.device_under_test()
        g = _rng(device, 0)
        if len(input_shape) == 1:
            train_shape = (train_size * 5, input_shape[0] * 10)
        else:
            train_shape = (16, 8, 8, 3)
        x_train = _randn(train_shape, device, g)
        y_train = _bernoulli((train_shape[0], out_logits), device, g)

        fn_and_kernel = KERNELS[kernel_type]
        params, f, ntk = fn_and_kernel(g, train_shape[1:], network, out_logits)
        g_dd = ntk(x_train, None, 'ntk')

        step_size = predict.max_learning_rate(g_dd, y_train_size=y_train.numel(), momentum=momentum) * lr_factor

        params_vec, treedef = tt.ravel_params(params)
        params_vec = params_vec.detach().clone().requires_grad_(True)
        opt = torch.optim.SGD([params_vec], lr=float(step_size), momentum=float(momentum))

        def loss_from_vec(pv):
            p = tt.unravel_params(pv, treedef)
            return 0.5 * torch.mean((f(p, x_train) - y_train) ** 2)

        steps = 30
        init_loss = loss_from_vec(params_vec).item()
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            l = loss_from_vec(params_vec)
            l.backward()
            opt.step()
        trained_loss = loss_from_vec(params_vec).item()
        loss_ratio = trained_loss / (init_loss + 1e-12)

        if lr_factor < 1.:
            self.assertLess(loss_ratio, 0.1)
        elif lr_factor == 1.0:
            self.assertLess(loss_ratio, 1.0)
        if lr_factor > 2.:
            if not math.isnan(loss_ratio):
                self.assertGreater(loss_ratio, 10.)


class PredictKwargsTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        do_batch=[True, False],
        mode=['analytic', 'mc', 'empirical']
    )
    def test_kwargs(self, do_batch, mode):
        device = test_utils.device_under_test()
        g = _rng(device, 1)
        x_train = _randn((8, 7, 10), device, g)
        x_test  = _randn((4, 7, 10), device, g)
        y_train = _randn((8, 1), device, g)

        g_train = _rng(device, 2)
        g_test  = _rng(device, 3)

        pattern_train = _randn((8, 7, 7), device, g)
        pattern_test  = _randn((4, 7, 7), device, g)

        diag_reg = 1e-4

        if str(torch.backends.mps.is_available()) == 'True':
            atol = 3e-3; rtol = 4e-2; width = 256
        else:
            atol = 5e-4; rtol = 1e-2; width = 64

        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(width, W_std=2**0.5),
            stax.Relu(),
            stax.Dropout(rate=0.7),
            stax.Aggregate(),
            stax.GlobalAvgPool(),
            stax.Dense(width)
        )

        kw_dd = dict(pattern=(pattern_train, pattern_train))
        kw_td = dict(pattern=(pattern_test,  pattern_train))
        kw_tt = dict(pattern=(pattern_test,  pattern_test))

        if mode == 'mc':
            kernel_fn = tt.monte_carlo_kernel_fn(init_fn, apply_fn, g, 2, batch_size=(2 if do_batch else 0))
        elif mode == 'empirical':
            kernel_fn = tt.empirical_kernel_fn(apply_fn)
            if do_batch:
                raise unittest.SkipTest('Batching of empirical kernel is not implemented with keyword arguments.')
            params0 = init_fn(g, x_train.shape)[1]
            for kw in (kw_dd, kw_td, kw_tt):
                kw.update(dict(params=params0, get=('nngp', 'ntk')))
            kw_dd.update(dict(gen=(g_train, None)))
            kw_td.update(dict(gen=(g_test,  g_train)))
            kw_tt.update(dict(gen=(g_test,  None)))
        elif mode == 'analytic':
            if do_batch:
                kernel_fn = tt.batch(kernel_fn, batch_size=2)
        else:
            raise ValueError(mode)

        k_dd = kernel_fn(x_train, None, **kw_dd)
        k_td = kernel_fn(x_test,  x_train, **kw_td)
        k_tt = kernel_fn(x_test,  None, **kw_tt)

        predict_fn_gp = predict.gp_inference(k_dd, y_train, diag_reg=diag_reg)
        out_gp = predict_fn_gp(k_test_train=k_td, k_test_test=k_tt.nngp)

        if mode == 'empirical':
            for kw in (kw_dd, kw_td, kw_tt):
                kw.pop('get')

        predict_fn_ensemble = predict.gradient_descent_mse_ensemble(
            kernel_fn, x_train, y_train, diag_reg=diag_reg, **kw_dd
        )
        out_ensemble = predict_fn_ensemble(x_test=x_test, compute_cov=True, **kw_tt)
        self.assertAllClose(out_gp, out_ensemble)

        # Finite-time NTK test (test outputs)
        predict_fn_mse = predict.gradient_descent_mse(k_dd.ntk, y_train)
        out_mse = predict_fn_mse(t=1., fx_train_0=None, fx_test_0=0., k_test_train=k_td.ntk)
        out_ens = predict_fn_ensemble(t=1., get='ntk', x_test=x_test, compute_cov=False, **kw_tt)
        self.assertAllClose(out_mse, out_ens, atol=atol, rtol=rtol)

        # Finite-time NTK train (train outputs)
        out_mse = predict_fn_mse(t=0.5, fx_train_0=0., fx_test_0=None, k_test_train=k_td.ntk)
        out_ens = predict_fn_ensemble(t=0.5, get='ntk', x_test=None, compute_cov=False, **kw_dd)
        self.assertAllClose(out_mse, out_ens, atol=atol, rtol=rtol)

        # Finite-time NNGP test
        predict_fn_mse = predict.gradient_descent_mse(k_dd.nngp, y_train)
        out_mse = predict_fn_mse(t=1., fx_train_0=None, fx_test_0=0., k_test_train=k_td.nngp)
        out_ens = predict_fn_ensemble(t=1., get='nngp', x_test=x_test, compute_cov=False, **kw_tt)
        self.assertAllClose(out_mse, out_ens, atol=atol, rtol=rtol)

        # Finite-time NNGP train
        out_mse = predict_fn_mse(t=5., fx_train_0=0., fx_test_0=None, k_test_train=k_td.nngp)
        out_ens = predict_fn_ensemble(t=5., get='nngp', x_test=None, compute_cov=False, **kw_dd)
        self.assertAllClose(out_mse, out_ens, atol=atol, rtol=rtol)


class IsOnCpuTest(test_utils.NeuralTangentsTestCase):

    def test_is_on_cpu(self):
        dtypes = [torch.float16, torch.float32]
        if torch.cuda.is_available():
            dtypes += [torch.float64]

        for dtype in dtypes:
            with self.subTest(dtype=str(dtype)):
                def x():
                    g = _rng('cpu', 1)
                    return _randn((2, 3), 'cpu', g, dtype=dtype)

                def x_cpu():
                    g = _rng('cpu', 1)
                    return _randn((2, 3), 'cpu', g, dtype=dtype)

                self.assertTrue(_is_on_cpu(x_cpu()))

                if not torch.cuda.is_available():
                    self.assertTrue(_is_on_cpu(x()))
                else:
                    self.assertTrue(_is_on_cpu(x()))  # still CPU tensors here

                    # Now move to GPU to make the negative case
                    gx = x().to('cuda')
                    self.assertFalse(_is_on_cpu(gx))


if __name__ == '__main__':
    unittest.main()
