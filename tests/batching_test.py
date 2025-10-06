# PyTorch/torchtangents rewrite of tests for `neural_tangents/_src/batching.py`.

from functools import partial
import unittest
import torch

import torchtangents as tt
from torchtangents import stax
from torchtangents._src import batching
from torchtangents._src.empirical import _DEFAULT_TESTING_NTK_IMPLEMENTATION
from tests import test_utils


FLAT = 'FLAT'
POOLING = 'POOLING'
INTERMEDIATE_CONV = 'INTERMEDIATE_CONV'

# TODO: Add a pooling test when multiple inputs are supported in Conv + Pooling.
TRAIN_SIZES = [2, 4, 8]
TEST_SIZES = [2, 16]
INPUT_SHAPES = [(16,), (4, 4, 3)]
NETWORK = [FLAT, INTERMEDIATE_CONV, POOLING]
OUTPUT_LOGITS = [1, 2, 3]
CONVOLUTION_CHANNELS = 2
WIDTH = 2
RTOL = 1e-2

test_utils.update_test_tolerance(f64_tol=5e-5)


def _rng(device: str, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _randn(shape, dtype, device, gen):
    return torch.randn(*shape, dtype=dtype, device=device, generator=gen)


def _jit(f):
    # Placeholder for JIT: keep semantics identical (no scripting/tracing).
    return f


def _tree_map(fn, tree):
    if isinstance(tree, (list, tuple)):
        t = [ _tree_map(fn, t) for t in tree ]
        return type(tree)(t)
    if isinstance(tree, dict):
        return {k: _tree_map(fn, v) for k, v in tree.items()}
    return fn(tree)


def _build_network(input_shape, network, out_logits, use_dropout):
    dropout = stax.Dropout(0.9, mode='train') if use_dropout else stax.Identity()

    if len(input_shape) == 1:
        if network != FLAT:
            raise unittest.SkipTest('Not testing pooling on FCN inputs.')
        return stax.serial(
            stax.Dense(WIDTH, W_std=2.0, b_std=0.5),
            dropout,
            stax.Dense(out_logits, W_std=2.0, b_std=0.5),
        )

    if len(input_shape) == 3:
        if network == POOLING:
            return stax.serial(
                stax.Conv(CONVOLUTION_CHANNELS, (2, 2), W_std=2.0, b_std=0.05),
                stax.GlobalAvgPool(),
                dropout,
                stax.Dense(out_logits, W_std=2.0, b_std=0.5),
            )
        if network == FLAT:
            return stax.serial(
                stax.Conv(CONVOLUTION_CHANNELS, (2, 2), W_std=2.0, b_std=0.05),
                stax.Flatten(),
                dropout,
                stax.Dense(out_logits, W_std=2.0, b_std=0.5),
            )
        if network == INTERMEDIATE_CONV:
            return stax.Conv(CONVOLUTION_CHANNELS, (2, 2), W_std=2.0, b_std=0.05)
        raise ValueError(f'Unexpected network type found: {network}')

    raise ValueError('Expected flat or image test input.')


def _empirical_kernel(gen, input_shape, network, out_logits, use_dropout):
    init_fn, f, _ = _build_network(input_shape, network, out_logits, use_dropout)
    # Initialize params with a dummy batch dim = -1 convention like NT; here we materialize with 1.
    _, params = init_fn(gen, (-1,) + tuple(input_shape))
    # Mirror NT’s empirical_ntk_fn (implementation knob preserved).
    kernel_fn = _jit(tt.empirical_ntk_fn(
        f,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
    ))
    # Return a function that binds params and RNG (as `gen`).
    return partial(kernel_fn, params=params, gen=gen)


def _theoretical_kernel(unused_gen, input_shape, network, just_theta, use_dropout):
    _, _, kernel_fn = _build_network(input_shape, network, 1, use_dropout)

    @_jit
    def _ker(x1, x2=None):
        get_all = None
        return kernel_fn(x1, x2, 'ntk') if just_theta else kernel_fn(x1, x2, get_all)

    return _ker


KERNELS = {}
for o in OUTPUT_LOGITS:
    KERNELS[f'empirical_logits_{o}'] = partial(
        _empirical_kernel, out_logits=o, use_dropout=False
    )
KERNELS['theoretical'] = partial(_theoretical_kernel, just_theta=True, use_dropout=True)
KERNELS['theoretical_pytree'] = partial(_theoretical_kernel, just_theta=False, use_dropout=True)


def _test_kernel_against_batched(
    cls,
    kernel_fn,
    batched_kernel_fn,
    train,
    test,
    is_parallel_only=False
):
    g = kernel_fn(train, None)
    g_b = batched_kernel_fn(train, None)

    if is_parallel_only and hasattr(g_b, 'x1_is_x2'):
        # Align x1_is_x2 metadata in parallel case when x1==x2.
        g_b = g_b.replace(x1_is_x2=g.x1_is_x2)

    cls.assertAllClose(g, g_b)

    g = kernel_fn(train, test)
    g_b = batched_kernel_fn(train, test)
    if is_parallel_only and hasattr(g_b, 'x1_is_x2'):
        g_b = g_b.replace(x1_is_x2=g.x1_is_x2)
    cls.assertAllClose(g, g_b)


class BatchTest(test_utils.NeuralTangentsTestCase):

    @classmethod
    def _get_data_and_kernel_fn(
        cls,
        input_shape,
        kernel_type,
        network,
        test_size,
        train_size,
        **kwargs
    ):
        # Simulate 2 devices (like JAX pmap stubbing).
        test_utils.stub_out_pmap(batching, 2)
        device = test_utils.device_under_test()
        dtype = torch.float32
        gen = _rng(device, 0)
        gen_self = _rng(device, 1)
        gen_other = _rng(device, 2)

        data_self = _randn((train_size, *input_shape), dtype, device, gen_self)
        data_other = _randn((test_size, *input_shape), dtype, device, gen_other)

        kernel_fn_ctor = KERNELS[kernel_type]
        kernel_fn = kernel_fn_ctor(gen, input_shape, network, **kwargs)
        return data_other, data_self, kernel_fn

    @test_utils.product(
        train_size=TRAIN_SIZES,
        test_size=TEST_SIZES,
        input_shape=INPUT_SHAPES,
        network=NETWORK,
        kernel_type=list(KERNELS.keys()),
        batch_size=[2, 8],
    )
    def testSerial(self, train_size, test_size, input_shape, network, kernel_type, batch_size):
        data_other, data_self, kernel_fn = self._get_data_and_kernel_fn(
            input_shape, kernel_type, network, test_size, train_size
        )
        kernel_batched = batching._serial(kernel_fn, batch_size=batch_size)
        _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self, data_other)

    # We also exclude tests for dropout + parallel. It is not clear what is the best way to handle this case.
    @test_utils.product(
        train_size=TRAIN_SIZES,
        test_size=TEST_SIZES,
        input_shape=INPUT_SHAPES,
        network=NETWORK,
        kernel_type=list(KERNELS.keys()),
    )
    def testParallel(self, train_size, test_size, input_shape, network, kernel_type):
        data_other, data_self, kernel_fn = self._get_data_and_kernel_fn(
            input_shape, kernel_type, network, test_size, train_size, use_dropout=False
        )
        kernel_batched = batching._parallel(kernel_fn)
        _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self, data_other, True)

    @test_utils.product(
        train_size=TRAIN_SIZES,
        test_size=TEST_SIZES,
        input_shape=INPUT_SHAPES,
        network=NETWORK,
        kernel_type=list(KERNELS.keys()),
        batch_size=[2, 8],
    )
    def testComposition(self, train_size, test_size, input_shape, network, kernel_type, batch_size):
        data_other, data_self, kernel_fn = self._get_data_and_kernel_fn(
            input_shape, kernel_type, network, test_size, train_size
        )

        kernel_batched = batching._parallel(batching._serial(kernel_fn, batch_size=batch_size))
        _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self, data_other)

        kernel_batched = batching._serial(batching._parallel(kernel_fn), batch_size=batch_size)
        _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self, data_other)

    @test_utils.product(
        train_size=TRAIN_SIZES,
        test_size=TEST_SIZES,
        input_shape=INPUT_SHAPES,
        network=NETWORK,
        kernel_type=list(KERNELS.keys()),
        batch_size=[2, 8],
    )
    def testAutomatic(self, train_size, test_size, input_shape, network, kernel_type, batch_size):
        data_other, data_self, kernel_fn = self._get_data_and_kernel_fn(
            input_shape, kernel_type, network, test_size, train_size
        )

        kernel_batched = batching.batch(kernel_fn, batch_size=batch_size)
        _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self, data_other)

        kernel_batched = batching.batch(kernel_fn, batch_size=batch_size, store_on_device=False)
        _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self, data_other)

    def _test_analytic_kernel_composition(self, batching_fn):
        # FC block
        device = test_utils.device_under_test()
        dtype = torch.float32
        gen = _rng(device, 0)
        gen_self = _rng(device, 3)
        gen_other = _rng(device, 4)

        x_self = _randn((8, 2), dtype, device, gen_self)
        x_other = _randn((2, 2), dtype, device, gen_other)

        Block = stax.serial(stax.Dense(256), stax.Relu())

        _, _, ker_fn = Block
        ker_fn = batching_fn(ker_fn)

        _, _, composed_ker_fn = stax.serial(Block, Block)

        ker_out = ker_fn(x_self)
        composed_ker_out = composed_ker_fn(x_self)
        if batching_fn == batching._parallel and hasattr(composed_ker_out, 'x1_is_x2'):
            composed_ker_out = composed_ker_out.replace(x1_is_x2=ker_out.x1_is_x2)
        self.assertAllClose(ker_out, composed_ker_out)

        ker_out = ker_fn(x_self, x_other)
        composed_ker_out = composed_ker_fn(x_self, x_other)
        if batching_fn == batching._parallel and hasattr(composed_ker_out, 'x1_is_x2'):
            composed_ker_out = composed_ker_out.replace(x1_is_x2=ker_out.x1_is_x2)
        self.assertAllClose(ker_out, composed_ker_out)

        # Conv + pooling composition
        x_self = _randn((8, 4, 4, 3), dtype, device, gen_self)
        x_other = _randn((2, 4, 4, 3), dtype, device, gen_other)

        Block = stax.serial(stax.Conv(256, (2, 2)), stax.Relu())
        Readout = stax.serial(stax.GlobalAvgPool(), stax.Dense(10))

        block_ker_fn, readout_ker_fn = Block[2], Readout[2]
        _, _, composed_ker_fn = stax.serial(Block, Readout)
        block_ker_fn = batching_fn(block_ker_fn)
        readout_ker_fn = batching_fn(readout_ker_fn)

        ker_out = readout_ker_fn(block_ker_fn(x_self))
        composed_ker_out = composed_ker_fn(x_self)
        if batching_fn == batching._parallel and hasattr(composed_ker_out, 'x1_is_x2'):
            composed_ker_out = composed_ker_out.replace(x1_is_x2=ker_out.x1_is_x2)
        self.assertAllClose(ker_out, composed_ker_out)

        ker_out = readout_ker_fn(block_ker_fn(x_self, x_other))
        composed_ker_out = composed_ker_fn(x_self, x_other)
        if batching_fn == batching._parallel and hasattr(composed_ker_out, 'x1_is_x2'):
            composed_ker_out = composed_ker_out.replace(x1_is_x2=ker_out.x1_is_x2)
        self.assertAllClose(ker_out, composed_ker_out)

    @test_utils.product(store_on_device=[True, False], batch_size=[2, 8])
    def testAnalyticKernelComposeSerial(self, store_on_device, batch_size):
        self._test_analytic_kernel_composition(
            partial(batching._serial, batch_size=batch_size, store_on_device=store_on_device)
        )

    def testAnalyticKernelComposeParallel(self):
        test_utils.stub_out_pmap(batching, 2)
        self._test_analytic_kernel_composition(batching._parallel)

    @test_utils.product(store_on_device=[True, False], batch_size=[2, 8])
    def testAnalyticKernelComposeAutomatic(self, store_on_device, batch_size):
        test_utils.stub_out_pmap(batching, 2)
        self._test_analytic_kernel_composition(
            partial(batching.batch, batch_size=batch_size, store_on_device=store_on_device)
        )

    @test_utils.product(same_inputs=[True, False])
    def test_parallel_in_out(self, same_inputs):
        test_utils.stub_out_pmap(batching, 2)
        device = test_utils.device_under_test()
        dtype = torch.float32
        gen = _rng(device, 0)
        g1 = _rng(device, 1)
        g2 = _rng(device, 2)

        x1_1 = _randn((3, 4, 1), dtype, device, g1)
        x1_2 = _randn((3, 4, 1), dtype, device, g1)
        x1_3 = _randn((3, 4, 1), dtype, device, g1)
        x1 = (x1_1, (x1_2, x1_3))

        if same_inputs:
            x2 = None
        else:
            x2_1 = _randn((3, 8, 1), dtype, device, g2)
            x2_2 = _randn((3, 8, 1), dtype, device, g2)
            x2_3 = _randn((3, 8, 1), dtype, device, g2)
            x2 = (x2_1, (x2_2, x2_3))

        N = WIDTH

        def net(N_out):
            return stax.parallel(
                stax.Dense(N_out),
                stax.parallel(stax.Dense(N_out + 1), stax.Dense(N_out + 2)),
            )

        readin = net(N)
        readout = net(1)

        K_readin_fn = _jit(readin[2])
        K_readout_fn = _jit(partial(readout[2], get='nngp'))

        batch_K_readin_fn = batching.batch(K_readin_fn, 2)
        batch_K_readout_fn = batching.batch(K_readout_fn, 2)

        test_utils.assert_close_matrices(
            self,
            K_readout_fn(K_readin_fn(x1, x2)),
            batch_K_readout_fn(batch_K_readin_fn(x1, x2)),
            RTOL,
        )

        # Check Both (NNGP, NTK)
        K_readin_fn = _jit(readin[2])
        K_readout_fn = _jit(partial(readout[2], get=('nngp', 'ntk')))

        batch_K_readin_fn = batching.batch(K_readin_fn, 2)
        batch_K_readout_fn = batching.batch(K_readout_fn, 2)

        test_utils.assert_close_matrices(
            self,
            K_readout_fn(K_readin_fn(x1, x2)),
            batch_K_readout_fn(batch_K_readin_fn(x1, x2)),
            RTOL,
        )

    @test_utils.product(same_inputs=[True, False])
    def test_parallel_in_out_empirical(self, same_inputs):
        test_utils.stub_out_pmap(batching, 2)
        device = test_utils.device_under_test()
        dtype = torch.float32
        gen = _rng(device, 0)
        g1 = _rng(device, 1)
        g2 = _rng(device, 2)
        gnet = _rng(device, 3)

        x1_1 = _randn((3, 4, 1), dtype, device, g1)
        x1_2 = _randn((3, 4, 1), dtype, device, g1)
        x1_3 = _randn((3, 4, 1), dtype, device, g1)
        x1 = (x1_1, (x1_2, x1_3))

        if same_inputs:
            x2 = None
        else:
            x2_1 = _randn((3, 8, 1), dtype, device, g2)
            x2_2 = _randn((3, 8, 1), dtype, device, g2)
            x2_3 = _randn((3, 8, 1), dtype, device, g2)
            x2 = (x2_1, (x2_2, x2_3))

        def net(N_out):
            return stax.parallel(
                stax.Dense(N_out),
                stax.parallel(stax.Dense(N_out + 1), stax.Dense(N_out + 2)),
            )

        # NNGP
        init_fn, apply_fn, _ = net(WIDTH)
        _, params = init_fn(gnet, ((-1, 1), ((-1, 1), (-1, 1))))
        kernel_fn = _jit(tt.empirical_nngp_fn(apply_fn))
        batch_kernel_fn = _jit(batching.batch(kernel_fn, 2))

        test_utils.assert_close_matrices(
            self,
            kernel_fn(x1, x2, params=params, gen=gen),
            batch_kernel_fn(x1, x2, params=params, gen=gen),
            RTOL,
        )

        # NTK
        init_fn, apply_fn, _ = stax.serial(net(WIDTH), net(1))
        _, params = init_fn(gnet, ((-1, 1), ((-1, 1), (-1, 1))))
        kernel_fn = _jit(tt.empirical_ntk_fn(
            apply_fn,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
        ))
        batch_kernel_fn = _jit(batching.batch(kernel_fn, 2))

        test_utils.assert_close_matrices(
            self,
            kernel_fn(x1, x2, params=params, gen=gen),
            batch_kernel_fn(x1, x2, params=params, gen=gen),
            RTOL,
        )

    @test_utils.product(
        same_inputs=[True, False],
        device_count=[-1, 0, 1, 2],
        trace_axes=[(-1,), (1, -1), ()],
        diagonal_axes=[(1,), (), (1, -1)],
    )
    def test_empirical_ntk_diagonal_outputs(
        self, same_inputs, device_count, trace_axes, diagonal_axes
    ):
        if any(t in diagonal_axes for t in trace_axes):
            raise unittest.SkipTest('Overlapping axes.')

        test_utils.stub_out_pmap(batching, 2)
        device = test_utils.device_under_test()
        dtype = torch.float32
        g1 = _rng(device, 1)
        g2 = _rng(device, 2)
        gnet = _rng(device, 3)

        init_fn, apply_fn, _ = stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(3))

        test_x1 = _randn((12, 4, 4), dtype, device, g1)
        test_x2 = None
        if same_inputs:
            test_x2 = _randn((9, 4, 4), dtype, device, g2)

        kernel_fn = tt.empirical_ntk_fn(
            apply_fn,
            trace_axes=trace_axes,
            diagonal_axes=diagonal_axes,
            vmap_axes=0,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        )

        _, params = init_fn(gnet, test_x1.shape)

        true_kernel = kernel_fn(test_x1, test_x2, params=params, gen=g1)
        batched_fn = batching.batch(kernel_fn, device_count=device_count, batch_size=3)
        batch_kernel = batched_fn(test_x1, test_x2, params=params, gen=g1)
        self.assertAllClose(true_kernel, batch_kernel)


if __name__ == '__main__':
    unittest.main()
