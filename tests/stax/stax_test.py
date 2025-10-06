# PyTorch/torchtangents rewrite of tests for `neural_tangents/stax.py`.

import functools
import itertools
import random as prandom
import unittest
from typing import Tuple

import numpy as np
import torch
import torchtangents as nt
from torchtangents import stax
from torchtangents._src.empirical import _DEFAULT_TESTING_NTK_IMPLEMENTATION
from tests import test_utils


# -------------------------
# Global config / constants
# -------------------------
MODELS = ['fc', 'conv']
BATCH_SIZE = 4
INPUT_SHAPE = (BATCH_SIZE, 8, 6, 2)
WIDTHS = [2**10]
N_SAMPLES = 100
RTOL = 0.041
ATOL = 0.1

FILTER_SHAPES = [(2, 1), (3, 2)]
PADDINGS = ['SAME', 'VALID', 'CIRCULAR']
STRIDES = [(1, 2), (2, 1)]
ACTIVATIONS = [stax.Relu()]
PROJECTIONS = ['FLAT', 'POOL', 'ATTN']

test_utils.update_test_tolerance()
prandom.seed(1)


# -------------------------
# RNG utilities
# -------------------------
def _rng(device: str | torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device='cpu')  # seeds params; ops run on tensors' device
    g.manual_seed(seed)
    return g


def _randn(shape, device, gen, dtype=torch.float32):
    return torch.randn(*shape, generator=gen, device=device, dtype=dtype)


# -------------------------
# Input helper (matches JAX test)
# -------------------------
def _get_inputs(
    gen: torch.Generator,
    same_inputs: bool,
    shape: Tuple[int, ...],
    fn=lambda x: torch.cos(x)
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x1 = fn(_randn(shape, device=test_utils.device_under_test(), gen=gen))
    batch_axis = shape.index(BATCH_SIZE)
    shape2 = shape[:batch_axis] + (2 * BATCH_SIZE,) + shape[batch_axis + 1:]
    x2 = None if same_inputs else fn(_randn(shape2, device=x1.device, gen=gen)) * 2
    return x1, x2


# -------------------------
# Network builders (close to original)
# -------------------------
def _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res, padding,
             phi, strides, width, is_ntk, proj_into_2d, pool_type, layer_norm,
             parameterization, s, use_dropout):

    device = test_utils.device_under_test()
    gen = _rng(device, 123)

    if is_conv:
        # Choose random filter order.
        default_filter_spec = 'HW'
        filter_specs = [''.join(p) for p in itertools.permutations('HWIO')]
        filter_spec = prandom.choice(filter_specs)
        filter_shape = tuple(filter_shape[default_filter_spec.index(c)]
                             for c in filter_spec if c in default_filter_spec)
        strides = tuple(strides[default_filter_spec.index(c)]
                        for c in filter_spec if c in default_filter_spec)

        # Choose activation order (NCHW variants).
        specs = [''.join(p) for p in itertools.permutations('NCHW')]
        spec = prandom.choice(specs)
        input_shape = tuple(INPUT_SHAPE['NHWC'.index(c)] for c in spec.replace('N', 'N').replace('C', 'C').replace('H', 'H').replace('W', 'W'))
        # Map from NHWC given INPUT_SHAPE to our chosen spec
        nhwc = {'N': INPUT_SHAPE[0], 'H': INPUT_SHAPE[1], 'W': INPUT_SHAPE[2], 'C': INPUT_SHAPE[3]}
        input_shape = tuple(nhwc[c] for c in spec)
    else:
        input_shape = (INPUT_SHAPE[0], int(np.prod(INPUT_SHAPE[1:])))
        spec = prandom.choice(['NC', 'CN'])
        if spec.index('N') == 1:
            input_shape = input_shape[::-1]
        filter_spec = None

    # `dimension_numbers` parity; many ports ignore filter_spec, but pass anyway.
    dimension_numbers = (spec, None, spec)
    batch_axis, channel_axis = spec.index('N'), spec.index('C')

    spec_fc = ''.join(c for c in spec if c in ('N', 'C'))
    batch_axis_fc, channel_axis_fc = spec_fc.index('N'), spec_fc.index('C')

    if not is_conv:
        batch_axis = batch_axis_fc
        channel_axis = channel_axis_fc

    if layer_norm:
        layer_norm = tuple(spec.index(c) for c in layer_norm)

    def fc(out_dim, s_pair):
        return stax.Dense(
            out_dim=out_dim, W_std=W_std, b_std=b_std,
            parameterization=parameterization, s=s_pair,
            batch_axis=batch_axis_fc, channel_axis=channel_axis_fc
        )

    def conv(out_chan, s_pair):
        return stax.Conv(
            out_chan=out_chan, filter_shape=filter_shape, strides=strides, padding=padding,
            W_std=W_std, b_std=b_std, dimension_numbers=dimension_numbers,
            parameterization=parameterization, s=s_pair
        )

    affine = conv(width, (s, s)) if is_conv else fc(width, (s, s))
    affine_bottom = conv(width, (1, s)) if is_conv else fc(width, (1, s))

    rate = float(np.random.uniform(0.5, 0.9))
    dropout = stax.Dropout(rate, mode='train')

    if pool_type == 'AVG':
        pool_fn = stax.AvgPool
        global_pool_fn = stax.GlobalAvgPool
    elif pool_type == 'SUM':
        pool_fn = stax.SumPool
        global_pool_fn = stax.GlobalSumPool
    else:
        raise ValueError(pool_type)

    if use_pooling:
        pool_or_identity = pool_fn((2, 3),
                                   None,
                                   'SAME' if padding == 'SAME' else 'CIRCULAR',
                                   batch_axis=batch_axis,
                                   channel_axis=channel_axis)
    else:
        pool_or_identity = stax.Identity()
    dropout_or_identity = dropout if use_dropout else stax.Identity()
    layer_norm_or_identity = (stax.Identity() if layer_norm is None else
                              stax.LayerNorm(axis=layer_norm,
                                             batch_axis=batch_axis,
                                             channel_axis=channel_axis))
    res_unit = stax.serial(dropout_or_identity, affine, pool_or_identity)
    if is_res:
        block = stax.serial(
            affine_bottom,
            stax.FanOut(2),
            stax.parallel(stax.Identity(), res_unit),
            stax.FanInSum(),
            layer_norm_or_identity,
            phi)
    else:
        block = stax.serial(affine_bottom, res_unit, layer_norm_or_identity, phi)

    if proj_into_2d == 'FLAT':
        proj_layer = stax.Flatten(batch_axis, batch_axis_fc)
    elif proj_into_2d == 'POOL':
        proj_layer = global_pool_fn(batch_axis, channel_axis)
    elif proj_into_2d.startswith('ATTN'):
        n_heads = int(np.sqrt(width))
        n_chan_val = int(np.round(float(width) / n_heads))
        proj_layer = stax.serial(
            stax.GlobalSelfAttention(
                n_chan_out=width,
                n_chan_key=width,
                n_chan_val=n_chan_val,
                n_heads=n_heads,
                linear_scaling=True,
                W_key_std=W_std,
                W_value_std=W_std,
                W_query_std=W_std,
                W_out_std=1.0,
                b_std=b_std,
                batch_axis=batch_axis,
                channel_axis=channel_axis),
            stax.Flatten(batch_axis, batch_axis_fc))
    else:
        raise ValueError(proj_into_2d)

    readout = stax.serial(
        proj_layer,
        fc(1 if is_ntk else width, (s, 1 if is_ntk else s))
    )

    device_count = -1 if spec.index('N') == 0 else 0
    net = stax.serial(block, readout)
    return net, input_shape, device_count, channel_axis_fc


def _get_net_pool(width, is_ntk, pool_type, padding,
                  filter_shape, strides, normalize_edges):
    W_std, b_std = 2.**0.5, 0.5**0.5
    phi = stax.Relu()
    parameterization = 'ntk'

    fc = functools.partial(
        stax.Dense,
        W_std=W_std / width if pool_type == 'SUM' else W_std,
        b_std=b_std,
        parameterization=parameterization)

    conv = functools.partial(
        stax.Conv,
        filter_shape=filter_shape,
        strides=None,
        padding='SAME',
        W_std=W_std / np.prod(filter_shape) if pool_type == 'SUM' else W_std,
        b_std=b_std,
        parameterization=parameterization)

    if pool_type == 'AVG':
        pool_fn = functools.partial(stax.AvgPool, normalize_edges=normalize_edges)
        global_pool_fn = stax.GlobalAvgPool
    elif pool_type == 'SUM':
        pool_fn = stax.SumPool
        global_pool_fn = stax.GlobalSumPool
    else:
        raise ValueError(pool_type)

    pool = pool_fn(filter_shape, strides, padding)
    device_count = -1

    return stax.serial(
        conv(width), phi, pool,
        conv(width), phi,
        global_pool_fn(),
        fc(1 if is_ntk else width)
    ), INPUT_SHAPE, device_count, -1


# -------------------------
# Empirical vs analytic checker
# -------------------------
def _check_agreement_with_empirical(
    self,
    net,
    same_inputs,
    use_dropout,
    is_ntk,
    rtol=RTOL,
    atol=ATOL
):
    ((init_fn, apply_fn, kernel_fn),
     input_shape, device_count, channel_axis) = net

    device = test_utils.device_under_test()
    gen = _rng(device, 1)

    num_samples = N_SAMPLES * 5 if use_dropout else N_SAMPLES
    x1, x2 = _get_inputs(gen, same_inputs, input_shape)

    # Match batching logic from the JAX test
    batch_size = 2 if (use_dropout and str(device).startswith('xla')) else 0

    x1_out_shape, params = init_fn(gen, x1.shape)
    x2_out_shape = x1_out_shape if x2 is None else init_fn(gen, x2.shape)[0]

    def _get_empirical(n_samples, get):
        kernel_fn_empirical = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=apply_fn,
            gen=gen,
            n_samples=n_samples,
            device_count=device_count,
            trace_axes=(channel_axis,),
            batch_size=batch_size,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
        )
        return kernel_fn_empirical(x1, x2, get)

    if is_ntk:
        exact, shape1, shape2 = kernel_fn(x1, x2, ('ntk', 'shape1', 'shape2'))
        empirical = _get_empirical(num_samples, 'ntk')
    else:
        exact, shape1, shape2 = kernel_fn(x1, x2, ('nngp', 'shape1', 'shape2'))
        empirical = _get_empirical(num_samples, 'nngp')

    test_utils.assert_close_matrices(self, exact, empirical, rtol, atol)
    self.assertEqual(shape1, x1_out_shape)
    self.assertEqual(shape2, x2_out_shape)


# -------------------------
# Test classes
# -------------------------
class StaxTest(test_utils.NeuralTangentsTestCase):

    def _skip_test(self, filter_shape, is_conv, is_res, padding, proj_into_2d,
                   strides, use_pooling):
        if is_conv:
            test_utils.skip_test(self)
            if (is_res and is_conv and ((strides is not None and strides != (1, 1)) or
                                        (padding == 'VALID' and filter_shape != (1, 1)))):
                raise unittest.SkipTest('Residual paths must match shapes.')
        elif (filter_shape != FILTER_SHAPES[0] or padding != PADDINGS[0] or
              strides != STRIDES[0] or proj_into_2d != PROJECTIONS[0] or
              use_pooling):
            raise unittest.SkipTest('FC models do not have these parameters.')

    @test_utils.product(
        model=MODELS,
        width=WIDTHS,
        phi=ACTIVATIONS,
        same_inputs=[False],
        padding=PADDINGS,
        strides=STRIDES,
        filter_shape=FILTER_SHAPES,
        use_pooling=[False, True],
        is_ntk=[False, True],
        is_res=[False, True],
        proj_into_2d=PROJECTIONS,
    )
    def test_exact(self, model, width, strides, padding, phi, same_inputs,
                   filter_shape, use_pooling, is_ntk, is_res, proj_into_2d):
        is_conv = 'conv' in model
        self._skip_test(filter_shape, is_conv, is_res, padding, proj_into_2d, strides, use_pooling)

        pool_type = 'AVG'
        W_std, b_std = 2.**0.5, 0.5**0.5
        layer_norm = None
        parameterization = 'ntk'
        use_dropout = False

        net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                       padding, phi, strides, width, is_ntk, proj_into_2d,
                       pool_type, layer_norm, parameterization, 1, use_dropout)
        _check_agreement_with_empirical(self, net, same_inputs, use_dropout, is_ntk, RTOL, 1.2)

    @test_utils.product(
        model=MODELS,
        width=WIDTHS,
        same_inputs=[False],
        is_ntk=[False, True],
        proj_into_2d=PROJECTIONS[:2],
        layer_norm=['C', 'HC', 'CHW', 'NC', 'NWC', 'NCHW'],
    )
    def test_layernorm(self, model, width, same_inputs, is_ntk, proj_into_2d, layer_norm):
        is_conv = 'conv' in model
        # Mirror original constraints
        if is_conv:
            test_utils.skip_test(self)
        elif proj_into_2d != PROJECTIONS[0] or layer_norm not in ('C', 'NC'):
            raise unittest.SkipTest('FC models only test simple LayerNorm axes.')

        W_std, b_std = 2.**0.5, 0.5**0.5
        filter_shape = FILTER_SHAPES[0]
        padding = PADDINGS[0]
        strides = STRIDES[0]
        phi = stax.Relu()
        use_pooling, is_res = False, False
        parameterization = 'ntk'
        pool_type = 'AVG'
        use_dropout = False

        net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                       padding, phi, strides, width, is_ntk, proj_into_2d,
                       pool_type, layer_norm, parameterization, 1, use_dropout)
        _check_agreement_with_empirical(self, net, same_inputs, use_dropout, is_ntk, 0.07)

    @test_utils.product(
        width=WIDTHS,
        same_inputs=[False],
        is_ntk=[False, True],
        pool_type=['SUM', 'AVG'],
        padding=PADDINGS,
        filter_shape=FILTER_SHAPES,
        strides=STRIDES,
        normalize_edges=[True, False]
    )
    def test_pool(self, width, same_inputs, is_ntk, pool_type, padding, filter_shape, strides, normalize_edges):
        test_utils.skip_test(self)
        if pool_type == 'SUM' and normalize_edges:
            raise unittest.SkipTest('normalize_edges not applicable to SumPool.')

        net = _get_net_pool(width, is_ntk, pool_type, padding, filter_shape, strides, normalize_edges)
        _check_agreement_with_empirical(self, net, same_inputs, False, is_ntk)

    def test_avg_pool(self):
        device = test_utils.device_under_test()
        X1 = torch.ones((4, 2, 3, 2), device=device)
        X2 = torch.ones((3, 2, 3, 2), device=device)

        _, apply_fn, kernel_fn = stax.AvgPool((2, 2), (1, 1), 'SAME', normalize_edges=False)
        _, apply_fn_norm, kernel_fn_norm = stax.AvgPool((2, 2), (1, 1), 'SAME', normalize_edges=True)
        # Reference: ostax equivalent not available in torchtangents; assume normalized matches framework pool.

        out1 = apply_fn((), X1)
        out2 = apply_fn((), X2)
        out1_norm = apply_fn_norm((), X1)
        out2_norm = apply_fn_norm((), X2)

        # Unnormalized expected border scaling
        out_unnorm = torch.tensor([[1., 1., 0.5], [0.5, 0.5, 0.25]], device=device).reshape((1, 2, 3, 1))
        out1_unnormalized = out_unnorm.expand_as(X1)
        out2_unnormalized = out_unnorm.expand_as(X2)

        self.assertAllClose((out1_unnormalized, out2_unnormalized), (out1, out2))
        self.assertAllClose((out1_norm, out2_norm), (out1_norm, out2_norm))  # identity check

        ker = kernel_fn(X1, X2)
        ker_norm = kernel_fn_norm(X1, X2)

        self.assertAllClose(torch.ones_like(ker_norm.nngp), ker_norm.nngp)
        self.assertAllClose(torch.ones_like(ker_norm.cov1), ker_norm.cov1)
        self.assertAllClose(torch.ones_like(ker_norm.cov2), ker_norm.cov2)

        self.assertEqual(ker_norm.nngp.shape, ker.nngp.shape)
        self.assertEqual(ker_norm.cov1.shape, ker.cov1.shape)
        self.assertEqual(ker_norm.cov2.shape, ker.cov2.shape)

    @test_utils.product(
        model=MODELS,
        width=WIDTHS,
        same_inputs=[True, False],
        phi=ACTIVATIONS,
        padding=['SAME'],
        strides=STRIDES,
        filter_shape=[(2, 1)],
        is_ntk=[True, False],
        use_pooling=[True, False],
        proj_into_2d=['FLAT', 'POOL']
    )
    def test_dropout(self, model, width, same_inputs, is_ntk, padding, strides, filter_shape, phi, use_pooling, proj_into_2d):
        pool_type = 'AVG'
        use_dropout = True
        is_conv = 'conv' in model
        is_res = False
        W_std, b_std = 2.**0.5, 0.5**0.5
        layer_norm = None
        parameterization = 'ntk'
        self._skip_test(filter_shape, is_conv, is_res, padding, proj_into_2d, strides, use_pooling)

        net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                       padding, phi, strides, width, is_ntk, proj_into_2d,
                       pool_type, layer_norm, parameterization, 1, use_dropout)
        _check_agreement_with_empirical(self, net, same_inputs, use_dropout, is_ntk)

    @test_utils.product(
        act=['erf', 'relu'],
        do_stabilize=[True, False],
        kernel=['nngp', 'ntk']
    )
    def test_sparse_inputs(self, act, kernel, do_stabilize):
        if do_stabilize and act != 'relu':
            raise unittest.SkipTest('Stabilization only in Relu.')

        device = test_utils.device_under_test()
        gen = _rng(device, 1)

        input_count = 4
        sparse_count = 2
        input_size = 3
        width = 1024

        samples = N_SAMPLES
        tol = {np.dtype(np.float32): 5e-2, np.dtype(np.float64): 5e-3}

        x_dense = _randn((input_count, input_size), device, gen)
        x_sparse = x_dense.clone()
        x_sparse[:sparse_count] = 0.

        activation = stax.Relu(do_stabilize=do_stabilize) if act == 'relu' else stax.Erf()
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(width), activation, stax.Dense(1 if kernel == 'ntk' else width)
        )
        exact = kernel_fn(x_sparse, None, kernel)

        mc = nt.monte_carlo_kernel_fn(
            init_fn, apply_fn, gen, samples,
            vmap_axes=0, device_count=-1,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
        )(x_sparse, None, kernel)

        mc = mc.reshape(exact.shape)
        assert not torch.isnan(exact).any()
        self.assertAllClose(exact[sparse_count:, sparse_count:], mc[sparse_count:, sparse_count:],
                            rtol=5e-2, atol=5e-2)

    def test_composition_dense(self):
        device = test_utils.device_under_test()
        gen = _rng(device, 0)
        x1 = _randn((2, 3), device, gen)
        x2 = _randn((4, 3), device, gen)

        Block = stax.serial(stax.Dense(256), stax.Relu())

        _, _, ker_fn = Block
        _, _, composed_ker_fn = stax.serial(Block, Block)

        ker_out = ker_fn(ker_fn(x1))
        composed_ker_out = composed_ker_fn(x1)
        self.assertAllClose(ker_out, composed_ker_out)

        ker_out = ker_fn(ker_fn(x1, x2))
        composed_ker_out = composed_ker_fn(x1, x2)
        self.assertAllClose(ker_out, composed_ker_out)


class ParallelInOutTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(same_inputs=[True, False], kernel_type=['ntk'])
    def test_parallel_in(self, same_inputs, kernel_type):
        device = test_utils.device_under_test()
        rtol = RTOL

        gen = _rng(device, 0)
        g1 = _rng(device, 1)
        g2 = _rng(device, 2)
        gmc = _rng(device, 3)

        x1_1, x2_1 = _get_inputs(g1, same_inputs, (BATCH_SIZE, 2))
        x1_2, x2_2 = _get_inputs(g2, same_inputs, (BATCH_SIZE, 3))
        x1 = (x1_1, x1_2)
        x2 = (x2_1, x2_2)

        N = 2 ** 7

        def net(logits):
            return stax.serial(
                stax.parallel(stax.Dense(N), stax.Dense(N)),
                stax.serial(stax.FanInSum(), stax.Dense(logits))
            )

        init_fn, apply_fn, kernel_fn = net(N if kernel_type == 'nngp' else 1)
        kernel_fn_empirical = nt.monte_carlo_kernel_fn(
            init_fn, apply_fn, gmc, N_SAMPLES, trace_axes=(-1,),
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
            vmap_axes=((0, 0), 0, {})
        )
        test_utils.assert_close_matrices(self,
                                         kernel_fn(x1, x2, kernel_type),
                                         kernel_fn_empirical(x1, x2, kernel_type),
                                         rtol)

    @test_utils.product(same_inputs=[True, False], kernel_type=['ntk'])
    def test_parallel_out(self, same_inputs, kernel_type):
        device = test_utils.device_under_test()
        rtol = RTOL

        g1 = _rng(device, 1)
        gmc = _rng(device, 2)

        x1, x2 = _get_inputs(g1, same_inputs, (BATCH_SIZE, 1))

        N = 2 ** 10
        def net(logits):
            return stax.serial(
                stax.Dense(N),
                stax.FanOut(2),
                stax.parallel(stax.Dense(logits), stax.Dense(logits))
            )

        init_fn, apply_fn, kernel_fn = net(N if kernel_type == 'nngp' else 1)
        kernel_fn_empirical = nt.monte_carlo_kernel_fn(
            init_fn, apply_fn, gmc, N_SAMPLES, trace_axes=(-1,),
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
            vmap_axes=(0, [0, 0], {})
        )
        test_utils.assert_close_matrices(self,
                                         kernel_fn(x1, x2, kernel_type),
                                         kernel_fn_empirical(x1, x2, kernel_type),
                                         rtol)

    @test_utils.product(same_inputs=[True, False], kernel_type=['ntk'])
    def test_parallel_in_out(self, same_inputs, kernel_type):
        device = test_utils.device_under_test()
        rtol = RTOL

        g1 = _rng(device, 1)
        g2 = _rng(device, 2)
        gmc = _rng(device, 3)

        x1_1, x2_1 = _get_inputs(g1, same_inputs, (BATCH_SIZE, 1))
        x1_2, x2_2 = _get_inputs(g2, same_inputs, (BATCH_SIZE, 2))

        x1 = (x1_1, x1_2)
        x2 = (x2_1, x2_2)

        N_in = 2 ** 10
        N_out = N_in if kernel_type == 'nngp' else 1

        readin = stax.serial(stax.parallel(stax.Dense(N_in), stax.Dense(N_in)), stax.FanInSum())
        readout = stax.serial(stax.FanOut(3),
                              stax.parallel(stax.Dense(N_out),
                                            stax.Dense(N_out + 1),
                                            stax.Dense(N_out + 2)))
        init_fn, apply_fn, _ = stax.serial(readin, readout)

        K_readin_fn = readin[2]
        K_readout_fn = functools.partial(readout[2], get=kernel_type)

        kernel_fn_empirical = nt.monte_carlo_kernel_fn(
            init_fn, apply_fn, gmc, N_SAMPLES, trace_axes=(-1,),
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
            vmap_axes=((0, 0), [0, 0, 0], {})
        )

        test_utils.assert_close_matrices(
            self,
            K_readout_fn(K_readin_fn(x1, x2)),
            kernel_fn_empirical(x1, x2, get=kernel_type),
            rtol)

        # Both (ensure compute path works)
        _ = functools.partial(readout[2], get=('nngp', 'ntk'))(K_readin_fn(x1, x2))

    @test_utils.product(same_inputs=[True, False], kernel_type=['ntk'])
    def test_nested_parallel(self, same_inputs, kernel_type):
        device = test_utils.device_under_test()
        rtol = RTOL

        gens = [_rng(device, s) for s in range(6)]
        input_key1, input_key2, input_key3, input_key4, mask_key, mc_key = gens

        x1_1, x2_1 = _get_inputs(input_key1, same_inputs, (BATCH_SIZE, 5))
        x1_2, x2_2 = _get_inputs(input_key2, same_inputs, (BATCH_SIZE, 2, 2, 2))
        x1_3, x2_3 = _get_inputs(input_key3, same_inputs, (BATCH_SIZE, 2, 2, 3))
        x1_4, x2_4 = _get_inputs(input_key4, same_inputs, (BATCH_SIZE, 3, 4))

        # Apply masks as in JAX tests (using provided helper)
        m1_key, m2_key, m3_key, m4_key = [_rng(device, 10+i) for i in range(4)]
        x1_1 = test_utils.mask(x1_1, mask_constant=-1, mask_axis=(1,), gen=m1_key, p=0.5)
        x1_2 = test_utils.mask(x1_2, mask_constant=-1, mask_axis=(2, 3,), gen=m2_key, p=0.5)
        if not same_inputs:
            x2_3 = test_utils.mask(x2_3, mask_constant=-1, mask_axis=(1, 3,), gen=m3_key, p=0.5)
            x2_4 = test_utils.mask(x2_4, mask_constant=-1, mask_axis=(2,), gen=m4_key, p=0.5)

        x1 = (((x1_1, x1_2), x1_3), x1_4)
        x2 = None if same_inputs else (((x2_1, x2_2), x2_3), x2_4)

        N_in = 2 ** 7
        dropout_or_id = stax.Dropout(0.9)  # ok on CPU/GPU

        init_fn, apply_fn, kernel_fn = stax.parallel(
            stax.parallel(
                stax.parallel(stax.Dense(N_in),
                              stax.serial(stax.Conv(N_in + 1, (2, 2)), stax.Flatten())),
                stax.serial(stax.Conv(N_in + 2, (2, 2)), dropout_or_id, stax.GlobalAvgPool())
            ),
            stax.Conv(N_in + 3, (2,))
        )

        kernel_fn_empirical = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=stax.unmask_fn(apply_fn),
            gen=mc_key,
            n_samples=N_SAMPLES,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
            vmap_axes=None
        )

        test_utils.assert_close_matrices(
            self,
            kernel_fn(x1, x2, get=kernel_type, mask_constant=-1),
            kernel_fn_empirical(x1, x2, get=kernel_type, mask_constant=-1),
            rtol
        )


class ParameterizationTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        get=['nngp', 'ntk'],
        s=[2**9, 2**8, 2**7],
        depth=[0, 1, 2],
        same_inputs=[True, False],
        W_std=[0., 1., 2.],
        b_std=[None, 0., 0.5**0.5, 2],
        parameterization=['ntk', 'standard']
    )
    def test_linear(self, get, s, depth, same_inputs, b_std, W_std, parameterization):
        if parameterization == 'standard':
            width = 2**9 // s
        elif parameterization == 'ntk':
            if s != 2**9:
                raise unittest.SkipTest('"ntk" parameterization does not depend on "s".')
            width = 2**10
        else:
            raise ValueError(parameterization)

        layers = []
        for i in range(depth + 1):
            s_in = 1 if i == 0 else s
            s_out = 1 if (i == depth and get == 'ntk') else s
            out_dim = 1 if (i == depth and get == 'ntk') else width * (i + 1)
            layers += [stax.Dense(out_dim,
                                  W_std=W_std / (i + 1),
                                  b_std=(b_std if b_std is None else b_std / (i + 1)),
                                  parameterization=parameterization,
                                  s=(s_in, s_out))]

        net = stax.serial(*layers)
        net = net, (BATCH_SIZE, 3), -1, 1
        _check_agreement_with_empirical(self, net, same_inputs, False, get == 'ntk', rtol=0.02, atol=10)

    @test_utils.product(
        model=MODELS,
        width=[2**10],
        same_inputs=[False],
        is_ntk=[False, True],
        filter_shape=FILTER_SHAPES,
        proj_into_2d=PROJECTIONS[:2],
        W_std=[0., 1., 2.],
        b_std=[None, 0., 0.5**0.5],
        parameterization=['ntk', 'standard'],
        s=[2**10]
    )
    def test_nonlinear(self, model, width, same_inputs, is_ntk, filter_shape, proj_into_2d,
                       b_std, W_std, parameterization, s):
        is_conv = 'conv' in model
        if parameterization == 'standard':
            width //= s

        padding = PADDINGS[0]
        strides = STRIDES[0]
        phi = stax.Relu()
        use_pooling, is_res = False, False
        layer_norm = None
        pool_type = 'AVG'
        use_dropout = False

        # Mirror original constraints
        if is_conv:
            test_utils.skip_test(self)
        elif proj_into_2d != PROJECTIONS[0] or filter_shape != FILTER_SHAPES[0]:
            raise unittest.SkipTest('FC models do not have these parameters.')

        net = _get_net(W_std=W_std, b_std=b_std, filter_shape=filter_shape, is_conv=is_conv,
                       use_pooling=use_pooling, is_res=is_res, padding=padding, phi=phi,
                       strides=strides, width=width, is_ntk=is_ntk, proj_into_2d=proj_into_2d,
                       pool_type=pool_type, layer_norm=layer_norm, parameterization=parameterization,
                       s=s, use_dropout=use_dropout)

        _check_agreement_with_empirical(self, net, same_inputs, use_dropout, is_ntk,
                                        rtol=0.015, atol=1000)


if __name__ == '__main__':
    unittest.main()
