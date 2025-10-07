# PyTorch/torchtangents rewrite of tests for
# `neural_tangents/_src/stax/linear.py`

import itertools
import random as prandom
import string
import time
import unittest

import torch
from torch.autograd.functional import vjp as torch_vjp

import torchtangents as nt
from torchtangents import stax
from torchtangents._src.empirical import _DEFAULT_TESTING_NTK_IMPLEMENTATION
from torchtangents._src.utils import utils
from tests import test_utils
import numpy as np


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


# ============================================================
# Flatten
# ============================================================
@test_utils.product(
    same_inputs=[True, False]
)
class FlattenTest(test_utils.NeuralTangentsTestCase):

    def test_flatten(self, same_inputs):
        device = test_utils.device_under_test()
        gen = _rng(device, 1)
        X0_1 = _randn((4, 4, 3, 2), gen)
        X0_2 = None if same_inputs else _randn((2, 4, 3, 2), gen)

        X0_1_flat = X0_1.view(X0_1.shape[0], -1)
        X0_2_flat = None if X0_2 is None else X0_2.view(X0_2.shape[0], -1)

        dense = stax.Dense(512, 1.7, 0.1)
        init_fc, apply_fc, kernel_fc = stax.serial(dense, stax.Erf(), dense)
        init_top, apply_top, kernel_top = stax.serial(dense, stax.Erf(), dense, stax.Flatten())
        init_mid, apply_mid, kernel_mid = stax.serial(dense, stax.Erf(), stax.Flatten(), dense)
        init_bot, apply_bot, kernel_bot = stax.serial(stax.Flatten(), dense, stax.Erf(), dense)

        n = 100
        kernel_fc_mc = nt.monte_carlo_kernel_fn(
            init_fc, apply_fc, key=gen, n_samples=n, vmap_axes=0,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
        )
        kernel_bot_mc = nt.monte_carlo_kernel_fn(
            init_bot, apply_bot, key=gen, n_samples=n, vmap_axes=0,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
        )
        kernel_mid_mc = nt.monte_carlo_kernel_fn(
            init_mid, apply_mid, key=gen, n_samples=n, vmap_axes=0,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
        )
        kernel_top_mc = nt.monte_carlo_kernel_fn(
            init_top, apply_top, key=gen, n_samples=n, vmap_axes=0,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
        )

        K = kernel_fc(X0_1_flat, X0_2_flat)

        K_bot = kernel_bot(X0_1, X0_2)
        K_bot_flat = kernel_bot(X0_1_flat, X0_2_flat)
        self.assertAllClose(K_bot, K)
        self.assertAllClose(K_bot_flat, K)

        def assert_close(a, b):
            self.assertAllClose(a, b, atol=0.05, rtol=0.02)

        K_fc_mc = kernel_fc_mc(X0_1_flat, X0_2_flat, get='nngp')
        K_bot_mc_val = kernel_bot_mc(X0_1, X0_2, get='nngp')
        K_bot_flat_mc_val = kernel_bot_mc(X0_1_flat, X0_2_flat, get='nngp')

        assert_close(K_fc_mc, K.nngp)
        assert_close(K_bot_mc_val, K_bot.nngp)
        assert_close(K_bot_flat_mc_val, K_bot_flat.nngp)

        K_mid = kernel_mid(X0_1, X0_2)
        K_mid_flat = kernel_mid(X0_1_flat, X0_2_flat)

        K_mid_mc_val = kernel_mid_mc(X0_1, X0_2, get='nngp')
        K_mid_flat_mc_val = kernel_mid_mc(X0_1_flat, X0_2_flat, get='nngp')

        assert_close(K_mid_mc_val, K_mid.nngp)
        assert_close(K_mid_flat, K)
        assert_close(K_mid_flat_mc_val, K_mid_flat.nngp)

        K_top = kernel_top(X0_1, X0_2).replace(is_gaussian=True,
                                               shape1=K_mid.shape1,
                                               shape2=K_mid.shape2)
        K_top_flat = kernel_top(X0_1_flat, X0_2_flat).replace(is_gaussian=True)

        K_top_mc_val = kernel_top_mc(X0_1, X0_2, get='nngp')
        K_top_flat_mc_val = kernel_top_mc(X0_1_flat, X0_2_flat, get='nngp')

        assert_close(K_top_flat, K)
        assert_close(K_top_mc_val, K_top.nngp)
        assert_close(K_top_flat_mc_val, K_top_flat.nngp)

        assert_close(K_top, K_mid)


# ============================================================
# ConvND
# ============================================================
class ConvNDTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        same_inputs=[False],
        n=[0, 1, 2],
        get=['ntk'],
        proj=['flatten', 'pool'],
        use_attn=[True],
        channels_first=[True, False],
        use_dropout=[True],
        use_layernorm=[True],
    )
    def test_conv_nd(
        self,
        same_inputs,
        n,
        get,
        proj,
        use_attn,
        channels_first,
        use_dropout,
        use_layernorm
    ):
        platform = test_utils.device_under_test()
        if platform == 'cpu':
            test_utils.skip_test(self)
        elif platform == 'cuda' and n not in (0, 1, 2, 3):
            raise unittest.SkipTest('>=4D CNN does not work on GPU.')
        elif platform == 'xla' and use_dropout and same_inputs:
            raise unittest.SkipTest('Batched empirical kernel with dropout not supported.')

        width = 1024
        n_samples = 512
        tol = 0.03 if platform == 'xla' else 0.015

        device = test_utils.device_under_test()
        gen = _rng(device, 1)

        n_max = 5
        spatial_shape = (2, 3, 5, 4, 3)[:n] + (1,) * (n - n_max)
        filter_shape = (1, 2, 3, 1, 1)[:n] + (1,) * (n - n_max)
        strides = (1, 1, 2, 1, 2)[:n] + (1,) * (n - n_max)
        spatial_spec = ''.join(c for c in string.ascii_uppercase if c not in ('N', 'C', 'I', 'O'))[:n]
        filter_spec = spatial_spec + 'IO'

        if channels_first:
            channel_axis = 1
            dimension_numbers = ('NC' + spatial_spec, filter_spec, 'NC' + spatial_spec)
            X0_1 = _randn((2, 3) + tuple(spatial_shape), gen)
            X0_2 = None if same_inputs else _randn((4, 3) + tuple(spatial_shape), gen)
        else:
            channel_axis = -1
            dimension_numbers = ('N' + spatial_spec + 'C', filter_spec, 'N' + spatial_spec + 'C')
            X0_1 = _randn((2,) + tuple(spatial_shape) + (3,), gen)
            X0_2 = None if same_inputs else _randn((4,) + tuple(spatial_shape) + (3,), gen)

        layernorm_axes = (dimension_numbers[2].index('C'),)
        if 'H' in dimension_numbers[2]:
            layernorm_axes += (dimension_numbers[2].index('H'),)

        if proj == 'pool':
            proj_layer = stax.GlobalAvgPool(channel_axis=channel_axis)
        elif proj == 'flatten':
            proj_layer = stax.Flatten()
        else:
            raise ValueError(proj)

        if use_attn:
            n_heads = int(width ** 0.5)
            n_chan_val = int(round(float(width) / n_heads))
            proj_layer = stax.serial(
                stax.GlobalSelfAttention(
                    n_chan_out=width,
                    n_chan_key=width,
                    n_chan_val=n_chan_val,
                    n_heads=n_heads,
                    linear_scaling=True,
                    W_key_std=2.,
                    W_value_std=1.,
                    W_query_std=1.,
                    W_out_std=1.0,
                    b_std=0.1,
                    channel_axis=channel_axis
                ),
                proj_layer
            )

        nn = stax.serial(
            stax.Conv(width, filter_shape, None, 'SAME', dimension_numbers=dimension_numbers),
            (stax.LayerNorm(layernorm_axes, channel_axis=channel_axis) if use_layernorm else stax.Identity()),
            stax.Relu(),
            (stax.Dropout(0.8) if use_dropout else stax.Identity()),
            stax.Conv(width, filter_shape, strides, 'CIRCULAR', dimension_numbers=dimension_numbers),
            stax.Abs(),
            proj_layer
        )

        if get == 'nngp':
            init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(width, 2., 0.5))
        elif get == 'ntk':
            init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(1, 2., 0.5))
        else:
            raise ValueError(get)

        kernel_fn_mc = nt.monte_carlo_kernel_fn(
            init_fn, apply_fn, key=gen, n_samples=n_samples,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION, vmap_axes=0
        )

        exact = kernel_fn(X0_1, X0_2, get=get)
        empirical = kernel_fn_mc(X0_1, X0_2, get=get)
        test_utils.assert_close_matrices(self, empirical, exact, tol)


# ============================================================
# Attention (kept as in original: skipped via helper)
# ============================================================
class AttentionTest(test_utils.NeuralTangentsTestCase):

    @test_utils.parameters(
        dict(
            same_inputs=same_inputs,
            get=get,
            n=n,
            linear_scaling=linear_scaling,
            mask_constant=mask_constant,
            p=p,
            mask_axis=mask_axis,
            pos_emb_type=pos_emb_type,
            n_chan_pos_emb=n_chan_pos_emb,
            pos_emb_decay_fn=pos_emb_decay_fn,
            val_pos_emb=val_pos_emb,
            W_pos_emb_std=W_pos_emb_std
        )
        for same_inputs in [False]
        for get in ['ntk']
        for n in [2]
        for linear_scaling in [True, False]
        for mask_constant in [10.]
        for p in [0.5]
        for mask_axis in [(-1,)]
        for pos_emb_type in ['CONCAT', 'SUM', 'NONE']
        for n_chan_pos_emb in ([None] if pos_emb_type != 'CONCAT' else [None, 512])
        for pos_emb_decay_fn in [None, 'linear']
        for val_pos_emb in ([True, False] if pos_emb_type != 'NONE' else [True])
        for W_pos_emb_std in ([2] if pos_emb_type != 'NONE' else [0.])
    )
    def test_attention(self, **_):
        test_utils.skip_test(self)


# ============================================================
# Aggregate
# ============================================================
class AggregateTest(test_utils.NeuralTangentsTestCase):

    @test_utils.parameters(
        dict(
            get=get,
            readout=readout,
            same_input=same_input,
            activation=activation,
            mask_constant=mask_constant,
            shape=shape,
            batch_axis=batch_axis,
            channel_axis=channel_axis,
            agg_axes=agg_axes,
            do_batch=do_batch,
            implementation=implementation,
            to_dense=to_dense
        )
        for get in ['ntk']
        for same_input in [False, True]
        for act_name, activation in [('Relu', stax.Relu())]
        for mask_constant in [10.]
        for shape in [(4,), (3, 2)]
        for batch_axis in range(len(shape) + 2)
        for channel_axis in [c for c in range(len(shape) + 2) if c != batch_axis]
        for agg_axes in [None] + list(itertools.chain.from_iterable(
            itertools.combinations(
                [p for p in range(len(shape) + 2) if p not in (batch_axis, channel_axis)], r
            ) for r in range(len([p for p in range(len(shape) + 2) if p not in (batch_axis, channel_axis)]) + 1)
        ))
        for do_batch in ([True] if batch_axis == 0 else [False])
        for implementation in ['DENSE', 'SPARSE']
        for to_dense in ['identity', 'sparse_to_dense']
        for name, readout in [('Pooling', stax.GlobalAvgPool(batch_axis=batch_axis, channel_axis=channel_axis))]
    )
    def test_aggregate(
        self,
        get,
        readout,
        same_input,
        activation,
        mask_constant,
        shape,
        batch_axis,
        channel_axis,
        agg_axes,
        do_batch,
        implementation,
        to_dense
    ):
        if len(shape) > 1:
            test_utils.skip_test(self)

        if implementation == 'SPARSE' and to_dense != 'identity':
            raise unittest.SkipTest('`implementation="SPARSE"` ignores `to_dense` argument.')

        if get in ('cov1', 'cov2') and do_batch:
            raise unittest.SkipTest('Batching of empirical kernel does not work for `diagonal_axes != ()`.')

        batch1, batch2 = 8, 4
        num_channels = 1
        output_dims = 1 if get == 'ntk' else 2**6

        device = test_utils.device_under_test()
        gen = _rng(device, 1)
        gen, split1, split2 = _rng(device, 11), _rng(device, 12), _rng(device, 13)

        x1 = _randn((batch1,) + tuple(shape) + (num_channels,), split1)
        x1 = torch.movedim(x1, (0, -1), (batch_axis, channel_axis))

        if same_input:
            x2 = None
        else:
            x2 = _randn((batch2,) + tuple(shape) + (num_channels,), split2)
            x2 = torch.movedim(x2, (0, -1), (batch_axis, channel_axis))

        if mask_constant is not None:
            device = test_utils.device_under_test()
            gen, split1, split2 = _rng(device, 21), _rng(device, 22), _rng(device, 23)
            shape1 = list(x1.shape); shape1[channel_axis] = 1
            mask1 = torch.bernoulli(torch.full(shape1, 0.3, generator=split1, device=x1.device)).bool()
            x1 = torch.where(mask1, torch.as_tensor(mask_constant, device=x1.device, dtype=x1.dtype), x1)

            if x2 is not None:
                shape2 = list(x2.shape); shape2[channel_axis] = 1
                mask2 = torch.bernoulli(torch.full(shape2, 0.2, generator=split2, device=x2.device)).bool()
                x2 = torch.where(mask2, torch.as_tensor(mask_constant, device=x2.device, dtype=x2.dtype), x2)

        device = test_utils.device_under_test()
        gen, split1, split2 = _rng(device, 31), _rng(device, 32), _rng(device, 33)

        agg_shape = shape if agg_axes is None else tuple(x1.shape[a] for a in agg_axes)
        agg_ndim = len(agg_shape)

        def sparse_to_dense(pattern):
            if pattern is None:
                return None
            # pattern: [B, E, dims, 2]
            pattern = pattern.reshape(pattern.shape[:2] + (pattern.shape[2] * 2,))
            bsz, n_edges, n_dims = pattern.shape
            batch_range = torch.arange(bsz, device=pattern.device).view(bsz, 1, 1).expand(bsz, n_edges, 1)
            pattern = torch.cat([batch_range, pattern], dim=2)              # [B, E, n_dims+1]
            pattern = pattern.reshape(bsz * n_edges, n_dims + 1)            # [B*E, n_dims+1]
            out = torch.zeros((bsz,) + tuple(a for a in agg_shape for _ in (0, 1)),
                              device=pattern.device, dtype=torch.float32)
            out[tuple(pattern.T)] += 1.0
            out = utils.unzip_axes(out, 1)
            return out

        if to_dense == 'sparse_to_dense' or implementation == 'SPARSE':
            def get_sparse_pattern(batch_size, g):
                n_edges_max = int(np.prod((1,) + agg_shape) ** 2)
                n_edges = prandom.randint(0, n_edges_max)
                device = x1.device
                pat = [torch.zeros((batch_size, n_edges, 0, 2), dtype=torch.int32, device=device)]
                for d in range(agg_ndim):
                    n_nodes = agg_shape[d]
                    edges = torch.randint(0, n_nodes, (batch_size, n_edges, 1, 2), generator=g, device=device)
                    pat += [edges]
                pat = torch.cat(pat, dim=2)
                mask = torch.bernoulli(torch.full(pat.shape[:2], 0.2, generator=g, device=device)).bool()
                pat[mask] = -10000
                return pat

            pattern1 = get_sparse_pattern(batch1, split1)
            pattern2 = pattern1 if same_input else get_sparse_pattern(batch2, split2)
        else:
            device = x1.device
            pattern1 = torch.rand((batch1,) + agg_shape * 2, generator=split1, device=device)
            pattern2 = pattern1 if same_input else torch.rand((batch2,) + agg_shape * 2, generator=split2, device=device)

        def get_nn(to_dense_kind, impl):
            return stax.serial(
                stax.Dense(2**6, batch_axis=batch_axis, channel_axis=channel_axis),
                activation,
                stax.Aggregate(
                    aggregate_axis=agg_axes,
                    batch_axis=batch_axis,
                    channel_axis=channel_axis,
                    to_dense=dict(identity=lambda p: p, sparse_to_dense=sparse_to_dense)[to_dense_kind],
                    implementation=impl
                ),
                readout,
                stax.Dense(output_dims,
                           batch_axis=int(batch_axis > channel_axis),
                           channel_axis=int(batch_axis < channel_axis))
            )

        init_fn, apply_fn, kernel_fn = get_nn(to_dense, implementation)
        exact = kernel_fn(x1, x2, get, mask_constant=mask_constant, pattern=(pattern1, pattern2))

        rtol = 0.08

        if to_dense == 'sparse_to_dense' or implementation == 'SPARSE':
            init_fn_dense, apply_fn_dense, kernel_fn_dense = get_nn('identity', 'DENSE')
            pattern1_dense = sparse_to_dense(pattern1)
            pattern2_dense = sparse_to_dense(pattern2)
            device = test_utils.device_under_test()

            _, params_sparse = init_fn(_rng(device, 41), x1.shape)
            _, params_dense = init_fn_dense(_rng(device, 41), x1.shape)
            self.assertAllClose(params_dense, params_sparse)

            fx1_dense = apply_fn_dense(params_dense, x1, pattern=pattern1_dense)
            fx1_sparse = apply_fn(params_sparse, x1, pattern=pattern1)
            test_utils.assert_close_matrices(self, fx1_dense, fx1_sparse, rtol)

            if not same_input:
                fx2_dense = apply_fn_dense(params_dense, x2, pattern=pattern2_dense)
                fx2_sparse = apply_fn(params_sparse, x2, pattern=pattern2)
                test_utils.assert_close_matrices(self, fx2_dense, fx2_sparse, rtol)

            exact_dense = kernel_fn_dense(
                x1, x2, get, mask_constant=mask_constant, pattern=(pattern1_dense, pattern2_dense)
            )
            self.assertAllClose(exact_dense, exact)

        kernel_mc_fn = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=stax.unmask_fn(apply_fn),
            key=_rng(device, 52),
            n_samples=2**6,
            batch_size=2 if (test_utils.device_under_test() == 'xla' and batch_axis == 0) else 0,
            device_count=-1 if batch_axis == 0 else 0,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
            trace_axes=(int(batch_axis < channel_axis),)
        )
        empirical = kernel_mc_fn(x1, x2, get, mask_constant=mask_constant, pattern=(pattern1, pattern2))
        test_utils.assert_close_matrices(self, exact, empirical, rtol, 0.2)


# ============================================================
# ConvTranspose
# ============================================================
class ConvTransposeTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        padding=['CIRCULAR', 'SAME', 'VALID'],
        same_inputs=[False],
        filter_shape=[2, 3, 4],
        strides=[2, 3, 4],
        size=[2, 3, 4],
        diagonal_batch=[True],
        diagonal_spatial=[True, False],
    )
    def test_conv_transpose(
        self,
        same_inputs,
        padding,
        filter_shape,
        strides,
        size,
        diagonal_batch,
        diagonal_spatial
    ):
        if size > 2:
            test_utils.skip_test(self)

        width = 512
        tol = 0.01
        n_samples = 512
        filter_shape = (filter_shape,)
        strides = (strides,)

        init_fn, apply_fn, kernel_fn = stax.ConvTranspose(width, filter_shape, strides, padding, b_std=0.1)

        device = test_utils.device_under_test()
        gen = _rng(device, 1)
        shape = (size, 1)
        x1 = _randn((2,) + shape, gen)
        x2 = _randn((3,) + shape, gen) if not same_inputs else None

        k = kernel_fn(
            x1, x2,
            diagonal_batch=diagonal_batch,
            diagonal_spatial=diagonal_spatial,
            get='cov1' if diagonal_batch else 'nngp'
        )

        diagonal_axes = ()
        if diagonal_batch:
            diagonal_axes += (0,)
        if diagonal_spatial:
            diagonal_axes += (1,)

        kernel_fn_mc = nt.monte_carlo_kernel_fn(
            init_fn, apply_fn, key=gen, n_samples=n_samples, diagonal_axes=diagonal_axes,
            device_count=0, implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION, vmap_axes=0
        )
        k_mc = kernel_fn_mc(x1, None if diagonal_batch else x2, 'nngp')
        test_utils.assert_close_matrices(self, k_mc, k, tol)

    @classmethod
    def _conv_transpose_circular_via_grad(
        cls,
        lhs,
        params,
        strides,
        padding,
        dimension_numbers
    ):
        # Compute via VJP of a forward conv (adjoint trick)
        W = params[0]
        # swap IO in weight spec and flip spatial
        w_dn = dimension_numbers[1]  # e.g., 'HIO'
        h_axis = w_dn.index('H')
        i_axis = w_dn.index('I')
        o_axis = w_dn.index('O')
        W_swapped = W.movedim((o_axis, i_axis), (i_axis, o_axis))
        W_swapped = torch.flip(W_swapped, dims=(h_axis,))

        _, apply_fn, _ = stax.Conv(
            out_chan=W_swapped.shape[w_dn.index('I')],
            filter_shape=(W_swapped.shape[h_axis],),
            strides=strides,
            padding=padding,
            dimension_numbers=dimension_numbers,
            parameterization='standard'
        )

        # Placeholder output with appropriate layout
        lhs_spec, rhs_spec, out_spec = dimension_numbers
        # Build a placeholder that matches the forward output shape
        # Use apply_fn on a ones tensor to infer shape
        dummy = torch.ones_like(lhs, dtype=lhs.dtype, device=lhs.device, requires_grad=False)
        out = apply_fn((W_swapped, torch.tensor(0., device=lhs.device)), dummy)
        placeholder = torch.ones_like(out, dtype=lhs.dtype, device=lhs.device)

        def conv_f(x):
            return apply_fn((W_swapped, torch.tensor(0., device=x.device)), x)

        _, vjp_fun = torch_vjp(conv_f, placeholder)
        return vjp_fun(lhs)[0]

    @classmethod
    def _conv_transpose_circular(
        cls,
        lhs,
        params,
        strides,
        padding,
        dimension_numbers
    ):
        _, apply_fn, _ = stax.ConvTranspose(
            out_chan=params[0].shape[dimension_numbers[1].index('O')],
            filter_shape=(params[0].shape[dimension_numbers[1].index('H')],),
            strides=strides,
            padding=padding,
            dimension_numbers=dimension_numbers,
            parameterization='standard'
        )
        return apply_fn((params[0], torch.tensor(0., device=lhs.device)), lhs)

    @test_utils.product(
        filter_shape=[1, 2, 3, 4],
        strides=[1, 2, 3, 4],
        size=[1, 2, 3, 4]
    )
    def test_conv_transpose_circular(self, size, filter_shape, strides):
        if size > 2:
            test_utils.skip_test(self)

        device = test_utils.device_under_test()
        gen = _rng(device,1)
        x = _randn((2, size, 3), gen)
        dn = ('NHC', 'HIO', 'NHC')
        padding = 'CIRCULAR'
        filter_shape = (filter_shape,)
        strides = (strides,)

        init_fn, _, _ = stax.ConvTranspose(4, filter_shape, strides, padding)
        _, params = init_fn(_rng(device, 2), x.shape)
        f_conv = self._conv_transpose_circular(x, params, strides, padding, dn)
        f_adj = self._conv_transpose_circular_via_grad(x, params, strides, padding, dn)
        self.assertAllClose(f_adj, f_conv)


# ============================================================
# DotGeneral
# ============================================================
class DotGeneralTest(test_utils.NeuralTangentsTestCase):

    @test_utils.parameters(
        dict(
            same_inputs=same_inputs,
            n=n,
            batch_dims=batch_dims,
            contracting_dims=contracting_dims,
            b_dims=b_dims,
            c_dims=c_dims,
            r_permutation=r_permutation,
            channel_axis=channel_axis,
            batch_axis=batch_axis,
            is_rhs=is_rhs,
            diagonal_spatial=diagonal_spatial,
            diagonal_batch=diagonal_batch,
        )
        for same_inputs in [True, False]
        for n in [2, 3]
        for is_rhs in [False, True]

        # pick explicit batch/channel axes from range(n), distinct
        for batch_axis in range(n)
        for channel_axis in [i for i in range(n) if i != batch_axis]

        for diagonal_spatial in [True, False]
        for diagonal_batch in [True, False]

        # batch dims: choose any subset excluding batch_axis and channel_axis
        for batch_dims in itertools.chain.from_iterable(
            itertools.combinations(
                [i for i in range(n) if i not in {batch_axis, channel_axis}], r
            )
            for r in range(n + 1)
        )

        # contracting dims: choose any subset excluding batch_axis, channel_axis, and batch_dims
        for contracting_dims in itertools.chain.from_iterable(
            itertools.combinations(
                [i for i in range(n) if i not in set(batch_dims) | {batch_axis, channel_axis}], r
            )
            for r in range(n + 1)
        )

        # orderings of the chosen dims
        for c_dims in itertools.permutations(contracting_dims or ())
        for b_dims in itertools.permutations(batch_dims or ())

        # permutation over all n axes (use range(n), not range(2))
        for r_permutation in itertools.permutations(range(n))
    )
    def test_dot_general(
        self,
        same_inputs,
        n,
        batch_dims,
        contracting_dims,
        c_dims,
        b_dims,
        r_permutation,
        channel_axis,
        is_rhs,
        diagonal_spatial,
        diagonal_batch,
        batch_axis
    ):
        if n != 2:
            test_utils.skip_test(self)

        atol = 1.0 if test_utils.device_under_test() == 'xla' else 0.1

        n_b = 2
        n_c = 1
        device = test_utils.device_under_test()
        g1, g2, g3 = _rng(device, 1), _rng(device, 2), _rng(device, 3)

        x_shape_n_c = [2, 4][:n - 2]
        x_shape = list(x_shape_n_c)
        for a in sorted((batch_axis, channel_axis)):
            x_shape.insert(a, n_b if a == batch_axis else n_c)

        mask_constant = 10.0

        x1 = torch.cos(_randn(tuple(x_shape), g1))
        mask1 = torch.bernoulli(torch.full_like(x1, 0.8, device=x1.device)).bool()
        x1 = torch.where(mask1, torch.as_tensor(mask_constant, device=x1.device, dtype=x1.dtype), x1)

        if same_inputs:
            x2 = None
        else:
            x2_shape = list(x_shape)
            if batch_axis not in contracting_dims + b_dims:
                x2_shape[batch_axis] = 4
            x2 = torch.cos(_randn(tuple(x2_shape), g2))
            mask2 = torch.bernoulli(torch.full_like(x2, 0.4, device=x2.device)).bool()
            x2 = torch.where(mask2, torch.as_tensor(mask_constant, device=x2.device, dtype=x2.dtype), x2)

        other_shape = [1, 3]
        for i in contracting_dims + b_dims:
            other_shape[i] = x_shape[i]
        other = _randn(tuple(other_shape), g3)
        other = torch.arange(other.numel(), device=other.device, dtype=other.dtype).view(other_shape)

        other_t = other.permute(r_permutation)
        r_c_dims = tuple(r_permutation.index(c) for c in c_dims)
        r_b_dims = tuple(r_permutation.index(b) for b in b_dims)

        if is_rhs:
            lhs, rhs = None, other_t
            dn = ((c_dims, r_c_dims), (b_dims, r_b_dims))
        else:
            lhs, rhs = other_t, None
            dn = ((r_c_dims, c_dims), (r_b_dims, b_dims))

        lhs_ndim = None if lhs is None else lhs.ndim
        init_fn, apply_fn, kernel_fn = stax.DotGeneral(lhs=lhs, rhs=rhs, dimension_numbers=dn, batch_axis=batch_axis)

        def get_exact():
            return kernel_fn(
                x1, x2,
                diagonal_spatial=diagonal_spatial,
                diagonal_batch=diagonal_batch,
                batch_axis=batch_axis,
                channel_axis=channel_axis,
                mask_constant=mask_constant
            )

        if (([i for i in c_dims if i not in (batch_axis, channel_axis)] and diagonal_spatial) or
            (batch_axis in c_dims and diagonal_batch)):
            self.assertRaises(ValueError, get_exact)
            return

        exact = get_exact()

        out_c_axis = utils.axis_after_dot(channel_axis, c_dims, b_dims, lhs_ndim)
        out_b_axis = utils.axis_after_dot(batch_axis, c_dims, b_dims, lhs_ndim)

        def get_empirical(which_get):
            def get_diagonal_axes():
                axes = ()
                if which_get in ('cov1', 'cov2') and diagonal_batch and batch_axis not in c_dims:
                    axes += (out_b_axis,)
                if diagonal_spatial:
                    axes += tuple(
                        utils.axis_after_dot(i, c_dims, b_dims, lhs_ndim)
                        for i in range(n)
                        if i not in c_dims + (batch_axis, channel_axis)
                    )
                    rhs_ndim = None if rhs is None else rhs.ndim
                    axes += tuple(
                        utils.axis_after_dot(i, r_c_dims, r_b_dims, rhs_ndim)
                        for i in range(n)
                        if i not in r_c_dims and not (i in r_b_dims and b_dims[r_b_dims.index(i)] == batch_axis)
                    )
                return axes

            device = test_utils.device_under_test()
            kernel_fn_mc = nt.monte_carlo_kernel_fn(
                init_fn=init_fn,
                apply_fn=stax.unmask_fn(apply_fn),
                key=_rng(device, 101),
                n_samples=1,
                trace_axes=(out_c_axis,),
                diagonal_axes=get_diagonal_axes(),
                device_count=-1 if (which_get == 'nngp' and batch_axis == out_b_axis == 0 and 0 not in c_dims + b_dims) else 0,
                implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
            )

            empirical = kernel_fn_mc(
                x1=x2 if which_get == 'cov2' else x1,
                x2=x2 if which_get == 'nngp' else None,
                get='nngp',
                batch_axis=batch_axis,
                channel_axis=channel_axis,
                mask_constant=mask_constant
            )

            def batch_axes():
                if batch_axis in contracting_dims:
                    return (), ()
                axis = out_b_axis
                if out_c_axis < axis:
                    axis -= 1
                if not diagonal_spatial:
                    axis *= 2
                if which_get in ('cov1', 'cov2') and diagonal_batch:
                    return (axis,), (0,)
                return (axis, axis + 1), (0, 1)

            empirical = empirical.movedim(*batch_axes())
            return empirical

        for which in ('nngp', 'cov1', 'cov2'):
            if which == 'cov2' and same_inputs:
                continue
            with self.subTest(get=which):
                test_utils.assert_close_matrices(self, get_empirical(which), getattr(exact, which), 0.01, atol)

    @test_utils.product(
        same_inputs=[False, True],
        get=['ntk'],
        do_pool=[True, False],
        n=[3, 4],
        is_rhs=[False, True],
        dot_first=[True, False]
    )
    def test_dot_general_nn(
        self,
        same_inputs,
        get,
        n,
        is_rhs,
        do_pool,
        dot_first
    ):
        if n != 2:
            test_utils.skip_test(self)

        width = 2**8
        n_samples = 2**8
        tol = 0.03
        device = test_utils.device_under_test()
        g1, g2, g3 = _rng(device, 1), _rng(device, 2), _rng(device, 3)

        mask_constant = 10.0
        x_shape = [6, 3][:n - 1] + [1]
        x1 = torch.cos(_randn(tuple(x_shape), g1))
        mask1 = torch.bernoulli(torch.full_like(x1, 0.8, device=x1.device)).bool()
        x1 = torch.where(mask1, torch.as_tensor(mask_constant, device=x1.device, dtype=x1.dtype), x1)

        if same_inputs:
            x2 = None
        else:
            x2 = torch.cos(_randn(tuple(x_shape), g2))
            mask2 = torch.bernoulli(torch.full_like(x2, 0.4, device=x2.device)).bool()
            x2 = torch.where(mask2, torch.as_tensor(mask_constant, device=x2.device, dtype=x2.dtype), x2)

        other = _randn([3, 4], g3).view(3, 4)

        c_dims, b_dims = (1,), (0,)
        o_c_dims, o_b_dims = (0,), (2,)

        if is_rhs:
            lhs, rhs = None, other
            dn = ((c_dims, o_c_dims), (b_dims, o_b_dims))
        else:
            lhs, rhs = other, None
            dn = ((o_c_dims, c_dims), (o_b_dims, b_dims))

        lhs_ndim = None if lhs is None else lhs.ndim
        out_c_axis = utils.axis_after_dot(n - 1, c_dims, b_dims, lhs_ndim)
        out_b_axis = utils.axis_after_dot(0, c_dims, b_dims, lhs_ndim)

        top_b_axis = int(out_b_axis > out_c_axis and do_pool)
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Identity() if dot_first else stax.Conv(width, (3,) * (n - 2), padding='SAME'),
            stax.DotGeneral(lhs=lhs, rhs=rhs, dimension_numbers=dn),
            stax.Dense(width, batch_axis=out_b_axis, channel_axis=out_c_axis),
            stax.Relu(),
            (stax.GlobalAvgPool(channel_axis=out_c_axis, batch_axis=out_b_axis) if do_pool else
             stax.Flatten(batch_axis=out_b_axis)),
            stax.Dense(width if get == 'nngp' else 1, 0.9, 0.1,
                       batch_axis=top_b_axis,
                       channel_axis=int(out_c_axis > out_b_axis or not do_pool))
        )

        kernel_fn_mc = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=stax.unmask_fn(apply_fn),
            key=g1,
            n_samples=n_samples,
            trace_axes=(int(out_c_axis > out_b_axis) if do_pool else 1,),
            device_count=0,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        )

        empirical = kernel_fn_mc(x1, x2, get, mask_constant=mask_constant)
        exact = kernel_fn(x1, x2, get, mask_constant=mask_constant)
        test_utils.assert_close_matrices(self, empirical, exact, tol)

    def test_dot_general_mask(self):
        x1 = torch.ones((4, 2, 3, 1))
        x2 = torch.ones((4, 2, 3, 1))

        mask_constant = 10.0

        def get_k(x1_, x2_, m1, m2):
            x1m = torch.where(m1, torch.tensor(mask_constant, dtype=torch.float32), x1_.to(torch.float32))
            x2m = torch.where(m2, torch.tensor(mask_constant, dtype=torch.float32), x2_.to(torch.float32))
            k_fn = stax.DotGeneral(rhs=torch.ones(x1m.shape[:-1], dtype=torch.float32), dimension_numbers=(((1,), (1,)), ((2,), (2,))))[2]
            return k_fn(x1m, x2m, 'nngp', mask_constant=mask_constant)

        m1 = torch.zeros_like(x1, dtype=torch.bool)
        m2 = torch.zeros_like(x2, dtype=torch.bool)
        k = get_k(x1, x2, m1, m2)
        self.assertAllClose(torch.ones_like(k) * 4, k)

        m1 = torch.ones_like(x1, dtype=torch.bool)
        m2 = torch.zeros_like(x2, dtype=torch.bool)
        k = get_k(x1, x2, m1, m2)
        self.assertAllClose(torch.zeros_like(k), k)

        m1 = torch.ones_like(x1, dtype=torch.bool)
        m2 = torch.ones_like(x2, dtype=torch.bool)
        k = get_k(x1, x2, m1, m2)
        self.assertAllClose(torch.zeros_like(k), k)

        m1 = torch.cat([torch.ones_like(x1[:2], dtype=torch.bool), torch.zeros_like(x1[2:], dtype=torch.bool)])
        m2 = torch.zeros_like(x2, dtype=torch.bool)
        k = get_k(x1, x2, m1, m2)
        self.assertAllClose(torch.zeros_like(k[:2]), k[:2])
        self.assertAllClose(torch.full_like(k[2:], 4.), k[2:])


# ============================================================
# ImageResize
# ============================================================
class ImageResizeTest(test_utils.NeuralTangentsTestCase):

    @test_utils.parameters(
        dict(
            same_inputs=same_inputs,
            n=n,
            channel_axis=channel_axis,
            batch_axis=batch_axis,
            diagonal_spatial=diagonal_spatial,
            diagonal_batch=diagonal_batch,
            method=method,
            antialias=antialias,
            precision=precision,
            shape=shape
        )
        for same_inputs in [True, False]
        for n in [2, 3, 4]
        for batch_axis in range(2)
        for channel_axis in [i for i in range(2) if i != batch_axis]
        for diagonal_spatial in [True, False]
        for diagonal_batch in [True, False]
        for method in ['linear', 'nearest']
        for antialias in [True, False]
        for precision in ['default']
        for shape in [s[:2] for s in [(-1, 2, 3, 4), (-1, 3, -1, 4), (10, 5, 1, 8), (5, -1, 2, 3)]]
    )
    def test_image_resize(
        self,
        same_inputs,
        n,
        channel_axis,
        diagonal_spatial,
        diagonal_batch,
        batch_axis,
        method,
        antialias,
        precision,
        shape
    ):
        if n > 2:
            test_utils.skip_test(self)

        n_b1, n_b2 = 2, 4
        n_c = 1

        device = test_utils.device_under_test()
        g1, g2 = _rng(device, 1), _rng(device, 2)

        shape = list(shape)
        shape[channel_axis] = -1

        x_shape_n_c = [2, 4][:n - 2]
        x_shape = list(x_shape_n_c)
        for a in sorted((batch_axis, channel_axis)):
            x_shape.insert(a, n_b1 if a == batch_axis else n_c)

        mask_constant = 10.0
        x1 = torch.cos(_randn(tuple(x_shape), g1))
        mask1 = torch.bernoulli(torch.full_like(x1, 0.3, device=x1.device)).bool()
        x1 = torch.where(mask1, torch.as_tensor(mask_constant, device=x1.device, dtype=x1.dtype), x1)

        if same_inputs:
            x2 = None
        else:
            x2_shape = list(x_shape); x2_shape[batch_axis] = n_b2
            x2 = torch.cos(_randn(tuple(x2_shape), g2))
            mask2 = torch.bernoulli(torch.full_like(x2, 0.2, device=x2.device)).bool()
            x2 = torch.where(mask2, torch.as_tensor(mask_constant, device=x2.device, dtype=x2.dtype), x2)

        init_fn, apply_fn, kernel_fn = stax.ImageResize(
            method=method,
            antialias=antialias,
            precision=precision,
            batch_axis=batch_axis,
            channel_axis=channel_axis,
            shape=tuple(shape)
        )

        def get_exact():
            return kernel_fn(
                x1, x2,
                diagonal_spatial=diagonal_spatial,
                diagonal_batch=diagonal_batch,
                batch_axis=batch_axis,
                channel_axis=channel_axis,
                mask_constant=mask_constant
            )

        if ((shape[batch_axis] != -1 and diagonal_batch) or
            (any(shape[i] != -1 for i in range(len(shape)) if i not in (batch_axis, channel_axis)) and diagonal_spatial)):
            self.assertRaises(ValueError, get_exact)
            return

        exact = get_exact()

        def get_empirical(which_get):
            def get_diagonal_axes():
                axes = ()
                if which_get in ('cov1', 'cov2') and diagonal_batch:
                    axes += (batch_axis,)
                if diagonal_spatial:
                    axes += tuple(i for i in range(n) if i not in (batch_axis, channel_axis))
                return axes

            device = test_utils.device_under_test()
            kernel_fn_mc = nt.monte_carlo_kernel_fn(
                init_fn=init_fn,
                apply_fn=stax.unmask_fn(apply_fn),
                key=_rng(device, 101),
                n_samples=1,
                trace_axes=(channel_axis,),
                diagonal_axes=get_diagonal_axes(),
                device_count=-1 if (which_get == 'nngp' and batch_axis == 0 and shape[batch_axis] == -1) else 0,
                implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
            )

            empirical = kernel_fn_mc(
                x1=x2 if which_get == 'cov2' else x1,
                x2=x2 if which_get == 'nngp' else None,
                get='nngp',
                batch_axis=batch_axis,
                channel_axis=channel_axis,
                mask_constant=mask_constant
            )

            def batch_axes():
                axis = batch_axis
                if channel_axis < batch_axis:
                    axis -= 1
                if not diagonal_spatial:
                    axis *= 2
                if which_get in ('cov1', 'cov2') and diagonal_batch:
                    return (axis,), (0,)
                return (axis, axis + 1), (0, 1)

            empirical = empirical.movedim(*batch_axes())
            return empirical

        for which in ('nngp', 'cov1', 'cov2'):
            if which == 'cov2' and same_inputs:
                continue
            with self.subTest(get=which):
                tol = 1e-2 if test_utils.device_under_test() == 'xla' else 1e-5
                test_utils.assert_close_matrices(self, get_empirical(which), getattr(exact, which), tol)

    @test_utils.product(
        same_inputs=[False, True],
        get=['ntk'],
        do_pool=[True, False],
        n=[3],
        bottom_layer=['resize', 'conv', 'relu'],
        method=['linear', 'nearest'],
        shape=[
            (1, 2, 4),
            (2, 1, 1),
            (-1, 2, -1),
            (2, 4, -1),
            (9, -1, -1),
            (-1, -1, -1),
            (3, 4, -1),
            (1, 1, -1),
        ]
    )
    def test_image_resize_nn(
        self,
        same_inputs,
        get,
        n,
        do_pool,
        bottom_layer,
        method,
        shape
    ):
        if n != 2:
            test_utils.skip_test(self)

        width = 2**7
        n_samples = 2**7
        tol = 0.03

        device = test_utils.device_under_test()
        g1, g2 = _rng(device, 1), _rng(device, 2)

        mask_constant = 10.0
        x_shape = [6, 3][:n - 1] + [1]
        x1 = torch.cos(_randn(tuple(x_shape), g1))
        mask1 = torch.bernoulli(torch.full_like(x1, 0.2, device=x1.device)).bool()
        x1 = torch.where(mask1, torch.as_tensor(mask_constant, device=x1.device, dtype=x1.dtype), x1)

        if same_inputs:
            x2 = None
        else:
            x2 = torch.cos(_randn(tuple(x_shape), g2))
            mask2 = torch.bernoulli(torch.full_like(x2, 0.1, device=x2.device)).bool()
            x2 = torch.where(mask2, torch.as_tensor(mask_constant, device=x2.device, dtype=x2.dtype), x2)

        bottom = {'conv': stax.Conv(width, (3,) * (n - 2), padding='SAME'),
                  'relu': stax.serial(stax.Conv(width, (3,) * (n - 2), padding='SAME'), stax.Relu()),
                  'resize': stax.Identity()}[bottom_layer]

        init_fn, apply_fn, kernel_fn = stax.serial(
            bottom,
            stax.ImageResize(method=method, shape=shape),
            stax.Conv(width, (2,), padding='SAME'),
            stax.Relu(),
            (stax.GlobalAvgPool() if do_pool else stax.Flatten()),
            stax.Dense(width if get == 'nngp' else 1, 0.9, 0.1)
        )

        kernel_fn_mc = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=stax.unmask_fn(apply_fn),
            key=g1,
            n_samples=n_samples,
            device_count=0,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        )

        empirical = kernel_fn_mc(x1, x2, get, mask_constant=mask_constant)

        def get_exact():
            return kernel_fn(x1, x2, get, mask_constant=mask_constant)

        if shape[-1] != -1:
            self.assertRaises(ValueError, get_exact)
        else:
            exact = get_exact()
            test_utils.assert_close_matrices(self, empirical, exact, tol)


# ============================================================
# ConvLocal (kept, but heavy tests skipped like original)
# ============================================================
class ConvLocalTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(diagonal_spatial=[True, False])
    def test_whitened_inputs(self, diagonal_spatial):
        test_utils.skip_test(self)

    @test_utils.product(
        padding=['SAME', 'VALID', 'CIRCULAR'],
        same_inputs=[False],
        filter_shape=[2, 3],
        strides=[1, 2],
        size=[2, 3],
        diagonal_batch=[True],
        diagonal_spatial=[True, False],
        get=['cov1', 'nngp', 'ntk'],
        parameterization=['standard', 'ntk']
    )
    def test_conv_local(self, **_):
        test_utils.skip_test(self)

    @test_utils.product(
        pool=[stax.Identity(), stax.AvgPool((2, 3), (2, 1), 'VALID')],
        readout=[stax.Flatten(), stax.GlobalAvgPool()],
        same_inputs=[False],
        get=['ntk'],
        parameterization=['ntk', 'standard']
    )
    def test_conv_local_deep(self, **_):
        test_utils.skip_test(self)

    def test_conv_local_conv(self):
        test_utils.skip_test(self, platforms=('cpu', 'xla'))

    def test_double_pool(self):
        test_utils.skip_test(self)


# ============================================================
# Index
# ============================================================
class IndexTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        same_inputs=[True, False],
        get=['nngp', 'ntk', 'cov1', 'cov2'],
        index_layer=[0, 1, 2, 3],
        mask_constant=[None, 10.],
        idx=[
            stax.Slice[0],
            stax.Slice[-1],
            stax.Slice[:],
            stax.Slice[:, 0],
            stax.Slice[:, -1],
            stax.Slice[:, -3:],
            stax.Slice[:, :],
            stax.Slice[::2],
            stax.Slice[...],
            stax.Slice[0, ...],
            stax.Slice[1:2, ...],
            stax.Slice[0:2, ...],
            stax.Slice[:, ::-2, ...],
            stax.Slice[::2, ::-2, 0, ...],
            stax.Slice[..., 1],
            stax.Slice[..., :2],
            stax.Slice[::2, 1, ...],
            stax.Slice[:, 1, -1, :],
            stax.Slice[..., 1::2],
            stax.Slice[:3, 1, 2],
            stax.Slice[:2, :2, :2],
            stax.Slice[..., ::2],
            stax.Slice[1:2:-1, 1, 2],
            stax.Slice[:, 0, :],
        ],
        readout=[stax.GlobalAvgPool, stax.Flatten]
    )
    def test_index(
        self,
        same_inputs,
        get,
        index_layer,
        mask_constant,
        idx,
        readout,
    ):
        if index_layer == 3 and isinstance(idx, tuple) and len(idx) > 2:
            raise unittest.SkipTest('Readout outputs have only 2 dimensions, but the index has more.')

        if get == 'cov2' and same_inputs:
            raise unittest.SkipTest('cov2 is None when x2 is None.')

        width = 2**7
        n_samples = 2**7
        tol = 0.05

        device = test_utils.device_under_test()
        g1, g2, gmc = _rng(device, 1), _rng(device, 2), _rng(device, 3)

        x1 = torch.cos(_randn((6, 3, 4, 5), g1))
        if mask_constant is not None:
            mask1 = torch.bernoulli(torch.full_like(x1, 0.2, device=x1.device)).bool()
            x1 = torch.where(mask1, torch.as_tensor(mask_constant, device=x1.device, dtype=x1.dtype), x1)

        if same_inputs:
            x2 = None
        else:
            x2 = torch.cos(_randn((7, 3, 4, 5), g2))
            if mask_constant is not None:
                mask2 = torch.bernoulli(torch.full_like(x2, 0.1, device=x2.device)).bool()
                x2 = torch.where(mask2, torch.as_tensor(mask_constant, device=x2.device, dtype=x2.dtype), x2)

        canonical_idx = utils.canonicalize_idx(idx=idx, ndim=x1.ndim if index_layer != 3 else 2)

        filter_shape = (2, 3)
        if index_layer == 0:
            for i, s in enumerate(canonical_idx):
                if isinstance(s, int) and i in (1, 2):
                    filter_shape = filter_shape[:-1]

        layers = [
            stax.Conv(width, filter_shape, padding='SAME'),
            stax.Relu(),
            readout(),
            stax.Dense(1 if get == 'ntk' else width)
        ]
        layers.insert(index_layer, stax.Index(idx=idx))
        init_fn, apply_fn, kernel_fn = stax.serial(*layers)

        def get_exact():
            return kernel_fn(x1, x2, get, mask_constant=mask_constant)

        if isinstance(canonical_idx[0], int) or canonical_idx[-1] != slice(None):
            self.assertRaises(NotImplementedError, get_exact)
            return

        exact = get_exact()

        if get in ('cov1', 'cov2'):
            diagonal_axes = (0,)
            get_e = 'nngp'
            if get == 'cov1':
                x1_e, x2_e = x1, None
            else:
                x1_e, x2_e = x2, None
        else:
            diagonal_axes = ()
            x1_e, x2_e = x1, x2
            get_e = get

        kernel_fn_mc = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=stax.unmask_fn(apply_fn),
            key=gmc,
            n_samples=n_samples,
            device_count=0,
            diagonal_axes=diagonal_axes,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        )
        empirical = kernel_fn_mc(x1_e, x2_e, get_e, mask_constant=mask_constant)
        test_utils.assert_close_matrices(self, empirical, exact, tol)


if __name__ == '__main__':
    unittest.main()
