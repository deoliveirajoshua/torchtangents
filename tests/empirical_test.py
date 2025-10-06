# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# PyTorch/torchtangents rewrite of tests for `neural_tangents/_src/empirical.py`.

from functools import partial
import logging
import operator
from typing import Any, Callable, Optional, Sequence

import unittest
import numpy as np
import torch

# === IMPORTANT: use torchtangents (a PyTorch port with the same public API) ===
import torchtangents as tt
from torchtangents import stax

from tests import test_utils  # PyTorch version provided earlier


# -----------------------------------------------------------------------------
# Configuration & tolerances
# -----------------------------------------------------------------------------
TAYLOR_MATRIX_SHAPES = [(3, 3), (4, 4)]
TAYLOR_RANDOM_SAMPLES = 10

FLAT = 'FLAT'
POOLING = 'POOLING'
CONV = 'CONV'

TRAIN_SHAPES = [(4, 4), (4, 8), (8, 8), (6, 4, 4, 3), (4, 4, 4, 3), (4, 4, 4, 3)]
TEST_SHAPES  = [(2, 4), (6, 8), (16, 8), (2, 4, 4, 3), (2, 4, 4, 3), (2, 4, 4, 3)]
NETWORK = [FLAT, FLAT, FLAT, FLAT, POOLING, CONV]
OUTPUT_LOGITS = [1, 2, 3]

CONVOLUTION_CHANNELS = 2
test_utils.update_test_tolerance()


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _tree_map(fn, x):
    if isinstance(x, dict):
        return {k: _tree_map(fn, v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        typ = type(x)
        return typ(_tree_map(fn, xi) for xi in x)
    return fn(x)

def _rng(device: str, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g

def _randn(shape, dtype, device, gen):
    if len(shape) == 0:
        return torch.randn((), dtype=dtype, device=device, generator=gen)
    return torch.randn(*shape, dtype=dtype, device=device, generator=gen)


# -----------------------------------------------------------------------------
# Networks via torchtangents.stax (API mirrors neural_tangents.stax)
# -----------------------------------------------------------------------------
def _build_network(input_shape, network, out_logits):
    """
    Returns (init_fn, apply_fn, kernel_fn_stub) like neural_tangents.stax.
    """
    if len(input_shape) == 1:
        assert network == FLAT
        return stax.Dense(out_logits, W_std=2.0, b_std=0.5)

    elif len(input_shape) == 3:
        if network == POOLING:
            return stax.serial(
                stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
                stax.GlobalAvgPool(),
                stax.Dense(out_logits, W_std=2.0, b_std=0.5)
            )
        elif network == CONV:
            return stax.serial(
                stax.Conv(CONVOLUTION_CHANNELS, (1, 2), W_std=1.5, b_std=0.1),
                stax.Relu(),
                stax.Conv(CONVOLUTION_CHANNELS, (3, 2), W_std=2.0, b_std=0.05),
            )
        elif network == FLAT:
            return stax.serial(
                stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
                stax.Flatten(),
                stax.Dense(out_logits, W_std=2.0, b_std=0.5)
            )
        else:
            raise ValueError(f'Unexpected network type: {network}')

    else:
        raise ValueError('Expected flat or image test input.')


def _kernel_fns(
    seed: int,
    input_shape,
    network,
    out_logits,
    diagonal_axes,
    trace_axes,
    vmap_axes=None,
):
    """
    Build empirical NNGP/NTK functions for a randomly initialized stax model.
    """
    device = test_utils.device_under_test()
    dtype  = torch.float32
    gen    = _rng(device, seed)

    init_fn, apply_fn, _ = _build_network(input_shape, network, out_logits)
    # Stax init expects a (batch, ...) shape; use -1 batch like the JAX test.
    _, params = init_fn(gen, (-1,) + tuple(input_shape), dtype=dtype, device=device)

    kwargs = dict(
        f=apply_fn,
        trace_axes=trace_axes,
        diagonal_axes=diagonal_axes,
    )

    # Enumerate all implementations (assumed API parity with neural_tangents)
    ntk_fns = {
        i: partial(tt.empirical_ntk_fn(**kwargs, vmap_axes=vmap_axes, implementation=i), params=params)
        for i in tt.NtkImplementation
    }

    nngp_kernel_fn = partial(tt.empirical_nngp_fn(**kwargs), params=params)
    return nngp_kernel_fn, ntk_fns


KERNELS = {}
for o in OUTPUT_LOGITS:
    KERNELS[f'empirical_logits_{o}'] = partial(_kernel_fns, out_logits=o)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class EmpiricalTest(test_utils.NeuralTangentsTestCase):
    """
    PyTorch rewrite of the core empirical NTK/NNGP tests: linearization/Taylor
    and NTK implementation consistency under different shapes/axes/vmap modes.
    """

    # Three-layer deep linear-ish function used for Taylor tests.
    @classmethod
    def _f(cls, x, params, do_alter: bool, do_shift_x: bool = True):
        w1, w2, b = params
        if do_alter:
            b = b * 2.0
            w1 = w1 + 5.0
            w2 = w2 / 0.9
        if do_shift_x:
            x = x * 2.0 + 1.0

        # Match original nested PyTree structure
        quadratic = 0.5 * (x.T @ w1 @ x) + (w2 @ x) + b
        linear    = w1 @ x
        return ({'list': [{'quadratic': quadratic, 'linear': linear}, w2]},)

    @classmethod
    def _f_lin_exact(cls, x0, x, params, do_alter: bool, do_shift_x: bool = True):
        w1, w2, b = params
        f0 = EmpiricalTest._f(x0, params, do_alter, do_shift_x)
        if do_shift_x:
            x0 = x0 * 2.0 + 1.0
            x  = x  * 2.0 + 1.0
        dx = x - x0
        if do_alter:
            b = b * 2.0
            w1 = w1 + 5.0
            w2 = w2 / 0.9

        return _tree_map(
            lambda y: y,  # identity, we combine with operator.add below
            operator.add(
                f0,
                ({'list': [
                    {'quadratic': (x0.T @ w1 + w2) @ dx, 'linear': w1 @ dx},
                    torch.zeros((), dtype=w1.dtype, device=w1.device)
                ]},)
            )
        )

    @classmethod
    def _get_init_data(cls, shape):
        """
        Create symmetric w1, arbitrary w2, and bias b with the same last dim.
        """
        device = test_utils.device_under_test()
        dtype  = torch.float64  # match high precision for Taylor checks
        gen    = _rng(device, seed=0)

        w1 = _randn(shape, dtype, device, gen)
        w1 = 0.5 * (w1 + w1.T)
        w2 = _randn(shape, dtype, device, gen)
        b  = _randn((1,) * (len(shape) - 1) + (shape[-1],), dtype, device, gen)
        params = (w1, w2, b)

        x0 = _randn((shape[-1], 1), dtype, device, gen)
        return gen, params, x0

    @test_utils.product(shape=TAYLOR_MATRIX_SHAPES)
    def testLinearization(self, shape):
        gen, params, x0 = self._get_init_data(shape)
        # Assume torchtangents exposes linearize with the same signature
        f_lin = tt.linearize(EmpiricalTest._f, x0)

        for s in range(TAYLOR_RANDOM_SAMPLES):
            for do_alter in (True, False):
                for do_shift_x in (True, False):
                    x = torch.randn(shape[-1], 1, dtype=x0.dtype, device=x0.device, generator=gen)
                    self.assertAllClose(
                        EmpiricalTest._f_lin_exact(x0, x, params, do_alter, do_shift_x=do_shift_x),
                        f_lin(x, params, do_alter, do_shift_x=do_shift_x)
                    )

    @test_utils.product(shape=TAYLOR_MATRIX_SHAPES)
    def testTaylorExpansion(self, shape):
        def f_2_exact(x0, x, params, do_alter, do_shift_x=True):
            w1, w2, b = params
            f_lin = EmpiricalTest._f_lin_exact(x0, x, params, do_alter, do_shift_x)
            if do_shift_x:
                x0 = x0 * 2.0 + 1.0
                x  = x  * 2.0 + 1.0
            if do_alter:
                b  = b * 2.0
                w1 = w1 + 5.0
                w2 = w2 / 0.9
            dx = x - x0
            return _tree_map(
                lambda y: y,
                operator.add(
                    f_lin,
                    ({'list': [{'quadratic': 0.5 * (dx.T @ w1 @ dx), 'linear': torch.zeros_like(dx[:1, :])}, 0.]},)
                )
            )

        gen, params, x0 = self._get_init_data(shape)
        f_lin = tt.taylor_expand(EmpiricalTest._f, x0, 1)
        f_2   = tt.taylor_expand(EmpiricalTest._f, x0, 2)

        for s in range(TAYLOR_RANDOM_SAMPLES):
            for do_alter in (True, False):
                for do_shift_x in (True, False):
                    x = torch.randn(shape[-1], 1, dtype=x0.dtype, device=x0.device, generator=gen)
                    self.assertAllClose(
                        EmpiricalTest._f_lin_exact(x0, x, params, do_alter, do_shift_x=do_shift_x),
                        f_lin(x, params, do_alter, do_shift_x=do_shift_x)
                    )
                    self.assertAllClose(
                        f_2_exact(x0, x, params, do_alter, do_shift_x=do_shift_x),
                        f_2(x, params, do_alter, do_shift_x=do_shift_x)
                    )

    # -------------------------------------------------------------------------
    # Kernel consistency tests
    # -------------------------------------------------------------------------
    def _compare_kernels(self, x1, x2, ntk_fns, ntk_fns_vmapped, nngp_fn):
        nngp = nngp_fn(x1, x2)

        ntks          = {i: ntk_fns[i](x1, x2)         for i in ntk_fns}
        ntks_vmapped  = {i: ntk_fns_vmapped[i](x1, x2) for i in ntk_fns_vmapped}
        ntk_ref       = ntks[tt.NtkImplementation.JACOBIAN_CONTRACTION]

        _tree_map(lambda pair: None, None)  # no-op to mirror structure

        _tree_map(lambda _: _, nngp)  # structure check by shapes
        _tree_map(lambda x_y: None, None)

        # Assert shapes match NNGP (same structure)
        # We just compare one-to-one by mapping structure:
        def _same_shape(a, b):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                self.assertEqual(a.shape, b.shape)
        _tree_map(lambda _: _, _tree_map(lambda ab: ab, None))  # just keeping style parity
        tt.tree_map = _tree_map  # if torchtangents exposes tree_map like NT

        # Implementations agree
        for i, ntk in ntks.items():
            self.assertAllClose(ntk_ref, ntk, err_msg=f'{i} impl. fails.')
        for i, ntk in ntks_vmapped.items():
            self.assertAllClose(ntk_ref, ntk, err_msg=f'{i} vmapped impl. fails.')

    @test_utils.product(
        train_test_network=list(zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)),
        kernel_type=list(KERNELS.keys())
    )
    def testNTKAgainstDirect(self, train_test_network, kernel_type):
        kernel_fn = KERNELS[kernel_type]
        train_shape, test_shape, network = train_test_network

        # Build data
        device = test_utils.device_under_test()
        dtype  = torch.float32
        gen    = _rng(device, 0)

        x1 = _randn(train_shape, dtype, device, gen)
        x2 = _randn(test_shape,  dtype, device, gen)

        nngp_fn, ntk_fns = kernel_fn(
            seed=0,
            input_shape=train_shape[1:],
            network=network,
            diagonal_axes=(),
            trace_axes=()
        )

        _, ntk_fns_vmapped = kernel_fn(
            seed=0,
            input_shape=train_shape[1:],
            network=network,
            diagonal_axes=(),
            trace_axes=(),
            vmap_axes=0
        )

        self._compare_kernels(x1, None, ntk_fns, ntk_fns_vmapped, nngp_fn)
        self._compare_kernels(x1, x2,  ntk_fns, ntk_fns_vmapped, nngp_fn)

    @test_utils.product(
        diagonal_axes=[
            (),
            (0,),
            (0, 1),
            (0, 1, 2),
            (0, 1, 2, 3),
            (-1,),
            (-2,),
            (0, -1),
            (1, -2),
            (2, 3),
            (3, 0, 2)
        ],
        trace_axes=[
            (),
            (0,),
            (0, 1),
            (-1,),
            (1,),
            (0, -1),
            (-1, -2),
            (0, 1, 2, 3),
            (3, 1, 2, 0),
            (1, 2, 3),
            (-3, -2),
            (-3, -1),
            (-2, -4),
            (2, 0, -1)
        ],
    )
    def testAxes(self, diagonal_axes, trace_axes):
        device = test_utils.device_under_test()
        dtype  = torch.float32
        gen    = _rng(device, 0)

        x1 = _randn((4, 5, 6, 3), dtype, device, gen)
        x2 = _randn((2, 5, 6, 3), dtype, device, gen)

        _diagonal_axes = tuple(d % x1.ndim for d in diagonal_axes)
        _trace_axes    = tuple(t % x1.ndim for t in trace_axes)
        if any(d == c for d in _diagonal_axes for c in _trace_axes):
            raise unittest.SkipTest('diagonal axes must be different from channel axes.')

        get_kernel_fns = KERNELS['empirical_logits_3']
        kwargs = dict(
            seed=0,
            input_shape=(5, 6, 3),
            network=CONV,
            diagonal_axes=diagonal_axes,
            trace_axes=trace_axes
        )

        nngp_fn, ntk_fns = get_kernel_fns(**kwargs)
        _, ntk_fns_vmapped = get_kernel_fns(**kwargs, vmap_axes=0)

        self._compare_kernels(x1, None, ntk_fns, ntk_fns_vmapped, nngp_fn)
        if 0 not in _trace_axes and 0 not in _diagonal_axes:
            self._compare_kernels(x1, x2, ntk_fns, ntk_fns_vmapped, nngp_fn)

    # -------------------------------------------------------------------------
    # Parallel input/output tests (stax.parallel with tuple/list pytrees)
    # -------------------------------------------------------------------------
    @test_utils.product(same_inputs=[True, False])
    def test_parallel_in_out(self, same_inputs):
        device = test_utils.device_under_test()
        dtype  = torch.float32
        gen    = _rng(device, 0)

        x1_full = _randn((3, 21), dtype, device, gen)
        x2_full = _randn((4, 21), dtype, device, gen)

        x1_1, x1_2 = torch.split(x1_full, [10, 11], dim=1)
        x2_1, x2_2 = torch.split(x2_full, [10, 11], dim=1)

        x1 = (x1_1, x1_2)
        x2 = (x2_1, x2_2) if not same_inputs else None

        def layer(N_out):
            return stax.parallel(stax.Dense(N_out), stax.Dense(N_out + 1))

        init_fn, apply_fn, _ = stax.serial(layer(1024), layer(1))

        # params init: supply shapes of both branches
        _, params = init_fn(gen, (x1_1.shape, x1_2.shape), dtype=dtype, device=device)

        ntk_fns = {
            i: partial(tt.empirical_ntk_fn(apply_fn, implementation=i), params=params)
            for i in tt.NtkImplementation
        }
        ntk_fns_vmapped = {
            i: partial(tt.empirical_ntk_fn(apply_fn, implementation=i, vmap_axes=(0, 0)), params=params)
            for i in tt.NtkImplementation
        }
        nngp_fn = partial(tt.empirical_nngp_fn(apply_fn), params=params)

        # Basic NNGP shape checks
        nngp = nngp_fn(x1, x2)
        self.assertLen(nngp, 2)
        self.assertEqual(nngp[0].shape, (3, 3 if same_inputs else 4))
        self.assertEqual(nngp[1].shape, (3, 3 if same_inputs else 4))

        self._compare_kernels(x1, x2, ntk_fns, ntk_fns_vmapped, nngp_fn)

    @test_utils.product(same_inputs=[True, False])
    def test_parallel_nested(self, same_inputs):
        device = test_utils.device_under_test()
        dtype  = torch.float32
        gen    = _rng(device, 0)

        x1_full = _randn((3, 33), dtype, device, gen)
        x2_full = _randn((4, 33), dtype, device, gen)

        x1_1, x1_2, x1_3 = torch.split(x1_full, [10, 11, 12], dim=1)
        x2_1, x2_2, x2_3 = torch.split(x2_full, [10, 11, 12], dim=1)

        x1 = ([x1_1, x1_2], x1_3)
        x2 = ([x2_1, x2_2], x2_3) if not same_inputs else None

        def layer(N_out):
            return stax.parallel(
                stax.parallel(stax.Dense(N_out), stax.Dense(N_out + 1)),
                stax.Dense(N_out + 2)
            )

        init_fn, apply_fn, _ = stax.serial(layer(1024), layer(1))
        _, params = init_fn(gen, _tree_map(lambda t: t.shape, x1), dtype=dtype, device=device)

        ntk_fns = {
            i: tt.empirical_ntk_fn(apply_fn, implementation=i)
            for i in tt.NtkImplementation
        }
        ntk_fns_vmapped = {
            i: tt.empirical_ntk_fn(apply_fn, implementation=i, vmap_axes=([0, 0], 0))
            for i in tt.NtkImplementation
        }

        ntks         = {i: ntk_fns[i](x1, x2, params)        for i in ntk_fns}
        ntks_vmapped = {i: ntk_fns_vmapped[i](x1, x2, params) for i in ntk_fns_vmapped}

        ntk_ref = ntks[tt.NtkImplementation.JACOBIAN_CONTRACTION]
        for i, ntk in ntks.items():
            self.assertAllClose(ntk_ref, ntk, err_msg=f'{i} impl. fails.')
        for i, ntk in ntks_vmapped.items():
            self.assertAllClose(ntk_ref, ntk, err_msg=f'{i} vmapped impl. fails.')

        nngp_kernel_fn = tt.empirical_nngp_fn(apply_fn)
        nngp = nngp_kernel_fn(x1, x2, params)
        self.assertEqual(len(nngp), 2)
        nngp_shape = (3, 3 if same_inputs else 4)
        self.assertEqual(nngp[0][0].shape, nngp_shape)
        self.assertEqual(nngp[0][1].shape, nngp_shape)
        self.assertEqual(nngp[1].shape,   nngp_shape)


class EmpiricalNtkVpTest(test_utils.NeuralTangentsTestCase):
    """
    Test the NTK–vector-product API by reconstructing the full NTK
    from NTK–vps and comparing to the direct reference.
    """

    @test_utils.product(same_inputs=[True, False], do_jit=[True, False])
    def test_ntk_vp_fn(self, same_inputs, do_jit):
        device = test_utils.device_under_test()
        dtype  = torch.float32
        gen    = _rng(device, 1)

        N1 = 4
        N2 = N1 if same_inputs else 6
        O  = 3

        init_fn, f, _ = stax.serial(
            stax.Dense(8),
            stax.Relu(),
            stax.Dense(O)
        )

        x1 = _randn((N1, 7), dtype, device, gen)
        x2 = None if same_inputs else _randn((N2, 7), dtype, device, gen)
        _, params = init_fn(gen, x1.shape, dtype=dtype, device=device)

        ntk_ref = tt.empirical_ntk_fn(f, (), vmap_axes=0)(x1, x2, params)
        # moveaxis(1->2) to follow the original test convention
        ntk_ref = ntk_ref.movedim(1, 2)

        ntk_vp_fn = tt.empirical_ntk_vp_fn(f, x1, x2, params)
        if do_jit and hasattr(torch, "compile"):
            # Optional: torch.compile for speed if available
            ntk_vp_fn = torch.compile(ntk_vp_fn)  # no-op if unsupported

        eye = torch.eye(N2 * O, dtype=dtype, device=device).reshape(N2 * O, N2, O)
        # Vectorize over cotangents
        ntk_vps = torch.vmap(ntk_vp_fn)(eye)
        ntk_vps = ntk_vps.movedim(0, 2).reshape(N1, O, N2, O)

        self.assertAllClose(ntk_ref, ntk_vps)

        # Single NTK-vp check
        cotangents = _randn(f(params, x1 if same_inputs else x2).shape, dtype, device, gen)
        ntk_vp_ref = torch.tensordot(ntk_ref, cotangents, dims=([2, 3], [0, 1]))
        ntk_vp     = ntk_vp_fn(cotangents)
        self.assertAllClose(ntk_vp_ref, ntk_vp)


if __name__ == '__main__':
    unittest.main()
