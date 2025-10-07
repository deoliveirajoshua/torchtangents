# PyTorch/torchtangents rewrite of tests for
# `neural_tangents/_src/stax/combinators.py`

import random as prandom
import unittest

import torch
from torchtangents import stax
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


class RepeatTest(test_utils.NeuralTangentsTestCase):

    def _test_repeat(self, x1, x2, layer, n, gen_params, **kwargs):
        # Build an explicit stack vs stax.repeat
        if n == 0:
            init_fn, apply_fn, kernel_fn = stax.Identity()
        else:
            init_fn, apply_fn, kernel_fn = stax.serial(*([layer] * n))

        init_fn_r, apply_fn_r, kernel_fn_r = stax.repeat(layer, n)

        out_shape, params = init_fn(gen_params, x1.shape)
        out_shape_r, params_r = init_fn_r(gen_params, x1.shape)
        self.assertEqual(out_shape, out_shape_r)

        # For apply, kwargs may be a (p1, p2) pair; use the first for forward.
        kwargs_apply = {k: (v[0] if isinstance(v, tuple) else v) for k, v in kwargs.items()}
        out = apply_fn(params, x1, **kwargs_apply)
        out_r = apply_fn_r(params_r, x1, **kwargs_apply)
        self.assertAllClose(out, out_r)

        # Compare kernels under multiple `get` arguments.
        for get in [None, 'ntk', 'nngp', 'cov1',
                    ('nngp', 'cov1'), ('cov1', 'ntk')]:
            with self.subTest(get=get):
                k = kernel_fn(x1, x2, get, **kwargs)
                k_r = kernel_fn_r(x1, x2, get, **kwargs)
                self.assertAllClose(k, k_r)

    # -------- Dense / simple stacks --------
    @test_utils.product(
        same_inputs=[False, True],
        n=[0, 1, 2, 3],
        layer=[
            stax.Identity(),
            stax.Dense(3),
            stax.serial(stax.Identity()),
            stax.serial(stax.Dense(3)),
            stax.GlobalAvgPool(),
            stax.serial(stax.Dense(3), stax.Relu()),
            stax.serial(stax.Dense(3), stax.Relu(), stax.Dense(3)),
        ],
    )
    def test_repeat(self, same_inputs, n, layer):
        device = test_utils.device_under_test()
        gen_inp, gen_params = _rng(device, 1), _rng(device, 2)
        x1 = torch.cos(_randn((2, 3), gen_inp))
        x2 = None if same_inputs else _randn((4, 3), gen_inp)
        self._test_repeat(x1, x2, layer, n, gen_params)

    # -------- Convolutional stacks --------
    @test_utils.product(
        same_inputs=[False, True],
        n=[0, 1, 2, 3],
        layer=[
            stax.serial(
                stax.Conv(3, (2, 2), padding='SAME'),
                stax.Relu(),
                stax.Conv(3, (2, 2), padding='SAME'),
                stax.Gelu(),
            )
        ],
    )
    def test_repeat_conv(self, same_inputs, n, layer):
        device = test_utils.device_under_test()
        gen_inp, gen_params = _rng(device, 3), _rng(device, 4)
        x1 = torch.cos(_randn((2, 4, 4, 3), gen_inp))
        x2 = None if same_inputs else _randn((4, 4, 4, 3), gen_inp)
        self._test_repeat(x1, x2, layer, n, gen_params)

    # -------- Aggregate stacks (with pattern kwarg) --------
    @test_utils.product(
        same_inputs=[False, True],
        n=[0, 1, 2, 3],
        layer=[
            stax.Aggregate(),
            stax.serial(stax.Dense(3), stax.Aggregate(), stax.Abs()),
            stax.serial(
                stax.Conv(3, (2, 2), padding='SAME'),
                stax.Aggregate(),
                stax.Abs(),
                stax.Conv(3, (1, 2), padding='SAME'),
            ),
        ],
    )
    def test_repeat_agg(self, same_inputs, n, layer):
        device = test_utils.device_under_test()
        gen_inp, gen_params, gen_p1, gen_p2 = _rng(device, 5), _rng(device, 6), _rng(device, 7), _rng(device, 8)
        x1 = torch.cos(_randn((2, 4, 3, 3), gen_inp))
        x2 = None if same_inputs else _randn((4, 4, 3, 3), gen_inp)

        # Construct patterns like the JAX test:
        # p has shape: x.shape[:-1] + x.shape[1:-1]  (batch + spatial, spatial)
        p1 = _randn(tuple(x1.shape[:-1]) + tuple(x1.shape[1:-1]), gen_p1)
        p2 = p1 if x2 is None else _randn(tuple(x2.shape[:-1]) + tuple(x2.shape[1:-1]), gen_p2)

        # With a concrete pattern pair
        self._test_repeat(x1, x2, layer, n, gen_params, pattern=(p1, p2))
        # And with no pattern
        self._test_repeat(x1, x2, layer, n, gen_params, pattern=(None, None))


if __name__ == '__main__':
    unittest.main()
