# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# PyTorch/torchtangents rewrite of tests for `neural_tangents/_src/monte_carlo.py`.

import unittest
import torch

# Use the PyTorch port that mirrors neural_tangents’ public API.
import torchtangents as tt
from torchtangents import stax
from torchtangents._src import batching
from torchtangents._src import monte_carlo

from tests import test_utils


BATCH_SIZES = [2, 4]
WIDTH = 192
DEVICE_COUNTS = [0, 1, 2]
STORE_ON_DEVICE = [True, False]
ALL_GET = ('nngp', 'ntk', ('nngp', 'ntk'), None)

test_utils.update_test_tolerance()


def _rng(device: str, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _randn(shape, dtype, device, gen):
    if isinstance(shape, int):
        shape = (shape,)
    if len(shape) == 0:
        return torch.randn((), dtype=dtype, device=device, generator=gen)
    return torch.randn(*shape, dtype=dtype, device=device, generator=gen)


def _get_inputs_and_model(width=1, n_classes=2, use_conv=True):
    """Return x1, x2, init_fn, apply_fn, kernel_fn, gen."""
    device = test_utils.device_under_test()
    dtype = torch.float32
    gen = _rng(device, 1)

    x1 = _randn((8, 4, 3, 2), dtype, device, gen)
    x2 = _randn((4, 4, 3, 2), dtype, device, gen)

    if not use_conv:
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Conv(width, (3, 3)) if use_conv else stax.Dense(width),
        stax.Relu(),
        stax.Flatten(),
        stax.Dense(n_classes, 2.0, 0.5),
    )
    return x1, x2, init_fn, apply_fn, kernel_fn, gen


class MonteCarloTest(test_utils.NeuralTangentsTestCase):

    @test_utils.product(
        batch_size=BATCH_SIZES,
        device_count=DEVICE_COUNTS,
        store_on_device=STORE_ON_DEVICE,
        get=ALL_GET,
    )
    def test_sample_once_batch(self, batch_size, device_count, store_on_device, get):
        # Simulate multi-device pmap with vmap-stub if needed.
        test_utils.stub_out_pmap(batching, device_count)

        x1, x2, init_fn, apply_fn, _, gen = _get_inputs_and_model()
        kernel_fn = tt.empirical_kernel_fn(apply_fn)

        sample_once_fn = monte_carlo._sample_once_kernel_fn(kernel_fn, init_fn)
        sample_once_batch_fn = monte_carlo._sample_once_kernel_fn(
            kernel_fn, init_fn, batch_size, device_count, store_on_device
        )

        one_sample = sample_once_fn(x1, x2, gen, get)
        one_sample_batch = sample_once_batch_fn(x1, x2, gen, get)
        self._assertAllClose(one_sample, one_sample_batch)

    @test_utils.product(
        batch_size=BATCH_SIZES,
        device_count=DEVICE_COUNTS,
        store_on_device=STORE_ON_DEVICE,
        get=ALL_GET,
    )
    def test_batch_sample_once(self, batch_size, device_count, store_on_device, get):
        test_utils.stub_out_pmap(batching, device_count)

        x1, x2, init_fn, apply_fn, _, gen = _get_inputs_and_model()
        kernel_fn = tt.empirical_kernel_fn(apply_fn)

        sample_once_fn = monte_carlo._sample_once_kernel_fn(
            kernel_fn, init_fn, device_count=0
        )
        batch_sample_once_fn = batching.batch(
            sample_once_fn, batch_size, device_count, store_on_device
        )

        one_sample = sample_once_fn(x1, x2, gen, get)
        one_batch_sample = batch_sample_once_fn(x1, x2, gen, get)
        self._assertAllClose(one_sample, one_batch_sample)

    @test_utils.product(
        batch_size=BATCH_SIZES,
        device_count=DEVICE_COUNTS,
        store_on_device=STORE_ON_DEVICE,
    )
    def test_sample_vs_analytic_nngp(self, batch_size, device_count, store_on_device):
        test_utils.stub_out_pmap(batching, device_count)

        # Keep the TPU-specific branching from the original (torch doesn’t expose TPU here);
        # we just pass `use_conv=True` unconditionally for practicality.
        use_conv = (test_utils.device_under_test() == 'tpu')
        x1, x2, init_fn, apply_fn, stax_kernel_fn, gen = _get_inputs_and_model(
            WIDTH, 256, use_conv or True
        )

        sample = monte_carlo.monte_carlo_kernel_fn(
            init_fn, apply_fn, gen, n_samples=200,
            batch_size=batch_size, device_count=device_count,
            store_on_device=store_on_device
        )

        ker_empirical = sample(x1, x2, 'nngp')
        ker_analytic = stax_kernel_fn(x1, x2, 'nngp')

        test_utils.assert_close_matrices(self, ker_analytic, ker_empirical, 2e-2)

    @test_utils.product(
        batch_size=BATCH_SIZES,
        device_count=DEVICE_COUNTS,
        store_on_device=STORE_ON_DEVICE,
    )
    def test_monte_carlo_vs_analytic_ntk(self, batch_size, device_count, store_on_device):
        test_utils.stub_out_pmap(batching, device_count)

        use_conv = (test_utils.device_under_test() == 'tpu')
        x1, x2, init_fn, apply_fn, stax_kernel_fn, gen = _get_inputs_and_model(
            WIDTH, 2, use_conv or True
        )

        sample = monte_carlo.monte_carlo_kernel_fn(
            init_fn, apply_fn, gen, n_samples=100,
            batch_size=batch_size, device_count=device_count,
            store_on_device=store_on_device, vmap_axes=0
        )

        ker_empirical = sample(x1, x2, 'ntk')
        ker_analytic = stax_kernel_fn(x1, x2, 'ntk')

        test_utils.assert_close_matrices(self, ker_analytic, ker_empirical, 2e-2)

    @test_utils.product(
        batch_size=BATCH_SIZES,
        device_count=DEVICE_COUNTS,
        store_on_device=STORE_ON_DEVICE,
        get=ALL_GET,
    )
    def test_monte_carlo_generator(self, batch_size, device_count, store_on_device, get):
        test_utils.stub_out_pmap(batching, device_count)

        x1, x2, init_fn, apply_fn, stax_kernel_fn, gen = _get_inputs_and_model(8, 1)
        x3, x4, _, _, _, _ = _get_inputs_and_model(8, 1)

        log_n_max = 4
        n_samples = [2 ** k for k in range(log_n_max)]

        sample_generator = monte_carlo.monte_carlo_kernel_fn(
            init_fn, apply_fn, gen, n_samples,
            batch_size, device_count, store_on_device, vmap_axes=0
        )

        if get is None:
            samples_12 = sample_generator(x1, x2)
            samples_34 = sample_generator(x3, x4)

            count = 0
            for n, s_12, s_34 in zip(n_samples, samples_12, samples_34):
                sample_fn = monte_carlo.monte_carlo_kernel_fn(
                    init_fn, apply_fn, gen, n,
                    batch_size, device_count, store_on_device, vmap_axes=0
                )
                sample_12 = sample_fn(x1, x2)
                sample_34 = sample_fn(x3, x4)
                self._assertAllClose(s_12, sample_12)
                self._assertAllClose(s_12, s_34)
                self._assertAllClose(s_12, sample_34)
                count += 1

            self.assertEqual(log_n_max, count)

            ker_analytic_12 = stax_kernel_fn(x1, x2, ('nngp', 'ntk'))
            ker_analytic_34 = stax_kernel_fn(x3, x4, ('nngp', 'ntk'))

        else:
            samples_12 = sample_generator(x1, x2, get)
            samples_34 = sample_generator(x3, x4, get)

            count = 0
            for n, s_12, s_34 in zip(n_samples, samples_12, samples_34):
                sample_fn = monte_carlo.monte_carlo_kernel_fn(
                    init_fn, apply_fn, gen, n,
                    batch_size, device_count, store_on_device, vmap_axes=0
                )
                sample_12 = sample_fn(x1, x2, get)
                sample_34 = sample_fn(x3, x4, get)
                self._assertAllClose(s_12, sample_12)
                self._assertAllClose(s_12, s_34)
                self._assertAllClose(s_12, sample_34)
                count += 1

            self.assertEqual(log_n_max, count)

            ker_analytic_12 = stax_kernel_fn(x1, x2, get)
            ker_analytic_34 = stax_kernel_fn(x3, x4, get)

        # Loose tolerances (matches original JAX test’s intent).
        self._assertAllClose(ker_analytic_12, s_12, atol=2.0, rtol=2.0)
        self._assertAllClose(ker_analytic_12, ker_analytic_34)

    @test_utils.product(same_inputs=[True, False], batch_size=[1, 2])
    def test_parallel_in_out_mc(self, same_inputs, batch_size):
        device = test_utils.device_under_test()
        dtype = torch.float32
        gen = _rng(device, 0)

        # Build nested inputs ((B,2,5), (B,4,5)) with tuple nesting like original.
        x1_1 = _randn((3, 2, 5), dtype, device, gen)
        x1_2 = _randn((3, 4, 5), dtype, device, gen)
        x1_3 = _randn((3, 4, 5), dtype, device, gen)
        x1 = (x1_1, (x1_2, x1_3))

        if same_inputs:
            x2 = None
        else:
            x2_1 = _randn((3, 4, 5), dtype, device, gen)
            x2_2 = _randn((3, 4, 5), dtype, device, gen)
            x2_3 = _randn((3, 4, 5), dtype, device, gen)
            x2 = (x2_1, (x2_2, x2_3))

        def net(N_out):
            return stax.parallel(
                stax.Dense(N_out),
                stax.parallel(stax.Dense(N_out + 1), stax.Dense(N_out + 2)),
            )

        # NNGP: compare batched vs non-batched Monte Carlo estimators.
        init_fn, apply_fn, _ = net(WIDTH)

        nb_kernel_fn = monte_carlo.monte_carlo_kernel_fn(
            init_fn, apply_fn, gen, n_samples=4, trace_axes=(-1,)
        )
        kernel_fn = monte_carlo.monte_carlo_kernel_fn(
            init_fn, apply_fn, gen, n_samples=4, batch_size=batch_size, trace_axes=(-1,)
        )

        self._assertAllClose(kernel_fn(x1, x2, 'nngp'), nb_kernel_fn(x1, x2, 'nngp'))


if __name__ == "__main__":
    unittest.main()
