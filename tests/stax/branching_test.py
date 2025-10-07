# PyTorch/torchtangents rewrite of tests for
# `neural_tangents/_src/stax/branching.py`

import random as prandom
import unittest
import torch

import torchtangents as nt
from torchtangents import stax
from tests import test_utils


test_utils.update_test_tolerance()
prandom.seed(1)


def _rng(device: str | torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)  # seeds params; ops run on tensors' device
    g.manual_seed(seed)
    return g


def _randn(shape, gen, dtype=torch.float32, device=None):
    device = device or test_utils.device_under_test()
    return torch.randn(*shape, generator=gen, dtype=dtype, device=device)


class FanInTest(test_utils.NeuralTangentsTestCase):

    @classmethod
    def _get_phi(cls, i):
        return {
            0: stax.Relu(),
            1: stax.Erf(),
            2: stax.Abs(),
        }[i % 3]

    @test_utils.product(
        same_inputs=[False],
        axis=[0, 1],
        n_branches=[3],
        get=['ntk'],
        branch_in=['dense_before_branch_in', 'dense_after_branch_in'],
        fan_in_mode=['FanInSum', 'FanInConcat', 'FanInProd'],
    )
    def test_fan_in_fc(
        self,
        same_inputs,
        axis,
        n_branches,
        get,
        branch_in,
        fan_in_mode,
    ):
        # Match the conditional skips/constraints from the JAX test.
        if fan_in_mode in ['FanInSum', 'FanInProd']:
            if axis != 0:
                self.skipTest('`FanInSum`/`FanInProd` only supported with axis=0 for this test.')
            axis = None
        if (fan_in_mode == 'FanInSum' or axis == 0) and branch_in == 'dense_after_branch_in':
            self.skipTest('`FanInSum` and `FanInConcat(0)` require is_gaussian downstream.')
        if ((axis == 1 or fan_in_mode == 'FanInProd') and branch_in == 'dense_before_branch_in'):
            self.skipTest('Feature-axis concat/prod requires a dense after FanIn.')

        if n_branches != 3:
            test_utils.skip_test(self)

        device = test_utils.device_under_test()
        gen = _rng(device, 1)
        x1 = torch.cos(_randn((4, 3), gen))
        x2 = None if same_inputs else _randn((8, 3), gen)

        width = 1024
        n_samples = 512
        tol = 0.02

        dense = stax.Dense(width, 1.25, 0.1)

        # Input fork
        input_layers = [dense, stax.FanOut(n_branches)]

        # Build branches of increasing depth/width like the original
        branches = []
        for b in range(n_branches):
            bl = [FanInTest._get_phi(b)]
            for i in range(b):
                multiplier = 1 if axis not in (1, -1) else (1 + 0.25 * i)
                bl += [
                    stax.Dense(int(width * multiplier), 1.0 + 2 * i, 0.5 + i),
                    FanInTest._get_phi(i),
                ]
            if branch_in == 'dense_before_branch_in':
                bl += [dense]
            branches.append(stax.serial(*bl))

        # Fan-in choice
        if fan_in_mode == 'FanInSum':
            fan_in_layer = stax.FanInSum()
        elif fan_in_mode == 'FanInProd':
            fan_in_layer = stax.FanInProd()
        else:
            fan_in_layer = stax.FanInConcat(axis)

        output_layers = [fan_in_layer, stax.Relu()]
        if branch_in == 'dense_after_branch_in':
            output_layers.insert(1, dense)

        nn = stax.serial(*(input_layers + [stax.parallel(*branches)] + output_layers))

        if get == 'nngp':
            init_fn, apply_fn, kernel_fn = nn
        elif get == 'ntk':
            init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(1, 1.25, 0.5))
        else:
            raise ValueError(get)

        # MC kernel
        mc_kernel_fn = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=apply_fn,
            key=42,                 # torchtangents accepts ints/seeds
            n_samples=n_samples,
            device_count=-1,
            implementation=nt._DEFAULT_TESTING_NTK_IMPLEMENTATION,  # keep parity with originals
            vmap_axes=0 if axis not in (0, -2) else None,
        )

        exact = kernel_fn(x1, x2, get=get)
        empirical = mc_kernel_fn(x1, x2, get=get)
        test_utils.assert_close_matrices(self, empirical, exact, tol)

    @test_utils.product(
        same_inputs=[False],
        axis=[0, 1, 2, 3],
        n_branches=[2],
        get=['ntk'],
        branch_in=['dense_before_branch_in', 'dense_after_branch_in'],
        readout=['pool', 'flatten'],
        fan_in_mode=['FanInSum', 'FanInConcat', 'FanInProd'],
    )
    def test_fan_in_conv(
        self,
        same_inputs,
        axis,
        n_branches,
        get,
        branch_in,
        readout,
        fan_in_mode,
    ):
        # Keep parity with the original which skips this suite in many settings.
        test_utils.skip_test(self)

        if fan_in_mode in ['FanInSum', 'FanInProd']:
            if axis != 0:
                self.skipTest('`FanInSum`/`FanInProd` only supported with axis=0 for this test.')
            axis = None
        if (fan_in_mode == 'FanInSum' or axis in [0, 1, 2]) and branch_in == 'dense_after_branch_in':
            self.skipTest('FanInSum and FanInConcat(0/1/2) require is_gaussian.')
        if ((axis == 3 or fan_in_mode == 'FanInProd') and branch_in == 'dense_before_branch_in'):
            self.skipTest('Feature-axis concat/prod requires dense after FanIn.')

        device = test_utils.device_under_test()
        gen = _rng(device, 3)
        x1 = _randn((2, 5, 6, 3), gen)
        x2 = None if same_inputs else _randn((3, 5, 6, 3), gen)

        width = 1024
        n_samples = 512
        tol = 0.015

        conv = stax.Conv(
            out_chan=width,
            filter_shape=(3, 3),
            padding='SAME',
            W_std=1.25,
            b_std=0.1,
        )

        input_layers = [conv, stax.FanOut(n_branches)]

        branches = []
        for b in range(n_branches):
            bl = [FanInTest._get_phi(b)]
            for i in range(b):
                multiplier = 1 if axis not in (3, -1) else (1 + 0.25 * i)
                bl += [
                    stax.Conv(
                        out_chan=int(width * multiplier),
                        filter_shape=(i + 1, 4 - i),
                        padding='SAME',
                        W_std=1.25 + i,
                        b_std=0.1 + i,
                    ),
                    FanInTest._get_phi(i),
                ]
            if branch_in == 'dense_before_branch_in':
                bl += [conv]
            branches.append(stax.serial(*bl))

        if fan_in_mode == 'FanInSum':
            fan_in_layer = stax.FanInSum()
        elif fan_in_mode == 'FanInProd':
            fan_in_layer = stax.FanInProd()
        else:
            fan_in_layer = stax.FanInConcat(axis)

        readout_layer = stax.GlobalAvgPool() if readout == 'pool' else stax.Flatten()
        output_layers = [fan_in_layer, stax.Relu(), readout_layer]
        if branch_in == 'dense_after_branch_in':
            output_layers.insert(1, conv)

        nn = stax.serial(*(input_layers + [stax.parallel(*branches)] + output_layers))
        init_fn, apply_fn, kernel_fn = stax.serial(
            nn, stax.Dense(1 if get == 'ntk' else width, 1.25, 0.5)
        )

        mc_kernel_fn = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=apply_fn,
            key=24,
            n_samples=n_samples,
            device_count=-1,
            implementation=nt._DEFAULT_TESTING_NTK_IMPLEMENTATION,
            vmap_axes=0 if axis not in (0, -4) else None,
        )

        exact = kernel_fn(x1, x2, get=get)
        empirical = mc_kernel_fn(x1, x2, get=get)
        test_utils.assert_close_matrices(self, empirical, exact, tol)


if __name__ == '__main__':
    unittest.main()
