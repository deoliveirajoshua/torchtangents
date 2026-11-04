import jax
jax.config.update("jax_enable_x64", True)
import unittest
import neural_tangents.stax as jax_stax
from torchtangents import stax as tt_stax
from test_parity import _test_parity

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' 

class TestResnetParity(unittest.TestCase):
    def test_resnet(self):
        def WideResnetBlock_jax(channels, strides=(1, 1), channel_mismatch=False):
            Main = jax_stax.serial(
                jax_stax.Relu(), jax_stax.Conv(channels, (3, 3), strides, padding='SAME'),
                jax_stax.Relu(), jax_stax.Conv(channels, (3, 3), padding='SAME'))
            Shortcut = jax_stax.Identity() if not channel_mismatch else jax_stax.Conv(
                channels, (3, 3), strides, padding='SAME')
            return jax_stax.serial(jax_stax.FanOut(2),
                                   jax_stax.parallel(Main, Shortcut),
                                   jax_stax.FanInSum())

        def WideResnetGroup_jax(n, channels, strides=(1, 1)):
            blocks = []
            blocks += [WideResnetBlock_jax(channels, strides, channel_mismatch=True)]
            for _ in range(n - 1):
                blocks += [WideResnetBlock_jax(channels, (1, 1))]
            return jax_stax.serial(*blocks)

        def WideResnet_jax(block_size, k, num_classes):
            return jax_stax.serial(
                jax_stax.Conv(16, (3, 3), padding='SAME'),
                WideResnetGroup_jax(block_size, int(16 * k)),
                WideResnetGroup_jax(block_size, int(32 * k), (2, 2)),
                WideResnetGroup_jax(block_size, int(64 * k), (2, 2)),
                jax_stax.AvgPool((8, 8)),
                jax_stax.Flatten(),
                jax_stax.Dense(num_classes, 1., 0.))

        # PyTorch Equivalent
        def WideResnetBlock_tt(channels, strides=(1, 1), channel_mismatch=False):
            Main = tt_stax.Sequential(
                tt_stax.ReLU(), tt_stax.Conv(channels, (3, 3), strides, padding='SAME'),
                tt_stax.ReLU(), tt_stax.Conv(channels, (3, 3), padding='SAME'))
            Shortcut = tt_stax.Identity() if not channel_mismatch else tt_stax.Conv(
                channels, (3, 3), strides, padding='SAME')
            return tt_stax.Sequential(tt_stax.FanOut(2),
                                  tt_stax.parallel(Main, Shortcut),
                                  tt_stax.FanInSum())

        def WideResnetGroup_tt(n, channels, strides=(1, 1)):
            blocks = []
            blocks += [WideResnetBlock_tt(channels, strides, channel_mismatch=True)]
            for _ in range(n - 1):
                blocks += [WideResnetBlock_tt(channels, (1, 1))]
            return tt_stax.Sequential(*blocks)

        def WideResnet_tt(block_size, k, num_classes):
            return tt_stax.Sequential(
                tt_stax.Conv(16, (3, 3), padding='SAME'),
                WideResnetGroup_tt(block_size, int(16 * k)),
                WideResnetGroup_tt(block_size, int(32 * k), (2, 2)),
                WideResnetGroup_tt(block_size, int(64 * k), (2, 2)),
                tt_stax.AvgPool((8, 8)),
                tt_stax.Flatten(),
                tt_stax.Linear(num_classes, 1., 0.))

        # Limit block size to 2 to keep matrix ops small/quick for testing.
        _test_parity(
            WideResnet_jax(block_size=2, k=1, num_classes=10),
            WideResnet_tt(block_size=2, k=1, num_classes=10),
            input_shape=(2, 64, 64, 3),
        )

if __name__ == '__main__':
    unittest.main()
