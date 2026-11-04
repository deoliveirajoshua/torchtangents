import jax
import pytest
jax.config.update("jax_enable_x64", True)
import numpy as np
import torch
import jax.numpy as jnp
from jax import random as jax_random
import neural_tangents.stax as jax_stax

import torchtangents as tt
from torchtangents import stax as tt_stax
from torchtangents.stax import batch

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

class TestBatching:
    def _run_batching_parity(self, architecture, input_shape, num_classes, batch_size):
        rng = jax_random.PRNGKey(0)
        rng, key1, key2 = jax_random.split(rng, 3)
        
        n_train, n_test = 100, 60
        shape = (n_train,) + input_shape
        x_train = jax_random.normal(key1, shape)
        x_test = jax_random.normal(key2, (n_test,) + input_shape)
        
        if len(input_shape) > 1:
            x_train_tt = torch.from_numpy(np.array(x_train)).permute(0, 3, 1, 2)
            x_test_tt = torch.from_numpy(np.array(x_test)).permute(0, 3, 1, 2)
        else:
            x_train_tt = torch.from_numpy(np.array(x_train))
            x_test_tt = torch.from_numpy(np.array(x_test))
            
        if architecture == 'mlp':
            _, _, kernel_fn_jax = jax_stax.serial(jax_stax.Dense(32), jax_stax.Relu(), jax_stax.Dense(num_classes))
            kernel_fn_tt = tt_stax.Sequential(tt_stax.Linear(32), tt_stax.ReLU(), tt_stax.Linear(num_classes))
        elif architecture == 'cnn':
            _, _, kernel_fn_jax = jax_stax.serial(jax_stax.Conv(8, (3, 3), padding='SAME'), jax_stax.Flatten(), jax_stax.Dense(num_classes))
            kernel_fn_tt = tt_stax.Sequential(tt_stax.Conv(8, (3, 3), padding='SAME'), tt_stax.Flatten(), tt_stax.Linear(num_classes))

        # Full Memory JAX
        k_train_train_jax = kernel_fn_jax(x_train, None, get=('nngp', 'ntk'))
        k_test_train_jax = kernel_fn_jax(x_test, x_train, get=('nngp', 'ntk'))
        
        # Batched TorchTangents
        batched_kernel_fn_tt = batch(kernel_fn_tt, batch_size=batch_size, store_on_device=True)
        
        k_train_train_tt = batched_kernel_fn_tt(x_train_tt, None, get=('nngp', 'ntk'))
        k_test_train_tt = batched_kernel_fn_tt(x_test_tt, x_train_tt, get=('nngp', 'ntk'))
        
        np.testing.assert_allclose(np.array(k_train_train_tt.ntk), np.array(k_train_train_jax.ntk), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.array(k_train_train_tt.nngp), np.array(k_train_train_jax.nngp), rtol=1e-4, atol=1e-5)
        
        np.testing.assert_allclose(np.array(k_test_train_tt.ntk), np.array(k_test_train_jax.ntk), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.array(k_test_train_tt.nngp), np.array(k_test_train_jax.nngp), rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("architecture, input_shape", [
        ('mlp', (10,)),
        ('cnn', (6, 6, 3))
    ])
    @pytest.mark.parametrize("batch_size", [2, 5])
    def test_batching_parity(self, architecture, input_shape, batch_size):
        self._run_batching_parity(architecture, input_shape, num_classes=2, batch_size=batch_size)
