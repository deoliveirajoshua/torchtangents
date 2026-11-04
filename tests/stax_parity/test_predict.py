import jax
import pytest
jax.config.update("jax_enable_x64", True)
import unittest
import numpy as np
import torch
import jax.numpy as jnp
from jax import random as jax_random
import neural_tangents as nt
import neural_tangents.stax as jax_stax

import torchtangents as tt
from torchtangents import stax as tt_stax

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

class TestPredict:
    def _run_predict_parity(self, architecture, input_shape, num_classes, diag_reg, learning_rate):
        # Generate random data
        rng = jax_random.PRNGKey(0)
        rng, key1, key2 = jax_random.split(rng, 3)
        
        n_train, n_test = 5, 3
        shape = (n_train,) + input_shape
        x_train = jax_random.normal(key1, shape)
        x_test = jax_random.normal(key2, (n_test,) + input_shape)
        y_train = jax_random.normal(rng, (n_train, num_classes))
        
        
        if len(input_shape) > 1:
            # For images, torch expects NCHW while JAX uses NHWC
            x_train_tt = torch.from_numpy(np.array(x_train)).permute(0, 3, 1, 2).requires_grad_(False)
            x_test_tt = torch.from_numpy(np.array(x_test)).permute(0, 3, 1, 2).requires_grad_(False)
        else:
            x_train_tt = torch.from_numpy(np.array(x_train)).requires_grad_(False)
            x_test_tt = torch.from_numpy(np.array(x_test)).requires_grad_(False)
            
        y_train_tt = torch.from_numpy(np.array(y_train)).requires_grad_(False)
        
        # Build architecture logic based on type
        if architecture == 'mlp':
            _, _, kernel_fn_jax = jax_stax.serial(jax_stax.Dense(16), jax_stax.Relu(), jax_stax.Dense(num_classes))
            kernel_fn_tt = tt_stax.Sequential(tt_stax.Linear(16), tt_stax.ReLU(), tt_stax.Linear(num_classes))
        elif architecture == 'cnn':
            _, _, kernel_fn_jax = jax_stax.serial(jax_stax.Conv(8, (3, 3), padding='SAME'), jax_stax.Flatten(), jax_stax.Dense(num_classes))
            kernel_fn_tt = tt_stax.Sequential(tt_stax.Conv(8, (3, 3), padding='SAME'), tt_stax.Flatten(), tt_stax.Linear(num_classes))
        elif architecture == 'deep_mlp':
            _, _, kernel_fn_jax = jax_stax.serial(jax_stax.Dense(32), jax_stax.Relu(), jax_stax.Dense(32), jax_stax.Relu(), jax_stax.Dense(num_classes))
            kernel_fn_tt = tt_stax.Sequential(tt_stax.Linear(32), tt_stax.ReLU(), tt_stax.Linear(32), tt_stax.ReLU(), tt_stax.Linear(num_classes))
            
        k_train_train_jax = kernel_fn_jax(x_train, None, get=('nngp', 'ntk'))
        k_test_train_jax = kernel_fn_jax(x_test, x_train, get=('nngp', 'ntk'))
        k_test_test_jax = kernel_fn_jax(x_test, None, get=('nngp', 'ntk'))
        
        k_train_train_tt = kernel_fn_tt(x_train_tt, None, get=('nngp', 'ntk'))
        k_test_train_tt = kernel_fn_tt(x_test_tt, x_train_tt, get=('nngp', 'ntk'))
        k_test_test_tt = kernel_fn_tt(x_test_tt, None, get=('nngp', 'ntk'))

        # gp_inference evaluations
        predict_fn_jax = nt.predict.gp_inference(k_train_train_jax, y_train, diag_reg=diag_reg)
        predict_fn_tt = tt.predict.gp_inference(k_train_train_tt, y_train_tt, diag_reg=diag_reg)
        
        for get in ('nngp', 'ntk'):
            mean_jax, cov_jax = predict_fn_jax(get=get, k_test_train=k_test_train_jax, k_test_test=k_test_test_jax)
            out_tt = predict_fn_tt(get=get, k_test_train=k_test_train_tt, k_test_test=k_test_test_tt)
            np.testing.assert_allclose(np.array(out_tt.mean), np.array(mean_jax), rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(np.array(out_tt.covariance), np.array(cov_jax), rtol=1e-4, atol=1e-6)

        # gradient_descent_mse evaluations
        gd_predict_fn_jax = nt.predict.gradient_descent_mse(k_train_train_jax.ntk, y_train, learning_rate=learning_rate, diag_reg=diag_reg)
        gd_predict_fn_tt = tt.predict.gradient_descent_mse(k_train_train_tt.ntk, y_train_tt, learning_rate=learning_rate, diag_reg=diag_reg)
        
        fx_train_0_jax = jnp.zeros_like(y_train)
        fx_test_0_jax = jnp.zeros((n_test, num_classes))
        
        fx_train_0_tt = torch.zeros_like(y_train_tt)
        fx_test_0_tt = torch.zeros((n_test, num_classes), dtype=y_train_tt.dtype)
        
        for t in [None, 0.5, 10.0]:
            fx_train_t_jax, fx_test_t_jax = gd_predict_fn_jax(t, fx_train_0_jax, fx_test_0_jax, k_test_train_jax.ntk)
            fx_train_t_tt, fx_test_t_tt = gd_predict_fn_tt(t, fx_train_0_tt, fx_test_0_tt, k_test_train_tt.ntk)
            np.testing.assert_allclose(np.array(fx_train_t_tt), np.array(fx_train_t_jax), rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(np.array(fx_test_t_tt), np.array(fx_test_t_jax), rtol=1e-4, atol=1e-6)
    @pytest.mark.parametrize("architecture, input_shape", [
        ('mlp', (10,)),
        ('deep_mlp', (10,)),
        ('cnn', (6, 6, 3))
    ])
    @pytest.mark.parametrize("diag_reg", [0.0, 1e-4, 0.1])
    @pytest.mark.parametrize("learning_rate", [1.0, 0.05])
    def test_predict_parity(self, architecture, input_shape, diag_reg, learning_rate):
        self._run_predict_parity(architecture, input_shape, num_classes=2, diag_reg=diag_reg, learning_rate=learning_rate)
