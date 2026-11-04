import jax
jax.config.update("jax_enable_x64", True)
import unittest
import numpy as np
import torch
import jax.numpy as jnp
import jax.numpy as jnp
from jax import random as jax_random
import neural_tangents.stax as jax_stax
from torchtangents import stax as tt_stax

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # Prevents JAX from preallocating memory, which can causes some OOM errors

def assert_trees_allclose(jax_out, tt_out, rtol=1e-5, atol=1e-7, verbose=False):
    if verbose:
        print("Testing NNGP parity...")
    if hasattr(jax_out, 'nngp'):
        nngp_jax = np.array(jax_out.nngp)
        nngp_tt = np.array(tt_out.nngp)
        if verbose:
            print("JAX NNGP:")
            print(nngp_jax)
            print("Torch NNGP:")
            print(nngp_tt)
        np.testing.assert_allclose(nngp_tt, nngp_jax, rtol=rtol, atol=atol)
    if verbose:
        print("Testing NTK parity...")
    if hasattr(jax_out, 'ntk') and getattr(jax_out, 'ntk') is not None:
        ntk_jax = np.array(jax_out.ntk)
        ntk_tt = np.array(tt_out.ntk)
        if verbose:
            print("JAX NTK:")
            print(ntk_jax)
            print("Torch NTK:")
            print(ntk_tt)
        np.testing.assert_allclose(ntk_tt, ntk_jax, rtol=rtol, atol=atol)

def _test_parity(jax_layer, tt_layer, input_shape=(5, 10), is_same_input=False, is_branching=False, tuple_in=False, verbose=False, rtol=1e-4, atol=1e-6):
    rng = jax_random.PRNGKey(0)
    
    if tuple_in:
        x1_jax = tuple(jax_random.normal(jax_random.PRNGKey(i), shape) for i, shape in enumerate(input_shape))
    else:
        x1_jax = jax_random.normal(rng, input_shape)

    if is_same_input:
        x2_jax = None
        if tuple_in:
            x1_tt = tuple(torch.from_numpy(np.array(x)).requires_grad_(False) for x in x1_jax)
            x1_tt = tuple(x.permute(0, 3, 1, 2) if x.ndim == 4 else x for x in x1_tt)
        else:
            x1_tt = torch.from_numpy(np.array(x1_jax)).requires_grad_(False)
            if len(input_shape) == 4:
                x1_tt = x1_tt.permute(0, 3, 1, 2)
        x2_tt = None
    else:
        if tuple_in:
            x2_jax = tuple(jax_random.normal(jax_random.PRNGKey(10 + i), shape) for i, shape in enumerate(input_shape))
            x1_tt = tuple(torch.from_numpy(np.array(x)).requires_grad_(False) for x in x1_jax)
            x2_tt = tuple(torch.from_numpy(np.array(x)).requires_grad_(False) for x in x2_jax)
            
            x1_tt = tuple(x.permute(0, 3, 1, 2) if x.ndim == 4 else x for x in x1_tt)
            x2_tt = tuple(x.permute(0, 3, 1, 2) if x.ndim == 4 else x for x in x2_tt)
        else:
            x2_jax = jax_random.normal(jax_random.PRNGKey(1), input_shape)
            x1_tt = torch.from_numpy(np.array(x1_jax)).requires_grad_(False)
            x2_tt = torch.from_numpy(np.array(x2_jax)).requires_grad_(False)
            if len(input_shape) == 4:
                x1_tt = x1_tt.permute(0, 3, 1, 2)
                x2_tt = x2_tt.permute(0, 3, 1, 2)

    # JAX still uses the (init_fn, apply_fn, kernel_fn) triple
    if isinstance(jax_layer, tuple):
        _, _, j_kernel_fn = jax_layer
    else:
        j_kernel_fn = jax_layer

    # TT layers are now single StaxKernelModule objects (which are callable)
    tt_kernel_fn = tt_layer

    get = ('nngp', 'ntk')
    jax_out = j_kernel_fn(x1_jax, x2_jax, get=get)
    tt_out = tt_kernel_fn(x1_tt, x2_tt, get=get)

    if is_branching:
        for j_out, t_out in zip(jax_out, tt_out):
            assert_trees_allclose(j_out, t_out, verbose=verbose, rtol=rtol, atol=atol)
    else:
        assert_trees_allclose(jax_out, tt_out, verbose=verbose, rtol=rtol, atol=atol)


class TestParity(unittest.TestCase):
    def test_dense(self):
        _test_parity(jax_stax.Dense(16), tt_stax.Linear(16))

    def test_relu(self):
        layer_jax = jax_stax.serial(jax_stax.Dense(16), jax_stax.Relu())
        layer_tt = tt_stax.Sequential(tt_stax.Linear(16), tt_stax.ReLU())
        _test_parity(layer_jax, layer_tt)

    def test_conv_same(self):
        _test_parity(
            jax_stax.Conv(16, (3, 3), padding='SAME'),
            tt_stax.Conv(16, (3, 3), padding='SAME'),
            input_shape=(2, 64, 64, 3),
        )

    def test_conv_valid(self):
        _test_parity(
            jax_stax.Conv(16, (3, 3), padding='VALID'),
            tt_stax.Conv(16, (3, 3), padding='VALID'),
            input_shape=(2, 64, 64, 3),
        )
    
    def test_conv_circular(self):
        _test_parity(
            jax_stax.Conv(16, (3, 3), padding='CIRCULAR'),
            tt_stax.Conv(16, (3, 3), padding='CIRCULAR'),
            input_shape=(2, 64, 64, 3),
        )

    
    def test_conv_transpose_same(self):
        _test_parity(
            jax_stax.ConvTranspose(4, (3, 3), padding='SAME', strides=(2, 2)),
            tt_stax.ConvTranspose(4, (3, 3), padding='SAME', strides=(2, 2)),
            input_shape=(2, 8, 8, 3),
        )

    def test_conv_transpose_valid(self):
        _test_parity(
            jax_stax.ConvTranspose(4, (3, 3), padding='VALID'),
            tt_stax.ConvTranspose(4, (3, 3), padding='VALID'),
            input_shape=(2, 8, 8, 3),
        )

    
    def test_conv_transpose_circular(self):
        _test_parity(
            jax_stax.ConvTranspose(4, (3, 3), padding='CIRCULAR', strides=(2, 2)),
            tt_stax.ConvTranspose(4, (3, 3), padding='CIRCULAR', strides=(2, 2)),
            input_shape=(2, 8, 8, 3),
        )


    def test_conv_local_same(self):
        _test_parity(
            jax_stax.ConvLocal(4, (3, 3), padding='SAME'),
            tt_stax.ConvLocal(4, (3, 3), padding='SAME'),
            input_shape=(2, 8, 8, 3),
        )


    def test_conv_local_valid(self):
        _test_parity(
            jax_stax.ConvLocal(4, (3, 3), padding='VALID'),
            tt_stax.ConvLocal(4, (3, 3), padding='VALID'),
            input_shape=(2, 8, 8, 3),
        )


    def test_conv_local_circular(self):
        _test_parity(
            jax_stax.ConvLocal(4, (3, 3), padding='CIRCULAR'),
            tt_stax.ConvLocal(4, (3, 3), padding='CIRCULAR'),
            input_shape=(2, 8, 8, 3),
        )

    def test_flatten(self):
        _test_parity(
            jax_stax.serial(jax_stax.Conv(16, (3, 3)), jax_stax.Flatten(), jax_stax.Dense(16)),
            tt_stax.Sequential(tt_stax.Conv(16, (3, 3)), tt_stax.Flatten(), tt_stax.Linear(16)),
            input_shape=(2, 64, 64, 3),
        )

    def test_layer_norm(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.LayerNorm()),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.LayerNorm(axis=1)),
            input_shape=(2, 64, 64, 3),
        )

    def test_identity(self):
        _test_parity(jax_stax.Identity(), tt_stax.Identity())
        
    def test_global_avg_pool(self):
        _test_parity(
            jax_stax.serial(jax_stax.Conv(4, (3, 3)), jax_stax.GlobalAvgPool()),
            tt_stax.Sequential(tt_stax.Conv(4, (3, 3)), tt_stax.GlobalAvgPool()),
            input_shape=(2, 64, 64, 3),
        )
        
    def test_dropout(self):
        # Dropout() evaluates to Identity() during infinite width analysis basically
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Dropout(0.5)),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.Dropout(0.5)),
            input_shape=(2, 64, 64, 3),
        )

    def test_abs(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Abs()),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.Abs()),
        )

    def test_erf(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Erf()),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.Erf()),
        )

    def test_gelu(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Gelu()),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.GeLU()),
        )

    def test_sign(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Sign()),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.Sign()),
        )

    def test_exp(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Exp()),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.Exp()),
        )

    def test_rbf(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Rbf()),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.Rbf()),
        )

    def test_leaky_relu(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.LeakyRelu(0.1)),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.LeakyReLU(0.1)),
        )

    def test_global_sum_pool(self):
        _test_parity(
            jax_stax.serial(jax_stax.Conv(4, (3, 3)), jax_stax.GlobalSumPool()),
            tt_stax.Sequential(tt_stax.Conv(4, (3, 3)), tt_stax.GlobalSumPool()),
            input_shape=(2, 64, 64, 3),
        )

    def test_avg_pool(self):
        _test_parity(
            jax_stax.serial(jax_stax.Conv(4, (3, 3), padding='SAME'), jax_stax.AvgPool((2, 2))),
            tt_stax.Sequential(tt_stax.Conv(4, (3, 3), padding='SAME'), tt_stax.AvgPool((2, 2))),
            input_shape=(2, 4, 4, 3)
        )

    def test_sum_pool(self):
        _test_parity(
            jax_stax.serial(jax_stax.Conv(4, (3, 3), padding='SAME'), jax_stax.SumPool((2, 2))),
            tt_stax.Sequential(tt_stax.Conv(4, (3, 3), padding='SAME'), tt_stax.SumPool((2, 2))),
            input_shape=(2, 4, 4, 3)
        )

    def test_sin(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Sin()),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.Sin()),
        )

    def test_cos(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Cos()),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.Cos()),
        )

    def test_gaussian(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Gaussian()),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.Gaussian()),
        )

    def test_abrelu(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.ABRelu(0.5, 0.5)),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.ABRelu(0.5, 0.5)),
        )

    def test_fan_out(self):
        _test_parity(
            jax_stax.FanOut(2),
            tt_stax.FanOut(2),
            is_branching=True
        )

    def test_fan_in_sum(self):
        _test_parity(
            jax_stax.serial(jax_stax.parallel(jax_stax.Dense(16), jax_stax.Dense(16)), jax_stax.FanInSum()),
            tt_stax.Sequential(tt_stax.parallel(tt_stax.Linear(16), tt_stax.Linear(16)), tt_stax.FanInSum()),
            input_shape=((5, 16), (5, 16)),
            tuple_in=True
        )

    def test_fan_in_prod(self):
        _test_parity(
            jax_stax.serial(jax_stax.parallel(jax_stax.Dense(16), jax_stax.Dense(16)), jax_stax.FanInProd()),
            tt_stax.Sequential(tt_stax.parallel(tt_stax.Linear(16), tt_stax.Linear(16)), tt_stax.FanInProd()),
            input_shape=((5, 16), (5, 16)),
            tuple_in=True
        )

    def test_fan_in_concat(self):
        _test_parity(
            jax_stax.serial(jax_stax.parallel(jax_stax.Dense(16), jax_stax.Dense(32)), jax_stax.FanInConcat(axis=-1)),
            tt_stax.Sequential(tt_stax.parallel(tt_stax.Linear(16), tt_stax.Linear(32)), tt_stax.FanInConcat(axis=1)),
            input_shape=((5, 16), (5, 32)),
            tuple_in=True
        )

    def test_parallel(self):
        _test_parity(
            jax_stax.parallel(jax_stax.Dense(16), jax_stax.Dense(32)),
            tt_stax.parallel(tt_stax.Linear(16), tt_stax.Linear(32)),
            input_shape=((5, 16), (5, 32)),
            is_branching=True,
            tuple_in=True
        )

    def test_global_self_attention(self):
        _test_parity(
            jax_stax.GlobalSelfAttention(
                n_chan_out=4, n_chan_key=8, n_chan_val=8, n_heads=2, pos_emb_type='NONE'
            ),
            tt_stax.GlobalSelfAttention(
                n_chan_out=4, n_chan_key=8, n_chan_val=8, n_heads=2, pos_emb_type='NONE'
            ),
            input_shape=(2, 16, 16, 3),
        )

    def test_polynomial(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.Polynomial([1.0, 0.5, 0.2])),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.Polynomial([1.0, 0.5, 0.2])),
        )

    def test_rectified_monomial(self):
        _test_parity(
            jax_stax.serial(jax_stax.Dense(16), jax_stax.RectifiedMonomial(1)),
            tt_stax.Sequential(tt_stax.Linear(16), tt_stax.RectifiedMonomial(1)),
        )

   

if __name__ == '__main__':
    unittest.main()
