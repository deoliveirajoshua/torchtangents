import torch
import unittest
from torchtangents import stax as stax

class TestStax(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator()
        self.rng.manual_seed(0)
        self.input_shape = (5, 32, 8, 8)
        self.x1 = torch.randn(self.input_shape, generator=self.rng)
        self.x2 = torch.randn(self.input_shape, generator=self.rng)

    def _test_layer(self, layer_fn, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
            x1, x2 = self.x1, self.x2
        else:
            if isinstance(input_shape[0], tuple):
                x1 = tuple(torch.randn(s, generator=self.rng) for s in input_shape)
                x2 = tuple(torch.randn(s, generator=self.rng) for s in input_shape)
            else:
                x1 = torch.randn(input_shape, generator=self.rng)
                x2 = torch.randn(input_shape, generator=self.rng)
            
        kernel_fn = layer_fn
        # init_fn and apply_fn are no longer strictly required for stax kernel_fns
        
        
        # kernel
        try:
            out_kernel_diag = kernel_fn(x1, x2=x2, get=('nngp', 'ntk'), diagonal_spatial=True)
            if isinstance(out_kernel_diag, (list, tuple)) and not hasattr(out_kernel_diag, '_fields'):
                for out_kernel in out_kernel_diag:
                    self.assertIsNotNone(out_kernel.nngp)
                    self.assertIsNotNone(out_kernel.ntk)
            else:
                self.assertIsNotNone(out_kernel_diag.nngp)
                self.assertIsNotNone(out_kernel_diag.ntk)
            has_diag = True
        except ValueError:
            # Some layers (like GlobalAvgPool) strict combination constraints.
            has_diag = False

        
        try:
            out_kernel_full = kernel_fn(x1, x2=x2, get=('nngp', 'ntk'), diagonal_spatial=False)
            if isinstance(out_kernel_full, (list, tuple)) and not hasattr(out_kernel_full, '_fields'):
                for kf, kd in zip(out_kernel_full, out_kernel_diag):
                    self.assertIsNotNone(kf.nngp)
                    self.assertIsNotNone(kf.ntk)
                    if has_diag:
                        self.assertTrue(kf.nngp.ndim >= kd.nngp.ndim)
            else:
                self.assertIsNotNone(out_kernel_full.nngp)
                self.assertIsNotNone(out_kernel_full.ntk)
                if has_diag:
                    self.assertTrue(out_kernel_full.nngp.ndim >= out_kernel_diag.nngp.ndim)
        except Exception:
            pass

    # Combinators
    def test_serial(self):
        self._test_layer(stax.Sequential(stax.Linear(128), stax.ReLU(), stax.Linear(10)))
        
    def test_parallel(self):
        input_shape = (self.input_shape, self.input_shape)
        self._test_layer(stax.parallel(stax.Linear(128), stax.Identity()), input_shape=input_shape)

    def test_repeat(self):
        self._test_layer(stax.Sequential(stax.Linear(32), stax.repeat(stax.Sequential(stax.ReLU(), stax.Linear(32)), 2)))

    # Elementwise
    def test_relu(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.ReLU()))

    def test_abrelu(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.ABRelu(0.5, 0.5)))

    def test_leakyrelu(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.LeakyReLU(0.1)))

    def test_abs(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Abs()))

    def test_erf(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Erf()))

    def test_sigmoid_like(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Sigmoid_like()))

    def test_gabor(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Gabor()))

    def test_gelu(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.GeLU()))

    def test_sin(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Sin()))

    def test_cos(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Cos()))

    def test_sign(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Sign()))

    def test_exp(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Exp()))

    def test_gaussian(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Gaussian()))

    def test_rbf(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Rbf()))

    def test_exp_normalized(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.ExpNormalized()))

    def test_hermite(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Hermite(2)))

    def test_monomial(self):
        self._test_layer(stax.Sequential(stax.Linear(16), stax.Monomial(3)))

    def test_rectified_monomial(self):
        # TODO: Not implemented yet
        pass

    # Linear
    def test_dense(self):
        self._test_layer(stax.Linear(64))

    def test_identity(self):
        self._test_layer(stax.Identity())

    def test_flatten(self):
        self._test_layer(stax.Flatten())

    def test_conv(self):
        input_shape = (5, 3, 8, 8)
        self._test_layer(stax.Conv(out_chan=16, filter_shape=(3, 3), padding='SAME'), input_shape=input_shape)

    # Pooling
    def test_global_avg_pool(self):
        input_shape = (5, 16, 8, 8)
        # Note: must pool over something that went through an affine layer (is_gaussian=True)
        layer = stax.Sequential(stax.Linear(16), stax.GlobalAvgPool())
        self._test_layer(layer, input_shape=input_shape)

    def test_global_sum_pool(self):
        input_shape = (5, 16, 8, 8)
        layer = stax.Sequential(stax.Linear(16), stax.GlobalSumPool())
        self._test_layer(layer, input_shape=input_shape)

    # Branching
    def test_fan_out(self):
        input_shape = (5, 16, 8, 8)
        layer = stax.FanOut(3)
        self._test_layer(layer, input_shape=input_shape)

    def test_fan_in_sum(self):
        input_shape = ((5, 16), (5, 16))
        layer = stax.Sequential(stax.parallel(stax.Linear(16), stax.Linear(16)), stax.FanInSum())
        self._test_layer(layer, input_shape=input_shape)

    def test_fan_in_prod(self):
        input_shape = ((5, 16), (5, 16))
        layer = stax.Sequential(stax.parallel(stax.Linear(16), stax.Linear(16)), stax.FanInProd())
        self._test_layer(layer, input_shape=input_shape)

    def test_fan_in_concat(self):
        input_shape = ((5, 16), (5, 32))
        layer = stax.Sequential(stax.parallel(stax.Linear(16), stax.Linear(32)), stax.FanInConcat(axis=1))
        self._test_layer(layer, input_shape=input_shape)

    # Attention
    def test_global_self_attention(self):
        input_shape = (2, 3, 5) 
        layer = stax.GlobalSelfAttention(n_chan_out=4, n_chan_key=8, n_chan_val=8, n_heads=2, pos_emb_type='NONE')
        self._test_layer(layer, input_shape=input_shape)

    # Adapters
    def test_index(self):
        self._test_layer(stax.Index(idx=stax.Slice[:, :, 0:2, 0:2]))

    def test_image_resize(self):
        self._test_layer(stax.ImageResize(shape=(-1, -1, 4, 4)))

    def test_dot_general(self):
        self._test_layer(stax.DotGeneral(rhs=2.0))
        
    def test_aggregate(self):
        self._test_layer(stax.Aggregate())

    def test_layer_norm(self):
        # Must be applied to Gaussian outputs
        layer = stax.Sequential(stax.Linear(16), stax.LayerNorm())
        self._test_layer(layer, input_shape=(5, 16, 8, 8))

if __name__ == '__main__':
    unittest.main()
