import unittest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from torchtangents import stax as tt_stax

def _test_gradients(model_tt, model_jax, x_train, x_test):
    # PyTorch computation
    x_train_torch = torch.tensor(x_train, requires_grad=True)
    x_test_torch = torch.tensor(x_test)
    
    if x_train.ndim == 4:
        x_train_torch = x_train_torch.permute(0, 3, 1, 2)
        x_test_torch = x_test_torch.permute(0, 3, 1, 2)

    def NTK_single_row(x_train_row):
        # StaxKernelModule is directly callable and performs the infinite-width pass
        out = model_tt(x_train_row.unsqueeze(0), x_test_torch, get=('ntk',)).ntk
        return out.squeeze(0)
        
    torch_jacobian = torch.func.vmap(torch.func.jacrev(NTK_single_row), randomness='same')(x_train_torch)
    
    # JAX computation
    x_train_jax = jnp.array(x_train)
    x_test_jax = jnp.array(x_test)
    
    def NTK_single_row_jax(x_train_row):
        return model_jax(jnp.expand_dims(x_train_row, 0), x_test_jax, get=('ntk',)).ntk.squeeze(0)
        
    jax_jacobian = jax.vmap(jax.jacrev(NTK_single_row_jax))(x_train_jax)

    # permute the torchtangents jacobian to match the neural-tangents shape 
    if torch_jacobian.ndim == 5:
        torch_jacobian = torch_jacobian.permute(0, 1, 3, 4, 2) 
    
    # verification
    np.testing.assert_allclose(
        torch_jacobian.detach().numpy(),
        np.array(jax_jacobian),
        rtol=1e-3, 
        atol=1e-3
    )


class TestNNGradients(unittest.TestCase):

    def test_infinite_width_kernel_gradients(self):
        """
        Tests whether the backpropagation graph is correctly maintained through the
        infinite-width kernel computations defined by tt_stax modules.
        """
        torch.manual_seed(42)

        # Using functional stax (stateless modules) directly
        model = tt_stax.Sequential(
            tt_stax.Linear(128),
            tt_stax.ReLU(),
            tt_stax.Linear(5)
        )
    
        x_train = torch.randn(2, 10, requires_grad=True)
        x_test = torch.randn(3, 10)
    
        # Calling the module performs the infinite-width kernel pass
        k_out = model(x_train, x_test, get=('ntk',))
        loss = k_out.ntk.sum()
        loss.backward()
        
        assert x_train.grad is not None, "Gradients were not computed for x_train!"
        assert x_train.grad.shape == (2, 10), f"Expected gradient shape (2, 10), got {x_train.grad.shape}"
        assert torch.any(x_train.grad != 0), "Gradients are all zero, graph might be disconnected."
    
        def NTK_single_row(x_train_row):
            out = model(x_train_row.unsqueeze(0), x_test, get=('ntk',)).ntk
            return out.squeeze(0)  
    
        torch_jacobian = torch.func.vmap(torch.func.jacrev(NTK_single_row), randomness='same')(x_train)
        assert torch_jacobian.shape == (2, 3, 10), f"Expected batched Jacobian shape (2, 3, 10), got {torch_jacobian.shape}"
    
        # verifying with neural_tangents
        from neural_tangents import stax as jax_stax
        
        _, _, jax_kernel_fn = jax_stax.serial(
            jax_stax.Dense(128),
            jax_stax.Relu(),
            jax_stax.Dense(5)
        )
        
        x_train_jax = jnp.array(x_train.detach().numpy())
        x_test_jax = jnp.array(x_test.detach().numpy())
        
        def NTK_single_row_jax(x_train_row):
            return jax_kernel_fn(jnp.expand_dims(x_train_row, 0), x_test_jax, get=('ntk',)).ntk.squeeze(0)
            
        jax_jacobian = jax.vmap(jax.jacrev(NTK_single_row_jax))(x_train_jax)
        
        np.testing.assert_allclose(
            torch_jacobian.detach().numpy(),
            np.array(jax_jacobian),
            rtol=1e-4, 
            atol=1e-6
        )

    def test_mlp_gradients(self):
        from neural_tangents import stax as jax_stax
        
        model_tt = tt_stax.Sequential(
            tt_stax.Linear(128),
            tt_stax.ReLU(),
            tt_stax.Linear(5)
        )
        
        _, _, model_jax = jax_stax.serial(
            jax_stax.Dense(128),
            jax_stax.Relu(),
            jax_stax.Dense(5)
        )
        
        np.random.seed(42)
        x_train = np.random.randn(2, 10).astype(np.float32)
        x_test = np.random.randn(3, 10).astype(np.float32)
        
        _test_gradients(model_tt, model_jax, x_train, x_test)

    def test_cnn_gradients(self):
        from neural_tangents import stax as jax_stax
        
        model_tt = tt_stax.Sequential(
            tt_stax.Conv(16, (3, 3), padding='SAME'),
            tt_stax.ReLU(),
            tt_stax.Flatten(),
            tt_stax.Linear(5)
        )
        
        _, _, model_jax = jax_stax.serial(
            jax_stax.Conv(16, (3, 3), padding='SAME'),
            jax_stax.Relu(),
            jax_stax.Flatten(),
            jax_stax.Dense(5)
        )
        
        np.random.seed(43)
        x_train = np.random.randn(2, 8, 8, 3).astype(np.float32) # (batch, height, width, channels)
        x_test = np.random.randn(3, 8, 8, 3).astype(np.float32) # (batch, height, width, channels)
        _test_gradients(model_tt, model_jax, x_train, x_test)

if __name__ == '__main__':
    unittest.main()
