# Copyright 2025 torchtangents contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example using torchtangents for infinite-width neural network inference."""

import torch
import torchtangents as tt
from torchtangents import stax


def main():
    """Run infinite FCN example."""
    print("TorchTangents Infinite FCN Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate synthetic data
    train_size = 100
    test_size = 20
    input_dim = 10
    output_dim = 1
    
    x_train = torch.randn(train_size, input_dim)
    y_train = torch.randn(train_size, output_dim)
    x_test = torch.randn(test_size, input_dim)
    y_test = torch.randn(test_size, output_dim)
    
    print(f"Training data: {x_train.shape}, {y_train.shape}")
    print(f"Test data: {x_test.shape}, {y_test.shape}")
    
    # Build infinite width network
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512),
        stax.Relu(),
        stax.Dense(256),
        stax.Relu(),
        stax.Dense(output_dim)
    )
    
    print("\nBuilt infinite-width network architecture")
    
    # Initialize parameters for empirical computation
    rng = torch.Generator()
    rng.manual_seed(123)
    params = init_fn(rng, x_train.shape)
    
    print("Initialized network parameters")
    
    # Compute empirical kernels
    print("\nComputing empirical kernels...")
    
    try:
        # Create empirical kernel function
        empirical_kernel_fn = tt.empirical_kernel_fn(apply_fn)
        
        # Compute train-train kernel
        k_train_train = empirical_kernel_fn(x_train, None, 'ntk', params)
        print(f"Train-train NTK shape: {k_train_train.shape}")
        print(f"Train-train NTK eigenvalues: {torch.linalg.eigvals(k_train_train).real[:5]}...")
        
        # Compute test-train kernel  
        k_test_train = empirical_kernel_fn(x_test, x_train, 'ntk', params)
        print(f"Test-train NTK shape: {k_test_train.shape}")
        
        # Make predictions using NTK
        print("\nMaking NTK predictions...")
        predict_fn = tt.predict.gradient_descent_mse(
            k_train_train, y_train, diag_reg=1e-2  # Increased regularization
        )
        
        # Get initial predictions
        fx_train_0 = apply_fn(params, x_train)
        fx_test_0 = apply_fn(params, x_test)
        
        # Predict at infinite time
        fx_train_inf, fx_test_inf = predict_fn(
            t=None, 
            fx_train_0=fx_train_0,
            fx_test_0=fx_test_0,
            k_test_train=k_test_train
        )
        
        print(f"Training predictions shape: {fx_train_inf.shape}")
        print(f"Test predictions shape: {fx_test_inf.shape}")
        
        # Compute MSE
        train_mse = torch.mean((fx_train_inf - y_train) ** 2).item()
        test_mse = torch.mean((fx_test_inf - y_test) ** 2).item()
        
        print(f"\nResults:")
        print(f"Training MSE: {train_mse:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        
        print("\n✓ Example completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during computation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()