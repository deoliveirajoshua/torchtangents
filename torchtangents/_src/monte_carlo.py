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

"""Monte Carlo kernel computation for torchtangents."""

from typing import Callable, Optional
import torch

from .utils.kernel import Kernel


def monte_carlo_kernel_fn(init_fn: Callable,
                         apply_fn: Callable,
                         key: torch.Generator,
                         n_samples: int,
                         device_count: int = -1,
                         store_on_device: bool = True,
                         trace_axes: tuple = (-1,),
                         diagonal_axes: tuple = ()) -> Callable:
    """Compute kernel using Monte Carlo sampling.
    
    Args:
        init_fn: function to initialize network parameters.
        apply_fn: function to apply network.
        key: random number generator.
        n_samples: number of Monte Carlo samples.
        device_count: number of devices (ignored in this implementation).
        store_on_device: whether to store on device.
        trace_axes: axes to trace over.
        diagonal_axes: axes to take diagonal of.
        
    Returns:
        Function that computes MC kernel estimates.
    """
    
    def kernel_fn(x1: torch.Tensor,
                 x2: Optional[torch.Tensor] = None,
                 get: Optional[str] = None) -> torch.Tensor:
        """Compute Monte Carlo kernel estimate."""
        if x2 is None:
            x2 = x1
        
        if get is None or 'nngp' in get.lower():
            # Compute NNGP via Monte Carlo
            outputs_1 = []
            outputs_2 = []
            
            for _ in range(n_samples):
                # Initialize random parameters
                params = init_fn(key, x1.shape)
                
                # Forward pass
                out_1 = apply_fn(params, x1)
                out_2 = apply_fn(params, x2)
                
                outputs_1.append(out_1)
                outputs_2.append(out_2)
            
            # Stack outputs
            outputs_1 = torch.stack(outputs_1)  # (n_samples, batch_1, ...)
            outputs_2 = torch.stack(outputs_2)  # (n_samples, batch_2, ...)
            
            # Compute empirical covariance
            # E[f(x1) f(x2)^T] = mean over samples of outer products
            batch_1, batch_2 = outputs_1.shape[1], outputs_2.shape[1]
            
            # Flatten non-batch dimensions
            outputs_1_flat = outputs_1.view(n_samples, batch_1, -1)
            outputs_2_flat = outputs_2.view(n_samples, batch_2, -1)
            
            # Compute covariances for each sample and average
            nngp = torch.zeros(batch_1, batch_2, device=x1.device)
            for i in range(n_samples):
                nngp += torch.matmul(outputs_1_flat[i], outputs_2_flat[i].T)
            nngp /= n_samples
            
            return nngp
        
        else:
            raise NotImplementedError("Monte Carlo NTK not implemented yet")
    
    return kernel_fn