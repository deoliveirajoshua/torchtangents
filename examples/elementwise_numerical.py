# Copyright 2022 Google LLC
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

"""Example of approximating the NNGP and NTK using quadrature and autodiff (PyTorch).

For details, see :obj:`~torchtangents.stax.ElementwiseNumerical` and
"Fast Neural Kernel Embeddings for General Activations"
(https://arxiv.org/abs/2209.04121).
"""

import torch
import torch.nn.functional as F
import torchtangents as tt


def main(unused_argv):
  # Random inputs analogous to JAX's random.normal with fixed seed.
  g = torch.Generator().manual_seed(1)
  x1 = torch.randn(10, 3, generator=g)
  x2 = torch.randn(20, 3, generator=g)

  # Consider a nonlinearity for which we know the closed-form expression (GeLU).
  _, _, kernel_fn_closed_form = tt.stax.serial(
      tt.stax.Dense(1),
      tt.stax.Gelu(),  # Contains the closed-form GeLU NNGP/NTK expression.
      tt.stax.Dense(1)
  )
  kernel_closed_form = kernel_fn_closed_form(x1, x2)

  # Construct the layer from only the elementwise forward-pass GeLU.
  _, _, kernel_fn_numerical = tt.stax.serial(
      tt.stax.Dense(1),
      # Approximation using Gaussian quadrature and autodiff.
      tt.stax.ElementwiseNumerical(F.gelu, deg=25),
      tt.stax.Dense(1)
  )
  kernel_numerical = kernel_fn_numerical(x1, x2)

  # The two kernels are close!
  assert torch.max(torch.abs(kernel_closed_form.nngp -
                             kernel_numerical.nngp)) < 1e-3
  assert torch.max(torch.abs(kernel_closed_form.ntk -
                             kernel_numerical.ntk)) < 1e-3
  print('Gaussian quadrature approximation of the kernel is accurate!')


if __name__ == '__main__':
  main()
