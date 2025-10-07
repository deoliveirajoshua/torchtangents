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

"""Example of using different implementations for empirical NTK computation (PyTorch).

All implementations apply to any differentiable functions, not necessarily ones
constructed with Torchtangents.

For details about the empirical (finite width) NTK computation, please see
"Fast Finite Width Neural Tangent Kernel" (https://arxiv.org/abs/2206.08720).
"""

import torch
import torchtangents as tt
from torchtangents import stax as stax


def main(unused_argv):
  # Seeded randomness analogous to jax.random.split/normal.
  g = torch.Generator().manual_seed(1)
  # Keep NHWC to mirror Neural Tangents stax default (channels-last).
  x1 = torch.randn(2, 8, 8, 3, generator=g)
  x2 = torch.randn(3, 8, 8, 3, generator=g)

  # A vanilla CNN.
  init_fn, f, _ = stax.serial(
      stax.Conv(8, (3, 3)),
      stax.Relu(),
      stax.Conv(8, (3, 3)),
      stax.Relu(),
      stax.Conv(8, (3, 3)),
      stax.Flatten(),
      stax.Dense(10)
  )

  # Initialize parameters. `init_fn` mirrors NT's (rng, input_shape) -> (out_shape, params).
  _, params = init_fn(g, x1.shape)

  kwargs = dict(
      f=f,
      trace_axes=(),
      vmap_axes=0,
  )

  # Default, baseline Jacobian contraction.
  jacobian_contraction = tt.empirical_ntk_fn(
      **kwargs,
      implementation=tt.NtkImplementation.JACOBIAN_CONTRACTION
  )
  # (6, 3, 10, 10) full test-train NTK expected shape.
  ntk_jc = jacobian_contraction(x2, x1, params)

  # NTK–vector-products-based implementation.
  ntk_vector_products = tt.empirical_ntk_fn(
      **kwargs,
      implementation=tt.NtkImplementation.NTK_VECTOR_PRODUCTS
  )
  ntk_vp = ntk_vector_products(x2, x1, params)

  # Structured-derivatives-based implementation.
  structured_derivatives = tt.empirical_ntk_fn(
      **kwargs,
      implementation=tt.NtkImplementation.STRUCTURED_DERIVATIVES
  )
  ntk_sd = structured_derivatives(x2, x1, params)

  # Auto-FLOPs-selecting implementation. (May not be optimal on CPU/GPU.)
  auto = tt.empirical_ntk_fn(
      **kwargs,
      implementation=tt.NtkImplementation.AUTO
  )
  ntk_auto = auto(x2, x1, params)

  # Check that implementations match.
  impls = [ntk_jc, ntk_vp, ntk_sd, ntk_auto]
  atol = 1e-4  # parity with JAX CPU/GPU threshold
  for i, ntk1 in enumerate(impls):
    for j, ntk2 in enumerate(impls):
      diff = torch.max(torch.abs(ntk1 - ntk2)).item()
      print(f'NTK implementation diff ({i},{j}) = {diff}.')
      assert diff < atol, diff

  print('All NTK implementations match.')


if __name__ == '__main__':
  main()
