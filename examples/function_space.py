# Copyright 2019 Google LLC
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

"""An example comparing training a neural network with the NTK dynamics (PyTorch).

In this example, we train a neural network on a small subset of MNIST using an
MSE loss and SGD. We compare this training with the analytic function space
prediction using the NTK. Data is loaded using examples.datasets (PyTorch port).
"""

import numpy as np
import torch
import torchtangents as tt
from torchtangents import stax
from examples import datasets
from examples import util


_LEARNING_RATE = 1.0   # Learning rate to use during training.
_TRAIN_SIZE   = 128    # Dataset size to use for training.
_TEST_SIZE    = 128    # Dataset size to use for testing.
_TRAIN_TIME   = 1000.0 # Continuous time denoting duration of training.


def _tree_map(fn, tree):
  if isinstance(tree, (tuple, list)):
    return type(tree)(_tree_map(fn, t) for t in tree)
  if isinstance(tree, dict):
    return {k: _tree_map(fn, v) for k, v in tree.items()}
  return fn(tree)


def _tree_zip_map(fn, t1, t2):
  if isinstance(t1, (tuple, list)):
    return type(t1)(_tree_zip_map(fn, a, b) for a, b in zip(t1, t2))
  if isinstance(t1, dict):
    return {k: _tree_zip_map(fn, t1[k], t2[k]) for k in t1}
  return fn(t1, t2)


def main(unused_argv):
  # --------------------------------------------------------------------------
  # Build data pipelines.
  # --------------------------------------------------------------------------
  print('Loading data.')
  x_train_np, y_train_np, x_test_np, y_test_np = datasets.get_dataset(
      'mnist', _TRAIN_SIZE, _TEST_SIZE)

  # Convert to torch tensors (float32).
  x_train = torch.tensor(x_train_np, dtype=torch.float32)
  y_train = torch.tensor(y_train_np, dtype=torch.float32)
  x_test  = torch.tensor(x_test_np,  dtype=torch.float32)
  y_test  = torch.tensor(y_test_np,  dtype=torch.float32)

  # --------------------------------------------------------------------------
  # Build the network (Erf MLP) using torchtangents.stax.
  # --------------------------------------------------------------------------
  init_fn, apply_fn, _ = stax.serial(
      stax.Dense(512, 1.0, 0.05),
      stax.Erf(),
      stax.Dense(10, 1.0, 0.05)
  )

  g = torch.Generator().manual_seed(0)
  _, params = init_fn(g, (-1, 784))  # match JAX example’s input shape signature

  # --------------------------------------------------------------------------
  # Create an MSE loss (matches original) and manual SGD on the param pytree.
  # --------------------------------------------------------------------------
  def loss_fn(params, x, y):
    fx = apply_fn(params, x)               # (N, C)
    return 0.5 * torch.mean((fx - y) ** 2)

  def sgd_step(params, lr, x, y):
    # Enable gradients.
    params = _tree_map(lambda p: p.detach().requires_grad_(True), params)
    loss = loss_fn(params, x, y)
    loss.backward()

    with torch.no_grad():
      new_params = _tree_map(lambda p: p - lr * p.grad, params)
    # Clear .grad references to avoid holding graph.
    _ = _tree_map(lambda p: p.detach().requires_grad_(False), params)
    return new_params

  # --------------------------------------------------------------------------
  # Create an MSE predictor to solve the NTK equation in function space.
  # --------------------------------------------------------------------------
  ntk = tt.batch(tt.empirical_ntk_fn(apply_fn, vmap_axes=0),
                 batch_size=64, device_count=0)
  g_dd = ntk(x_train, None, params)      # train-train kernel
  g_td = ntk(x_test,  x_train, params)   # test-train kernel
  predictor = tt.predict.gradient_descent_mse(g_dd, y_train)

  # Get initial network outputs (function space values).
  fx_train0 = apply_fn(params, x_train)
  fx_test0  = apply_fn(params, x_test)

  # --------------------------------------------------------------------------
  # Train the network with discrete-time SGD to match continuous time T = steps*lr.
  # --------------------------------------------------------------------------
  train_steps = int(_TRAIN_TIME // _LEARNING_RATE)
  print(f'Training for {train_steps} steps')

  for i in range(train_steps):
    params = sgd_step(params, _LEARNING_RATE, x_train, y_train)

  # --------------------------------------------------------------------------
  # Analytic prediction from NTK flow at continuous time _TRAIN_TIME.
  # --------------------------------------------------------------------------
  print('Computing analytic prediction.')
  fx_train_pred, fx_test_pred = predictor(_TRAIN_TIME, fx_train0, fx_test0, g_td)

  # --------------------------------------------------------------------------
  # Summaries comparing nonlinear training to linearized (NTK) dynamics.
  # --------------------------------------------------------------------------
  util.print_summary('train', y_train,
                     apply_fn(params, x_train),
                     fx_train_pred,
                     lambda fx, y: 0.5 * torch.mean((fx - y) ** 2))

  util.print_summary('test', y_test,
                     apply_fn(params, x_test),
                     fx_test_pred,
                     lambda fx, y: 0.5 * torch.mean((fx - y) ** 2))


if __name__ == '__main__':
  main()
