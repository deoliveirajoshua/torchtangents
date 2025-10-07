"""An example doing inference with an infinitely wide attention network on IMDb (PyTorch).

Adapted from
https://github.com/google/neural-tangents/blob/main/examples/infinite_fcn.py

By default, this example does inference on a very small subset, and uses small
word embeddings for performance. A 300/300 train/test split takes ~30s on a
machine with 2 Titan X Pascal GPUs, please adjust settings accordingly.

For details, please see "Infinite attention: NNGP and NTK for deep attention
networks" (https://arxiv.org/abs/2006.10540).
"""

import time
from typing import Tuple

import numpy as np
import torch
import torchtangents as tt
from torchtangents import stax
from examples import datasets
from examples import util


_TRAIN_SIZE = 300          # Dataset size to use for training.
_TEST_SIZE = 300           # Dataset size to use for testing.
_BATCH_SIZE = 15           # Batch size for kernel computation. 0 for no batching.
_MAX_SENTENCE_LENGTH = 500 # Pad/truncate sentences to this length.
_GLOVE_PATH = '/tmp/glove.6B.50d.txt'  # Path to GloVe word embeddings.
_IMDB_PATH = '/tmp/imdb_reviews'       # Path to imdb sentences.


def main(*args, use_dummy_data: bool = False, **kwargs) -> None:
  # Mask all padding with this value.
  mask_constant = 100.0

  if use_dummy_data:
    x_train_t, y_train_t, x_test_t, y_test_t = _get_dummy_data(mask_constant)
  else:
    # ------------------------------------------------------------------------
    # Build data pipelines.
    # ------------------------------------------------------------------------
    print('Loading IMDb data.')
    x_train_np, y_train_np, x_test_np, y_test_np = datasets.get_dataset(
        name='imdb_reviews',
        n_train=_TRAIN_SIZE,
        n_test=_TEST_SIZE,
        do_flatten_and_normalize=False,
        data_dir=_IMDB_PATH,
        input_key='text'
    )

    # Embed words and pad / truncate sentences to a fixed size. This returns a
    # map/iterator of two NumPy arrays (train, test) with masked padding.
    x_train_np, x_test_np = datasets.embed_glove(
        xs=[x_train_np, x_test_np],
        glove_path=_GLOVE_PATH,
        max_sentence_length=_MAX_SENTENCE_LENGTH,
        mask_constant=mask_constant
    )

    # Convert to torch tensors.
    x_train_t = torch.tensor(x_train_np, dtype=torch.float32)
    x_test_t  = torch.tensor(x_test_np,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_np,  dtype=torch.float32)

  # --------------------------------------------------------------------------
  # Build the infinite-width attention network’s kernel function.
  # Not using the finite model; width is set to 1 everywhere (as in original).
  # Data layout follows NT’s NHWC/sequence-last convention for stax.
  # x has shape (N, L, D) here (sequence length L, embedding dim D).
  # --------------------------------------------------------------------------
  _, _, kernel_fn = stax.serial(
      stax.Conv(out_chan=1, filter_shape=(9,), strides=(1,), padding='VALID'),
      stax.Relu(),
      stax.GlobalSelfAttention(
          n_chan_out=1,
          n_chan_key=1,
          n_chan_val=1,
          pos_emb_type='SUM',
          W_pos_emb_std=1.0,
          pos_emb_decay_fn=lambda d: 1 / (1 + d**2),
          n_heads=1
      ),
      stax.Relu(),
      stax.GlobalAvgPool(),
      stax.Dense(out_dim=1)
  )

  # Optionally, compute the kernel in batches, in parallel.
  kernel_fn = tt.batch(kernel_fn, device_count=-1, batch_size=_BATCH_SIZE)

  # --------------------------------------------------------------------------
  # Bayesian and infinite-time gradient descent inference with infinite network.
  # --------------------------------------------------------------------------
  start = time.time()

  predict = tt.predict.gradient_descent_mse_ensemble(
      kernel_fn=kernel_fn,
      x_train=x_train_t,
      y_train=y_train_t,
      diag_reg=1e-6,
      mask_constant=mask_constant
  )

  fx_test_nngp, fx_test_ntk = predict(x_test=x_test_t, get=('nngp', 'ntk'))

  # Torch is eager by default; synchronize if using CUDA for fair timing.
  if x_train_t.is_cuda:
    torch.cuda.synchronize()

  duration = time.time() - start
  print(f'Kernel construction and inference done in {duration:.3f} seconds.')

  # --------------------------------------------------------------------------
  # Print out accuracy and loss for infinite network predictions.
  # --------------------------------------------------------------------------
  loss = lambda fx, y_hat: 0.5 * torch.mean((fx - y_hat) ** 2)
  util.print_summary('NNGP test', y_test_t, fx_test_nngp, None, loss)
  util.print_summary('NTK test',  y_test_t, fx_test_ntk,  None, loss)


def _get_dummy_data(
    mask_constant: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return dummy data for when downloading embeddings is not feasible.

  Shapes mirror the GloVe-embedded setup: (N, L, D) with D=50.
  """
  n_train, n_test = 6, 6
  L, D = _MAX_SENTENCE_LENGTH, 50

  g = torch.Generator().manual_seed(1)

  def get_x(shape):
    x = torch.randn(shape, generator=g)
    mask = torch.bernoulli(torch.full(shape, 0.6), generator=g)
    # mask==1 => padding location; set to mask_constant
    x = torch.where(mask.bool(), torch.tensor(mask_constant, dtype=x.dtype), x)
    return x

  def get_y(x):
    # Convert masked tokens to zeros for label construction.
    x_clean = torch.where(x == mask_constant, torch.tensor(0.0, dtype=x.dtype), x)

    # Weighted sums over first and second halves of the embedding dimension.
    idx = torch.arange(x_clean.shape[1], dtype=x_clean.dtype)  # (L,)
    # Broadcast idx over batch and embedding dims.
    w = idx.view(1, L, 1)

    def weighted_sum(x, start, end):
      return torch.sum(x[..., start:end] * w, dim=(1, 2))

    y_label = torch.stack([
        weighted_sum(x_clean, 0, D // 2),
        weighted_sum(x_clean, D // 2, D)
    ], dim=-1) > 0

    # Map boolean labels {True, False} to {0.5, -0.5} as in original.
    y = torch.where(y_label, torch.tensor(0.5, dtype=x.dtype),
                    torch.tensor(-0.5, dtype=x.dtype))
    return y

  x_train = get_x((n_train, L, D))
  x_test  = get_x((n_test,  L, D))
  y_train = get_y(x_train)
  y_test  = get_y(x_test)
  return x_train, y_train, x_test, y_test


if __name__ == '__main__':
  main()
