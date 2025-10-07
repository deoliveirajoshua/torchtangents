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

"""Datasets used in examples (PyTorch version)."""

import gzip
import os
import shutil
import urllib.request
from typing import Iterable, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets as tvds
from torchvision import transforms as T


def _partial_flatten_and_normalize(x: np.ndarray) -> np.ndarray:
  """Flatten all but the first dimension of an `np.ndarray`."""
  x = np.reshape(x, (x.shape[0], -1))
  return (x - np.mean(x)) / np.std(x)


def _one_hot(x: np.ndarray, k: int, dtype=np.float32) -> np.ndarray:
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def get_dataset(
    name: str,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    permute_train: bool = False,
    do_flatten_and_normalize: bool = True,
    data_dir: Optional[str] = None,
    input_key: str = 'image'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Download, parse and process a dataset to unit scale and one-hot labels.

  Notes:
    * Keeps the same signature/return as the original TFDS-based function.
    * Supported names: 'mnist', 'fashion_mnist', 'cifar10', 'cifar100'.
      For other names (e.g., 'imdb_reviews'), raise a clear error since TFDS
      is not used here.

  Returns:
    (train_images, train_labels, test_images, test_labels) as NumPy arrays.
  """
  # 'input_key' kept for compatibility with the original stub.
  del input_key

  if data_dir is None:
    data_dir = os.path.join(os.path.expanduser('~'), '.torch_datasets')

  name_l = name.lower()

  # Choose torchvision dataset + meta.
  if name_l == 'mnist':
    make_ds = tvds.MNIST
    num_classes = 10
    to_numpy = lambda x: np.array(x, dtype=np.uint8)[..., None]  # (H,W,1)
  elif name_l in ('fashion_mnist', 'fashion-mnist'):
    make_ds = tvds.FashionMNIST
    num_classes = 10
    to_numpy = lambda x: np.array(x, dtype=np.uint8)[..., None]
  elif name_l == 'cifar10':
    make_ds = tvds.CIFAR10
    num_classes = 10
    to_numpy = lambda x: np.array(x, dtype=np.uint8)  # (H,W,3)
  elif name_l == 'cifar100':
    make_ds = tvds.CIFAR100
    num_classes = 100
    to_numpy = lambda x: np.array(x, dtype=np.uint8)
  elif name_l == 'imdb_reviews':
    raise NotImplementedError(
        "The PyTorch rewrite does not bundle TFDS IMDB. "
        "Use your own text loader (strings) + `embed_glove` below."
    )
  else:
    raise ValueError(f"Unsupported dataset name: {name!r}")

  # We fetch entire splits into memory to match original behavior
  # (which used tfds.as_numpy(..., batch_size=-1)).
  # Build transforms that keep PIL image (we will convert to numpy ourselves).
  # We do NOT normalize here; we do it below to reproduce original behavior.
  transform = T.Compose([])

  train_set = make_ds(root=data_dir, train=True, download=True, transform=transform)
  test_set  = make_ds(root=data_dir, train=False, download=True, transform=transform)

  # Optionally truncate.
  if n_train is not None:
    indices = np.arange(min(n_train, len(train_set)))
    train_subset = torch.utils.data.Subset(train_set, indices.tolist())
  else:
    train_subset = train_set

  if n_test is not None:
    indices = np.arange(min(n_test, len(test_set)))
    test_subset = torch.utils.data.Subset(test_set, indices.tolist())
  else:
    test_subset = test_set

  # Dump to numpy.
  def dataset_to_numpy(ds) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
    (batch_imgs, batch_labels) = next(iter(loader))
    # batch_imgs is a list of PIL if transform=[]; DataLoader with dataset returning (PIL, int)
    # actually stacks to a list; so re-extract properly:
    if isinstance(batch_imgs, list):
      xs = np.stack([to_numpy(img) for img in batch_imgs], axis=0)
    else:
      # Fallback: if a tensor snuck in, convert to HWC uint8.
      x = batch_imgs.numpy()
      if x.ndim == 4:  # NCHW
        xs = np.transpose(x, (0, 2, 3, 1))
      else:
        xs = x
    ys = np.array(batch_labels, dtype=np.int64)
    return xs, ys

  # The above branch with DataLoader(list) occurs with Subset; safer route:
  def iter_to_numpy(ds) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for img, y in ds:
      xs.append(to_numpy(img))
      ys.append(int(y))
    return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)

  train_images, train_labels_idx = iter_to_numpy(train_subset)
  test_images,  test_labels_idx  = iter_to_numpy(test_subset)

  if do_flatten_and_normalize:
    train_images = _partial_flatten_and_normalize(train_images)
    test_images  = _partial_flatten_and_normalize(test_images)

  train_labels = _one_hot(train_labels_idx, num_classes)
  test_labels  = _one_hot(test_labels_idx,  num_classes)

  if permute_train:
    rng = np.random.RandomState(0)
    perm = rng.permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels


def minibatch(x_train: np.ndarray,
              y_train: np.ndarray,
              batch_size: int,
              train_epochs: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
  """Generate minibatches of data for a set number of epochs (NumPy in / out)."""
  epoch = 0
  start = 0
  rng = np.random.RandomState(0)

  while epoch < train_epochs:
    end = start + batch_size

    if end > x_train.shape[0]:
      permutation = rng.permutation(x_train.shape[0])
      x_train = x_train[permutation]
      y_train = y_train[permutation]
      epoch += 1
      start = 0
      continue

    yield x_train[start:end], y_train[start:end]
    start = start + batch_size


def embed_glove(xs: List[np.ndarray],
                glove_path: str,
                max_sentence_length: int = 1000,
                mask_constant: float = 1000.) -> Iterable[np.ndarray]:
  """Embed a list of string arrays into GloVe word embeddings.

  Adapted from https://keras.io/examples/pretrained_word_embeddings/ (sans Keras).

  Args:
    xs:
      list of string numpy arrays to embed.

    glove_path:
      path to the GloVe embedding file.

    max_sentence_length:
      pad/truncate embeddings to this length.

    mask_constant:
      mask padding with this constant.

  Returns:
    Iterator over xs with words replaced by word embeddings, padded/truncated to
    a fixed length, with padding masked with the given constant.
  """
  xs = list(map(_decode, xs))
  tokenizer = _SimpleTokenizer()
  tokenizer.fit_on_texts(np.concatenate(xs))
  embedding_layer = _get_glove_embedding_layer(tokenizer,
                                               glove_path,
                                               max_sentence_length)

  def embed(x: np.ndarray) -> np.ndarray:
    # Replace strings with sequences of integer tokens.
    x_tok = tokenizer.texts_to_sequences(x)
    lengths = np.array([len(s) for s in x_tok], dtype=np.int64)

    # Pad all sentences to a fixed max sentence length.
    x_tok = _pad_sequences(
        x_tok,
        maxlen=max_sentence_length,
        padding='post',
        truncating='post'
    )  # shape: (N, L)

    # Replace integer tokens with word embeddings via torch Embedding.
    with torch.no_grad():
      xt = torch.as_tensor(x_tok, dtype=torch.long)
      x_emb_t = embedding_layer(xt)  # (N, L, D)
      x_emb = x_emb_t.detach().cpu().numpy()

    # Mask padding tokens.
    mask = np.arange(max_sentence_length)[None, :] >= lengths[:, None]
    x_emb[mask, ...] = mask_constant
    return x_emb

  return map(embed, xs)


def _get_glove_embedding_layer(tokenizer,
                               glove_path: str,
                               max_sentence_length: int):
  """Get a torch.nn.Embedding initialized from given GloVe embeddings.

  Adapted from https://keras.io/examples/pretrained_word_embeddings/.

  Args:
    tokenizer:
      the tokenizer used to tokenize inputs; must have `word_index`.

    glove_path:
      path to the GloVe embedding file (expects 50D glove.6B.50d).

    max_sentence_length:
      pad/truncate embeddings to this length (kept for API parity).

  Returns:
    A frozen torch.nn.Embedding layer whose weights are initialized from GloVe.
  """
  del max_sentence_length  # kept for stub parity; not needed by torch Embedding.

  embedding_dim = 50
  word_index = tokenizer.word_index
  print('Loading the embedding model')
  embeddings_index = {}

  # Ensure glove_path exists (optionally decompress .gz).
  if not os.path.exists(glove_path):
    if not os.path.exists(f'{glove_path}.gz'):
      print(f'Did not find {glove_path} word embeddings, downloading...')
      url = 'https://github.com/icml2020-attention/glove/raw/main/glove.6B.50d.txt.gz'
      urllib.request.urlretrieve(url, f'{glove_path}.gz')

    with gzip.open(f'{glove_path}.gz', 'rt') as f_in:
      with open(glove_path, 'wt') as f_out:
        shutil.copyfileobj(f_in, f_out)

  with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
      word, coefs = line.split(sep=' ', maxsplit=1)
      coefs = np.fromstring(coefs, dtype='f', sep=' ')
      embeddings_index[word] = coefs

  print(f'Found {len(embeddings_index)} word vectors.')
  print(f'Found {len(word_index)} unique tokens.')
  num_words = len(word_index) + 1  # reserve index 0

  emb_mat = np.zeros((num_words, embedding_dim), dtype=np.float32)
  for word, i in word_index.items():
    if i >= num_words:
      continue
    emb_vector = embeddings_index.get(word)
    if emb_vector is not None:
      emb_mat[i] = emb_vector

  emb = torch.nn.Embedding(num_words, embedding_dim)
  emb.weight.data.copy_(torch.from_numpy(emb_mat))
  emb.weight.requires_grad_(False)  # freeze to mimic Keras trainable=False
  return emb


def _decode(x: np.ndarray) -> np.ndarray:
  # Preserve original behavior: convert bytes -> str if necessary.
  if x.dtype.type is np.bytes_ or x.dtype == object:
    return np.array([s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                     for s in x])
  return x.astype(str)


# --------------------------
# Utilities: tokenizer / pad
# --------------------------

class _SimpleTokenizer:
  """Minimal tokenizer compatible with Keras Tokenizer API used above."""
  def __init__(self):
    self.word_index = {}  # 1-based indices
    self._fit = False

  def fit_on_texts(self, texts: Iterable[str]) -> None:
    # Very simple whitespace tokenizer; lowercases and strips punctuation at ends.
    from collections import Counter
    import re
    token_pattern = re.compile(r"\b\w+\b", flags=re.UNICODE)

    counter = Counter()
    for t in texts:
      tokens = token_pattern.findall(t.lower())
      counter.update(tokens)

    # Reserve 0 for padding; assign indices by frequency (desc) then alpha.
    # This roughly matches Keras' typical behavior.
    vocab = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    self.word_index = {w: i+1 for i, (w, _) in enumerate(vocab)}
    self._fit = True

  def texts_to_sequences(self, texts: Iterable[str]) -> List[List[int]]:
    if not self._fit:
      raise RuntimeError("Tokenizer must be fit before calling texts_to_sequences.")
    seqs = []
    for t in texts:
      tokens = t.lower().split()
      seqs.append([self.word_index.get(tok, 0) for tok in tokens if tok])
    return seqs


def _pad_sequences(sequences: List[List[int]],
                   maxlen: int,
                   padding: str = 'post',
                   truncating: str = 'post',
                   value: int = 0) -> np.ndarray:
  """Pad/truncate a list of int sequences to a fixed length (NumPy output)."""
  assert padding in ('pre', 'post')
  assert truncating in ('pre', 'post')
  n = len(sequences)
  out = np.full((n, maxlen), value, dtype=np.int64)
  for i, s in enumerate(sequences):
    if len(s) > maxlen:
      s = s[-maxlen:] if truncating == 'pre' else s[:maxlen]
    if padding == 'post':
      out[i, :len(s)] = s
    else:
      out[i, -len(s):] = s
  return out
