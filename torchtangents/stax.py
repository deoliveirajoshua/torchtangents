# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License');
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

"""Closed-form NNGP and NTK library.

This library contains layers mimicking those in
:obj:`jax.example_libraries.stax` with similar API apart from:

1) Instead of `(init_fn, apply_fn)` tuple, layers return a triple
`(init_fn, apply_fn, kernel_fn)`, where the added `kernel_fn` maps a
:class:`~neural_tangents.Kernel` to a new :class:`~neural_tangents.Kernel`, and
represents the change in the analytic NTK and NNGP kernels
(:attr:`~neural_tangents.Kernel.nngp`, :attr:`~neural_tangents.Kernel.ntk`).
These functions are chained / stacked together within the :obj:`serial` or
:obj:`parallel` combinators, similarly to `init_fn` and `apply_fn`.
For details, please see "`Neural Tangents: Fast and Easy Infinite Neural
Networks in Python <https://arxiv.org/abs/1912.02803>`_".

2) In layers with random weights, NTK parameterization is used by default
(see page 3 in
"`Neural Tangent Kernel: Convergence and Generalization in Neural Networks
<https://arxiv.org/abs/1806.07572>`_"). Standard parameterization can be
specified for :obj:`Conv` and :obj:`Dense` layers by a keyword argument
`parameterization`. For details, please see "`On the infinite width limit of
neural networks with a standard parameterization
<https://arxiv.org/abs/2001.07301>`_"

PyTorch implementation of neural-tangents stax layers.

This module provides a PyTorch implementation of the neural-tangents stax API,
allowing for computation of empirical and infinite-width Neural Tangent Kernels
using PyTorch tensors and operations.

Example:
    >>> import torch
    >>> import torchtangents as tt
    >>> from torchtangents import stax
    >>> #
    >>> x_train = torch.randn(20, 32, 32, 3)
    >>> y_train = torch.randn(20, 10)
    >>> x_test = torch.randn(5, 32, 32, 3)
    >>> #
    >>> init_fn, apply_fn, kernel_fn = stax.serial(
    >>>     stax.Dense(128),
    >>>     stax.Relu(),
    >>>     stax.Dense(256),
    >>>     stax.Relu(),
    >>>     stax.Flatten(),
    >>>     stax.Dense(10)
    >>> )
    >>> #
    >>> # Initialize network parameters
    >>> rng = torch.Generator()
    >>> params = init_fn(rng, x_train.shape)
    >>> #
    >>> # Compute empirical kernels
    >>> kernel_fn = tt.empirical_kernel_fn(apply_fn)
    >>> nngp = kernel_fn(x_test, x_train, 'nngp', params)
    >>> ntk = kernel_fn(x_test, x_train, 'ntk', params)
"""


# Layer combinators, combining multiple layers into a single layer.
from ._src.stax.combinators import (
    parallel,
    serial,
    repeat
)


# Elementwise nonlinearities.
from ._src.stax.elementwise import (
    ABRelu,
    Abs,
    Cos,
    Elementwise,
    ElementwiseNumerical,
    Erf,
    Exp,
    ExpNormalized,
    Gabor,
    Gaussian,
    Gelu,
    Hermite,
    LeakyRelu,
    Monomial,
    Polynomial,
    Rbf,
    RectifiedMonomial,
    Relu,
    Sigmoid,
    Sigmoid_like,
    Sign,
    Sin,
    Tanh,
)


# Linear layers.
from ._src.stax.linear import (
    Aggregate,
    AvgPool,
    Conv,
    ConvLocal,
    ConvTranspose,
    Dense,
    Identity,
    Index,
    DotGeneral,
    Dropout,
    Flatten,
    GlobalAvgPool,
    GlobalSelfAttention,
    GlobalSumPool,
    ImageResize,
    LayerNorm,
    SumPool,
)


# Helper object for the `Index` layer.
from ._src.stax.linear import (
    Slice
)


# Branching layers.
from ._src.stax.branching import (
    FanInConcat,
    FanInProd,
    FanInSum,
    FanOut,
)


# Enums to specify layer behavior.
from ._src.stax.linear import (
    AggregateImplementation,
    AttentionMechanism,
    Padding,
    PositionalEmbedding,
)


# Decorators and classes for constructing your own layers.
from ._src.stax.requirements import (
    Bool,
    Diagonal,
    MaskedArray,
    layer,
    requires,
    supports_masking,
    unmask_fn,
)