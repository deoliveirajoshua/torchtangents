# Copyright 2025
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

"""Common Type Definitions (PyTorch version)."""

from typing import (
    Any,
    Generator,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import torch

from .kernel import Kernel


PyTree = Any
"""A (PyTorch) PyTree-like structure: nested Python containers (lists/tuples/dicts)
with leaves such as tensors or user-registered pytree nodes.
"""


Axes = Union[int, Sequence[int]]
"""Axes specification, can be integers (e.g., ``axis=-1``) or sequences (e.g., ``axis=(1, 3)``).
"""


T = TypeVar('T')

if TYPE_CHECKING:
    NTTree = Union[T, list['NTTree[T]'], tuple['NTTree[T]', ...], T]
    NTTrees = Union[list['NTTree[T]'], tuple['NTTree[T]', ...]]
else:
    # Can't use recursive types with `sphinx-autodoc-typehints`.
    NTTree = Union[list[T], tuple[T, ...], T]
    """Neural Tangents Tree.

    Trees of kernels and arrays naturally emerge in certain neural network computations
    (e.g., nested parallel layers).

    Mimicking JAX, we use a lightweight tree structure called an :class:`NTTree`.
    :class:`NTTree` has internal nodes that are either lists or tuples and leaves
    which are either :class:`torch.Tensor` or :class:`~neural_tangents.Kernel` objects.
    """

    NTTrees = Union[list[T], tuple[T, ...]]
    """A list or tuple of :class:`NTTree` s.
    """


Shapes = NTTree[tuple[int, ...]]
"""A shape — a tuple of integers, or an :class:`NTTree` of such tuples.
"""


# ---- Layer Definition ----

# In PyTorch, RNG is commonly handled via torch.Generator. We allow a few
# practical variants for compatibility (e.g., integer seed, Tensor seed).
PRNGKey = Union[torch.Generator, int, torch.Tensor]


class InitFn(Protocol):
    """A type alias for initialization functions.

    Initialization functions construct parameters for neural networks given an RNG
    and an input shape. They produce a tuple: (output_shape, params_pytree).
    """

    def __call__(
        self,
        rng: PRNGKey,
        input_shape: Shapes,
        **kwargs
    ) -> tuple[Shapes, PyTree]:
        ...


class ApplyFn(Protocol):
    """A type alias for apply functions.

    Apply functions do computations with finite-width neural networks. They take a
    PyTree of parameters and a (possibly tree-structured) input of tensors and
    produce a (possibly tree-structured) output of tensors.
    """

    def __call__(
        self,
        params: PyTree,
        inputs: NTTree[torch.Tensor],
        *args,
        **kwargs
    ) -> NTTree[torch.Tensor]:
        ...


class MaskFn(Protocol):
    """A type alias for masking functions.

    Forward-propagate a mask in a layer of a finite-width network.
    """

    def __call__(
        self,
        mask: Union[torch.Tensor, Sequence[torch.Tensor]],
        input_shape: Shapes,
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        ...


KernelOrInput = Union[NTTree[Kernel], NTTree[torch.Tensor]]


Get = Union[None, str, tuple[str, ...]]


class LayerKernelFn(Protocol):
    """A type alias for pure kernel functions.

    A pure kernel function takes a PyTree of :class:`Kernel` object(s) and produces
    a PyTree of :class:`Kernel` object(s). These functions are used to define new
    layer types.
    """

    def __call__(
        self,
        k: NTTree[Kernel]
    ) -> NTTree[Kernel]:
        ...


class AnalyticKernelFn(Protocol):
    """A type alias for analytic kernel functions.

    A kernel function that computes an analytic kernel. Takes either a
    :class:`~neural_tangents.Kernel` or :class:`torch.Tensor` inputs and a `get`
    argument that specifies which quantities should be computed by the kernel.
    Returns either a :class:`~neural_tangents.Kernel` object or :class:`torch.Tensor`
    leaves for kernels specified by `get`.
    """

    def __call__(
        self,
        x1: KernelOrInput,
        x2: Optional[NTTree[torch.Tensor]] = None,
        get: Get = None,
        **kwargs
    ) -> Union[NTTree[Kernel], NTTree[torch.Tensor]]:
        ...


class EmpiricalGetKernelFn(Protocol):
    """A type alias for empirical kernel functions accepting a `get` argument.

    Produces an empirical kernel from a single instantiation of a neural network
    specified by its parameters.

    Equivalent to `EmpiricalKernelFn`, but accepts a `get` argument,
    e.g. `get=("nngp", "ntk")`, to compute both kernels together.
    """

    def __call__(
        self,
        x1: NTTree[torch.Tensor],
        x2: Optional[NTTree[torch.Tensor]],
        get: Get,
        params: PyTree,
        **kwargs
    ) -> NTTree[torch.Tensor]:
        ...


class EmpiricalKernelFn(Protocol):
    """A type alias for empirical kernel functions computing either NTK or NNGP.

    Produces an empirical kernel from a single network instantiation specified by
    its parameters.

    Equivalent to `EmpiricalGetKernelFn` with `get="nngp"` or `get="ntk"`.
    """

    def __call__(
        self,
        x1: NTTree[torch.Tensor],
        x2: Optional[NTTree[torch.Tensor]],
        params: PyTree,
        **kwargs
    ) -> NTTree[torch.Tensor]:
        ...


class MonteCarloKernelFn(Protocol):
    """A type alias for Monte Carlo kernel functions.

    A kernel function that produces an estimate of an `AnalyticKernel` by Monte Carlo
    sampling given an RNG (typically passed via `kwargs`, e.g., a `torch.Generator`).
    """

    def __call__(
        self,
        x1: NTTree[torch.Tensor],
        x2: Optional[NTTree[torch.Tensor]],
        get: Get = None,
        **kwargs
    ) -> Union[NTTree[torch.Tensor], Generator[NTTree[torch.Tensor], None, None]]:
        ...


KernelFn = Union[
    AnalyticKernelFn,
    EmpiricalKernelFn,
    EmpiricalGetKernelFn,
    MonteCarloKernelFn,
]


InternalLayer = tuple[InitFn, ApplyFn, LayerKernelFn]
InternalLayerMasked = tuple[InitFn, ApplyFn, LayerKernelFn, MaskFn]


Layer = tuple[InitFn, ApplyFn, AnalyticKernelFn]


Kernels = Union[list[Kernel], tuple[Kernel, ...]]
"""Kernel inputs/outputs of `FanOut`, `FanInSum`, etc.
"""


_VMapAxis = Optional[PyTree]
"""A `PyTree` of integers specifying per-leaf vmap axes (or `None` to skip)."""

VMapAxisTriple = tuple[_VMapAxis, _VMapAxis, dict[str, _VMapAxis]]
VMapAxes = Union[_VMapAxis, VMapAxisTriple]
"""Specifies `(input, output, kwargs)` axes for vectorization in empirical NTK."""
