# Copyright 2020 Google LLC
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

"""Common Type Definitions for PyTorch Fork."""

from typing import Any, Generator, Optional, Sequence, TYPE_CHECKING, TypeVar, Union
from typing_extensions import Protocol

import torch

from .kernel import Kernel


PyTree = Any


Axes = Union[int, Sequence[int]]


T = TypeVar('T')

if TYPE_CHECKING:
  NTTree = Union[T, list['NTTree[T]'], tuple['NTTree[T]', ...], T]
  NTTrees = Union[list['NTTree[T]'], tuple['NTTree[T]', ...]]
else:
  NTTree = Union[list[T], tuple[T, ...], T]

  NTTrees = Union[list[T], tuple[T, ...]]


Shapes = NTTree[tuple[int, ...]]


# Layer Definition.


class InitFn(Protocol):
  def __call__(
      self,
      rng: torch.Generator,
      input_shape: Shapes,
      **kwargs
  ) -> tuple[Shapes, PyTree]:
    ...


class ApplyFn(Protocol):
  def __call__(
      self,
      params: PyTree,
      inputs: NTTree[torch.Tensor],
      *args,
      **kwargs
  ) -> NTTree[torch.Tensor]:
    ...


class MaskFn(Protocol):
  def __call__(
      self,
      mask: Union[torch.Tensor, Sequence[torch.Tensor]],
      input_shape: Shapes,
  ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    ...


KernelOrInput = Union[NTTree[Kernel], NTTree[torch.Tensor]]


Get = Union[None, str, tuple[str, ...]]


class LayerKernelFn(Protocol):
  def __call__(
      self,
      k: NTTree[Kernel]
  ) -> NTTree[Kernel]:
    ...


class AnalyticKernelFn(Protocol):
  def __call__(
      self,
      x1: KernelOrInput,
      x2: Optional[NTTree[torch.Tensor]] = None,
      get: Get = None,
      **kwargs
  ) -> Union[NTTree[Kernel], NTTree[torch.Tensor]]:
    ...


class EmpiricalGetKernelFn(Protocol):
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
  def __call__(
      self,
      x1: NTTree[torch.Tensor],
      x2: Optional[NTTree[torch.Tensor]],
      params: PyTree,
      **kwargs
  ) -> NTTree[torch.Tensor]:
    ...


class MonteCarloKernelFn(Protocol):
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


InternalLayer = AnalyticKernelFn
InternalLayerMasked = AnalyticKernelFn


Layer = AnalyticKernelFn


Kernels = Union[list[Kernel], tuple[Kernel, ...]]


_VMapAxis = Optional[PyTree]

VMapAxisTriple = tuple[_VMapAxis, _VMapAxis, dict[str, _VMapAxis]]
VMapAxes = Union[_VMapAxis, VMapAxisTriple]
