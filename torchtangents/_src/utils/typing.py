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

"""Type definitions for torchtangents."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
from ..utils.kernel import Kernel

# Type aliases
PyTree = Union[torch.Tensor, Dict[str, Any], Tuple[Any, ...], Any]
Axes = Union[int, Sequence[int]]

# Layer function types
InitFn = Callable[[torch.Generator, Tuple[int, ...]], PyTree]
ApplyFn = Callable[[PyTree, torch.Tensor, Optional[torch.Generator]], torch.Tensor]
KernelFn = Callable[[Kernel], Kernel]

# Layer tuple types
InternalLayer = Tuple[InitFn, ApplyFn, KernelFn]
InternalLayerMasked = Tuple[InitFn, ApplyFn, KernelFn]  # For consistency with neural-tangents

# Other useful types
ActivationFn = Callable[[torch.Tensor], torch.Tensor]
Shape = Tuple[int, ...]