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

"""Utilities for defining dataclasses that can be used with torch transforms.

This module mirrors the behavior of the JAX/Flax-friendly dataclass helper,
but targets PyTorch's pytree utilities so objects are compatible with:
- torch.compile (Dynamo/AOTAutograd)
- torch.func transforms (e.g., vmap), and
- torch.utils._pytree tree flattening.

Example:
    >>> import torch
    >>> import torch.utils._pytree as pytree
    >>> from torch import compile as torch_compile
    >>> #
    >>> import my_dataclasses as tdataclasses  # this module
    >>> @tdataclasses.dataclass
    >>> class Data:
    >>>     x: torch.Tensor
    >>>     use_x: bool = tdataclasses.field(pytree_node=False)
    >>> #
    >>> data = Data(torch.tensor([1.0]), True)
    >>> # data.x = torch.tensor([2.0])  # frozen; will raise dataclasses.FrozenInstanceError
    >>> data = data.replace(x=torch.tensor([2.0]))  # OK
    >>> #
    >>> def f(d: Data):
    >>>     return d.x if d.use_x else torch.zeros_like(d.x)
    >>> g = torch.compile(f)  # Works because Data is a registered pytree.
    >>> out = g(data)
"""

import dataclasses
import functools
from typing import Any, Dict, Tuple

import torch
from torch.utils._pytree import register_pytree_node


def dataclass(clz):
    """Create a class which can be passed to PyTorch transformations.

    Fields with metadata `pytree_node=False` are treated as static meta-data
    (placed in the pytree context); remaining `init=True` fields are treated
    as pytree children. The class is frozen (immutable); use `.replace(...)`
    to create modified copies.

    Args:
        clz: the class that will be transformed by the decorator.

    Returns:
        The new (frozen) dataclass, registered as a pytree node.
    """
    data_clz = dataclasses.dataclass(frozen=True)(clz)
    meta_fields = []
    data_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
        # Default behavior mirrors the original: pytree_node defaults to True.
        is_pytree_node = field_info.metadata.get("pytree_node", True)
        init = field_info.metadata.get("init", True)
        if init:
            if is_pytree_node:
                data_fields.append(name)
            else:
                meta_fields.append(name)

    # PyTree flatten/unflatten follow the same (children, context) convention
    # as JAX. Children are traversed; context is static meta-data.
    def _flatten_fn(x: Any) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        meta = tuple(getattr(x, name) for name in meta_fields)
        children = tuple(getattr(x, name) for name in data_fields)
        return children, meta

    def _unflatten_fn(meta: Tuple[Any, ...], children: Tuple[Any, ...]) -> Any:
        meta_args = tuple(zip(meta_fields, meta))
        data_args = tuple(zip(data_fields, children))
        kwargs: Dict[str, Any] = dict(meta_args + data_args)
        return data_clz(**kwargs)

    # Register with PyTorch's pytree system.
    register_pytree_node(data_clz, _flatten_fn, _unflatten_fn)

    # Instance helpers for ergonomic parity with the original code.
    @functools.wraps(
        dataclasses.replace,
        assigned=("__module__", "__name__", "__qualname__", "__annotations__"),
    )
    def replace(self: Any, **kwargs) -> Any:
        """Instance method alternative to `dataclasses.replace`."""
        return dataclasses.replace(self, **kwargs)

    @functools.wraps(
        dataclasses.asdict,
        assigned=("__module__", "__name__", "__qualname__", "__annotations__"),
    )
    def asdict(self: Any) -> Dict[str, Any]:
        """Instance method alternative to `dataclasses.asdict`."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @functools.wraps(
        dataclasses.astuple,
        assigned=("__module__", "__name__", "__qualname__", "__annotations__"),
    )
    def astuple(self: Any) -> Tuple[Any, ...]:
        """Instance method alternative to `dataclasses.astuple`."""
        return tuple(getattr(self, f.name) for f in dataclasses.fields(self))

    data_clz.replace = replace
    data_clz.asdict = asdict
    data_clz.astuple = astuple

    return data_clz


def field(pytree_node: bool = True, **kwargs):
    """Wrapper for `dataclasses.field` that carries pytree metadata.

    Args:
        pytree_node: If False, the field is placed in the pytree context (static).
        **kwargs: Forwarded to `dataclasses.field(...)`. If `init=False` is provided,
                  the field is excluded from both children and meta.

    Returns:
        A `dataclasses.field` with appropriate `metadata`.
    """
    metadata = {"pytree_node": pytree_node}
    if "init" in kwargs:
        metadata["init"] = kwargs["init"]
    return dataclasses.field(metadata=metadata, **kwargs)
