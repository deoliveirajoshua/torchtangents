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

"""Layer combinators (PyTorch port)."""

from __future__ import annotations

import operator as op
from typing import Any, Callable, List, Tuple
import warnings

import frozendict
import torch

from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer, Layer, LayerKernelFn, NTTree, NTTrees, Shapes
from .requirements import Diagonal, get_req, layer, requires

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _tree_map(fn: Callable[..., Any], *trees, is_leaf=None):
    if not trees:
        return None
    ref = trees[0]
    leaf = is_leaf(ref) if is_leaf is not None else (type(ref) not in (list, tuple))
    if leaf:
        return fn(*trees)
    t = type(ref)
    return t(_tree_map(fn, *(xs[i] for xs in trees), is_leaf=is_leaf) for i in range(len(ref)))

def _tree_zip(*trees):
    ref = trees[0]
    if type(ref) in (list, tuple):
        t = type(ref)
        return t(_tree_zip(*(xs[i] for xs in trees)) for i in range(len(ref)))
    return trees


def _generator_device(gen: torch.Generator) -> torch.device:
    # PyTorch exposes generator.device on CUDA (>=1.10-ish); default to CPU
    dev = getattr(gen, "device", None)
    return torch.device(dev) if dev is not None else torch.device("cpu")


def _split_generator(gen: torch.Generator) -> Tuple[torch.Generator, torch.Generator]:
    """Deterministically split a torch.Generator into (gen, new_child)."""
    # Draw a new seed from `gen` and create a child generator.
    # This keeps things purely PyTorch and compile-friendly.
    dev = _generator_device(gen)
    seed = int(torch.randint(0, 2**63 - 1, (1,), generator=gen).item())
    child = torch.Generator(device=dev)
    child.manual_seed(seed)
    # return original for further use (already advanced), and the child
    return gen, child

# -----------------------------------------------------------------------------
# serial
# -----------------------------------------------------------------------------

@layer
def serial(*layers: Layer) -> InternalLayer:
    """Compose layers in serial (PyTorch version)."""

    init_fns, apply_fns, kernel_fns = zip(*layers)

    def init_fn(rng: torch.Generator, input_shape: Shapes):
        """Initialize each layer in turn; params is a list with same structure."""
        params: List[Any] = []
        cur_shape = input_shape
        cur_rng = rng
        for f in init_fns:
            cur_rng, child = _split_generator(cur_rng)
            out_shape, p = f(child, cur_shape)
            params.append(p)
            cur_shape = out_shape
        return cur_shape, params

    def apply_fn(params, inputs, **kwargs):
        """Apply each layer sequentially."""
        x = inputs
        for f, p in zip(apply_fns, params):
            x = f(p, x, **kwargs)
        return x

    @requires(**_get_input_req_attr(kernel_fns, fold=op.rshift))
    def kernel_fn(k: NTTree[Kernel], **kwargs) -> NTTree[Kernel]:
        for f in kernel_fns:
            k = f(k, **kwargs)
        return k

    return init_fn, apply_fn, kernel_fn

# -----------------------------------------------------------------------------
# repeat
# -----------------------------------------------------------------------------

@layer
def repeat(layer: Layer, n: int) -> InternalLayer:
    """Compose `layer` in a compiled-friendly Python loop `n` times.

    Assumes the layer keeps activation shape and Kernel metadata unchanged.
    """

    init_f, apply_f, kernel_f = layer

    def init_fn_repeat(rng: torch.Generator, input_shape: Shapes):
        # Check shape invariance and build params for n copies.
        out_shape, _ = init_f(rng, input_shape)
        if out_shape != input_shape:
            raise ValueError(
                f'`init_fn` produces a different output shape {out_shape} than the '
                f'input shape {input_shape}. Please use `serial(*([layer] * n))`.'
            )

        params_list: List[Any] = []
        cur_rng = rng
        for _ in range(n):
            cur_rng, child = _split_generator(cur_rng)
            out_shape_i, p = init_f(child, input_shape)
            if out_shape_i != input_shape:
                raise ValueError(
                    'Intermediate layer in `repeat` changed shape; use `serial`.'
                )
            params_list.append(p)
        return input_shape, params_list

    def apply_fn_repeat(params, inputs, **kwargs):
        x = inputs
        for p in params:
            x = apply_f(p, x, **kwargs)
        return x

    @requires(**get_req(kernel_f))
    def kernel_fn_repeat(k: NTTree[Kernel], **kwargs) -> NTTree[Kernel]:
        if n <= 0:
            return k
        for _ in range(n):
            k = kernel_f(k, **kwargs)
        return k

    return init_fn_repeat, apply_fn_repeat, kernel_fn_repeat

# -----------------------------------------------------------------------------
# parallel
# -----------------------------------------------------------------------------

@layer
def parallel(*layers: Layer) -> InternalLayer:
    """Compose layers in parallel (zip structure in, zip structure out)."""

    init_fns, apply_fns, kernel_fns = zip(*layers)

    def init_fn(rng: torch.Generator, input_shape: Shapes):
        # Preserve the container type of `input_shape` (list/tuple)
        if not isinstance(input_shape, (list, tuple)):
            raise TypeError('`parallel` expects a sequence of input shapes.')
        shapes_out = []
        params_out = []
        cur_rng = rng
        for f, s in zip(init_fns, input_shape):
            cur_rng, child = _split_generator(cur_rng)
            so, p = f(child, s)
            shapes_out.append(so)
            params_out.append(p)
        t = type(input_shape)
        return t(shapes_out), t(params_out)

    def apply_fn(params, inputs, **kwargs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`parallel` expects a sequence of inputs.')
        outs = [f(p, x, **kwargs) for f, p, x in zip(apply_fns, params, inputs)]
        return type(inputs)(outs)

    @requires(**_get_input_req_attr(kernel_fns, fold=op.and_))
    def kernel_fn(ks: NTTrees[Kernel], **kwargs) -> NTTrees[Kernel]:
        if not isinstance(ks, (list, tuple)):
            raise TypeError('`parallel` expects a sequence of Kernels.')
        outs = [f(k, **kwargs) for k, f in zip(ks, kernel_fns)]
        return type(ks)(outs)

    return init_fn, apply_fn, kernel_fn

# -----------------------------------------------------------------------------
# INTERNAL UTILITIES
# -----------------------------------------------------------------------------

def _get_input_req_attr(
    kernel_fns: list[LayerKernelFn],
    fold: Callable[[Diagonal, Diagonal], Diagonal]
) -> dict[str, Any]:
    """Combine requirements/allowances across layers (serial or parallel)."""
    req: dict[str, Any] = {}

    for f in kernel_fns:
        req_f = get_req(f, default=frozendict.frozendict())

        for k, v in req_f.items():
            if k == 'use_dropout':
                if k in req and req[k] != v:
                    raise ValueError('`use_dropout` is a whole-network attribute '
                                     'and cannot differ across layers.')
                req[k] = v

            elif k in ('batch_axis', 'channel_axis'):
                if k not in req:
                    req[k] = v
                else:
                    if fold is op.and_:
                        if k in req and req[k] != v:
                            if (req[k] >= 0 and v >= 0) or (req[k] < 0 and v < 0):
                                warnings.warn(
                                    f'For `kernel_fn`, `{k}` must match in all parallel branches, '
                                    f'got {req[k]} and {v}. This WILL lead to [silent] errors.'
                                )
                            else:
                                warnings.warn(
                                    f'Got potentially mismatching `{k}` values in '
                                    f'parallel branches: {req[k]} and {v}.'
                                )
                    elif fold is not op.rshift:
                        raise ValueError(fold)

            elif k in ('diagonal_batch', 'diagonal_spatial'):
                if k in req:
                    req[k] = fold(req[k], v)
                else:
                    req[k] = v

            else:
                raise NotImplementedError(k)

    return req
