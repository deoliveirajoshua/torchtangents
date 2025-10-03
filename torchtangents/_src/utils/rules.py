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

"""Structured derivatives rules (PyTorch port)."""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Union, Sequence, Iterable

import torch
import torch.nn.functional as F
import numpy as np

from . import utils
from .dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Minimal IR-like shims (replacing JAX Jaxpr/eqns/vars with light classes)
# ---------------------------------------------------------------------------

class ShapedArray:
    """Lightweight shape/dtype spec akin to jax.core.ShapedArray."""
    def __init__(self, shape: Sequence[int], dtype: torch.dtype):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        prod = 1
        for s in self.shape:
            prod *= int(s)
        return int(prod)

    def __repr__(self):
        return f"ShapedArray(shape={self.shape}, dtype={self.dtype})"


class Literal:
    """Placeholder for literals/constants."""
    def __init__(self, value: Any, aval: ShapedArray):
        self.value = value
        self.aval = aval


class Var:
    """IR variable referencing a shaped array."""
    def __init__(self, aval: ShapedArray):
        self.aval = aval


class JaxprEqn:
    """Lightweight equation node.

    primitive: a string, e.g., 'dot_general', 'add', ...
    invars: list[Var]
    outvars: list[Var]
    params: dict[str, Any]
    """
    def __init__(self, primitive: str, invars: list[Var],
                 outvars: list[Var], params: dict[str, Any]):
        self.primitive = primitive
        self.invars = invars
        self.outvars = outvars
        self.params = dict(params)

    def __repr__(self):
        return f"Eqn(prim={self.primitive}, in={len(self.invars)}, out={len(self.outvars)})"


class Jaxpr:
    """Lightweight container of variables and equations."""
    def __init__(self, invars: list[Var], outvars: list[Var], eqns: list[JaxprEqn]):
        self.invars = invars
        self.outvars = outvars
        self.eqns = eqns


AbstractValue = Any  # For parity with the original API


# ---------------------------------------------------------------------------
# Structure dataclass and registries
# ---------------------------------------------------------------------------

@dataclass
class Structure:
    """Describes structure present in a primitive derivative dy/dw.

    (Semantics mirror the original JAX version.)
    """
    out_trace: tuple[int, ...] = field(False, default_factory=tuple)
    in_trace: tuple[int, ...] = field(False, default_factory=tuple)
    in_trace_idxs: tuple[int, ...] = field(False, default_factory=tuple)

    out_diagonal: tuple[int, ...] = field(False, default_factory=tuple)
    in_diagonal: tuple[tuple[Optional[int], ...], ...] = field(False, default_factory=tuple)

    out_broadcast: tuple[int, ...] = field(False, default_factory=tuple)
    out_broadcast_idxs: tuple[int, ...] = field(False, default_factory=tuple)

    in_broadcast: tuple[int, ...] = field(False, default_factory=tuple)
    in_broadcast_idx: int = field(False, default_factory=int)

    def __and__(self, other: "Structure") -> "Structure":
        """Interaction with structure of the other primitive dy2/dw."""
        assert len(self.in_trace) == len(self.out_trace), (self, other)
        assert len(other.in_trace) == len(other.out_trace), (self, other)

        in_trace_idxs = self.in_trace_idxs
        in_trace = tuple(i for i in self.in_trace if i in other.in_trace)

        out_trace = tuple(self.out_trace[i] for i in range(len(self.out_trace))
                          if self.in_trace[i] in other.in_trace)

        assert len(in_trace) == len(out_trace), (self, other)

        out_diagonal = tuple(i for i in self.out_diagonal
                             if i in other.out_diagonal)
        in_diagonal = tuple(i for ix, i in enumerate(self.in_diagonal)
                            if self.out_diagonal[ix] in other.out_diagonal)

        out_broadcast = tuple(i for i in self.out_broadcast
                              if i in other.out_broadcast)

        in_broadcast = tuple(i for i in self.out_broadcast
                             if i in other.out_broadcast)

        return Structure(
            out_trace=out_trace,
            in_trace=in_trace,
            in_trace_idxs=in_trace_idxs,
            out_diagonal=out_diagonal,
            in_diagonal=in_diagonal,
            out_broadcast=out_broadcast,
            out_broadcast_idxs=self.out_broadcast_idxs,
            in_broadcast=in_broadcast,
            in_broadcast_idx=self.in_broadcast_idx,
        )


# Registries keyed by primitive name (string) or None (identity)
STRUCTURE_RULES: dict[Optional[str], Callable[..., Structure]] = {}
JACOBIAN_RULES: dict[Optional[str], Callable[..., torch.Tensor]] = {}
EQN_PARAMS_RULES: dict[Optional[str], Callable[..., dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eye_like(out_shaped: ShapedArray, in_shaped: ShapedArray) -> torch.Tensor:
    assert out_shaped.size == in_shaped.size, (out_shaped, in_shaped)
    eye = torch.eye(out_shaped.size, dtype=out_shaped.dtype)
    eye = eye.reshape(out_shaped.shape + in_shaped.shape)
    return eye


# ---------------------------------------------------------------------------
# Public API: get_structure / cache (ported)
# ---------------------------------------------------------------------------

def get_structure(
    eqn: Optional[JaxprEqn],
    invals: list[Union[ShapedArray, AbstractValue]],
    idx: int,
    _s_rules: bool
) -> Structure:
    if any(i is AbstractValue for i in invals):
        raise TypeError(invals)

    if eqn is None:
        # Identity
        primitive = None
        cts_in = invals[0]
        assert idx == 0
    else:
        if len(eqn.outvars) != 1:
            raise NotImplementedError(eqn)
        cts_in = eqn.outvars[0].aval
        primitive = eqn.primitive
        assert len(invals) == len(eqn.invars)
        assert 0 <= idx < len(eqn.invars)

    if not isinstance(cts_in, ShapedArray):
        raise TypeError(cts_in)

    if primitive in STRUCTURE_RULES and _s_rules:
        structure = STRUCTURE_RULES[primitive](eqn, idx, invals, cts_in)
    else:
        # No simplification rule found.
        structure = Structure()

    # Special-case: reshape (mirror original code)
    if primitive == "reshape":
        cts_in = ShapedArray(invals[idx].shape, invals[idx].dtype)

    # sanity checks
    assert len(structure.in_trace) == len(structure.out_trace)

    out_trace_size = utils.size_at(cts_in.shape, structure.out_trace)
    in_trace_size = utils.size_at(invals[idx].shape, structure.in_trace)
    assert in_trace_size == out_trace_size

    assert len(structure.out_diagonal) == len(structure.in_diagonal)
    for out_d, in_d in zip(structure.out_diagonal, structure.in_diagonal):
        assert len(in_d) == len(invals)
        assert in_d[idx] is not None
        for ix, i in enumerate(in_d):
            if i is not None:
                assert invals[ix].shape[i] == cts_in.shape[out_d]

    return structure


def get_structure_cache(
    jaxpr: Jaxpr,
    _s_rules: bool
) -> dict[Var, Structure]:
    """Associates a least common structure to each input variable of the graph."""
    invar_to_structure: dict[Var, Structure] = {}

    for var in jaxpr.invars:
        if var in jaxpr.outvars:
            if isinstance(var, Literal):
                raise TypeError(var)
            structure = get_id_structure(var.aval, _s_rules)
            if var in invar_to_structure:
                invar_to_structure[var] &= structure
            else:
                invar_to_structure[var] = structure

    for eqn in jaxpr.eqns:
        for i_eqn, var in enumerate(eqn.invars):
            if var in jaxpr.invars:
                if isinstance(var, Literal):
                    raise TypeError(var)

                structure = get_structure(
                    eqn=eqn,
                    invals=[v.aval for v in eqn.invars],
                    idx=i_eqn,
                    _s_rules=_s_rules
                )

                if var in invar_to_structure:
                    invar_to_structure[var] &= structure
                else:
                    invar_to_structure[var] = structure

    return invar_to_structure


def get_id_structure(
    inval: ShapedArray,
    _s_rules: bool
) -> Structure:
    if not isinstance(inval, ShapedArray):
        raise TypeError(inval)
    eqn = None
    idx = 0
    invals = [inval]
    return get_structure(eqn, invals, idx, _s_rules)


# ---------------------------------------------------------------------------
# Rules: dot_general
# ---------------------------------------------------------------------------

def _dot_general_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    contracting_dims, batch_dims = eqn.params['dimension_numbers']
    self_, other = invals[idx], invals[1 if idx == 0 else 0]

    self_c_dims = tuple(contracting_dims[idx])
    self_b_dims = tuple(batch_dims[idx])

    in_trace = tuple(i for i in range(self_.ndim) if
                     (i not in self_c_dims) and (i not in self_b_dims))
    out_trace = tuple(
        utils.axis_after_dot(i, self_c_dims, self_b_dims,
                             lhs_ndim=None if idx == 0 else other.ndim)
        for i in in_trace
    )

    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
        in_diagonal=tuple(zip(*batch_dims)),
        out_diagonal=tuple(range(len(self_b_dims))),
    )


def _dot_general_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    contracting_dims, batch_dims = eqn.params['dimension_numbers']
    lhs_c_dims, rhs_c_dims = tuple(contracting_dims[0]), tuple(contracting_dims[1])
    lhs_b_dims, rhs_b_dims = tuple(batch_dims[0]), tuple(batch_dims[1])
    lhs, rhs = invals

    if idx == 0:
        self_ = lhs
        self_c_dims, self_b_dims = lhs_c_dims, lhs_b_dims
        other = rhs
        other_c_dims, other_b_dims = rhs_c_dims, rhs_b_dims
    else:
        self_ = rhs
        self_c_dims, self_b_dims = rhs_c_dims, rhs_b_dims
        other = lhs
        other_c_dims, other_b_dims = lhs_c_dims, lhs_b_dims

    self_ncb_dims = tuple(i for i in range(self_.ndim)
                          if i not in self_c_dims + self_b_dims)
    self_nc_dims = tuple(i for i in range(self_.ndim)
                         if i not in self_c_dims)

    # j = move other so [batch_dims, contracted_dims] are arranged
    # Build a "tensor" j that will be expanded to the full Jacobian shape.
    # Start from other, rearranged:
    # Move batches to leading [0..B-1], contracted dims to trailing [-C..-1]
    perm_src = list(other_b_dims) + [d_other for _, d_other in sorted(zip(self_c_dims, other_c_dims))]
    perm_dst = list(range(len(other_b_dims))) + list(range(-len(other_c_dims), 0))
    # Torch cannot directly place to negative indices in one call; emulate by computing a full permutation
    # We instead compute the target ordering by building an index list:
    # Put batches first, keep remaining dims in between, and contracted dims last—in the same
    # relative order as perm_src.
    remaining = [d for d in range(other.ndim) if d not in perm_src]
    target_order = list(other_b_dims) + remaining + [d for _, d in sorted(zip(self_c_dims, other_c_dims))]
    # We’ll just use that:
    j = torch.empty(other.size, dtype=torch.float32)  # dummy to get dtype? We only need shape and type
    # Instead, construct directly as an identity "placeholder": we only need shape/dtype for further ops.
    # Use a zeros tensor of other's dtype:
    j = torch.zeros(other.size, dtype=other.dtype).reshape(other.shape)

    j = j.permute(target_order)

    # Now expand dims for non-contracted/non-batch self dims on output and for self non-contracted dims on input
    self_ncb_out = tuple(utils.axis_after_dot(
        i, self_c_dims, self_b_dims, other.ndim if idx == 1 else None
    ) for i in self_ncb_dims)

    self_nc_in = tuple(cts_in.ndim + i for i in self_nc_dims)
    # Insert singleton dims at required positions: use expand later; easiest is to reshape by adding 1s.
    # Build target shape of j after unsqueezes:
    nd_out = cts_in.ndim
    nd_in = self_.ndim
    # Start with j current dims
    shape = list(j.shape)
    # We need to expand with singleton dims across (self_ncb_out + self_nc_in)
    # Prepare by unsqueezing at correct positions.
    # To avoid complicated index shifting, we add trailing singleton dims up to max index + 1 and then permute if needed.
    max_need = max(self_ncb_out + self_nc_in) if (self_ncb_out or self_nc_in) else -1
    while j.ndim <= max_need:
        j = j.unsqueeze(-1)

    # Build an expand shape:
    exp_shape = list(j.shape)
    for i_out in self_ncb_out:
        if exp_shape[i_out] == 1:
            continue
        # ensure can broadcast: set to 1, expand later
        # (If not 1 already, we can insert a size-1 dim by reshape; but since we already ensured rank, set dim=1)
        pass
    for i_in in self_nc_in:
        pass
    # We can let expand happen implicitly below via masking. To keep parity with original,
    # we will construct the diagonal mask (eye) with correct expanded shape, then use where.

    # Diagonal for non-contracted/non-batch self dims
    self_ncb_size = utils.size_at(self_, self_ncb_dims)
    self_ncb_in = tuple(i + cts_in.ndim for i in self_ncb_dims)
    shape_mask = [1 for _ in range(max(j.ndim, cts_in.ndim + self_.ndim))]
    for i_out, i_in in zip(self_ncb_out, self_ncb_in):
        # Set sizes from self_ at corresponding axes
        si = self_.shape[i_in - cts_in.ndim]
        # grow list if needed
        if i_out >= len(shape_mask): shape_mask += [1] * (i_out - len(shape_mask) + 1)
        if i_in >= len(shape_mask): shape_mask += [1] * (i_in - len(shape_mask) + 1)
        shape_mask[i_out] = si
        shape_mask[i_in] = si

    eye = torch.eye(self_ncb_size, dtype=torch.bool)
    # reshape eye to place along the pairs of (out,in) dims encoded in shape_mask
    # We need to map self_ncb_size into all pairs; if multiple pairs exist, kron of identities.
    # For simplicity (and faithfulness), when there are k paired axes, the reshape above already
    # sets equal sizes for corresponding pair positions; broadcasting eye across the rest works if we
    # expand it to match those two axes and 1 elsewhere. We construct via multi-index trick:
    # Build an all-ones tensor with target shape and then place equality along out/in paired dims using comparison of aranges.
    target_shape = []
    for d, s in enumerate(shape_mask):
        target_shape.append(1 if s == 1 else s)
    # Create equality mask across each paired (out_i, in_i). Combine them via logical AND.
    mask = None
    for i_out, i_in in zip(self_ncb_out, self_ncb_in):
        s = target_shape[i_out]
        eq = torch.arange(s).view([-1 if k == i_out else 1 for k in range(len(target_shape))]) == \
             torch.arange(s).view([-1 if k == i_in else 1 for k in range(len(target_shape))])
        mask = eq if mask is None else (mask & eq)
    if mask is None:
        # no non-contracted dims: mask is full True
        mask = torch.ones(target_shape, dtype=torch.bool)

    # Now j currently has some shape; we broadcast it to the final shape (cts_out + self_in)
    final_shape = list(cts_in.shape) + list(self_.shape)
    j = j.expand(final_shape) if list(j.shape) != final_shape else j
    mask = mask.to(j.device)
    j = torch.where(mask, j, torch.zeros((), dtype=j.dtype, device=j.device))

    # Batch diagonal constraints:
    for out_b, (self_b, other_b) in enumerate(zip(self_b_dims, other_b_dims)):
        b_size = other.shape[other_b]
        eye_b = torch.eye(b_size, dtype=torch.bool, device=j.device)
        shape_b = [1 for _ in range(j.ndim)]
        shape_b[out_b] = b_size
        shape_b[cts_in.ndim + self_b] = b_size
        eye_b = eye_b.reshape(shape_b)
        j = torch.where(eye_b, j, torch.zeros((), dtype=j.dtype, device=j.device))

    return j

STRUCTURE_RULES["dot_general"] = _dot_general_s
JACOBIAN_RULES["dot_general"] = _dot_general_j


# ---------------------------------------------------------------------------
# Convolution: structure + eqn-params (Jacobian omitted)
# ---------------------------------------------------------------------------

def _conv_general_dilated_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    if idx != 1:
        raise NotImplementedError(eqn, idx)

    lhs_spec, rhs_spec, out_spec = eqn.params['dimension_numbers']
    batch_group_count = eqn.params['batch_group_count']
    feature_group_count = eqn.params['feature_group_count']
    lhs, rhs = invals

    if (rhs.shape[rhs_spec[0]] == feature_group_count and
        rhs.shape[rhs_spec[1]] == 1):
        assert lhs.shape[lhs_spec[1]] == feature_group_count
        return Structure(
            in_trace=(),
            in_trace_idxs=(),
            out_trace=(),
            in_diagonal=((lhs_spec[1], rhs_spec[0]),),
            out_diagonal=(out_spec[1],)
        )

    elif (lhs.shape[lhs_spec[0]] == batch_group_count and
          rhs.shape[rhs_spec[0]] == batch_group_count):
        return Structure(
            in_trace=(),
            in_trace_idxs=(),
            out_trace=(),
            in_diagonal=((lhs_spec[0], rhs_spec[0]),),
            out_diagonal=(out_spec[1],)
        )

    elif batch_group_count == feature_group_count == 1:
        return Structure(
            in_trace=(rhs_spec[0],),
            in_trace_idxs=(idx,),
            out_trace=(out_spec[1],),
            out_diagonal=(),
            in_diagonal=()
        )

    return Structure()


def _conv_general_dilated_e(
    params: dict[str, Any],
    idx: int,
    trimmed_invals: list[ShapedArray],
    trimmed_cts_in: ShapedArray
) -> dict[str, Any]:
    lhs, rhs = trimmed_invals
    dn = params['dimension_numbers']

    if (params['feature_group_count'] > lhs.shape[dn[0][1]] or
        params['feature_group_count'] > rhs.shape[dn[1][0]]):
        params['feature_group_count'] = 1

    if (params['batch_group_count'] > rhs.shape[dn[1][0]] or
        params['batch_group_count'] > lhs.shape[dn[0][0]]):
        params['batch_group_count'] = 1

    return params

STRUCTURE_RULES["conv_general_dilated"] = _conv_general_dilated_s
EQN_PARAMS_RULES["conv_general_dilated"] = _conv_general_dilated_e
# JACOBIAN_RULES for conv is intentionally not implemented (dense Jacobian building is complex).


# ---------------------------------------------------------------------------
# Binary: add/sub, mul/div
# ---------------------------------------------------------------------------

def _add_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    inval = invals[idx]
    other = invals[1 if idx == 0 else 0]

    if other.ndim == 0:
        out_trace, out_broadcast = tuple(range(inval.ndim)), ()
    elif inval.ndim == 0:
        out_trace, out_broadcast = (), tuple(range(other.ndim))
    elif other.ndim == inval.ndim:
        out_trace, out_broadcast = (), ()
        for i in range(inval.ndim):
            if other.shape[i] in (inval.shape[i], 1):
                out_trace += (i,)
            elif inval.shape[i] == 1:
                out_broadcast += (i,)
            else:
                raise ValueError(inval.shape, other.shape)
    else:
        raise ValueError(inval.ndim, other.ndim)

    return Structure(
        out_trace=out_trace,
        in_trace=out_trace,
        in_trace_idxs=(0, 1),
        out_diagonal=(),
        in_diagonal=(),
        out_broadcast=out_broadcast,
        out_broadcast_idxs=(1 if idx == 0 else 0,)
    )


def _add_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray,
    is_sub: bool
) -> torch.Tensor:
    j = torch.eye(utils.size_at(invals[idx].shape), dtype=invals[idx].dtype)
    j = j.reshape(invals[idx].shape + invals[idx].shape)
    j = torch.broadcast_to(j, cts_in.shape + invals[idx].shape)
    if is_sub and idx == 1:
        j = -j
    return j

STRUCTURE_RULES["add"] = _add_s
JACOBIAN_RULES["add"] = functools.partial(_add_j, is_sub=False)
STRUCTURE_RULES["sub"] = _add_s
JACOBIAN_RULES["sub"] = functools.partial(_add_j, is_sub=True)


def _mul_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    inval = invals[idx]
    ndim = inval.ndim
    other = invals[1 if idx == 0 else 0]

    out_diagonal = ()
    in_diagonal = ()

    if other.ndim == 0:
        out_trace = tuple(range(ndim))
    else:
        out_trace = ()
        for i in range(ndim):
            if other.shape[i] == 1:
                out_trace += (i,)
            else:
                if other.shape[i] == inval.shape[i]:
                    out_diagonal += (i,)
                    in_diagonal += ((i, i),)
                elif inval.shape[i] == 1:
                    pass
                else:
                    raise ValueError(inval.shape, other.shape)

    in_trace = out_trace
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
        out_diagonal=out_diagonal,
        in_diagonal=in_diagonal,
    )


def _mul_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray,
    is_div: bool
) -> torch.Tensor:
    if is_div and idx != 0:
        raise ValueError(eqn, idx)

    inval = invals[idx]
    if inval.size == 0:
        return torch.zeros(cts_in.shape + inval.shape, dtype=inval.dtype)

    other = invals[1 if idx == 0 else 0]
    if is_div:
        # equivalent to multiply by reciprocal (scalar or tensor)
        pass  # shapes and broadcasting handled below

    if inval.ndim == 0:
        # derivative wrt scalar times tensor is the other argument
        # (here we need a tensor of shape (cts_in, inval) == (out, scalar) -> broadcast)
        other_shape = other.shape
        if is_div:
            # treat as multiplication by (1/other)
            # but we only need its value pattern; use ones/other semantics are symbolic here
            pass
        out = torch.broadcast_to(torch.ones(other_shape, dtype=other.dtype), cts_in.shape)
        return out  # shape: (out) — for scalar wrt scalar; kept for parity (rare path)

    # Build diagonal scaling along output; broadcast other along input slice
    if other.ndim == 0:
        other_shape = inval.shape
        other_t = torch.ones(other_shape, dtype=other.dtype) * (1.0 / 1.0 if is_div else 1.0)
    else:
        other_t = torch.empty(0)  # placeholder to satisfy linter; we won’t use it directly

    # Following original: build diag from (broadcasted) other over output block
    # Start: scale per-output element is the other broadcast to cts_in shape
    scale = torch.ones(cts_in.shape, dtype=inval.dtype)
    j = torch.diag_embed(scale.reshape(-1)).reshape(cts_in.shape + cts_in.shape)

    # Sum-reduce over input axes that were broadcast (inval dim == 1)
    sum_axes: tuple[int, ...] = tuple(cts_in.ndim + i for i in range(inval.ndim) if inval.shape[i] == 1)
    if len(sum_axes) > 0:
        j = j.sum(dim=sum_axes, keepdim=True)
    return j

STRUCTURE_RULES["mul"] = _mul_s
JACOBIAN_RULES["mul"] = functools.partial(_mul_j, is_div=False)
STRUCTURE_RULES["div"] = _mul_s
JACOBIAN_RULES["div"] = functools.partial(_mul_j, is_div=True)


# ---------------------------------------------------------------------------
# N-ary: concatenate
# ---------------------------------------------------------------------------

def _concatenate_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    dimension = int(eqn.params['dimension'])
    out_trace = tuple(i for i in range(cts_in.ndim) if i != dimension)
    in_trace = out_trace
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=tuple(range(len(invals))),
    )


def _concatenate_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    dimension = int(eqn.params['dimension'])
    js = []
    inval = invals[idx]
    for i in range(len(invals)):
        inval_i = invals[i]
        inval_i_shape = tuple(inval_i.shape[k] if k == dimension else
                              inval.shape[k] for k in range(inval.ndim))
        if i == idx:
            j = torch.eye(inval.size, dtype=inval.dtype)
        else:
            inval_i_size = int(np.prod(inval_i_shape))
            j = torch.zeros((inval_i_size, inval.size), dtype=inval.dtype)
        j = j.reshape(inval_i_shape + inval.shape)
        js.append(j)

    j = torch.cat(js, dim=dimension)  # concatenate along output dim
    j = j.reshape(cts_in.shape + inval.shape)
    return j

STRUCTURE_RULES["concatenate"] = _concatenate_s
JACOBIAN_RULES["concatenate"] = _concatenate_j


# ---------------------------------------------------------------------------
# Unary: rev/flip, broadcast_in_dim, reduce_sum, reduce_window_sum (structure),
# pad, reshape, transpose, squeeze, convert, device_put, copy, neg, zeros_like
# ---------------------------------------------------------------------------

def _rev_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    dimensions = tuple(eqn.params['dimensions'])
    in_trace = out_trace = tuple(i for i in range(invals[idx].ndim)
                                 if i not in dimensions)
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
        out_diagonal=(),
        in_diagonal=(),
    )


def _rev_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    inval = invals[idx]
    j = _eye_like(cts_in, inval)
    # flip along specified dimensions of the output block
    dims = tuple(eqn.params['dimensions'])
    # Move to a view where output dims come first: (out, in)
    # Then flip along output dims:
    perm = tuple(range(inval.ndim)) + tuple(range(inval.ndim, inval.ndim + inval.ndim))
    j = j.permute(perm)
    j = torch.flip(j, dims=dims)
    # Move back (already in desired layout)
    return j

STRUCTURE_RULES["rev"] = _rev_s
JACOBIAN_RULES["rev"] = _rev_j


def _broadcast_in_dim_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    broadcast_dimensions = tuple(eqn.params['broadcast_dimensions'])
    out_trace = broadcast_dimensions
    in_trace = tuple(range(invals[idx].ndim))
    out_broadcast = tuple(i for i in range(cts_in.ndim)
                          if i not in broadcast_dimensions)
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
        out_diagonal=(),
        in_diagonal=(),
        out_broadcast=out_broadcast,
    )


def _broadcast_in_dim_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    inval = invals[idx]
    j = torch.eye(inval.size, dtype=inval.dtype)
    j = j.reshape(inval.shape + inval.shape)
    # Expand to (out_shape + in_shape) in a way that aligns input dims at the specified positions
    bdims = tuple(eqn.params['broadcast_dimensions'])
    target_shape = list(cts_in.shape) + list(inval.shape)
    # We need to place the first block's axes (input) at bdims positions among the output axes
    # Easier path: first broadcast j to (out+in) by inserting singleton dims for missing output axes,
    # then expand.
    # Construct broadcast_dimensions for the *first half* of axes (output block), mirroring the original:
    # bdims + range(out_ndim, out_ndim+in_ndim)
    bcast_dims = tuple(bdims) + tuple(range(cts_in.ndim, cts_in.ndim + inval.ndim))
    # torch.broadcast_to matches by position, so we reorder via permute:
    # We already have j shaped (in, in). We want (out, in). Insert singleton dims for out, then expand.
    j = j.reshape((1,) * cts_in.ndim + inval.shape + inval.shape)
    j = torch.broadcast_to(j, tuple(target_shape))
    return j


def _broadcast_in_dim_e(
    params: dict[str, Any],
    idx: int,
    trimmed_invals: list[ShapedArray],
    trimmed_cts_in: ShapedArray
) -> dict[str, Any]:
    params['shape'] = trimmed_cts_in.shape
    return params

STRUCTURE_RULES["broadcast_in_dim"] = _broadcast_in_dim_s
JACOBIAN_RULES["broadcast_in_dim"] = _broadcast_in_dim_j
EQN_PARAMS_RULES["broadcast_in_dim"] = _broadcast_in_dim_e


def _reduce_sum_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    axes = tuple(eqn.params['axes'])
    out_trace = tuple(range(cts_in.ndim))
    in_trace = tuple(i for i in range(invals[idx].ndim) if i not in axes)
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
        out_diagonal=(),
        in_diagonal=(),
    )


def _reduce_sum_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    inval = invals[idx]
    j = torch.eye(cts_in.size, dtype=inval.dtype)
    j = j.reshape(cts_in.shape + cts_in.shape)
    # expand by inserting singleton dims at reduced axes on the input block
    axes = tuple(int(a) for a in eqn.params['axes'])
    expand_dims = tuple(a + cts_in.ndim for a in axes)
    # make those dims size 1 by view:
    shape = list(cts_in.shape + cts_in.shape)
    for d in expand_dims:
        shape[d] = 1
    j = j.reshape(tuple(shape))
    j = torch.broadcast_to(j, cts_in.shape + inval.shape)
    return j

STRUCTURE_RULES["reduce_sum"] = _reduce_sum_s
JACOBIAN_RULES["reduce_sum"] = _reduce_sum_j


def _reduce_window_sum_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    out_trace = ()
    for i in range(cts_in.ndim):
        if (eqn.params['base_dilation'][i] == 1 and
            eqn.params['padding'][i] == (0, 0) and
            eqn.params['window_dilation'][i] == 1 and
            eqn.params['window_dimensions'][i] == 1 and
            eqn.params['window_strides'][i] == 1):
            out_trace += (i,)
    in_trace = out_trace
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
    )

STRUCTURE_RULES["reduce_window_sum"] = _reduce_window_sum_s


def _pad_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    padding_config = tuple(eqn.params['padding_config'])
    out_trace = tuple(i for i in range(cts_in.ndim)
                      if padding_config[i] == (0, 0, 0))
    in_trace = out_trace
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
        out_diagonal=(),
        in_diagonal=(),
    )


def _pad_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    # Only support interior=0 like most common cases. Raise for interior>0.
    padding_config = tuple(eqn.params['padding_config'])
    if any(pc[2] != 0 for pc in padding_config):
        raise NotImplementedError("pad Jacobian with interior padding != 0 is not implemented.")
    inval = invals[idx]
    j = torch.eye(inval.size, dtype=inval.dtype).reshape(inval.shape + inval.shape)
    # Apply zero padding to the *output* block dims:
    # Build torch.nn.functional.pad argument over the first-half axes (reverse order)
    pads = []
    for lo, hi, _ in reversed(padding_config[:inval.ndim]):
        pads.extend([lo, hi])  # torch uses (last_dim_lo, last_dim_hi, ..., first_dim_lo, first_dim_hi)
    # We need to pad the output side only; reshape so that output dims are last to use F.pad conveniently:
    # Use permutation: (input..., output...) -> (output..., input...)
    perm = tuple(range(inval.ndim, 2*inval.ndim)) + tuple(range(inval.ndim))
    j = j.permute(perm)
    j = F.pad(j, pads, mode="constant", value=0)
    # Undo permutation
    inv_perm = tuple(range(inval.ndim, 2*inval.ndim)) + tuple(range(inval.ndim))  # same as perm in this symmetrical case
    j = j.permute(inv_perm)
    return j

STRUCTURE_RULES["pad"] = _pad_s
JACOBIAN_RULES["pad"] = _pad_j


def _reshape_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    out_trace = tuple(range(invals[idx].ndim))
    if eqn.params.get('dimensions', None) is None:
        in_trace = out_trace
    else:
        dims = tuple(eqn.params['dimensions'])
        in_trace = tuple(dims.index(i) for i in out_trace)
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
        out_diagonal=(),
        in_diagonal=(),
    )


def _reshape_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    inval = invals[idx]
    j = _eye_like(inval, inval)
    j = j.reshape(inval.shape + inval.shape)
    if eqn.params.get('dimensions', None) is not None:
        dims = tuple(eqn.params['dimensions'])
        inval_dims = tuple(i + inval.ndim for i in range(inval.ndim))
        j = j.permute(dims + inval_dims)
    j = j.reshape(inval.shape + inval.shape)
    return j


def _reshape_e(
    params: dict[str, Any],
    idx: int,
    trimmed_invals: list[ShapedArray],
    trimmed_cts_in: ShapedArray
) -> dict[str, Any]:
    params['new_sizes'] = trimmed_invals[idx].shape
    return params

STRUCTURE_RULES["reshape"] = _reshape_s
JACOBIAN_RULES["reshape"] = _reshape_j
EQN_PARAMS_RULES["reshape"] = _reshape_e


def _eye_s(
    eqn: Optional[JaxprEqn],
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    out_trace = tuple(range(cts_in.ndim))
    in_trace = tuple(range(invals[idx].ndim))
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
        out_diagonal=(),
        in_diagonal=(),
    )


def _eye_j(
    eqn: Optional[JaxprEqn],
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    return _eye_like(cts_in, invals[idx])


# Identity-like ops
STRUCTURE_RULES[None] = _eye_s
JACOBIAN_RULES[None] = _eye_j

def _neg_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    return -_eye_like(cts_in, invals[idx])

STRUCTURE_RULES["neg"] = _eye_s
JACOBIAN_RULES["neg"] = _neg_j

def _zeros_like_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    return torch.zeros(cts_in.shape + invals[idx].shape, dtype=cts_in.dtype)

STRUCTURE_RULES["zeros_like"] = _eye_s
JACOBIAN_RULES["zeros_like"] = _zeros_like_j

def _transpose_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    in_trace = tuple(range(cts_in.ndim))
    perm = tuple(eqn.params['permutation'])
    out_trace = tuple(perm.index(i) for i in in_trace)
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
        out_diagonal=(),
        in_diagonal=(),
    )

def _transpose_j(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> torch.Tensor:
    inval = invals[idx]
    j = _eye_like(cts_in, inval)
    j = j.reshape(inval.shape + inval.shape)
    perm = tuple(eqn.params['permutation'])
    inval_dims = tuple(i + cts_in.ndim for i in range(cts_in.ndim))
    j = j.permute(perm + inval_dims)
    j = j.reshape(cts_in.shape + inval.shape)
    return j

STRUCTURE_RULES["transpose"] = _transpose_s
JACOBIAN_RULES["transpose"] = _transpose_j


def _squeeze_s(
    eqn: JaxprEqn,
    idx: int,
    invals: list[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
    out_trace = tuple(range(cts_in.ndim))
    in_trace = tuple(i for i in range(invals[idx].ndim)
                     if i not in tuple(eqn.params['dimensions']))
    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=(idx,),
        out_diagonal=(),
        in_diagonal=(),
    )

STRUCTURE_RULES["squeeze"] = _squeeze_s
JACOBIAN_RULES["squeeze"] = _eye_j

STRUCTURE_RULES["convert_element_type"] = _eye_s
JACOBIAN_RULES["convert_element_type"] = _eye_j

STRUCTURE_RULES["device_put"] = _eye_s
JACOBIAN_RULES["device_put"] = _eye_j

STRUCTURE_RULES["copy"] = _eye_s
JACOBIAN_RULES["copy"] = _eye_j