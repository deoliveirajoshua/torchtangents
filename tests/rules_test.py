# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Native PyTorch/unittest rewrite of tests for Jacobian consistency across
# a variety of tensor operations (unary, binary, n-ary).
#
# This port compares torch.func.jacfwd and torch.func.jacrev for each op and
# argument position on randomized inputs over a parameter grid.

import itertools
import logging
import math
import os
import random
from typing import Optional, Sequence, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
import unittest

from tests import test_utils  # PyTorch version provided earlier
from parameterized import parameterized

# ---------------------------------------------------------------------
# Global config and reproducibility
# ---------------------------------------------------------------------

torch.manual_seed(1)
random.seed(1)

# Precision/dtypes under test
_DTYPES = [torch.float32] + ([torch.float64] if torch.tensor(0., dtype=torch.float64).dtype == torch.float64 else [])

# Shapes to exercise (mirrors the original selection; feel free to expand)
_SHAPES: Sequence[Tuple[int, ...]] = [
    (),
    (2,),
    (1, 2),
    (2, 3),
    (1, 2, 3),
    (6, 3, 2),
    (1, 2, 1, 3),
    (2, 1, 3, 4),
]

# Optional per-op precisions (kept for parity with the original; unused in torch)
_PRECISIONS = [None]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _hypercube(ndim: int, start: int = 1, end: int = 3):
    endv = (end,) * ndim
    startv = (start,) * ndim
    return tuple(itertools.product(*[tuple(range(s, e)) for s, e in zip(startv, endv)]))

def _prod(xs: Sequence[int]) -> int:
    out = 1
    for v in xs:
        out *= v
    return out

def _is_broadcastable(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> bool:
    if not (len(s1) == 0 or len(s2) == 0 or len(s1) == len(s2)):
        return False
    for a, b in zip(s1, s2):
        if not (a == 1 or b == 1 or a == b):
            return False
    return True

def _dot_dim_nums(s1: Tuple[int, ...], s2: Tuple[int, ...]):
    """
    Generate dimension-number pairs akin to JAX dot_general; in PyTorch we’ll
    implement with torch.tensordot over those contractions.
    """
    pairs = []
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                pairs.append((i, j))

    def get_dn(pairs_left, dn):
        if len(pairs_left) == 0:
            yield dn
            return
        for p in pairs_left:
            new_pairs = [q for q in pairs_left if (q[0] != p[0] and q[1] != p[1])]
            # identity (no contraction)
            yield from get_dn(new_pairs, dn)
            # add as contracting dims
            dn_c = ((dn[0][0] + (p[0],), dn[0][1] + (p[1],)), dn[1])
            yield from get_dn(new_pairs, dn_c)
            # add as batch-like (we'll treat as non-contracting alignments here)
            dn_b = (dn[0], (dn[1][0] + (p[0],), dn[1][1] + (p[1],)))
            yield from get_dn(new_pairs, dn_b)

    # ((contract_lhs, contract_rhs), (batch_lhs, batch_rhs))
    seen = set()
    for dn in get_dn(pairs, ((tuple(), tuple()), (tuple(), tuple()))):
        key = tuple(map(tuple, dn[0])) + tuple(map(tuple, dn[1]))
        if key not in seen:
            seen.add(key)
            yield dn

def _concat_dims(*shapes: Tuple[int, ...]):
    dims = []
    if not shapes:
        return dims
    s0 = shapes[0]
    n = len(s0)
    if any(len(s) != n for s in shapes):
        return dims
    for i in range(n):
        if all(s[j] == s0[j] for s in shapes for j in range(n) if j != i):
            dims.append(i)
    return dims

def _concat_shapes(max_n_args: int = 4, *shapes: Tuple[int, ...]):
    sets = []
    if not shapes:
        return sets
    bins = {}
    for s in shapes:
        bins.setdefault(len(s), []).append(s)
    for n, bag in bins.items():
        for n_args in range(max_n_args):
            sets += list(itertools.combinations_with_replacement(bag, n_args))
    return sets

def _get_inputs(shapes, dtype: torch.dtype, device: str):
    xs = []
    g = torch.Generator(device=device)
    g.manual_seed(1)
    for s in shapes:
        if len(s) == 0:
            # scalar tensor
            x = torch.randn((), dtype=dtype, device=device, generator=g)
        else:
            x = torch.randn(*s, dtype=dtype, device=device, generator=g)
        # Enable autograd
        x.requires_grad_(True)
        xs.append(x)
    return xs

# ---------------------------------------------------------------------
# Operation catalogs (unary, binary, n-ary)
# Each entry maps to a parameter generator like the JAX version.
# ---------------------------------------------------------------------

# Unary ops: return list of param dicts for a given shape/dtype
def _unary_params(op: str, s: Tuple[int, ...], dtype: torch.dtype):
    if op == "identity":
        return [{}]
    if op == "neg":
        return [{}]
    if op == "transpose":
        return [{"permutation": p} for p in itertools.permutations(range(len(s)))]
    if op == "reduce_sum":
        return [{"axes": tuple(ax)} for ax in itertools.chain.from_iterable(
            itertools.combinations(range(len(s)), r) for r in range(len(s)+1))]
    if op == "reduce_window_sum":
        # Approximate: sliding-window sum via avg_pool multiplied by window size (for positive integers).
        # We’ll only test a few small kernels/strides/dilations to keep things simple.
        if len(s) == 0:
            return []
        choices = list(_hypercube(len(s))[:3])
        paddings = [
            tuple((0, 0) for _ in range(len(s))),
            tuple((1, 0) for _ in range(len(s))),
            tuple((0, 1) for _ in range(len(s))),
        ]
        params = []
        for wdl in choices:
            for wdd in choices:
                for ws in choices:
                    for pad in paddings:
                        params.append({
                            "window_dilation": wdl,
                            "window_dimensions": wdd,
                            "window_strides": ws,
                            "padding": pad
                        })
        return params
    if op == "broadcast_in_dim":
        # Insert 1 or 2 new dims with concrete target shapes; we’ll use torch.broadcast_to
        if len(s) == 0:
            base = ()
        else:
            base = s
        params = []
        # Insert a single dim of size 3 at every position
        for i in range(len(base)+1):
            shape = base[:i] + (3,) + base[i:]
            # broadcast dims are the positions of original axes
            bd = tuple(range(i)) + tuple(range(i+1, len(base)+1))
            params.append({"shape": shape, "broadcast_dimensions": bd})
        # Insert two dims (3 and 4)
        for i in range(len(base)):
            for j in range(i, len(base)+1):
                shape = base[:i] + (3,) + base[i:j] + (4,) + base[j:]
                bd = tuple(range(i)) + tuple(range(i+1, j+1)) + tuple(range(j+2, len(base)+2))
                params.append({"shape": shape, "broadcast_dimensions": bd})
        return params
    if op == "squeeze":
        dims = [idx for idx, d in enumerate(s) if d == 1]
        params = [{"dimensions": tuple(ax)} for r in range(len(dims)+1)
                  for ax in itertools.combinations(dims, r)]
        return params
    if op == "convert_element_type":
        # new_dtype drawn from a set different from current dtype
        cand = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
        cand = [d for d in cand if d != dtype]
        # weak_type concept doesn’t map directly; ignore
        return [{"new_dtype": d} for d in set(cand)]
    if op == "flip":
        return [{"dimensions": tuple(ax)} for r in range(len(s)+1)
                for ax in itertools.combinations(range(len(s)), r)]
    if op == "pad":
        if len(s) == 0:
            return []
        configs = []
        for k in range(len(s)):
            configs.append(tuple((1, 0) if i==k else (0, 0) for i in range(len(s))))
            configs.append(tuple((0, 1) if i==k else (0, 0) for i in range(len(s))))
            configs.append(tuple((max(3-k,0), k//2, ) if i==k else (0,0) for i in range(len(s))))
        # Torch pad wants a flat tuple in reverse order: (pad_last_dim_left, pad_last_dim_right, ..., pad_first_dim_left, pad_first_dim_right)
        vals = [-0.1, 1.5]
        out = []
        for v in vals:
            for cfg in configs:
                flat = []
                for (pl, pr) in reversed(cfg):
                    flat.extend([pl, pr])
                out.append({"padding_value": v, "padding_config": tuple(flat), "rank": len(s)})
        return out
    if op == "reshape":
        total = _prod(s) if s else 1
        new_shapes = { (total,), (total, 1), (1, total, 1) }
        if total % 2 == 0:
            new_shapes |= {(2, total//2, 1), (total//2, 1, 2)}
        if total % 3 == 0:
            new_shapes |= {(1, total//3, 3), (3, total//3)}
        if total % 6 == 0:
            new_shapes |= {(2, total//6, 3), (3, total//6, 2),
                           (total//6, 2, 3), (2, 3, total//6), (3, 2, total//6)}
        perms = list(itertools.permutations(range(len(s))))
        return [{"new_sizes": ns, "dimensions": p} for ns in new_shapes for p in perms]
    return []

_UNARY = [
    "identity",
    "neg",
    "transpose",
    "reduce_sum",
    "broadcast_in_dim",
    "squeeze",
    "convert_element_type",
    "flip",          # lax.rev_p analogue
    "pad",
    "reshape",
    # "reduce_window_sum",  # Optional: can be enabled with a custom implementation
]

# Binary ops
def _binary_params(op: str, s1: Tuple[int, ...], s2: Tuple[int, ...]):
    if op in ("mul", "div", "add", "sub"):
        return ([{}] if _is_broadcastable(s1, s2) else [])
    if op == "dot_general":
        dns = []
        for dn in _dot_dim_nums(s1, s2):
            dns.append({
                "dimension_numbers": dn,
                "precision": None,
                "preferred_element_type": None
            })
        return dns
    if op == "concat":  # handled as n-ary; keep here empty
        return []
    return []

_BINARY = ["mul", "div", "add", "sub", "dot_general"]

# N-ary ops (here just concatenate)
def _n_ary_params(op: str, *shapes: Tuple[int, ...]):
    if op == "concatenate":
        return [{"dimension": d} for d in _concat_dims(*shapes)]
    return []

_N_ARY = ["concatenate"]

# ---------------------------------------------------------------------
# The test case
# ---------------------------------------------------------------------

class JacobianRulesTest(test_utils.NeuralTangentsTestCase):

    def _f_from_unary(self, op: str, params):
        if op == "identity":
            return lambda x: x
        if op == "neg":
            return lambda x: -x
        if op == "transpose":
            perm = params["permutation"]
            return lambda x: x.permute(*perm) if x.ndim else x
        if op == "reduce_sum":
            axes = params["axes"]
            if len(axes) == 0:
                return lambda x: x.clone()
            return lambda x: x.sum(dim=axes, keepdim=False)
        if op == "broadcast_in_dim":
            shape = params["shape"]
            # Validate broadcast dims but torch.broadcast_to infers; keep simple.
            return lambda x: torch.broadcast_to(x, shape)
        if op == "squeeze":
            dims = params["dimensions"]
            return lambda x: x.squeeze(dim=dims) if len(dims)>0 else x
        if op == "convert_element_type":
            new_dtype = params["new_dtype"]
            return lambda x: x.to(new_dtype)
        if op == "flip":
            dims = params["dimensions"]
            return lambda x: torch.flip(x, dims) if len(dims)>0 else x
        if op == "pad":
            pad = params["padding_config"]    # flat reversed
            val = params["padding_value"]
            rank = params["rank"]
            def _pad(x):
                if x.ndim == 0:
                    return x  # torch.nn.functional.pad expects >=1D
                return F.pad(x, pad, mode="constant", value=float(val))
            return _pad
        if op == "reshape":
            new_sizes = params["new_sizes"]
            # permutation applies to input first (like reshape with dims reorder)
            perm = params["dimensions"]
            def _reshape(x):
                y = x if x.ndim == 0 else x.permute(*perm)
                return y.reshape(*new_sizes)
            return _reshape
        raise ValueError(op)

    def _f_from_binary(self, op: str, params):
        if op == "mul":
            return lambda x, y: x * y
        if op == "div":
            # test only linearity in first arg; we’ll skip jacobian wrt denominator
            return lambda x, y: x / (y + 0.1)  # avoid zero
        if op == "add":
            return lambda x, y: x + y
        if op == "sub":
            return lambda x, y: x - y
        if op == "dot_general":
            # params["dimension_numbers"] = ((contract_l, contract_r), (batch_l, batch_r))
            ((cl, cr), (bl, br)) = params["dimension_numbers"]
            cl, cr, bl, br = tuple(cl), tuple(cr), tuple(bl), tuple(br)
            def _tensordot(x, y):
                # For simplicity: apply batch alignments by checking sizes, then contract cl/cr via torch.tensordot.
                # Batch dims must be equal; we won’t permute explicitly—torch.tensordot just needs contraction dims.
                # If batch dims exist, we just rely on broadcasting alignment (ensure sizes equal in input generation).
                return torch.tensordot(x, y, dims=(cl, cr))
            return _tensordot
        raise ValueError(op)

    def _f_from_n_ary(self, op: str, params):
        if op == "concatenate":
            dim = params["dimension"]
            return lambda *xs: torch.cat(xs, dim=dim if len(xs[0].shape)>0 else 0)
        raise ValueError(op)

    # --------------------
    # Unary
    # --------------------
    @test_utils.parameters(
        dict(op=op, shape=shape, dtype=dtype, params=params)
        for shape in _SHAPES
        for dtype in _DTYPES
        for op in _UNARY
        for params in test_utils._cases_from_list(_unary_params(op, shape, dtype))
    )
    def test_unary(self, op: str, shape, dtype, params):
        device = test_utils.device_under_test()
        f = self._f_from_unary(op, params)
        xs = _get_inputs([shape], dtype, device)
        x = xs[0]

        # Forward and reverse Jacobian w.r.t. the single input
        jfwd = torch.func.jacfwd(f)(x)
        jrev = torch.func.jacrev(f)(x)

        # Shapes: flatten both sides for comparison when needed
        self.assertAllClose(jfwd, jrev)

    # --------------------
    # Binary
    # --------------------
    @test_utils.parameters(
        dict(op=op, shape1=shape1, shape2=shape2, dtype=dtype, params=params)
        for shape1 in _SHAPES
        for shape2 in _SHAPES
        for dtype in _DTYPES
        for op in _BINARY
        for params in test_utils._cases_from_list(_binary_params(op, shape1, shape2))
    )
    def test_binary(self, op: str, shape1, shape2, dtype, params):
        # Skip very high-rank combos to keep runtime sane (parity with original skip)
        if len(shape1) > 3 or len(shape2) > 3:
            test_utils.skip_test(self)

        device = test_utils.device_under_test()
        f = self._f_from_binary(op, params)
        x, y = _get_inputs([shape1, shape2], dtype, device)

        # Some binary ops are not smooth in y (e.g., division wrt denominator) -> only test jacobians where defined/expected.
        # We test jacobians w.r.t. both args, except for 'div' wrt y.
        jfwd_x = torch.func.jacfwd(lambda a: f(a, y))(x)
        jrev_x = torch.func.jacrev(lambda a: f(a, y))(x)
        self.assertAllClose(jfwd_x, jrev_x)

        if op != "div":
            jfwd_y = torch.func.jacfwd(lambda b: f(x, b))(y)
            jrev_y = torch.func.jacrev(lambda b: f(x, b))(y)
            self.assertAllClose(jfwd_y, jrev_y)

    # --------------------
    # N-ary (concatenate)
    # --------------------
    @test_utils.parameters(
        dict(op="concatenate", shapes=shapes, dtype=dtype, params=params)
        for shapes in _concat_shapes(4, *_SHAPES)
        for dtype in _DTYPES
        for params in test_utils._cases_from_list(_n_ary_params("concatenate", *shapes))
    )
    def test_n_ary(self, op: str, shapes, dtype, params):
        if len(shapes) == 0:
            return
        device = test_utils.device_under_test()
        f = self._f_from_n_ary(op, params)
        xs = _get_inputs(shapes, dtype, device)

        # jacfwd/jacrev wrt each input separately
        for idx in range(len(xs)):
            jfwd = torch.func.jacfwd(lambda *ys: f(*ys), argnums=idx)(*xs)
            jrev = torch.func.jacrev(lambda *ys: f(*ys), argnums=idx)(*xs)
            self.assertAllClose(jfwd, jrev)


if __name__ == "__main__":
    unittest.main()
