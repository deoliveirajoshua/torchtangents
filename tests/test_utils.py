# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# This file is a native PyTorch reimplementation of JAX-based test utilities.

import dataclasses
import itertools
import logging
import os
from types import ModuleType
from typing import Callable, Optional, Sequence, Any

import numpy as np
import torch
import unittest
from parameterized import parameterized as _parameterized  # alias for clarity




# -----------------------------------------------------------------------------
# Configuration & device detection
# -----------------------------------------------------------------------------

NT_TEST_DUT = os.getenv("NT_TEST_DUT", "").strip()
NT_NUM_GENERATED_CASES = int(os.getenv("NT_NUM_GENERATED_CASES", "4"))


# --- Add this probe right after the imports ---
try:
    _NP_BFLOAT16 = np.dtype("bfloat16")  # will raise on older NumPy
    _HAS_NP_BFLOAT16 = True
except TypeError:
    _NP_BFLOAT16 = None
    _HAS_NP_BFLOAT16 = False

def device_under_test() -> str:
    if NT_TEST_DUT:
        return NT_TEST_DUT
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

# -----------------------------------------------------------------------------
# Dtype helpers & tolerances
# -----------------------------------------------------------------------------

_python_scalar_dtypes = {
    bool: np.dtype("bool"),
    int: np.dtype("int64"),
    float: np.dtype("float64"),
    complex: np.dtype("complex128"),
}

def _dtype(x) -> np.dtype:
    if hasattr(x, "dtype"):
        if isinstance(x, torch.Tensor):
            # Map torch dtypes to numpy dtypes without going through .numpy() for bf16
            td = x.dtype
            if td is torch.float32:
                return np.dtype(np.float32)
            if td is torch.float64:
                return np.dtype(np.float64)
            if td is torch.float16:
                return np.dtype(np.float16)
            if td is torch.bfloat16:
                # Use np.bfloat16 if available; otherwise compare as float32
                return _NP_BFLOAT16 if _HAS_NP_BFLOAT16 else np.dtype(np.float32)
            if td is torch.int64:
                return np.dtype(np.int64)
            if td is torch.int32:
                return np.dtype(np.int32)
            if td is torch.bool:
                return np.dtype(np.bool_)
            # Fallback: go through numpy (works for supported dtypes)
            return np.dtype(str(x.detach().cpu().to(torch.float32).numpy().dtype))
        # NumPy array
        return np.dtype(x.dtype)
    elif type(x) in _python_scalar_dtypes:
        return np.dtype(_python_scalar_dtypes[type(x)])
    else:
        return np.asarray(x).dtype


_DEFAULT_TOLERANCE = {
    np.dtype(np.bool_): 0.0,
    np.dtype(np.int32): 0.0,
    np.dtype(np.int64): 0.0,
    np.dtype(np.float16): 5e-3,
    np.dtype(np.float32): 5e-3,
    np.dtype(np.float64): 1e-5,
}

# --- Only register bf16 tolerance if NumPy supports it ---
if _HAS_NP_BFLOAT16:
    _DEFAULT_TOLERANCE[_NP_BFLOAT16] = 5e-3


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        # --- Convert bf16 to f32 if NumPy doesn't support bf16 ---
        if t.dtype is torch.bfloat16 and not _HAS_NP_BFLOAT16:
            t = t.to(torch.float32)
        return t.numpy()
    return np.asarray(x)


def _default_tolerance() -> dict[np.dtype, float]:
    # Keep a TPU branch for completeness; PyTorch doesn't target TPU.
    dut = device_under_test()
    if dut not in ("tpu",):
        return _DEFAULT_TOLERANCE
    tol = _DEFAULT_TOLERANCE.copy()
    tol[np.dtype(np.float32)] = 5e-2
    tol[np.dtype(np.complex64)] = 5e-2
    return tol

def _tolerance(dtype: np.dtype, tol: Optional[float] = None) -> float:
    tol_mapping = {} if tol is None else tol
    if not isinstance(tol_mapping, dict):
        return tol_mapping
    tol_mapping = {np.dtype(k): v for k, v in tol_mapping.items()}
    dtype = np.dtype(dtype)
    return tol_mapping.get(dtype, _default_tolerance()[dtype])

def update_test_tolerance(f32_tol: float = 5e-3, f64_tol: float = 1e-5):
    _DEFAULT_TOLERANCE[np.dtype(np.float32)] = f32_tol
    _DEFAULT_TOLERANCE[np.dtype(np.float64)] = f64_tol

# -----------------------------------------------------------------------------
# Parameter generation (sampling cartesian product)
# -----------------------------------------------------------------------------

_CACHED_INDICES: dict[int, Sequence[int]] = {}

def _cases_from_list(xs):
    xs = list(xs)
    n = len(xs)
    if n <= NT_NUM_GENERATED_CASES:
        return xs
    k = min(n, NT_NUM_GENERATED_CASES)
    indices = _CACHED_INDICES.get(n)
    if indices is None:
        rng = np.random.RandomState(42)
        _CACHED_INDICES[n] = indices = rng.permutation(n)
    return [xs[i] for i in indices[:k]]

def product(*kwargs_seqs, **testgrid):
    """
    Test case decorator to randomly subset a cartesian product of parameters.
    Mirrors absltest.parameterized.product behavior.
    """
    for name, values in testgrid.items():
        if not isinstance(values, (list, tuple)):
            raise AssertionError(
                f"Values of {name} must be list/tuple, found {type(values)}"
            )

    prior_arg_names = set()
    for kwargs_seq in kwargs_seqs:
        if not (isinstance(kwargs_seq, (list, tuple)) and
                all(isinstance(kwargs, dict) for kwargs in kwargs_seq)):
            raise AssertionError(
                "Positional parameters must be a sequence of kwargs dicts")
        if kwargs_seq:
            arg_names = set(kwargs_seq[0])
            if not all(set(kwargs) == arg_names for kwargs in kwargs_seq):
                raise AssertionError("All kwargs dicts must share the same keys")
            if arg_names & prior_arg_names:
                dup = sorted(arg_names & prior_arg_names)
                raise AssertionError(f"Duplicate arg names across sequences: {dup}")
            prior_arg_names |= arg_names

    if prior_arg_names & set(testgrid):
        dup = sorted(prior_arg_names & set(testgrid))
        raise AssertionError(f"Duplicate arg names between positional and named: {dup}")

    testgrid_seq = (tuple({k: v} for v in vs) for k, vs in testgrid.items())
    testgrid_seq = tuple(kwargs_seqs) + tuple(testgrid_seq)

    testcases = [
        dict(itertools.chain.from_iterable(case.items() for case in cases))
        for cases in itertools.product(*testgrid_seq)
    ]
    return parameters(testcases)


def parameters(testcases):
    subset = _cases_from_list(testcases)
    return _parameterized.expand(subset)

# -----------------------------------------------------------------------------
# Tree utilities
# -----------------------------------------------------------------------------

def _is_sequence(x) -> bool:
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True

def _tree_map(fn: Callable[[Any], Any], x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _tree_map(fn, v) for k, v in x.items()}
    elif _is_sequence(x) and not hasattr(x, "__array__") and not isinstance(x, (str, bytes)):
        typ = type(x)
        return typ(_tree_map(fn, xi) for xi in x)
    else:
        return fn(x)

def _tree_zip(x, y):
    if isinstance(x, dict):
        assert isinstance(y, dict) and set(x.keys()) == set(y.keys())
        return {k: _tree_zip(x[k], y[k]) for k in x.keys()}
    elif _is_sequence(x) and not hasattr(x, "__array__"):
        assert _is_sequence(y) and not hasattr(y, "__array__")
        assert len(x) == len(y)
        return type(x)( _tree_zip(a, b) for a, b in zip(x, y) )
    else:
        return (x, y)

# -----------------------------------------------------------------------------
# Assertion helpers (NumPy semantics, Torch-aware)
# -----------------------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _assert_numpy_allclose(
    a: np.ndarray,
    b: np.ndarray,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    err_msg: str = ""
):
    kw = {}
    if atol is not None: kw["atol"] = atol
    if rtol is not None: kw["rtol"] = rtol
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)

# -----------------------------------------------------------------------------
# TestCase
# -----------------------------------------------------------------------------

class NeuralTangentsTestCase(unittest.TestCase):

    def assertDtypesMatch(self, x, y, *, canonicalize_dtypes: bool = True):
        dx, dy = _dtype(x), _dtype(y)
        # PyTorch does not have a global x64 toggle like JAX; we keep canonicalization trivial.
        if canonicalize_dtypes:
            self.assertEqual(np.dtype(dx), np.dtype(dy))
        else:
            self.assertEqual(dx, dy)

    def assertArraysAllClose(
        self,
        x,
        y,
        *,
        check_dtypes: bool = True,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        err_msg: str = ""
    ):
        ax, ay = _to_numpy(x), _to_numpy(y)
        self.assertEqual(ax.shape, ay.shape)
        atol_eff = max(_tolerance(_dtype(ax), atol), _tolerance(_dtype(ay), atol))
        rtol_eff = max(_tolerance(_dtype(ax), rtol), _tolerance(_dtype(ay), rtol))
        _assert_numpy_allclose(ax, ay, atol=atol_eff, rtol=rtol_eff, err_msg=err_msg)
        if check_dtypes:
            self.assertDtypesMatch(x, y)

    def _assertAllClose(
        self,
        x,
        y,
        *,
        check_dtypes: bool = True,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        canonicalize_dtypes: bool = True,
        err_msg: str = ""
    ):
        if isinstance(x, dict):
            self.assertIsInstance(y, dict)
            self.assertEqual(set(x.keys()), set(y.keys()))
            for k in x:
                self._assertAllClose(x[k], y[k],
                                     check_dtypes=check_dtypes,
                                     atol=atol, rtol=rtol,
                                     canonicalize_dtypes=canonicalize_dtypes,
                                     err_msg=err_msg)
        elif _is_sequence(x) and not hasattr(x, "__array__"):
            self.assertTrue(_is_sequence(y) and not hasattr(y, "__array__"))
            self.assertEqual(len(x), len(y))
            for xe, ye in zip(x, y):
                self._assertAllClose(xe, ye,
                                     check_dtypes=check_dtypes,
                                     atol=atol, rtol=rtol,
                                     canonicalize_dtypes=canonicalize_dtypes,
                                     err_msg=err_msg)
        elif hasattr(x, "__array__") or np.isscalar(x) or isinstance(x, torch.Tensor):
            self.assertTrue(hasattr(y, "__array__") or np.isscalar(y) or isinstance(y, torch.Tensor))
            if check_dtypes:
                self.assertDtypesMatch(x, y, canonicalize_dtypes=canonicalize_dtypes)
            self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol, err_msg=err_msg)
        elif x == y:
            return
        else:
            raise TypeError((type(x), type(y)))

    def assertAllClose(
        self,
        x,
        y,
        *,
        check_dtypes: bool = True,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        canonicalize_dtypes: bool = True,
        check_finite: bool = True,
        err_msg: str = ""
    ):
        if check_finite:
            def _is_finite(z):
                if isinstance(z, torch.Tensor):
                    self.assertTrue(torch.isfinite(z).all())
                elif hasattr(z, "__array__"):
                    self.assertTrue(np.isfinite(np.asarray(z)).all())
                elif np.isscalar(z):
                    self.assertTrue(np.isfinite(z))
                return z
            _tree_map(_is_finite, x)
            _tree_map(_is_finite, y)

        def _close(a, b):
            self._assertAllClose(
                a, b,
                check_dtypes=check_dtypes,
                atol=atol,
                rtol=rtol,
                canonicalize_dtypes=canonicalize_dtypes,
                err_msg=err_msg,
            )

        # dataclass support
        if dataclasses.is_dataclass(x):
            self.assertIs(type(y), type(x))
            for field in dataclasses.fields(x):
                key = field.name
                x_value, y_value = getattr(x, key), getattr(y, key)
                is_pytree_node = field.metadata.get("pytree_node", True)
                if is_pytree_node:
                    _close(x_value, y_value)
                else:
                    self.assertEqual(x_value, y_value, key)
        else:
            _close(x, y)

# -----------------------------------------------------------------------------
# Vectorization & "pmap" stubs
# -----------------------------------------------------------------------------

def _has_torch_vmap() -> bool:
    return hasattr(torch, "vmap")

def _jit_vmap(f: Callable) -> Callable:
    """
    In JAX this is jit(vmap(f)).
    In PyTorch we provide:
      - If torch.vmap exists, return torch.vmap(f).
      - Otherwise, return a wrapper that maps f over the leading batch dim.
    """
    if _has_torch_vmap():
        return torch.vmap(f)

    def _wrapper(*args, **kwargs):
        # Assume first tensor argument has batch dim on dim 0
        # Map f over leading dimension by simple Python loop + stack
        # Works for small batch sizes in tests.
        # Find batch size from first tensor-like arg
        bs = None
        flat_args = list(args)
        for a in flat_args:
            if isinstance(a, torch.Tensor):
                bs = a.shape[0]
                break
        if bs is None:
            # No tensor inputs; just call once
            return f(*args, **kwargs)
        outs = []
        for i in range(bs):
            sliced = []
            for a in flat_args:
                if isinstance(a, torch.Tensor) and a.shape[:1] == (bs,):
                    sliced.append(a[i])
                else:
                    sliced.append(a)
            outs.append(f(*sliced, **kwargs))
        # Stack outputs if tensor; else return list
        if isinstance(outs[0], torch.Tensor):
            return torch.stack(outs, dim=0)
        return outs
    return _wrapper

def stub_out_pmap(batch: ModuleType, count: int):
    """
    Simulate JAX pmap with vmap on CPU/CUDA.
    """
    if count > 0:
        class torch_device_stub:
            def device_count(self) -> int:
                return count
        platform = device_under_test()
        if platform in ("cpu", "cuda", "mps"):
            batch.pmap = _jit_vmap
            batch.xla_bridge = torch_device_stub()

# -----------------------------------------------------------------------------
# Logging & matrix closeness
# -----------------------------------------------------------------------------

def _log(relative_error: float, absolute_error: float, expected, actual, did_pass: bool):
    msg = "PASSED" if did_pass else "FAILED"
    logging.info(
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
        f"\n{msg} with {relative_error} relative error \n"
        f"and {absolute_error} absolute error: \n"
        "---------------------------------------------\n"
        "EXPECTED: \n"
        f"{expected}\n"
        "---------------------------------------------\n"
        "ACTUAL: \n"
        f"{actual}\n"
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    )

def assert_close_matrices(self: unittest.TestCase, expected, actual, rtol: float, atol: float = 0.1):
    def _to_tensor(z) -> torch.Tensor:
        if isinstance(z, torch.Tensor):
            return z
        return torch.as_tensor(z)

    def _assert_close(e, a):
        e_t, a_t = _to_tensor(e).detach(), _to_tensor(a).detach()
        self.assertEqual(e_t.shape, a_t.shape)
        diff = a_t - e_t
        denom = torch.linalg.norm(e_t)
        denom = denom if denom > 1e-12 else torch.tensor(1e-12, device=e_t.device, dtype=e_t.dtype)
        relative_error = (torch.linalg.norm(diff) / denom).item()
        absolute_error = torch.mean(torch.abs(diff)).item()

        if (np.isnan(relative_error) or
            relative_error > rtol or
            absolute_error > atol):
            _log(relative_error, absolute_error, e_t, a_t, False)
            self.fail(
                self.failureException(
                    "Relative ERROR: ",
                    float(relative_error),
                    "EXPECTED:" + " " * 50,
                    e_t,
                    "ACTUAL:" + " " * 50,
                    a_t,
                    " " * 50,
                    "Absolute ERROR: ",
                    float(absolute_error),
                )
            )
        else:
            _log(relative_error, absolute_error, e_t, a_t, True)

    xz = _tree_zip(expected, actual)
    def _apply(pair):
        if isinstance(pair, tuple) and len(pair) == 2:
            _assert_close(pair[0], pair[1])
        else:
            raise TypeError("Tree structure mismatch.")
    _tree_map(_apply, xz)

# -----------------------------------------------------------------------------
# Skipping utility
# -----------------------------------------------------------------------------

def skip_test(
    self: unittest.TestCase,
    msg: str = "Skipping large tests for speed.",
    platforms: tuple[str, ...] = ("cpu",)
):
    if device_under_test() in platforms:
        raise unittest.SkipTest(msg)

# -----------------------------------------------------------------------------
# Masking utility
# -----------------------------------------------------------------------------

def mask(
    x: torch.Tensor,
    mask_constant: Optional[float],
    mask_axis: Sequence[int],
    key: Optional[torch.Generator],
    p: float
) -> torch.Tensor:
    """
    Replaces entries along mask_axis with mask_constant with prob p and sorts dim=1.
    `key` is an optional torch.Generator for reproducibility.
    """
    if mask_constant is not None:
        mask_shape = [1 if i in mask_axis else s for i, s in enumerate(x.shape)]
        if key is None:
            mask_mat = torch.bernoulli(torch.full(mask_shape, p, device=x.device, dtype=x.dtype)).to(torch.bool)
        else:
            probs = torch.full(mask_shape, p, device=x.device, dtype=x.dtype)
            mask_mat = torch.bernoulli(probs, generator=key).to(torch.bool)
        x = torch.where(mask_mat, torch.as_tensor(mask_constant, dtype=x.dtype, device=x.device), x)
        # Following the original: sort along dim=1
        if x.dim() >= 2:
            x, _ = torch.sort(x, dim=1)
    return x
