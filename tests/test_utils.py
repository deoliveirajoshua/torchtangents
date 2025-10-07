# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# This file is a native PyTorch reimplementation of JAX-based test utilities.

from __future__ import annotations

import dataclasses
import inspect
import itertools
import logging
import os
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import unittest

# Optional: only needed for class-level parameterization
try:
    from parameterized import parameterized_class as _parameterized_class
except Exception:  # pragma: no cover
    _parameterized_class = None

# =============================================================================
# Config / device
# =============================================================================

NT_TEST_DUT = os.getenv("NT_TEST_DUT", "").strip()
NT_NUM_GENERATED_CASES = int(os.getenv("NT_NUM_GENERATED_CASES", "4"))

def device_under_test() -> str:
    """
    Returns a concrete device string:
      - "cuda:0" if CUDA is available
      - "cpu" otherwise
    Honors NT_TEST_DUT if set (e.g., "cpu", "cuda", "cuda:1").
    """
    if NT_TEST_DUT:
        return "cuda:0" if NT_TEST_DUT == "cuda" else NT_TEST_DUT
    if torch.cuda.is_available():
        # pick an explicit index
        try:
            idx = torch.cuda.current_device()
        except Exception:
            idx = 0
        return f"cuda:{idx}"
    return "cpu"

# =============================================================================
# NumPy dtype probe (bf16)
# =============================================================================

try:
    _NP_BFLOAT16 = np.dtype("bfloat16")  # may raise on older NumPy
    _HAS_NP_BFLOAT16 = True
except TypeError:
    _NP_BFLOAT16 = None
    _HAS_NP_BFLOAT16 = False

# =============================================================================
# Type / tolerance helpers
# =============================================================================

_python_scalar_dtypes = {
    bool: np.dtype("bool"),
    int: np.dtype("int64"),
    float: np.dtype("float64"),
    complex: np.dtype("complex128"),
}

def _dtype(x) -> np.dtype:
    if hasattr(x, "dtype"):
        if isinstance(x, torch.Tensor):
            td = x.dtype
            if td is torch.float32:
                return np.dtype(np.float32)
            if td is torch.float64:
                return np.dtype(np.float64)
            if td is torch.float16:
                return np.dtype(np.float16)
            if td is torch.bfloat16:
                return _NP_BFLOAT16 if _HAS_NP_BFLOAT16 else np.dtype(np.float32)
            if td is torch.int64:
                return np.dtype(np.int64)
            if td is torch.int32:
                return np.dtype(np.int32)
            if td is torch.bool:
                return np.dtype(np.bool_)
            # Fallback: coerce to f32 on CPU then infer
            return np.dtype(str(x.detach().cpu().to(torch.float32).numpy().dtype))
        # NumPy array
        return np.dtype(x.dtype)
    elif type(x) in _python_scalar_dtypes:
        return np.dtype(_python_scalar_dtypes[type(x)])
    else:
        return np.asarray(x).dtype

_DEFAULT_TOLERANCE: Dict[np.dtype, float] = {
    np.dtype(np.bool_): 0.0,
    np.dtype(np.int32): 0.0,
    np.dtype(np.int64): 0.0,
    np.dtype(np.float16): 5e-3,
    np.dtype(np.float32): 5e-3,
    np.dtype(np.float64): 1e-5,
}
if _HAS_NP_BFLOAT16:
    _DEFAULT_TOLERANCE[_NP_BFLOAT16] = 5e-3

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.dtype is torch.bfloat16 and not _HAS_NP_BFLOAT16:
            t = t.to(torch.float32)
        return t.numpy()
    return np.asarray(x)

def _default_tolerance() -> Dict[np.dtype, float]:
    dut = device_under_test()
    if dut != "tpu":
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

# =============================================================================
# Case subsetting
# =============================================================================

_CACHED_INDICES: Dict[int, Sequence[int]] = {}

def _cases_from_list(xs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

# =============================================================================
# Parameterization (robust, subTest-based for methods)
# =============================================================================

def _build_cases(kwargs_seqs: Iterable[Iterable[dict]], **testgrid) -> List[Dict[str, Any]]:
    # validate named grid
    for name, values in testgrid.items():
        if not isinstance(values, (list, tuple)):
            raise AssertionError(f"Values of {name} must be list/tuple, found {type(values)}")

    # validate positional sequences of kwargs dicts
    prior_arg_names = set()
    for kwargs_seq in kwargs_seqs:
        if not (isinstance(kwargs_seq, (list, tuple)) and all(isinstance(kwargs, dict) for kwargs in kwargs_seq)):
            raise AssertionError("Positional parameters must be a sequence of kwargs dicts")
        if kwargs_seq:
            arg_names = set(kwargs_seq[0])
            if not all(set(kwargs) == arg_names for kwargs in kwargs_seq):
                raise AssertionError("All kwargs dicts in a positional block must share keys")
            if arg_names & prior_arg_names:
                dup = sorted(arg_names & prior_arg_names)
                raise AssertionError(f"Duplicate arg names across blocks: {dup}")
            prior_arg_names |= arg_names

    if prior_arg_names & set(testgrid):
        dup = sorted(prior_arg_names & set(testgrid))
        raise AssertionError(f"Duplicate arg names between positional and named: {dup}")

    # build cartesian list of dict testcases
    testgrid_seq = (tuple({k: v} for v in vs) for k, vs in testgrid.items())
    testgrid_seq = tuple(kwargs_seqs) + tuple(testgrid_seq)
    all_cases = [
        dict(itertools.chain.from_iterable(case.items() for case in cases))
        for cases in itertools.product(*testgrid_seq)
    ]
    return _cases_from_list(all_cases)

def parameters(testcases: List[Dict[str, Any]]):
    """
    METHOD/FUNCTION decorator: run once per dict testcase using subTest(**case).
    This binds by keyword reliably and avoids parameterized.expand pitfalls.
    """
    subset = _cases_from_list(testcases)

    def _decorator(func):
        arg_names = [n for n in inspect.signature(func).parameters if n != "self"]

        def _wrapped(self, *_, **__):
            for i, case in enumerate(subset):
                missing = [n for n in arg_names if n not in case]
                if missing:
                    raise AssertionError(f"Missing keys {missing} in testcase #{i}: {case}")
                with self.subTest(**case):
                    func(self, **case)

        _wrapped.__name__ = func.__name__
        _wrapped.__doc__ = func.__doc__
        return _wrapped

    return _decorator

def product(*kwargs_seqs, **testgrid):
    """
    Universal decorator:
      - If decorating a CLASS: use parameterized_class (one class per dict).
      - If decorating a METHOD/FUNCTION: run subTests with kwargs per dict.
    """
    subset = _build_cases(kwargs_seqs, **testgrid)

    def _decorator(obj):
        if inspect.isclass(obj):
            return _parameterized_class(subset)(obj)
        else:
            return parameters(subset)(obj)
    return _decorator

    return _decorator


import inspect
from typing import Any, Dict, List

def _parameterized_class(cases: List[Dict[str, Any]]):
    """Return a class decorator that clones the test class per `case` dict."""
    def _decorator(cls):
        module = cls.__module__
        base_name = cls.__name__
        new_classes = []

        for i, case in enumerate(cases):
            name = f"{base_name}_{i}"
            # Build a new dict of attributes/methods for the subclass
            attrs = dict(cls.__dict__)  # shallow copy
            # Remove __dict__/__weakref__ descriptors added by type
            attrs.pop('__dict__', None)
            attrs.pop('__weakref__', None)

            # Attach case values as class attributes so tests can use self.<key>
            for k, v in case.items():
                attrs[k] = v

            # Wrap test methods to pass kwargs if they declare them
            def _wrap_test(meth, case_dict):
                if not callable(meth):
                    return meth
                if not meth.__name__.startswith("test"):
                    return meth

                sig = inspect.signature(meth)

                def wrapped(self, *args, **kwargs):
                    # If method expects keyword params, filter from case
                    if any(p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                      inspect.Parameter.KEYWORD_ONLY)
                           for p in sig.parameters.values() if p.name != "self"):
                        accepted = {k: v for k, v in case_dict.items() if k in sig.parameters}
                        return meth(self, **accepted)
                    # Otherwise, call without injecting kwargs
                    return meth(self, *args, **kwargs)

                wrapped.__name__ = meth.__name__
                wrapped.__doc__ = meth.__doc__
                wrapped.__qualname__ = f"{name}.{meth.__name__}"
                return wrapped

            for attr_name, attr_val in list(attrs.items()):
                if attr_name.startswith("test") and callable(attr_val):
                    attrs[attr_name] = _wrap_test(attr_val, case)

            # Create the subclass
            new_cls = type(name, (cls,), attrs)
            new_cls.__module__ = module
            new_classes.append(new_cls)

            # Inject into the defining module so unittest/pytest discover them
            m = __import__(module, fromlist=[module.split(".")[-1]])
            setattr(m, name, new_cls)

        # Return a benign placeholder (the original class won’t be collected)
        return None
    return _decorator

# =============================================================================
# Simple pytree helpers
# =============================================================================

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
        return type(x)(_tree_zip(a, b) for a, b in zip(x, y))
    else:
        return (x, y)

# =============================================================================
# Assertions
# =============================================================================

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

class NeuralTangentsTestCase(unittest.TestCase):

    def assertDtypesMatch(self, x, y, *, canonicalize_dtypes: bool = True):
        dx, dy = _dtype(x), _dtype(y)
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
            err_msg: str = "",
    ):
        """
        Dataclass-aware deep allclose:
          - Recurses through dict / list / tuple
          - For dataclasses, only recurses into fields with metadata pytree_node=True (default)
          - Leaves (torch.Tensor | numpy array | scalar) are compared with per-dtype tolerances
          - Optional finiteness pre-check over leaves only
        """

        # ---------- helpers ----------
        def _is_sequence(v) -> bool:
            # Only treat plain list/tuple as sequences; avoid iterating custom classes/dataclasses.
            return isinstance(v, (list, tuple))

        def _finite_leaf(z):
            if isinstance(z, torch.Tensor):
                self.assertTrue(torch.isfinite(z).all(), "Non-finite values in tensor leaf.")
            elif hasattr(z, "__array__"):
                self.assertTrue(np.isfinite(np.asarray(z)).all(), "Non-finite values in array leaf.")
            elif np.isscalar(z):
                self.assertTrue(np.isfinite(z), "Non-finite scalar.")
            return z

        def _finite_traverse(z):
            if dataclasses.is_dataclass(z):
                for field in dataclasses.fields(z):
                    if field.metadata.get("pytree_node", True):
                        _finite_traverse(getattr(z, field.name))
                return
            if isinstance(z, dict):
                for v in z.values():
                    _finite_traverse(v)
                return
            if _is_sequence(z) and not hasattr(z, "__array__"):
                for v in z:
                    _finite_traverse(v)
                return
            _finite_leaf(z)

        def _cmp_leaves(a, b):
            # Compare leaves with dtype-aware tolerances and optional dtype checks.
            ax, ay = _to_numpy(a), _to_numpy(b)
            self.assertEqual(ax.shape, ay.shape, f"Shape mismatch: {ax.shape} vs {ay.shape}. {err_msg}")
            atol_eff = max(_tolerance(_dtype(ax), atol), _tolerance(_dtype(ay), atol))
            rtol_eff = max(_tolerance(_dtype(ax), rtol), _tolerance(_dtype(ay), rtol))
            _assert_numpy_allclose(ax, ay, atol=atol_eff, rtol=rtol_eff, err_msg=err_msg)
            if check_dtypes:
                dx, dy = _dtype(a), _dtype(b)
                if canonicalize_dtypes:
                    self.assertEqual(np.dtype(dx), np.dtype(dy), f"Dtype mismatch: {dx} vs {dy}. {err_msg}")
                else:
                    self.assertEqual(dx, dy, f"Dtype mismatch: {dx} vs {dy}. {err_msg}")

        def _assert_close(a, b):
            # Dataclass (e.g., Kernel): recurse into pytree_node fields
            if dataclasses.is_dataclass(a):
                self.assertIs(type(a), type(b), f"Dataclass types differ: {type(a)} vs {type(b)}. {err_msg}")
                for field in dataclasses.fields(a):
                    name = field.name
                    a_val, b_val = getattr(a, name), getattr(b, name)
                    if field.metadata.get("pytree_node", True):
                        _assert_close(a_val, b_val)
                    else:
                        self.assertEqual(a_val, b_val, f"Non-pytree field `{name}` mismatch. {err_msg}")
                return

            # Dict
            if isinstance(a, dict):
                self.assertIsInstance(b, dict, f"Type mismatch: dict vs {type(b)}. {err_msg}")
                self.assertEqual(set(a.keys()), set(b.keys()), f"Dict keys differ. {err_msg}")
                for k in a:
                    _assert_close(a[k], b[k])
                return

            # Sequence (list/tuple)
            if _is_sequence(a) and not hasattr(a, "__array__"):
                self.assertTrue(_is_sequence(b) and not hasattr(b, "__array__"),
                                f"Type mismatch: sequence vs {type(b)}. {err_msg}")
                self.assertEqual(len(a), len(b), f"Sequence lengths differ. {err_msg}")
                for ae, be in zip(a, b):
                    _assert_close(ae, be)
                return

            # Leaves: tensors/ndarrays/scalars
            if isinstance(a, torch.Tensor) or hasattr(a, "__array__") or np.isscalar(a):
                self.assertTrue(isinstance(b, torch.Tensor) or hasattr(b, "__array__") or np.isscalar(b),
                                f"Leaf type mismatch: {type(a)} vs {type(b)}. {err_msg}")
                _cmp_leaves(a, b)
                return

            # Fallback to equality for other comparable types.
            self.assertEqual(a, b, f"Values differ: {a} vs {b}. {err_msg}")

        # ---------- run ----------
        if check_finite:
            _finite_traverse(x)
            _finite_traverse(y)

        _assert_close(x, y)


# =============================================================================
# Vectorization & pseudo-pmap
# =============================================================================

def _has_torch_vmap() -> bool:
    return hasattr(torch, "vmap")

def _jit_vmap(f: Callable) -> Callable:
    if _has_torch_vmap():
        return torch.vmap(f)

    def _wrapper(*args, **kwargs):
        bs = None
        for a in args:
            if isinstance(a, torch.Tensor):
                bs = a.shape[0]
                break
        if bs is None:
            return f(*args, **kwargs)
        outs = []
        for i in range(bs):
            sliced = []
            for a in args:
                if isinstance(a, torch.Tensor) and a.shape[:1] == (bs,):
                    sliced.append(a[i])
                else:
                    sliced.append(a)
            outs.append(f(*sliced, **kwargs))
        return torch.stack(outs, dim=0) if isinstance(outs[0], torch.Tensor) else outs
    return _wrapper

def stub_out_pmap(batch: ModuleType, count: int):
    if count > 0:
        class torch_device_stub:
            def device_count(self) -> int:
                return count
        platform = device_under_test()
        if platform in ("cpu", "cuda", "mps"):
            batch.pmap = _jit_vmap
            batch.xla_bridge = torch_device_stub()

# =============================================================================
# Logging & matrix closeness
# =============================================================================

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
        return z if isinstance(z, torch.Tensor) else torch.as_tensor(z)

    def _assert_close(e, a):
        e_t, a_t = _to_tensor(e).detach(), _to_tensor(a).detach()
        self.assertEqual(e_t.shape, a_t.shape)
        diff = a_t - e_t
        denom = torch.linalg.norm(e_t)
        denom = denom if denom > 1e-12 else torch.tensor(1e-12, device=e_t.device, dtype=e_t.dtype)
        relative_error = (torch.linalg.norm(diff) / denom).item()
        absolute_error = torch.mean(torch.abs(diff)).item()
        if (np.isnan(relative_error) or relative_error > rtol or absolute_error > atol):
            _log(relative_error, absolute_error, e_t, a_t, False)
            self.fail(self.failureException(
                "Relative ERROR: ", float(relative_error),
                "EXPECTED:" + " " * 50, e_t,
                "ACTUAL:" + " " * 50, a_t,
                " " * 50, "Absolute ERROR: ", float(absolute_error),
            ))
        else:
            _log(relative_error, absolute_error, e_t, a_t, True)

    xz = _tree_zip(expected, actual)
    def _apply(pair):
        if isinstance(pair, tuple) and len(pair) == 2:
            _assert_close(pair[0], pair[1])
        else:
            raise TypeError("Tree structure mismatch.")
    _tree_map(_apply, xz)

# =============================================================================
# Skips
# =============================================================================

def skip_test(
    self: unittest.TestCase,
    msg: str = "Skipping large tests for speed.",
    platforms: Tuple[str, ...] = ("cpu",)
):
    if device_under_test() in platforms:
        raise unittest.SkipTest(msg)

# =============================================================================
# RNG helper for mask (device-aware generator)
# =============================================================================

def _ensure_generator_on(gen: Optional[torch.Generator | int], device: torch.device | str) -> Optional[torch.Generator]:
    if gen is None:
        return None
    dev = torch.device(device)
    if isinstance(gen, int):
        g = torch.Generator(device=dev)
        g.manual_seed(int(gen))
        return g
    if isinstance(gen, torch.Generator):
        gdev = getattr(gen, "device", torch.device("cpu"))
        if torch.device(gdev).type == dev.type:
            return gen
        seed = gen.initial_seed() if hasattr(gen, "initial_seed") else int(torch.seed())
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        return g
    raise TypeError(f"Unsupported generator type: {type(gen)}")

# =============================================================================
# Masking utility
# =============================================================================

def mask(
    x: torch.Tensor,
    mask_constant: Optional[float],
    mask_axis: Sequence[int],
    gen: Optional[torch.Generator] = None,  # NOTE: named 'gen' to match call sites in tests
    p: float = 0.5
) -> torch.Tensor:
    """
    Replaces entries along mask_axis with mask_constant with prob p and sorts dim=1.
    `gen` is an optional torch.Generator for reproducibility (device-aware).
    """
    if mask_constant is not None:
        mask_shape = [1 if i in mask_axis else s for i, s in enumerate(x.shape)]
        probs = torch.full(mask_shape, p, device=x.device, dtype=x.dtype)
        g = _ensure_generator_on(gen, x.device)
        mask_mat = torch.bernoulli(probs, generator=g).to(torch.bool)
        x = torch.where(mask_mat, torch.as_tensor(mask_constant, dtype=x.dtype, device=x.device), x)
        if x.dim() >= 2:
            x, _ = torch.sort(x, dim=1)
    return x
