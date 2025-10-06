# empirical_torch.py
# Copyright 2019 Google LLC
# Port to PyTorch (c) 2025
#
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import enum
import functools
import operator
from typing import Callable, Iterable, KeysView, Optional, Tuple, Union, Any, Dict

import numpy as np
import torch
from torch import Tensor
try:
    # PyTorch 2.x
    from torch.func import vmap, jvp, vjp, jacrev
except Exception:  # pragma: no cover
    # functorch fallback
    try:
        from functorch import vmap, jvp, vjp, jacrev
    except Exception:
        raise ImportError("Requires PyTorch 2.x or functorch package.")

from torch.utils._pytree import (
    tree_map, tree_flatten, tree_unflatten, tree_all, tree_structure
)

# ---- assume these exist (ported non-jax helpers) ----
from .utils import utils
from .utils.typing import ApplyFn, Axes, EmpiricalGetKernelFn, EmpiricalKernelFn, PyTree, VMapAxes, VMapAxisTriple


# -----------------------
# LINEARIZATION / TAYLOR
# -----------------------

def linearize(f: ApplyFn, params: PyTree) -> ApplyFn:
    """
    First-order Taylor approximation of f at `params`.
    Works with PyTorch tensors and PyTrees of tensors.
    """

    def f_lin(p, *args, **kwargs):
        # dparams := p - params (PyTree)
        dparams = tree_map(lambda a, b: a - b, p, params)

        def f_params_fn(pr):
            return f(pr, *args, **kwargs)

        # jvp expects primals and tangents to have same structure
        y0 = f_params_fn(params)
        _, proj = jvp(f_params_fn, (params,), (dparams,))
        return tree_map(operator.add, y0, proj)

    return f_lin


def taylor_expand(f: ApplyFn, params: PyTree, degree: int) -> ApplyFn:
    """
    Degree-k Taylor approximation via recursive JVPs.
    """

    def taylorize_r(g, p0, dp, deg, cur):
        if cur == deg:
            return g(p0)

        def g_jvp(p_):
            _, val_jvp = jvp(g, (p_,), (dp,))
            return val_jvp

        df = taylorize_r(g_jvp, p0, dp, deg, cur + 1)
        base = g(p0)
        # add df / (cur+1)
        return tree_map(lambda a, b: a + b / (cur + 1), base, df)

    def f_tayl(p, *args, **kwargs):
        dp = tree_map(lambda a, b: a - b, p, params)

        def g(pr):
            return f(pr, *args, **kwargs)

        return taylorize_r(g, params, dp, degree, 0)

    return f_tayl


# -----------
#   NNGP
# -----------

def empirical_nngp_fn(
    f: ApplyFn,
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = ()
) -> EmpiricalKernelFn:
    """
    Single-sample empirical NNGP: f(x1) @ f(x2)^T with optional tracing/diagonals.
    """

    def nngp_fn(
        x1: PyTree,
        x2: Optional[PyTree],
        params: PyTree,
        **apply_fn_kwargs
    ) -> PyTree:
        def output(x, **kwargs):
            return f(params, x, **kwargs)

        kwargs1, kwargs2 = utils.split_kwargs(apply_fn_kwargs, x1, x2)

        out1 = output(x1, **kwargs1)
        out2 = out1 if utils.all_none(x2) else output(x2, **kwargs2)

        def contract(a: Tensor, b: Tensor) -> Tensor:
            dot = _dot_general(a, b, trace_axes, diagonal_axes)
            return dot / utils.size_at(a, trace_axes)

        return tree_map(contract, out1, out2)

    return nngp_fn


# -----------
#   NTK
# -----------

class NtkImplementation(enum.IntEnum):
    AUTO = 0
    JACOBIAN_CONTRACTION = 1
    NTK_VECTOR_PRODUCTS = 2
    STRUCTURED_DERIVATIVES = 3


DEFAULT_NTK_IMPLEMENTATION = NtkImplementation.JACOBIAN_CONTRACTION
_DEFAULT_TESTING_NTK_IMPLEMENTATION = NtkImplementation.STRUCTURED_DERIVATIVES

# The structured-derivatives and auto/Jaxpr paths are not available in PyTorch.
_DEFAULT_NTK_J_RULES: bool = True
_DEFAULT_NTK_S_RULES: bool = True
_DEFAULT_NTK_FWD: Optional[bool] = None


def _jacobian_contraction_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes,
    diagonal_axes: Axes,
    vmap_axes: VMapAxes,
    **kwargs
) -> EmpiricalKernelFn:
    """
    NTK via Jacobian contraction: sum_p J1[...,p] * J2[...,p]
    Implemented with torch.func.jacrev; supports optional vmap over a single batch axis.
    Limitations vs JAX version: `vmap_axes` supports `None` or a single int (e.g. 0).
    """

    def sum_and_contract(fx: Tensor, j1: PyTree, j2: PyTree) -> Tensor:
        # j1/j2 are PyTrees with leaves shaped like fx.shape + param_leaf.shape
        # Contract param dims; trace/diag over requested output axes.
        def leaf_contract(a: Tensor, b: Tensor) -> Tensor:
            # Move param dims to the end, flatten them, inner product
            fx_ndim = fx.ndim
            a_flat = a.reshape(a.shape[:fx_ndim] + (-1,))
            b_flat = b.reshape(b.shape[:fx_ndim] + (-1,))
            # einsum over last dim
            # a_flat: [N1, C, P], b_flat: [N2, C, P]
            #print(f"a_flat shape: {a_flat.shape}, b_flat shape: {b_flat.shape}")
            core = torch.einsum("n c p, m d p -> n c m d", a_flat, b_flat)
            #print(f"core shape: {core.shape}")
            # core: [N1, N2, C, C]

            return core

        contracted = tree_reduce_add(tree_map(leaf_contract, j1, j2))
        # Now contracted has shape like fx.shape (output shape). Apply trace/diag policy:
        return _trace_and_diagonal(contracted, trace_axes, diagonal_axes) / utils.size_at(fx, trace_axes)

    def tree_reduce_add(pytree: PyTree) -> Tensor:
        leaves, _ = tree_flatten(pytree)
        out = None
        for l in leaves:
            out = l if out is None else out + l
        assert out is not None
        return out

    def ntk_fn(
        x1: PyTree,
        x2: Optional[PyTree],
        params: PyTree,
        **apply_fn_kwargs
    ) -> Tensor:
        args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
            f, apply_fn_kwargs, params, vmap_axes, x1, x2
        )

        # Build a function g(p) that returns squeezed f(p, x, **kw) for the
        # appropriate side, matching original API expectations
        def j_fn(x, kw_tuple):
            kw = {k: v for k, v in zip(keys, kw_tuple)}
            def fx_of_p(p):
                return _get_f_params(f, x, x_axis, fx_axis, kw_axes, **kw)(p)
            # jacobian wrt params (argnums=0 equivalently). Returns PyTree
            return jacrev(fx_of_p)(params)

        # Optionally vmapped over a single batch axis
        if _is_int_axis(x_axis) or _is_int_axis(kw_axes):
            in_axes = [x_axis] + [kw_axes.get(k, None) for k in keys]
            j_fn_vm = vmap(j_fn, in_dims=(in_axes[0], tuple(in_axes[1:])), out_dims=fx_axis)
        else:
            j_fn_vm = j_fn

        j1 = j_fn_vm(x1, args1)
        j2 = j1 if utils.all_none(x2) else j_fn_vm(x2, args2)
        ntk = tree_map(sum_and_contract, fx1, j1, j2)
        return ntk

    return ntk_fn


def _ntk_vector_products_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes,
    diagonal_axes: Axes,
    vmap_axes: VMapAxes,
    **kwargs
) -> EmpiricalKernelFn:
    """
    NTK via NTK-vector products (no explicit Jacobian materialization).
    """

    def ntk_fn(
        x1: PyTree,
        x2: Optional[PyTree],
        params: PyTree,
        **apply_fn_kwargs
    ) -> Tensor:
        args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
            f, apply_fn_kwargs, params, vmap_axes, x1, x2
        )

        f1, f2 = _get_f1_f2(f, keys, x_axis, fx_axis, kw_axes, args1 + args2, x1, x2)

        def get_ntk(lhs_x, rhs_x, *all_args):
            # Split the kw-args tuple back
            half = len(all_args) // 2
            args_l, args_r = all_args[:half], all_args[half:]
            kw_l = {k: v for k, v in zip(keys, args_l)}
            kw_r = {k: v for k, v in zip(keys, args_r)}

            g1 = _get_f_params(f, lhs_x, x_axis, fx_axis, kw_axes, **kw_l)
            g2 = _get_f_params(f, rhs_x, x_axis, fx_axis, kw_axes, **kw_r)

            def delta_vjp_jvp(delta):
                # vjp wrt params at rhs
                _, vjp_fun = vjp(lambda p: g2(p), (params,))
                v = vjp_fun(delta)[0]  # tangent in params PyTree
                # jvp wrt params at lhs
                return jvp(lambda p: g1(p), (params,), (v,))[1]

            # Build standard basis for fx2 and push through linear_transpose analogue
            fx1_sh = _eval_f_shape(g1, params)
            fx2_sh = _eval_f_shape(g2, params)
            eye = _std_basis_from_shape(fx2_sh)  # PyTree basis matching fx2
            # Map linear transpose over basis vectors
            ntk_cols = tree_map(lambda e: _linear_transpose_like(delta_vjp_jvp, fx2_sh, e), eye)
            # Gather diagonal blocks matching fx1 structure
            ntk = _diagonal(ntk_cols, fx1_sh)
            return ntk

        if _is_int_axis(x_axis) or _is_int_axis(kw_axes):
            x2_eff = x1 if utils.all_none(x2) else x2
            kw_in_axes = [kw_axes.get(k, None) for k in keys]
            in_axes1 = [x_axis, None] + kw_in_axes + [None] * len(kw_in_axes)
            in_axes2 = [None, x_axis] + [None] * len(kw_in_axes) + kw_in_axes
            get_ntk_vm = vmap(vmap(get_ntk, in_axes1, fx_axis), in_axes2, _add_axes(fx_axis, _ndim_like(fx1)))
            ntk = get_ntk_vm(x1, x2_eff, *args1, *args2)
        else:
            ntk = get_ntk(x1, x2 if not utils.all_none(x2) else x1, *args1, *args2)

        ntk = tree_map(lambda x: _trace_and_diagonal(x, trace_axes, diagonal_axes), ntk)
        return ntk

    return ntk_fn


_implementation_to_ntk_fn = {
    NtkImplementation.JACOBIAN_CONTRACTION: _jacobian_contraction_ntk_fn,
    NtkImplementation.NTK_VECTOR_PRODUCTS: _ntk_vector_products_ntk_fn,
    # AUTO/STRUCTURED_DERIVATIVES are not supported in this PyTorch port.
}


def empirical_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = (),
    vmap_axes: VMapAxes = None,
    implementation: Union[NtkImplementation, int] = DEFAULT_NTK_IMPLEMENTATION,
    _j_rules: bool = _DEFAULT_NTK_J_RULES,
    _s_rules: bool = _DEFAULT_NTK_S_RULES,
    _fwd: Optional[bool] = _DEFAULT_NTK_FWD,
) -> EmpiricalKernelFn:
    """
    Returns a function computing a single-sample empirical NTK.
    In this PyTorch port, `implementation` can be 1 (Jacobian contraction) or 2 (NTK-vector products).
    """
    impl = NtkImplementation(int(implementation))
    if impl not in _implementation_to_ntk_fn:
        raise NotImplementedError(
            f"{impl.name} is not supported in the PyTorch port (depends on JAX Jaxprs). "
            "Use JACOBIAN_CONTRACTION (1) or NTK_VECTOR_PRODUCTS (2)."
        )
    return _implementation_to_ntk_fn[impl](
        f=f,
        trace_axes=trace_axes,
        diagonal_axes=diagonal_axes,
        vmap_axes=vmap_axes,
        _j_rules=_j_rules,
        _s_rules=_s_rules,
        _fwd=_fwd
    )


# -----------------------
# JOINT NNGP / NTK KERNEL
# -----------------------

def empirical_kernel_fn(
    f: ApplyFn,
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = (),
    vmap_axes: VMapAxes = None,
    implementation: Union[NtkImplementation, int] = DEFAULT_NTK_IMPLEMENTATION,
    _j_rules: bool = _DEFAULT_NTK_J_RULES,
    _s_rules: bool = _DEFAULT_NTK_S_RULES,
    _fwd: Optional[bool] = _DEFAULT_NTK_FWD,
) -> EmpiricalGetKernelFn:
    """
    Returns a function computing single draws from the NNGP and/or NTK.
    """

    nngp = empirical_nngp_fn(f, trace_axes=trace_axes, diagonal_axes=diagonal_axes)
    ntk = empirical_ntk_fn(
        f,
        trace_axes=trace_axes,
        diagonal_axes=diagonal_axes,
        vmap_axes=vmap_axes,
        implementation=implementation,
        _j_rules=_j_rules,
        _s_rules=_s_rules,
        _fwd=_fwd,
    )

    def kernel_fn(
        x1: PyTree,
        x2: Optional[PyTree],
        get: Union[None, str, Tuple[str, ...]],
        params: PyTree,
        **apply_fn_kwargs
    ) -> PyTree:
        if get is None:
            get = ("nngp", "ntk")
        if isinstance(get, str):
            get = (get,)

        out_dict = {}
        for g in get:
            if g == "nngp":
                out_dict["nngp"] = nngp(x1, x2, params, **apply_fn_kwargs)
            elif g == "ntk":
                out_dict["ntk"] = ntk(x1, x2, params, **apply_fn_kwargs)
            else:
                raise ValueError(f"Unknown kernel type {g}.")

        # Convert dict-of-pytrees into pytree-of-dicts like the JAX version
        return _dict_of_tree_to_tree_of_dict(out_dict, get)

    return kernel_fn


# -------------------------
#   NTK–VECTOR PRODUCT FN
# -------------------------

def empirical_ntk_vp_fn(
    f: ApplyFn,
    x1: PyTree,
    x2: Optional[PyTree],
    params: PyTree,
    **apply_fn_kwargs
) -> Callable[[PyTree], PyTree]:
    """
    Returns function for NTK-vector product (no explicit NTK build).
    """

    args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
        f, apply_fn_kwargs, params, None, x1, x2
    )
    f1, f2 = _get_f1_f2(f, keys, x_axis, fx_axis, kw_axes, args1 + args2, x1, x2)

    def ntk_vp_fn(cotangents: PyTree) -> PyTree:
        # VJP wrt params at x2
        _, vjp_fun = vjp(lambda p: f2(p), (params,))
        v = vjp_fun(cotangents)[0]  # param-space cotangent
        # JVP wrt params at x1
        return jvp(lambda p: f1(p), (params,), (v,))[1]

    return ntk_vp_fn


# -----------------
# INTERNAL HELPERS
# -----------------

def _trace_and_diagonal(
    ntk: torch.Tensor,
    trace_axes: tuple[int, ...],
    diagonal_axes: tuple[int, ...],
) -> torch.Tensor:
    """
    Input layout: [N1, N2, <feature axes left half>, <feature axes right half>]
    (i.e., halves are NOT interleaved). Only trace/diag over the feature block.

    With fx of shape [N, C], this means input is [N1, N2, C, C].
    """
    if ntk.ndim % 2 != 0:
        raise ValueError("Expected an even-dimensional kernel.")

    half = ntk.ndim // 2
    BATCH_NDIMS = 1  # number of example axes per half (here: just N)
    feat_ndim = half - BATCH_NDIMS
    if feat_ndim < 0:
        raise ValueError("Invalid layout: half < BATCH_NDIMS.")
    if feat_ndim == 0:
        # No feature axes remain: nothing to trace/diag.
        return ntk

    # Canonicalize trace/diag indices RELATIVE TO THE FEATURE BLOCK length (feat_ndim),
    # not relative to the whole half. This makes -1 mean "last FEATURE axis".
    def _canon_feat(axes):
        if axes is None:
            return ()
        out = []
        for a in axes:
            a = a if a >= 0 else feat_ndim + a
            if not (0 <= a < feat_ndim):
                raise IndexError(f"feature axis {a} out of range for feat_ndim={feat_ndim}")
            out.append(a)
        return tuple(out)

    trace_feat = _canon_feat(trace_axes)          # e.g. (-1,) -> (0,) when feat_ndim == 1
    diag_feat  = _canon_feat(diagonal_axes)

    # Map feature-relative index a to absolute axis in first half:
    # [N1, <feat...>]  ⇒ absolute = BATCH_NDIMS + a
    trace_abs = tuple(BATCH_NDIMS + a for a in trace_feat)
    diag_abs  = tuple(BATCH_NDIMS + a for a in diag_feat)

    # Normalization factor = product of sizes of traced feature axes (first half).
    contract_size = 1
    for a in trace_abs:
        contract_size *= ntk.shape[a]

    # --- Trace over requested feature pairs ---
    # Pair absolute axis a with its counterpart in second half: (a, half + a)
    for i, a in enumerate(sorted(trace_abs)):
        ntk = ntk.diagonal(dim1=a, dim2=half + a - i).sum(-1)

    # --- Take diagonals over requested feature pairs ---
    for i, d in enumerate(sorted(diag_abs)):
        axis1 = d - i
        axis2 = half + d - (2 * i) - len(trace_abs)
        ntk = torch.diagonal(ntk, dim1=axis1, dim2=axis2)

    # No zipping/interleaving for 2D blocks; guard larger cases too.
    n_diag = len(diag_abs)
    end_axis = ntk.ndim - n_diag
    if end_axis > 2:
        ntk = utils.zip_axes(ntk, 0, end_axis)  # ensure this routine has the no-op + tuple fixes

    # Move extracted diagonals, if any (n_diag==0 in your test, so this is skipped)
    if n_diag > 0:
        res_diagonal_axes = _get_res_batch_dims(trace_abs, diag_abs)
        ntk = ntk.movedim(list(range(ntk.ndim - n_diag, ntk.ndim)), res_diagonal_axes)

    return ntk / contract_size



def _dict_of_tree_to_tree_of_dict(out_dict: Dict[str, PyTree], get: Tuple[str, ...]) -> PyTree:
    # Convert dict of pytrees -> pytree of dicts (parallel outputs case)
    return tree_map(lambda *xs: {g: v for g, v in zip(get, xs)},
                    *[out_dict[g] for g in get])


def _get_f_params(
    f: Callable[..., PyTree],
    x: PyTree,
    x_axis: Optional[int],
    fx_axis: Optional[int],
    kw_axes: Dict[str, Optional[int]],
    **apply_fn_kwargs: Any
) -> Callable[[PyTree], PyTree]:
    def _f(p: PyTree) -> PyTree:
        fx = f(p, x, **apply_fn_kwargs)
        return fx  # `fx_axis` squeeze kept as identity in this port
    return _f


def _get_args(
    f: Callable[..., PyTree],
    apply_fn_kwargs: Dict[str, Any],
    params: PyTree,
    vmap_axes: Optional[VMapAxes],
    x1: PyTree,
    x2: Optional[PyTree]
):
    kwargs1, kwargs2 = utils.split_kwargs(apply_fn_kwargs, x1, x2)

    fx1 = f(params, x1, **kwargs1)
    fx2 = fx1 if utils.all_none(x2) else f(params, x2, **kwargs2)

    x_axis, fx_axis, kw_axes = _canonicalize_axes(vmap_axes, x1, fx1, **kwargs1)

    keys = list(apply_fn_kwargs.keys())
    args1 = tuple(kwargs1[k] for k in keys)
    args2 = tuple(kwargs2[k] for k in keys)
    return args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis


def _get_f1_f2(
    f: Callable[..., PyTree],
    keys: KeysView[str],
    x_axis: Optional[int],
    fx_axis: Optional[int],
    kw_axes: Dict[str, Optional[int]],
    args: Tuple[Any, ...],
    x1: PyTree,
    x2: Optional[PyTree]
) -> Tuple[Callable[[PyTree], PyTree], Callable[[PyTree], PyTree]]:
    half = len(args) // 2
    args1, args2 = args[:half], args[half:]
    _kwargs1 = {k: v for k, v in zip(keys, args1)}
    _kwargs2 = {k: v for k, v in zip(keys, args2)}
    f1 = _get_f_params(f, x1, x_axis, fx_axis, kw_axes, **_kwargs1)
    f2 = f1 if utils.all_none(x2) else _get_f_params(f, x2, x_axis, fx_axis, kw_axes, **_kwargs2)
    return f1, f2


def _is_int_axis(ax) -> bool:
    if isinstance(ax, int):
        return True
    if isinstance(ax, dict) and any(isinstance(v, int) for v in ax.values()):
        return True
    return False


def _canonicalize_axes(
    vmap_axes: Optional[VMapAxes],
    x: PyTree,
    fx: PyTree,
    **kwargs
) -> VMapAxisTriple:
    """
    Simplified: supports None or a single int for x and fx; kwargs_axes may be a dict of ints.
    """
    if isinstance(vmap_axes, tuple) and len(vmap_axes) == 3:
        x_axis, fx_axis, kw_axes = vmap_axes
    else:
        x_axis, fx_axis, kw_axes = vmap_axes, vmap_axes, {}

    return x_axis, fx_axis, kw_axes


def _add_axes(a: Optional[int], b: PyTree) -> Optional[int]:
    return a  # identity in this simplified port


def _ndim_like(x: PyTree) -> PyTree:
    return tree_map(lambda t: t.ndim, x)


def _diagonal(ntk_cols: PyTree, fx: PyTree) -> PyTree:
    leaves_ntk, _ = tree_flatten(ntk_cols)
    leaves_fx, fx_tree = tree_flatten(fx)
    n = len(leaves_fx)
    # pick diagonal blocks
    diag = [leaves_ntk[i * (n + 1)] for i in range(n)]
    return tree_unflatten(fx_tree, diag)


def _eval_f_shape(g: Callable[[PyTree], PyTree], params: PyTree) -> PyTree:
    return g(params)


def _linear_transpose_like(linear_op: Callable[[PyTree], PyTree], ref_out: PyTree, delta: PyTree) -> PyTree:
    # In JAX we’d use linear_transpose; here `linear_op` is already linear in delta.
    return linear_op(delta)


def _std_basis_from_shape(pytree: PyTree) -> PyTree:
    leaves, treedef = tree_flatten(pytree)
    sizes = [l.numel() for l in leaves]
    ndim = sum(sizes)
    if ndim == 0:
        return tree_unflatten(treedef, leaves)

    eye = torch.eye(ndim, dtype=leaves[0].dtype, device=leaves[0].device)
    # Split and reshape slices back into pytree leaf shapes on axis 0.
    idx = 0
    parts = []
    for s, leaf in zip(sizes, leaves):
        part = eye[:, idx: idx + s]
        part = part.T.reshape((s,) + leaf.shape)
        parts.append(part)
        idx += s
    # Broadcast a basis per-leaf; caller typically vmap over this.
    # Here we return a PyTree whose leaves are (s, *shape)
    return tree_unflatten(treedef, parts)


def _get_res_batch_dims(contracting_dims: Iterable[int], batch_dims: Iterable[int]) -> list[int]:
    res_batch_dims = [2 * b - i for i, b in enumerate(batch_dims)]
    for i, b in enumerate(batch_dims):
        for c in contracting_dims:
            if b > c:
                res_batch_dims[i] -= 2
    return res_batch_dims


def _dot_general(
    lhs: Tensor,
    rhs: Tensor,
    contracting_dims: Axes,
    batch_dims: Axes,
) -> Tensor:
    """
    torch tensordot + axis reordering to match the JAX semantics preserved in original code.
    """
    if lhs.ndim != rhs.ndim:
        raise ValueError("lhs and rhs must have the same rank.")

    contracting_dims_ = utils.canonicalize_axis(contracting_dims, tuple(range(lhs.ndim)))
    batch_dims_ = utils.canonicalize_axis(batch_dims, tuple(range(lhs.ndim)))

    # tensordot over contracting dims and batch dims simultaneously
    # Build lists of dims
    lhs_cont = list(contracting_dims_)
    rhs_cont = list(contracting_dims_)
    lhs_batch = list(batch_dims_)
    rhs_batch = list(batch_dims_)

    # Use einsum to keep control of resulting axis order
    # Build labels
    import string
    letters = string.ascii_lowercase + string.ascii_uppercase
    assert lhs.ndim <= len(letters) // 2, "Rank too high for simple labeling."
    L = list(letters[:lhs.ndim])
    R = list(letters[lhs.ndim:lhs.ndim*2])
    out = []

    # Tie batch dims
    for lbi, rbi in zip(lhs_batch, rhs_batch):
        R[rbi] = L[lbi]

    # Contract dims -> remove from output
    for lci, rci in zip(lhs_cont, rhs_cont):
        R[rci] = L[lci]

    # Output keeps non-contracted, non-batch dims from lhs and rhs
    for i in range(lhs.ndim):
        if i not in lhs_cont and i not in lhs_batch:
            out.append(L[i])
    for j in range(rhs.ndim):
        if j not in rhs_cont and j not in rhs_batch:
            out.append(R[j])

    expr = f"{''.join(L)},{''.join(R)}->{''.join(out)}"
    return torch.einsum(expr, lhs, rhs)