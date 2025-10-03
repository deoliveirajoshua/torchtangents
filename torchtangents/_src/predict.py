# Copyright 2019 Google LLC
# Apache-2.0

from __future__ import annotations

import collections
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Generator, Iterable, NamedTuple, Optional, Protocol, Union, Tuple, Dict

import torch
from torch import Tensor

# Local utilities assumed already ported to PyTorch versions.
from .utils import dataclasses  # If you replaced with stdlib dataclasses, adjust imports.
from .utils import utils
from .utils.typing import Axes, Get, KernelFn

PyTree = Any
ArrayOrScalar = Union[None, int, float, Tensor]


class PredictFn(Protocol):
    def __call__(
        self,
        t: Optional[ArrayOrScalar] = None,
        fx_train_0: ArrayOrScalar = 0.0,
        fx_test_0: Optional[ArrayOrScalar] = None,
        k_test_train: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        ...


def gradient_descent_mse(
    k_train_train: Tensor,
    y_train: Tensor,
    learning_rate: float = 1.0,
    diag_reg: float = 0.0,
    diag_reg_absolute_scale: bool = False,
    trace_axes: Axes = (-1,),
) -> PredictFn:
    """
    PyTorch translation of NTK/NNGP gradient-flow MSE predictor (closed-form).
    """

    _, odd, first, _ = _get_axes(k_train_train)
    trace_axes = utils.canonicalize_axis(trace_axes, y_train)
    trace_axes = tuple(-y_train.ndim + a for a in trace_axes)
    n_t_axes, n_non_t_axes = len(trace_axes), y_train.ndim - len(trace_axes)
    last_t_axes = tuple(range(-n_t_axes, 0))
    non_t_axes = tuple(range(-y_train.ndim, -n_t_axes))

    @lru_cache(1)
    def get_predict_fn_inf():
        solve = _get_cho_solve(k_train_train, diag_reg, diag_reg_absolute_scale)

        def predict_fn_inf(fx_train_0, fx_test_0, k_test_train):
            fx_train_t = y_train.to(k_train_train.dtype)
            if fx_test_0 is None:
                return fx_train_t

            rhs = y_train if fx_train_0 is None else (y_train - fx_train_0)
            dfx_test = torch.tensordot(k_test_train, solve(rhs, trace_axes), (odd, first))
            dfx_test = torch.movedim(dfx_test, last_t_axes, trace_axes)
            fx_test_t = fx_test_0 + dfx_test

            if fx_train_0 is None:
                return fx_test_t
            return fx_train_t, fx_test_t

        return predict_fn_inf

    @lru_cache(1)
    def get_predict_fn_finite():
        expm1_fn, inv_expm1_fn = _get_fns_in_eigenbasis(
            k_train_train,
            diag_reg,
            diag_reg_absolute_scale,
            (
                _make_expm1_fn(y_train.numel()),
                _make_inv_expm1_fn(y_train.numel()),
            ),
        )

        rhs_shape = tuple(y_train.shape[a] for a in trace_axes)

        def predict_fn_finite(t, fx_train_0, fx_test_0, k_test_train):
            t = torch.as_tensor(t, dtype=k_train_train.dtype, device=k_train_train.device) * learning_rate
            t_shape, t_ndim = t.shape, t.ndim
            first_t_axes = tuple(range(t_ndim))
            t = t.reshape((-1, 1))

            rhs = -y_train if fx_train_0 is None else (fx_train_0 - y_train)
            rhs = torch.movedim(rhs, trace_axes, last_t_axes).reshape((-1,) + rhs_shape)
            shape = t_shape + k_train_train.shape[1::2] + rhs_shape

            if fx_train_0 is not None:
                dfx_train = expm1_fn(rhs, t).reshape(shape)
                dfx_train = torch.movedim(dfx_train, last_t_axes, trace_axes)
                fx_train_t = fx_train_0.unsqueeze(first_t_axes).expand_as(dfx_train) + dfx_train

            if fx_test_0 is not None:
                dfx_test = inv_expm1_fn(rhs, t).reshape(shape)
                dfx_test = torch.tensordot(k_test_train, dfx_test, (odd, non_t_axes))
                dfx_test = torch.movedim(
                    dfx_test,
                    tuple(range(n_non_t_axes, n_non_t_axes + t_ndim)) + last_t_axes,
                    tuple(range(t_ndim)) + trace_axes,
                )
                fx_test_t = fx_test_0.unsqueeze(first_t_axes).expand_as(dfx_test) + dfx_test

            if fx_train_0 is not None and fx_test_0 is not None:
                return fx_train_t, fx_test_t
            if fx_test_0 is None:
                return fx_train_t
            return fx_test_t

        return predict_fn_finite

    def predict_fn(
        t: Optional[ArrayOrScalar] = None,
        fx_train_0: ArrayOrScalar = 0.0,
        fx_test_0: Optional[ArrayOrScalar] = None,
        k_test_train: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        _check_inputs(fx_train_0, fx_test_0, k_test_train)

        if t is None:
            return get_predict_fn_inf()(fx_train_0, fx_test_0, k_test_train)
        return get_predict_fn_finite()(t, fx_train_0, fx_test_0, k_test_train)

    return predict_fn


@dataclass
class ODEState:
    fx_train: Optional[Tensor] = None
    fx_test: Optional[Tensor] = None
    qx_train: Optional[Tensor] = None
    qx_test: Optional[Tensor] = None


class PredictFnODE(Protocol):
    def __call__(
        self,
        t: Optional[ArrayOrScalar] = None,
        fx_train_or_state_0: Union[ArrayOrScalar, ODEState] = 0.0,
        fx_test_0: Optional[ArrayOrScalar] = None,
        k_test_train: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor], ODEState]:
        ...


def gradient_descent(
    loss: Callable[[Tensor, Tensor], Tensor],
    k_train_train: Tensor,
    y_train: Tensor,
    learning_rate: float = 1.0,
    momentum: Optional[float] = None,
    trace_axes: Axes = (-1,),
) -> PredictFnODE:
    """
    PyTorch translation of function-space GD / momentum dynamics via ODE solve.
    We use an in-module RK4 `odeint` to avoid external dependencies and keep
    `torch.compile`-compatibility.
    """
    _, odd, _, _ = _get_axes(k_train_train)
    trace_axes = utils.canonicalize_axis(trace_axes, y_train)
    non_t_axes = tuple(a for a in range(y_train.ndim) if a not in trace_axes)
    last_t_axes = range(-len(trace_axes), 0)

    dtype = k_train_train.dtype
    device = k_train_train.device

    def grad_loss(fx: Tensor) -> Tensor:
        # d/d(fx) loss(fx, y_train)
        with torch.enable_grad():
            fx = fx.requires_grad_(True)
            l = loss(fx, y_train)
            (g,) = torch.autograd.grad(l, fx, retain_graph=False, create_graph=False)
        return g

    mom = momentum
    if mom is not None:
        learning_rate = learning_rate ** 0.5
        mom = (mom - 1.0) / learning_rate

    def get_state_0(fx_train_or_state_0, fx_test_0, fx_test_shape):
        if isinstance(fx_train_or_state_0, ODEState):
            fx_train_0 = fx_train_or_state_0.fx_train
            fx_test_0_ = fx_train_or_state_0.fx_test
            qx_train_0 = fx_train_or_state_0.qx_train
            qx_test_0 = fx_train_or_state_0.qx_test
            if fx_test_0 is not None:
                raise ValueError("`fx_test_0` must be None if passing ODEState.")
            fx_test_0 = fx_test_0_
        else:
            fx_train_0 = fx_train_or_state_0
            qx_train_0 = None
            qx_test_0 = None

        if fx_train_0 is None:
            fx_train_0 = torch.zeros_like(y_train, dtype=dtype, device=device)
        else:
            fx_train_0 = torch.broadcast_to(fx_train_0, y_train.shape)

        if fx_test_0 is not None:
            fx_test_0 = torch.broadcast_to(fx_test_0, fx_test_shape)

        if mom is None:
            if qx_train_0 is not None or qx_test_0 is not None:
                raise ValueError("Momentum state provided but `momentum is None`.")
        else:
            qx_train_0 = (
                torch.zeros_like(y_train, dtype=dtype, device=device)
                if qx_train_0 is None
                else torch.broadcast_to(qx_train_0, y_train.shape)
            )
            qx_test_0 = (
                None
                if fx_test_0 is None
                else (
                    torch.zeros(fx_test_shape, dtype=dtype, device=device)
                    if qx_test_0 is None
                    else torch.broadcast_to(qx_test_0, fx_test_shape)
                )
            )
        return ODEState(fx_train_0, fx_test_0, qx_train_0, qx_test_0)

    def get_dstate_dt(k_test_train):
        def dstate_dt(state_t: ODEState, _t_scalar: Tensor) -> ODEState:
            fx_train_t, fx_test_t, qx_train_t, qx_test_t = (
                state_t.fx_train,
                state_t.fx_test,
                state_t.qx_train,
                state_t.qx_test,
            )

            dy_df_t = grad_loss(fx_train_t)

            fx_train_rhs = -torch.movedim(
                torch.tensordot(k_train_train, dy_df_t, (odd, non_t_axes)),
                last_t_axes,
                trace_axes,
            )
            fx_test_rhs = None
            if fx_test_t is not None:
                fx_test_rhs = -torch.movedim(
                    torch.tensordot(k_test_train, dy_df_t, (odd, non_t_axes)),
                    last_t_axes,
                    trace_axes,
                )

            if mom is None:
                return ODEState(fx_train_rhs, fx_test_rhs)
            # Momentum: two-state system
            fx_train_rhs = fx_train_rhs + mom * qx_train_t
            if qx_test_t is not None:
                fx_test_rhs = fx_test_rhs + mom * qx_test_t
            return ODEState(qx_train_t, qx_test_t, fx_train_rhs, fx_test_rhs)

        return dstate_dt

    def predict_fn(
        t: Optional[ArrayOrScalar] = None,
        fx_train_or_state_0: Union[ArrayOrScalar, ODEState] = 0.0,
        fx_test_0: Optional[ArrayOrScalar] = None,
        k_test_train: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor], ODEState]:
        _check_inputs(fx_train_or_state_0, fx_test_0, k_test_train)

        t = torch.as_tensor(
            float("inf") if t is None else t,
            dtype=dtype,
            device=device,
        ) * learning_rate
        t_shape = t.shape
        t = t.reshape((-1,))

        # ODE solver expects t[0] present; we ensure strictly increasing by adding epsilon if needed.
        t0 = torch.where(t[0] == 0, torch.tensor([-1e-24], dtype=dtype, device=device), torch.zeros((1,), dtype=dtype, device=device))
        t = torch.cat([t0, t], dim=0)

        fx_test_shape = _get_fx_test_shape(y_train, k_test_train, trace_axes)
        state_0 = get_state_0(fx_train_or_state_0, fx_test_0, fx_test_shape)

        state_t = _odeint_rk4(get_dstate_dt(k_test_train), state_0, t)

        # trim artificial t0 and reshape time axes back
        def _trim(x: Optional[Tensor]) -> Optional[Tensor]:
            if x is None:
                return None
            return x[1:].reshape(t_shape + x.shape[2:])

        state_t = ODEState(
            _trim(state_t.fx_train),
            _trim(state_t.fx_test),
            _trim(state_t.qx_train),
            _trim(state_t.qx_test),
        )

        if isinstance(fx_train_or_state_0, ODEState):
            return state_t

        fx_train_t, fx_test_t = state_t.fx_train, state_t.fx_test
        if fx_train_or_state_0 is not None and fx_test_0 is None:
            return fx_train_t
        if fx_test_0 is not None and fx_train_or_state_0 is None:
            return fx_test_t
        return fx_train_t, fx_test_t

    return predict_fn


class Gaussian(NamedTuple):
    mean: Tensor
    covariance: Tensor


def gp_inference(
    k_train_train,
    y_train: Tensor,
    diag_reg: float = 0.0,
    diag_reg_absolute_scale: bool = False,
    trace_axes: Axes = (-1,),
):
    """
    Compute posterior mean / covariance for NNGP/NTK/NTKGP in PyTorch.
    """
    even, odd, first, last = _get_axes(_get_first(k_train_train))
    trace_axes = utils.canonicalize_axis(trace_axes, y_train)

    @lru_cache(2)
    def solve(g: str):
        k_dd = _get_attr(k_train_train, g)
        return _get_cho_solve(k_dd, diag_reg, diag_reg_absolute_scale)

    @lru_cache(2)
    def k_inv_y(g: str):
        return solve(g)(y_train, trace_axes)

    @utils.get_namedtuple("Gaussians")
    def predict_fn(
        get: Optional[Get] = None,
        k_test_train=None,
        k_test_test=None,
    ) -> Dict[str, Union[Tensor, Gaussian]]:
        if get is None:
            get = ("nngp", "ntk")

        out: Dict[str, Union[Tensor, Gaussian]] = {}

        for g in get:
            k = g if g != "ntkgp" else "ntk"
            k_dd = _get_attr(k_train_train, k)
            k_td = None if k_test_train is None else _get_attr(k_test_train, k)

            if k_td is None:
                y = y_train.to(k_dd.dtype)
            else:
                y = torch.tensordot(k_td, k_inv_y(k), (odd, first))
                y = torch.movedim(y, range(-len(trace_axes), 0), trace_axes)

            if k_test_test is not None:
                if k_td is None:
                    out[g] = Gaussian(y, torch.zeros_like(k_dd, dtype=k_dd.dtype))
                else:
                    if (g == "ntk") and (not hasattr(k_train_train, "nngp") or not hasattr(k_test_train, "nngp")):
                        raise ValueError(
                            'If `"ntk" in get` and you request test-set posterior covariance, '
                            "train-train and test-train must include both `nngp` and `ntk`."
                        )

                    g_init = "nngp" if g != "ntkgp" else "ntk"
                    k_td_g_inv_y = solve(k)(_get_attr(k_test_train, g_init), even)
                    k_tt = _get_attr(k_test_test, g_init)

                    if g in ("nngp", "ntkgp"):
                        cov = torch.tensordot(k_td, k_td_g_inv_y, (odd, first))
                        cov = k_tt - utils.zip_axes(cov)
                        out[g] = Gaussian(y, cov)
                    elif g == "ntk":
                        term_1 = solve(g)(k_td, even)
                        cov = torch.tensordot(_get_attr(k_train_train, "nngp"), term_1, (odd, first))
                        cov = torch.tensordot(term_1, cov, (first, first))

                        term_2 = torch.tensordot(k_td, k_td_g_inv_y, (odd, first))
                        term_2 = term_2 + torch.movedim(term_2, 0, -1)  # symmetrize along last/first per original
                        cov = utils.zip_axes(cov - term_2) + k_tt
                        out[g] = Gaussian(y, cov)
                    else:
                        raise ValueError(g)
            else:
                out[g] = y

        return out

    return predict_fn


_Kernel = collections.namedtuple("Kernel", "nngp ntk")
_Kernel.__new__.__defaults__ = (None,) * len(_Kernel._fields)


def gradient_descent_mse_ensemble(
    kernel_fn: KernelFn,
    x_train: Tensor,
    y_train: Tensor,
    learning_rate: float = 1.0,
    diag_reg: float = 0.0,
    diag_reg_absolute_scale: bool = False,
    trace_axes: Axes = (-1,),
    **kernel_fn_train_train_kwargs,
):
    """
    PyTorch translation of GD-on-MSE ensemble predictor (finite/infinite time).
    """
    expm1 = _make_expm1_fn(y_train.numel())
    inv_expm1 = _make_inv_expm1_fn(y_train.numel())

    trace_axes = utils.canonicalize_axis(trace_axes, y_train)
    trace_axes = tuple(-y_train.ndim + a for a in trace_axes)
    n_trace_axes = len(trace_axes)
    last_t_axes = range(-n_trace_axes, 0)
    trace_shape = tuple(y_train.shape[a] for a in trace_axes)

    y_train_flat = torch.movedim(y_train, trace_axes, last_t_axes).reshape((-1,) + trace_shape)

    k_dd_cache: Dict[str, Tensor] = {}

    def get_k_train_train(get: Tuple[str, ...]) -> _Kernel:
        if len(get) == 1:
            g = get[0]
            if g not in k_dd_cache:
                k_dd_cache[g] = kernel_fn(x_train, None, g, **kernel_fn_train_train_kwargs)
        elif len(get) == 2:
            if not any(g in k_dd_cache for g in get):
                k = kernel_fn(x_train, None, get, **kernel_fn_train_train_kwargs)._asdict()
                k_dd_cache.update(k)
            else:
                for g in get:
                    if g not in k_dd_cache:
                        k_dd_cache[g] = kernel_fn(x_train, None, g, **kernel_fn_train_train_kwargs)
        else:
            raise ValueError(get)
        return _Kernel(**k_dd_cache)

    @lru_cache(2)
    def eigenspace(get: str):
        k_dd = getattr(get_k_train_train((get,)), get)
        k_dd = _add_diagonal_regularizer(utils.make_2d(k_dd), diag_reg, diag_reg_absolute_scale)
        evals, evecs = torch.linalg.eigh(k_dd)
        evals = evals.unsqueeze(0)
        return evals, evecs

    @lru_cache(4)
    def predict_inf(get: Get):
        _, get_tuple = utils.canonicalize_get(get)
        k_dd = get_k_train_train(get_tuple)
        return gp_inference(k_dd, y_train, diag_reg, diag_reg_absolute_scale, trace_axes)

    def get_kernels(get: Get, x_test: Optional[Tensor], compute_cov: bool, **kernel_fn_test_test_kwargs):
        get_dep = _get_dependency(get, compute_cov)
        k_dd = get_k_train_train(get_dep)
        if x_test is None:
            k_td = None
            nngp_tt = compute_cov or None
        else:
            args_train, _ = utils.split_kwargs(kernel_fn_train_train_kwargs, x_train)
            args_test, _ = utils.split_kwargs(kernel_fn_test_test_kwargs, x_test)

            def is_array(x):
                return utils.tree_all(utils.tree_map(lambda z: isinstance(z, Tensor), x))

            kwargs_td = dict(kernel_fn_train_train_kwargs)
            kwargs_tt = dict(kernel_fn_train_train_kwargs)

            for k in kernel_fn_test_test_kwargs:
                v_tt = kernel_fn_test_test_kwargs[k]
                v_dd = kernel_fn_train_train_kwargs[k]
                if is_array(v_dd) and is_array(v_tt):
                    if isinstance(v_dd, tuple) and len(v_dd) == 2 and isinstance(v_tt, tuple) and len(v_tt) == 2:
                        v_td = (args_test[k], args_train[k])
                    else:
                        v_td = v_tt
                elif v_dd != v_tt:
                    raise ValueError(
                        f"Same keyword argument {k} set to different values {v_dd} != {v_tt} for train/test kernels."
                    )
                else:
                    v_td = v_tt
                kwargs_td[k] = v_td
                kwargs_tt[k] = v_tt

            k_td = kernel_fn(x_test, x_train, get_dep, **kwargs_td)
            nngp_tt = kernel_fn(x_test, None, "nngp", **kwargs_tt) if compute_cov else None
        return k_dd, k_td, nngp_tt

    @utils.get_namedtuple("Gaussians")
    def predict_fn(
        t: Optional[ArrayOrScalar] = None,
        x_test: Optional[Tensor] = None,
        get: Optional[Get] = None,
        compute_cov: bool = False,
        **kernel_fn_test_test_kwargs,
    ) -> Dict[str, Union[Tensor, Gaussian]]:
        if get is None:
            get = ("nngp", "ntk")

        k_dd, k_td, nngp_tt = get_kernels(get, x_test, compute_cov, **kernel_fn_test_test_kwargs)

        if t is None:
            return predict_inf(get)(get=get, k_test_train=k_td, k_test_test=nngp_tt)

        t = torch.as_tensor(t, dtype=y_train.dtype, device=y_train.device) * learning_rate
        t_shape = t.shape
        t = t.reshape((-1, 1))

        def reshape_mean(mean: Tensor) -> Tensor:
            k = _get_first(k_dd if k_td is None else k_td)
            mean = mean.reshape(t_shape + k.shape[::2] + trace_shape)
            return torch.movedim(mean, last_t_axes, trace_axes)

        def reshape_cov(cov: Tensor) -> Tensor:
            k = _get_first(k_dd if k_td is None else k_td)
            cov_shape_t = t_shape + k.shape[::2] * 2
            return utils.zip_axes(cov.reshape(cov_shape_t), len(t_shape))

        out: Dict[str, Union[Tensor, Gaussian]] = {}

        for g in utils.canonicalize_get(get)[1]:
            evals, evecs = eigenspace(g)

            if k_td is None:
                mean = torch.einsum("ji,ti,ki,k...->tj...", evecs, -expm1(evals, t), evecs, y_train_flat)
            else:
                neg_inv_expm1 = -inv_expm1(evals, t)
                ktd_g = utils.make_2d(getattr(k_td, g))
                mean = torch.einsum("lj,ji,ti,ki,k...->tl...", ktd_g, evecs, neg_inv_expm1, evecs, y_train_flat)

            mean = reshape_mean(mean)

            if nngp_tt is not None:
                nngp_dd = utils.make_2d(k_dd.nngp)

                if k_td is None:
                    if g == "nngp":
                        cov = torch.einsum(
                            "ji,ti,ki->tjk",
                            evecs,
                            (torch.clamp_min(evals, 0.0) * torch.exp(-2 * torch.clamp_min(evals, 0.0) * t / y_train.numel())),
                            evecs,
                        )
                    elif g == "ntk":
                        exp = torch.einsum(
                            "mi,ti,ki->tmk",
                            evecs,
                            torch.exp(-torch.clamp_min(evals, 0.0) * t / y_train.numel()),
                            evecs,
                        )
                        cov = torch.einsum("tmk,kl,tnl->tmn", exp, nngp_dd, exp)
                    else:
                        raise ValueError(g)
                else:
                    _nngp_tt = utils.make_2d(nngp_tt).unsqueeze(0)
                    ktd_g = utils.make_2d(getattr(k_td, g))
                    neg_inv_expm1 = -inv_expm1(evals, t)

                    if g == "nngp":
                        cov = _nngp_tt - torch.einsum("mj,ji,ti,ki,lk->tml", ktd_g, evecs, -inv_expm1(evals, 2 * t), evecs, ktd_g)
                    elif g == "ntk":
                        term_1 = torch.einsum("mi,ti,ki,lk->tml", evecs, neg_inv_expm1, evecs, ktd_g)
                        term_2 = torch.einsum(
                            "mj,ji,ti,ki,lk->tml",
                            ktd_g,
                            evecs,
                            neg_inv_expm1,
                            evecs,
                            utils.make_2d(k_td.nngp),
                        )
                        term_2 = term_2 + term_2.movedim(1, 2)
                        cov = torch.einsum("tji,jk,tkl->til", term_1, nngp_dd, term_1) - term_2 + _nngp_tt
                    else:
                        raise ValueError(g)

                out[g] = Gaussian(mean, reshape_cov(cov))
            else:
                out[g] = mean

        return out

    return predict_fn


def max_learning_rate(
    ntk_train_train: Tensor,
    y_train_size: Optional[int] = None,
    momentum: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """
    Max feasible LR for infinite-width NTK under MSE with (mini)batch GD + momentum.
    """
    ntk_train_train = utils.make_2d(ntk_train_train)
    factor = ntk_train_train.shape[0] if y_train_size is None else y_train_size
    max_eva = torch.linalg.eigvalsh(ntk_train_train)[-1]
    lr = 2 * (1 + momentum) * factor / (max_eva + eps)
    return float(lr)


# =========================
# ====== INTERNALS ========
# =========================

def _odeint_rk4(
    f: Callable[[ODEState, Tensor], ODEState],
    y0: ODEState,
    t: Tensor,
) -> ODEState:
    """
    Simple fixed-step RK4 along provided monotonically-increasing times `t`.
    Returns stacked states with leading time dimension equal to len(t).
    Torch-compile friendly: pure-Tensor loops.
    """
    assert t.ndim == 1 and t.numel() >= 1
    # allocate outputs
    def _stack_first(x_list: list[Tensor]) -> Tensor:
        return torch.stack(x_list, dim=0)

    def _zeros_like(x: Optional[Tensor]) -> Optional[Tensor]:
        return None if x is None else torch.zeros_like(x)

    ys_fx_train = [y0.fx_train]
    ys_fx_test = [y0.fx_test]
    ys_qx_train = [y0.qx_train]
    ys_qx_test = [y0.qx_test]

    y = y0
    for i in range(1, t.numel()):
        t0 = t[i - 1]
        t1 = t[i]
        h = t1 - t0

        def add(a: Optional[Tensor], b: Optional[Tensor]) -> Optional[Tensor]:
            if a is None or b is None:
                return None
            return a + b

        def scale(a: Optional[Tensor], s: Tensor) -> Optional[Tensor]:
            if a is None:
                return None
            return a * s

        k1 = f(y, t0)
        y2 = ODEState(
            add(y.fx_train, scale(k1.fx_train, h / 2)),
            add(y.fx_test, scale(k1.fx_test, h / 2)),
            add(y.qx_train, scale(k1.qx_train, h / 2)),
            add(y.qx_test, scale(k1.qx_test, h / 2)),
        )
        k2 = f(y2, t0 + h / 2)
        y3 = ODEState(
            add(y.fx_train, scale(k2.fx_train, h / 2)),
            add(y.fx_test, scale(k2.fx_test, h / 2)),
            add(y.qx_train, scale(k2.qx_train, h / 2)),
            add(y.qx_test, scale(k2.qx_test, h / 2)),
        )
        k3 = f(y3, t0 + h / 2)
        y4 = ODEState(
            add(y.fx_train, scale(k3.fx_train, h)),
            add(y.fx_test, scale(k3.fx_test, h)),
            add(y.qx_train, scale(k3.qx_train, h)),
            add(y.qx_test, scale(k3.qx_test, h)),
        )
        k4 = f(y4, t1)

        def combine(yc: Optional[Tensor], k1c, k2c, k3c, k4c) -> Optional[Tensor]:
            if yc is None:
                return None
            return yc + (h / 6) * (k1c + 2 * k2c + 2 * k3c + k4c)

        y = ODEState(
            combine(y.fx_train, k1.fx_train, k2.fx_train, k3.fx_train, k4.fx_train),
            combine(y.fx_test, k1.fx_test, k2.fx_test, k3.fx_test, k4.fx_test),
            combine(y.qx_train, k1.qx_train, k2.qx_train, k3.qx_train, k4.qx_train),
            combine(y.qx_test, k1.qx_test, k2.qx_test, k3.qx_test, k4.qx_test),
        )

        ys_fx_train.append(y.fx_train)
        ys_fx_test.append(y.fx_test)
        ys_qx_train.append(y.qx_train)
        ys_qx_test.append(y.qx_test)

    return ODEState(
        _stack_first(ys_fx_train),
        _stack_first(ys_fx_test) if ys_fx_test[0] is not None else None,
        _stack_first(ys_qx_train) if ys_qx_train[0] is not None else None,
        _stack_first(ys_qx_test) if ys_qx_test[0] is not None else None,
    )


def _get_dependency(get: Get, compute_cov: bool) -> Tuple[str, ...]:
    _, g = utils.canonicalize_get(get)
    for s in g:
        if s not in ["nngp", "ntk"]:
            raise NotImplementedError(f'Can only get "nngp"/"ntk"; got {s}.')
    dep: Tuple[str, ...] = ()
    if ("nngp" in g) or ("ntk" in g and compute_cov):
        dep += ("nngp",)
    if "ntk" in g:
        dep += ("ntk",)
    return dep


def _get_fns_in_eigenbasis(
    k_train_train: Tensor,
    diag_reg: float,
    diag_reg_absolute_scale: bool,
    fns: Iterable[Callable[[Tensor, Tensor], Tensor]],
) -> Generator[Callable[[Tensor, Tensor], Tensor], None, None]:
    A = utils.make_2d(k_train_train)
    A = _add_diagonal_regularizer(A, diag_reg, diag_reg_absolute_scale)
    evals, evecs = torch.linalg.eigh(A)
    evals = evals.unsqueeze(0)

    def to_eigenbasis(fn):
        def new_fn(y_train, t):
            return torch.einsum("ji,ti,ki,k...->tj...", evecs, fn(evals, t), evecs, y_train)
        return new_fn

    return (to_eigenbasis(fn) for fn in fns)


def _add_diagonal_regularizer(A: Tensor, diag_reg: float, diag_reg_absolute_scale: bool) -> Tensor:
    n = A.shape[0]
    if not diag_reg_absolute_scale:
        diag_reg = diag_reg * torch.trace(A) / n
    return A + diag_reg * torch.eye(n, dtype=A.dtype, device=A.device)


def _get_cho_solve(
    A: Tensor,
    diag_reg: float,
    diag_reg_absolute_scale: bool,
    lower: bool = False,
) -> Callable[[Tensor, Axes], Tensor]:
    x_non_channel_shape = A.shape[1::2]
    A2 = utils.make_2d(A)
    A2 = _add_diagonal_regularizer(A2, diag_reg, diag_reg_absolute_scale)
    L = torch.linalg.cholesky(A2, upper=not lower)

    def cho_solve(b: Tensor, b_axes: Axes) -> Tensor:
        b_axes = utils.canonicalize_axis(b_axes, b)
        last_b_axes = range(-len(b_axes), 0)
        x_shape = x_non_channel_shape + tuple(b.shape[a] for a in b_axes)
        b2 = torch.movedim(b, b_axes, last_b_axes).reshape((A2.shape[1], -1))
        x2 = torch.cholesky_solve(b2, L, upper=not lower)
        return x2.reshape(x_shape)

    return cho_solve


def _get_fx_test_shape(y_train: Tensor, k_test_train: Optional[Tensor], y_axes: Axes) -> Tuple[int, ...]:
    if k_test_train is None:
        return tuple(y_train.shape)
    shape = list(k_test_train.shape[::2])
    y_axes = utils.canonicalize_axis(y_axes, y_train)
    for i, c in enumerate(y_train.shape):
        if i in y_axes:
            shape.insert(i, c)
    return tuple(shape)


def _make_expm1_fn(normalization: float):
    def expm1_fn(evals: Tensor, t: Tensor):
        return torch.expm1(-torch.clamp_min(evals, 0.0) * t / normalization)
    return expm1_fn


def _make_inv_expm1_fn(normalization: float):
    expm1_fn = _make_expm1_fn(normalization)
    def _inv_expm1_fn(evals: Tensor, t: Tensor):
        # divide by |evals| (broadcasted). Avoid divide-by-zero by clamping evals?
        return expm1_fn(evals, t) / torch.clamp(evals, min=torch.finfo(evals.dtype).tiny).abs()
    return _inv_expm1_fn


def _check_inputs(
    fx_train_or_state_0: Union[ArrayOrScalar, ODEState],
    fx_test_0: ArrayOrScalar,
    k_test_train: Optional[Tensor],
):
    if isinstance(fx_train_or_state_0, ODEState):
        if fx_test_0 is not None:
            raise ValueError("`fx_test_0` is included in `ODEState` and must be set to `None`.")
        fx_train_0 = fx_train_or_state_0.fx_train
        fx_test_0 = fx_train_or_state_0.fx_test
    else:
        fx_train_0 = fx_train_or_state_0

    if fx_train_0 is None and fx_test_0 is None:
        raise ValueError("Both `fx_train_0` and `fx_test_0` are `None` — no predictions requested.")
    if fx_test_0 is not None and k_test_train is None:
        raise ValueError("To get test-set predictions, provide `k_test_train`.")


def _get_axes(x: Tensor):
    n = x.ndim
    return (
        tuple(range(0, n, 2)),
        tuple(range(1, n, 2)),
        tuple(range(0, n // 2)),
        tuple(range(n // 2, n)),
    )


def _get_first(k) -> Tensor:
    if isinstance(k, Tensor):
        return k
    for g in ("nngp", "ntk"):
        if hasattr(k, g):
            v = getattr(k, g)
            if v is not None:
                return v
    raise ValueError(k)


def _get_attr(k, g: str) -> Tensor:
    if isinstance(k, Tensor):
        return k
    return getattr(k, g)
