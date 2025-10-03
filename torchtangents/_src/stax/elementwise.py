# stax_elementwise_torch.py
# PyTorch port of Neural Tangents elementwise activations + kernel transforms.
# Compatible with torch.compile (pure-Torch math, no side effects in forward paths).

from __future__ import annotations
import functools
import operator as op
from typing import Callable, Optional, Sequence, Tuple
import warnings
import math

import torch
import numpy as np

from ..utils import utils
from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer, LayerKernelFn  # optional: mirror type names
from .requirements import (
    Diagonal,
    get_diagonal,
    get_diagonal_outer_prods,
    layer,
    requires,
    supports_masking,
)

# -----------------------
# Numerics helpers
# -----------------------

def _sqrt(x: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
    # Safe sqrt with clamp; autograd works fine in Torch
    if tol > 0:
        return torch.sqrt(torch.clamp(x, min=tol))
    return torch.sqrt(x)

def _arctan2(x: torch.Tensor, y: torch.Tensor, fill_zero: Optional[float] = None) -> torch.Tensor:
    if fill_zero is None:
        return torch.atan2(x, y)
    return torch.where(
        (x == 0) & (y == 0),
        torch.as_tensor(fill_zero, dtype=x.dtype, device=x.device),
        torch.atan2(x, y),
    )

def _factorial(n: int) -> int:
    out = 1
    for i in range(1, n + 1):
        out *= i
    return out

def _double_factorial(n: int) -> int:
    out = 1
    for i in range(n, 0, -2):
        out *= i
    return out

# A small autodiff helper: derivative wrt `cov12` for elementwise tensor fn
def _grad_wrt_cov12(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cov12: torch.Tensor,
    var1: torch.Tensor,
    var2: torch.Tensor,
) -> torch.Tensor:
    cov12_req = cov12.detach().requires_grad_(True)
    y = fn(cov12_req, var1, var2)
    # Ensure scalar for autograd
    (g,) = torch.autograd.grad(y.sum(), cov12_req, retain_graph=False, create_graph=False, allow_unused=False)
    return g

# Vectorized 2D “vmap” that relies on broadcasted elementwise fn (Torch ops)
def _vmap_2d(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    cov12: torch.Tensor,
    var1: torch.Tensor,
    var2: Optional[torch.Tensor],
    diagonal_batch: bool,
    diagonal_spatial: bool
) -> torch.Tensor:
    """
    Torch version of the JAX helper. Assumes `fn` supports broadcasted tensors.
    Shapes/layouts match your utils.make_2d / utils.zip_axes conventions.
    """
    batch_ndim = 1 if diagonal_batch else 2
    start = 2 - batch_ndim
    cov_end = batch_ndim if diagonal_spatial else cov12.ndim
    _cov12 = utils.make_2d(cov12, start, cov_end)

    var_end = 1 if diagonal_spatial else var1.ndim
    var1 = var1.reshape(var1.shape[:start] + (-1,) + var1.shape[var_end:])
    var2 = var1 if var2 is None else var2.reshape(var2.shape[:start] + (-1,) + var2.shape[var_end:])

    out = fn(_cov12, var1, var2)  # rely on broadcasting
    out_shape = (
        cov12.shape[:start] +
        cov12.shape[start:cov_end:2] +
        cov12.shape[start + 1:cov_end:2] +
        cov12.shape[cov_end:]
    )
    out = out.reshape(out_shape)
    out = utils.zip_axes(out, start, cov_end)
    return out


# -----------------------
# Core elementwise wrapper
# -----------------------

def _elementwise(
    fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    name: str,
    kernel_fn: Optional[LayerKernelFn],
) -> InternalLayer:
    init_fn = lambda rng, input_shape: (input_shape, ())

    def apply_fn(params, inputs: torch.Tensor, **kwargs):
        if fn is None:
            raise NotImplementedError(fn)
        return fn(inputs)

    @requires(diagonal_spatial=Diagonal())
    def new_kernel_fn(k: Kernel, **kwargs) -> Kernel:
        if kernel_fn is None:
            raise NotImplementedError(kernel_fn)
        if not k.is_gaussian:
            raise ValueError(
                'The input to the activation must be Gaussian; apply a random affine transform before the nonlinearity.'
            )
        k2 = kernel_fn(k)
        return k2.replace(is_gaussian=False)

    init_fn.__name__ = apply_fn.__name__ = new_kernel_fn.__name__ = name
    return init_fn, apply_fn, new_kernel_fn


# -----------------------
# Activations + kernel transforms
# -----------------------

@layer
@supports_masking(remask_kernel=True)
def Erf(a: float = 1., b: float = 1., c: float = 0.) -> InternalLayer:
    """a * erf(b * x) + c"""
    def fn(x: torch.Tensor) -> torch.Tensor:
        return a * torch.special.erf(b * x) + c

    def kernel_fn(k: Kernel) -> Kernel:
        k = k * b

        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        cov1_denom = 1 + 2 * cov1
        cov2_denom = None if cov2 is None else 1 + 2 * cov2

        prod11, prod12, prod22 = get_diagonal_outer_prods(
            cov1_denom, cov2_denom, k.diagonal_batch, k.diagonal_spatial, op.mul
        )

        factor = 2.0 / math.pi

        def nngp_ntk_fn(nngp: torch.Tensor, prod: torch.Tensor, ntk: Optional[torch.Tensor] = None):
            square_root = _sqrt(prod - 4 * nngp**2)
            new_nngp = factor * torch.atan2(2 * nngp, square_root)
            if ntk is not None:
                dot_sigma = 2 * factor / square_root
                ntk = ntk * dot_sigma
            return new_nngp, ntk

        def nngp_fn_diag(v: torch.Tensor) -> torch.Tensor:
            return factor * torch.atan2(v, torch.sqrt(v + 0.25))

        nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2)
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, prod22, None)

        k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
        return a * k + c

    return _elementwise(fn, f'Erf({a}, {b}, {c})', kernel_fn)


def Sigmoid_like():
    return Erf(a=0.5, b=1/2.4020563531719796, c=0.5)


@layer
@supports_masking(remask_kernel=False)
def Gabor() -> InternalLayer:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x * x) * torch.sin(x)

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        prod11, prod12, prod22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)
        sum11, sum12, sum22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.add)

        def nngp_ntk_fn(nngp, prod, sum_, ntk=None):
            diff = 4 * (prod - nngp**2)
            denom = 2 * sum_ + diff + 1
            num = sum_ + diff + 2 * nngp
            exp_left = torch.exp(-num / (2 * denom))
            exp_right = torch.exp(2 * nngp / denom)

            if ntk is not None:
                shared_term = 1 + 2 * sum_ + 4 * (nngp**2 + prod)
                diff_term = 4 * nngp * (diff + 3 * sum_ + 2)
                lhs = shared_term - diff_term
                rhs = shared_term + diff_term
                t_dot = exp_left * (lhs + exp_right * rhs) / denom.pow(2.5)
                ntk = ntk * (t_dot / 2)
            nngp = exp_left * (exp_right - 1) / (2 * _sqrt(denom))
            return nngp, ntk

        def nngp_fn_diag(v):
            denom = 1 + 4 * v
            return (1 - torch.exp(-2 * v / denom)) / (2 * _sqrt(denom))

        nngp, ntk = nngp_ntk_fn(nngp, prod12, sum12, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            cov2 = None if cov2 is None else nngp_fn_diag(cov2)
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11, sum11, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, prod22, sum22, None)

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, 'Gabor', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def Gelu(approximate: bool = False) -> InternalLayer:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate='tanh' if approximate else 'none')

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        cov1_plus_1 = cov1 + 1
        cov2_plus_1 = None if cov2 is None else cov2 + 1

        prod11_p1, prod12_p1, prod22_p1 = get_diagonal_outer_prods(cov1_plus_1, cov2_plus_1, k.diagonal_batch, k.diagonal_spatial, op.mul)
        prod11, prod12, prod22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

        def nngp_ntk_fn(nngp, prod, prod_plus_1, ntk=None):
            delta_sq = prod_plus_1 - nngp**2
            delta = _sqrt(delta_sq, 1e-30)
            angles = torch.atan2(nngp, delta)
            new_nngp = (nngp**2 + prod * delta_sq) / (prod_plus_1 * delta)
            new_nngp += nngp * angles
            new_nngp /= (2 * math.pi)
            new_nngp += 0.25 * nngp

            if ntk is not None:
                second = 0.25 + angles / (2 * math.pi)
                first = 1 / delta_sq + (1 - prod) / prod_plus_1 + 1
                first = first * nngp / delta / (2 * math.pi)
                dot_sigma = first + second
                ntk = ntk * dot_sigma
            return new_nngp, ntk

        def nngp_fn_diag(v):
            root = torch.sqrt(1 + 2 * v)
            out = v / ((v + 1) * torch.sqrt(1 + 2 * v))
            out += torch.atan2(v, root) / 2
            out /= math.pi
            out += 0.25
            out *= v
            return out

        nngp, ntk = nngp_ntk_fn(nngp, prod12, prod12_p1, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2)
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11, prod11_p1, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, prod22, prod22_p1, None)

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, 'Gelu', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Sin(a: float = 1., b: float = 1., c: float = 0.) -> InternalLayer:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return a * torch.sin(b * x + c)

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        sum11, sum12, sum22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.add)
        half_a_sq = a * a / 2.0

        def nngp_ntk_fn(nngp, sum_, ntk=None):
            s1 = torch.exp(b*b * (-0.5 * sum_ + nngp))
            s2 = torch.exp(b*b * (-0.5 * sum_ - nngp)) * math.cos(2 * c)
            new = half_a_sq * (s1 - s2)
            if ntk is not None:
                ntk = ntk * (half_a_sq * b*b * (s1 + s2))
            return new, ntk

        def nngp_fn_diag(v):
            return half_a_sq * (1 - torch.exp(-2 * b*b * v) * math.cos(2 * c))

        nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2)
        else:
            cov1, _ = nngp_ntk_fn(cov1, sum11, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, sum22, None)

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'Sin({a}, {b}, {c})', kernel_fn)


def Cos(a: float = 1., b: float = 1., c: float = 0.) -> InternalLayer:
    return Sin(a=a, b=b, c=c + math.pi / 2)


@layer
@supports_masking(remask_kernel=True)
def Rbf(gamma: float = 1.0) -> InternalLayer:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(2.0) * torch.sin(math.sqrt(2.0 * gamma) * x + math.pi / 4.0)

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        sum11, sum12, sum22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.add)

        def nngp_ntk_fn(nngp, sum_, ntk):
            new = torch.exp(gamma * (-sum_ + 2 * nngp))
            if ntk is not None:
                ntk = ntk * (2 * gamma * new)
            return new, ntk

        def nngp_fn_diag(v):
            return torch.ones_like(v)

        nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2)
        else:
            cov1, _ = nngp_ntk_fn(cov1, sum11, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, sum22, None)

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'Rbf({gamma})', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def ABRelu(a: float, b: float, do_stabilize: bool = False) -> InternalLayer:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return a * torch.minimum(x, torch.zeros_like(x)) + b * torch.maximum(x, torch.zeros_like(x))

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

        if do_stabilize:
            # scale by max magnitude; avoid /0
            factor = torch.maximum(torch.max(torch.abs(nngp)), torch.tensor(1e-12, device=nngp.device, dtype=nngp.dtype))
            nngp = nngp / factor
            cov1 = cov1 / factor
            if cov2 is not None:
                cov2 = cov2 / factor

        prod11, prod12, prod22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

        def nngp_ntk_fn(nngp, prod, ntk=None):
            square_root = _sqrt(prod - nngp**2, 1e-30)
            angles = _arctan2(square_root, nngp, fill_zero=math.pi / 2)

            fac = (a - b)**2 / (2 * math.pi)
            dot_sigma = (a*a + b*b) / 2 - fac * angles
            new = fac * square_root + dot_sigma * nngp
            if ntk is not None:
                ntk = ntk * dot_sigma
            return new, ntk

        def nngp_fn_diag(v):
            return (a*a + b*b) / 2 * v

        nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2)
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, prod22, None)

        if do_stabilize:
            nngp = nngp * factor
            cov1 = cov1 * factor
            if cov2 is not None:
                cov2 = cov2 * factor

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'ABRelu({a}, {b})', kernel_fn)


def Relu(do_stabilize: bool = False) -> InternalLayer:
    return ABRelu(0.0, 1.0, do_stabilize)


def LeakyRelu(alpha: float, do_stabilize: bool = False) -> InternalLayer:
    return ABRelu(alpha, 1.0, do_stabilize)


def Abs(do_stabilize: bool = False) -> InternalLayer:
    return ABRelu(-1.0, 1.0, do_stabilize)


@layer
@supports_masking(remask_kernel=False)
def Sign() -> InternalLayer:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x)

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        if ntk is not None:
            ntk = torch.zeros_like(ntk)
        _, prod12, _ = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)
        angles = _arctan2(_sqrt(prod12 - nngp**2, 1e-30), nngp, fill_zero=math.pi / 2)
        nngp = 1 - angles * 2 / math.pi
        cov1 = torch.where(cov1 == 0., torch.zeros_like(cov1), torch.ones_like(cov1))
        cov2 = cov2 if cov2 is None else torch.where(cov2 == 0., torch.zeros_like(cov2), torch.ones_like(cov2))
        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, 'Sign', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Exp(a: float = 1., b: float = 1.) -> InternalLayer:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return a * torch.exp(b * x)

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        sum11, sum12, sum22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.add)

        def nngp_ntk_fn(nngp, sum_, ntk):
            new = torch.exp(b*b * (sum_ / 2 + nngp))
            if ntk is not None:
                ntk = ntk * (b*b * new)
            return new, ntk

        def nngp_fn_diag(v):
            return torch.exp(2 * b*b * v)

        nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2)
        else:
            cov1, _ = nngp_ntk_fn(cov1, sum11, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, sum22, None)

        return (k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)) * a

    return _elementwise(fn, f'Exp({a}, {b})', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Gaussian(a: float = 1., b: float = -1.) -> InternalLayer:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return a * torch.exp(b * x * x)

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        cov1_denom = 1 - 2 * b * cov1
        cov2_denom = None if cov2 is None else 1 - 2 * b * cov2

        prod11, prod12, prod22 = get_diagonal_outer_prods(
            cov1_denom, cov2_denom, k.diagonal_batch, k.diagonal_spatial, op.mul
        )

        factor = 4 * b * b

        def nngp_ntk_fn(nngp, prod, ntk=None):
            det = _sqrt(prod - factor * nngp**2, 1e-30)
            if ntk is not None:
                ntk = ntk * (factor * nngp / det**3)
            return 1 / det, ntk

        def nngp_fn_diag(v):
            return 1 / _sqrt(1 - 4 * b * v, 1e-30)

        nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2)
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, prod22, None)

        return (k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)) * a

    return _elementwise(fn, f'Gaussian({a}, {b})', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def ExpNormalized(gamma: float = 1., shift: float = -1., do_clip: bool = False) -> InternalLayer:
    """Simulates the Gaussian normalized kernel. Finite-width apply_fn is not supported."""
    def kernel_fn(k: Kernel) -> Kernel:
        cov1, cov2, nngp, ntk = k.cov1, k.cov2, k.nngp, k.ntk
        prod11, prod12, prod22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

        tol = 1e-30
        prod11 = _sqrt(prod11, tol)
        prod12 = _sqrt(prod12, tol)
        prod22 = _sqrt(prod22, tol) if prod22 is not None else None

        def exp_part(cov, prod):
            if cov is not None:
                cov = cov / prod
                if do_clip:
                    cov = torch.clamp(cov, -1.0, 1.0)
                cov = torch.exp(gamma * (cov + shift))
            return cov

        exp12 = exp_part(nngp, prod12)

        return k.replace(
            nngp=prod12 * exp12,
            cov1=prod11 * exp_part(cov1, prod11),
            cov2=None if cov2 is None else prod22 * exp_part(cov2, prod22),
            ntk=ntk if ntk is None else gamma * ntk * exp12
        )

    return _elementwise(None, 'ExpNormalized', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Hermite(degree: int) -> InternalLayer:
    if degree < 0:
        raise NotImplementedError('`degree` must be non-negative.')

    # Probabilists’ Hermite-e coefficients, normalized
    p = np.polynomial.hermite_e.herme2poly([0] * degree + [1])[::-1]
    coeff = float(_factorial(degree)) ** 0.5
    p_t = torch.as_tensor(p, dtype=torch.get_default_dtype())

    def fn(x: torch.Tensor) -> torch.Tensor:
        # torch.polyval-like (Horner) with descending coeffs
        out = torch.zeros_like(x)
        for c in p_t:
            out = out * x + c
        return out / coeff

    def kernel_fn(k: Kernel) -> Kernel:
        warnings.warn(
            'Inputs to Hermite are assumed to have unit norm across channels/features.'
        )
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        if ntk is not None:
            if degree == 0:
                ntk = torch.zeros_like(ntk)
            else:
                ntk = degree * (nngp ** (degree - 1)) * ntk

        def _power(mat: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if mat is None else mat ** degree

        nngp, cov1, cov2 = map(_power, (nngp, cov1, cov2))
        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'{degree}-Hermite polynomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def Monomial(degree: int) -> InternalLayer:
    if degree not in [0, 1, 2, 3, 4, 5]:
        raise NotImplementedError('degree must be in {0,1,2,3,4,5}.')

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x ** degree

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        prod11, prod12, prod22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

        def nngp_ntk_fn(nngp, prod, ntk=None):
            def nngp_fn(nngp, degree: int):
                if degree == -1:
                    return torch.zeros_like(nngp)
                elif degree == 0:
                    return torch.ones_like(nngp)
                elif degree == 1:
                    return nngp
                elif degree == 2:
                    return 2 * nngp**2 + prod
                elif degree == 3:
                    return 6 * nngp**3 + 9 * nngp * prod
                elif degree == 4:
                    return 3 * (8 * nngp**4 + 3 * prod * (8 * nngp**2 + prod))
                elif degree == 5:
                    return 15 * nngp * (8 * nngp**4 + 5 * prod * (8 * nngp**2 + 3 * prod))
                else:
                    raise NotImplementedError(degree)

            if ntk is not None:
                ntk = ntk * (degree**2 * nngp_fn(nngp, degree - 1))
            return nngp_fn(nngp, degree), ntk

        def nngp_fn_diag(v):
            return _double_factorial(2 * degree - 1) * v ** degree

        nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2)
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, prod22, None)

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'{degree}-monomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def RectifiedMonomial(degree: int) -> InternalLayer:
    if degree < 0:
        raise NotImplementedError('`degree` must be non-negative.')

    def fn(x: torch.Tensor) -> torch.Tensor:
        return (x >= 0).to(x.dtype) * (x ** degree)

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
        prod11, prod12, prod22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

        # The `j` helper uses autograd for its derivative when needed (NTK path).
        def j(nngp: torch.Tensor, sqrt_prod: torch.Tensor) -> torch.Tensor:
            theta = torch.arccos(torch.clamp(nngp / sqrt_prod, -1.0, 1.0))

            def f0(theta: torch.Tensor) -> torch.Tensor:
                return (math.pi - theta) / torch.sin(theta)

            # f_{k+1} = (d/dtheta f_k) / sin(theta)
            def diff(f: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
                def df(theta: torch.Tensor) -> torch.Tensor:
                    theta_req = theta.detach().requires_grad_(True)
                    val = f(theta_req)
                    (g,) = torch.autograd.grad(val.sum(), theta_req, retain_graph=False, create_graph=False)
                    return g / torch.sin(theta)
                return df

            f = f0
            for _ in range(degree):
                f = diff(f)

            return ((-1)**degree) * (torch.sin(theta))**(2 * degree + 1) * f(theta)

        def nngp_ntk_fn(nngp, prod, ntk=None):
            sqrt_prod = _sqrt(prod, 1e-30)
            coeff = sqrt_prod**degree / (2 * math.pi)

            if ntk is not None:
                if degree == 0:
                    ntk = torch.zeros_like(ntk)
                else:
                    # d/d nngp j(nngp, sqrt_prod)
                    def j_wrt_cov12(c, s): return j(c, s)
                    j_dot = _grad_wrt_cov12(j_wrt_cov12, nngp, sqrt_prod, sqrt_prod)  # var placeholders
                    ntk = ntk * (coeff * j_dot)

            new = coeff * j(nngp, sqrt_prod)
            return new, ntk

        def nngp_fn_diag(v):
            return _double_factorial(2 * degree - 1) * v**degree / 2

        nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2)
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, prod22, None)

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'{degree}-rectified-monomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def Polynomial(coef: Sequence[float]) -> InternalLayer:
    coef = torch.as_tensor(np.array(coef)[::-1], dtype=torch.get_default_dtype())  # descending for Horner

    def fn(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for c in coef:
            out = out * x + c
        return out

    degree = len(coef)

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

        def r(n: Optional[torch.Tensor], l: int) -> Optional[torch.Tensor]:
            if n is None:
                return None
            # Build coef_l as in JAX code
            # coef is reversed here (descending), but below we build by degree index
            np_coef = coef.detach().cpu().numpy()[::-1]  # back to ascending for the polynomial logic
            coef_dict = {
                2 * i + l: np_coef[2 * i + l] * _factorial(2 * i + l) / (2**i * _factorial(i) * _factorial(l)**0.5)
                for i in range(0, (degree - 1 - l) // 2 + 1)
            }
            coef_l = np.array([coef_dict.get(i, 0.0) for i in range(degree)], dtype=np.float64)[::-1]  # descending
            coef_l_t = torch.as_tensor(coef_l, dtype=n.dtype, device=n.device)
            # Horner with descending coef_l_t
            out = torch.zeros_like(n)
            for c in coef_l_t:
                out = out * n.sqrt() + c
            return out

        if degree == 0:
            rs11, rs12, rs22 = [], [], []
        else:
            rs11, rs12, rs22 = list(zip(*[
                get_diagonal_outer_prods(
                    r(cov1, l), r(cov2, l), k.diagonal_batch, k.diagonal_spatial, op.mul
                )
                for l in range(degree)
            ]))

        prod11, prod12, prod22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

        def nngp_ntk_fn(nngp, prod, r_prods, ntk=None):
            ratio = nngp / _sqrt(prod, 1e-30)

            if ntk is not None:
                t_dot = torch.zeros_like(ntk)
                for l in range(1, degree):
                    t_dot = t_dot + l * r_prods[l] * ratio ** (l - 1)
                ntk = ntk * (t_dot / _sqrt(prod, 1e-30))

            out = torch.zeros_like(nngp)
            for l in range(degree):
                out = out + r_prods[l] * ratio ** l
            return out, ntk

        def nngp_fn_diag(v, r_prods):
            out = torch.zeros_like(v)
            for l in range(degree):
                out = out + r_prods[l]
            return out

        nngp, ntk = nngp_ntk_fn(nngp, prod12, rs12, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1, rs11)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2, rs22)
        else:
            cov1, _ = nngp_ntk_fn(cov1, prod11, rs11, None)
            if cov2 is not None:
                cov2, _ = nngp_ntk_fn(cov2, prod22, rs22, None)

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'{coef.tolist()}-polynomial', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Elementwise(
    fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    nngp_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    d_nngp_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None
) -> InternalLayer:
    """General elementwise layer given fn and its 2D integral nngp_fn."""
    if fn is not None:
        name = getattr(fn, "__name__", "elementwise_fn")
    elif nngp_fn is not None:
        name = getattr(nngp_fn, "__name__", "nngp_fn")
    else:
        raise ValueError('Provide at least `fn` or `nngp_fn`.')

    if nngp_fn is None:
        kernel_fn = None
    else:
        # If derivative not provided, compute via autograd w.r.t. cov12
        if d_nngp_fn is None:
            warnings.warn('Using Torch autodiff to compute d/dcov12 nngp_fn for NTK (may affect stability).')

            def d_nngp_autograd(c12, v1, v2):
                return _grad_wrt_cov12(nngp_fn, c12, v1, v2)
            d_nngp = d_nngp_autograd
        else:
            d_nngp = d_nngp_fn

        def kernel_fn(k: Kernel) -> Kernel:
            cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

            var1 = get_diagonal(cov1, k.diagonal_batch, k.diagonal_spatial)
            var2 = get_diagonal(cov2, k.diagonal_batch, k.diagonal_spatial)

            if ntk is not None:
                def d_wrap(c, v1, v2): return d_nngp(c, v1, v2)
                ntk = ntk * _vmap_2d(d_wrap, nngp, var1, var2, False, k.diagonal_spatial)

            def n_wrap(c, v1, v2): return nngp_fn(c, v1, v2)
            nngp2 = _vmap_2d(n_wrap, nngp, var1, var2, False, k.diagonal_spatial)

            cov1_ = _vmap_2d(n_wrap, cov1, var1, None, k.diagonal_batch, k.diagonal_spatial)
            cov2_ = None if cov2 is None else _vmap_2d(n_wrap, cov2, var2, None, k.diagonal_batch, k.diagonal_spatial)

            return k.replace(cov1=cov1_, nngp=nngp2, cov2=cov2_, ntk=ntk)

    return _elementwise(fn, name, kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def ElementwiseNumerical(
    fn: Callable[[torch.Tensor], torch.Tensor],
    deg: int,
    df: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
) -> InternalLayer:
    """Numerical Gauss–Hermite quadrature for general activations."""
    warnings.warn(
        f'Numerical Activation Layer with fn={fn}, deg={deg}. Accuracy depends on `deg` and smoothness of `fn`.'
    )

    # Use numpy.polynomial.hermite.hermgauss to avoid SciPy dependency
    xs_np, ws_np = np.polynomial.hermite.hermgauss(deg)
    xs = torch.as_tensor(xs_np, dtype=torch.get_default_dtype())
    ws = torch.as_tensor(ws_np, dtype=torch.get_default_dtype())

    if df is None:
        warnings.warn('Using Torch autodiff to get df for NTK; provide `df` for speed/stability.')
        def df(x: torch.Tensor) -> torch.Tensor:  # type: ignore[no-redef]
            x_req = x.detach().requires_grad_(True)
            y = fn(x_req)
            (g,) = torch.autograd.grad(y.sum(), x_req, retain_graph=False, create_graph=False)
            return g

    def kernel_fn(k: Kernel) -> Kernel:
        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

        d1 = get_diagonal(cov1, k.diagonal_batch, k.diagonal_spatial)
        d2 = get_diagonal(cov2, k.diagonal_batch, k.diagonal_spatial)

        end_axis = 1 if k.diagonal_spatial else cov1.ndim
        q11 = utils.interleave_ones(d1, 0, end_axis, True)
        q22 = utils.interleave_ones(d1 if d2 is None else d2, 0, end_axis, False)

        def nngp_ntk_fn(nngp, q11, q22, ntk=None):
            # Build quadrature grid
            grid = torch.outer(ws, ws)  # (deg, deg)
            x = xs.view(xs.shape[0], *([1] * (nngp.ndim + 1)))
            y = xs.view(1, xs.shape[0], *([1] * nngp.ndim))
            xy_axes = (0, 1)

            nngp_e = nngp.unsqueeze(0).unsqueeze(0)
            q11_e = q11.unsqueeze(0).unsqueeze(0)
            q22_e = q22.unsqueeze(0).unsqueeze(0)

            def integrate(f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
                # fvals shape (deg, deg, ...)
                fvals = f(_sqrt(2 * q11_e) * x) * (
                    f(nngp_e / _sqrt(q11_e / 2, 1e-30) * x + _sqrt(2 * (q22_e - nngp_e**2 / q11_e), 1e-30) * y)
                )
                # tensordot over the first two axes with the grid weights
                out = (grid.unsqueeze(-1) * fvals).sum(dim=xy_axes) / math.pi
                return out

            if ntk is not None:
                ntk = ntk * integrate(df)
            new_nngp = integrate(fn)
            return new_nngp, ntk

        def nngp_fn_diag(v):
            x = xs.view(xs.shape[0], *([1] * v.ndim))
            v_e = v.unsqueeze(0)
            fval = fn(_sqrt(2 * v_e) * x) ** 2
            return (ws.view(-1, *([1] * v.ndim)) * fval).sum(dim=0) / math.sqrt(math.pi)

        nngp, ntk = nngp_ntk_fn(nngp, q11, q22, ntk)

        if k.diagonal_batch and k.diagonal_spatial:
            cov1 = nngp_fn_diag(cov1)
            if cov2 is not None:
                cov2 = nngp_fn_diag(cov2)
        else:
            start_axis = 1 if k.diagonal_batch else 0
            q11 = utils.interleave_ones(d1, start_axis, end_axis, True)
            q22 = utils.interleave_ones(d1, start_axis, end_axis, False)
            cov1, _ = nngp_ntk_fn(cov1, q11, q22, None)

            if cov2 is not None:
                q11 = utils.interleave_ones(d2, start_axis, end_axis, True)
                q22 = utils.interleave_ones(d2, start_axis, end_axis, False)
                cov2, _ = nngp_ntk_fn(cov2, q11, q22, None)

        return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

    return _elementwise(fn, f'ElementwiseNumerical({getattr(fn, "__name__", "fn")},deg={deg})', kernel_fn)
