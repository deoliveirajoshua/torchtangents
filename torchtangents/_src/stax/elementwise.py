# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# limitations under the License.

"""Elementwise nonlinearities / activation functions ported to PyTorch."""

import operator as op
from typing import Callable, Optional, Sequence
import warnings

import torch
import math

from ..utils import utils
from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer
from ..utils.typing import LayerKernelFn
from .requirements import Bool, Diagonal
from .requirements import get_diagonal
from .requirements import get_diagonal_outer_prods
from .requirements import layer
from .requirements import requires
from .requirements import supports_masking


def _sqrt(x: torch.Tensor, tol: float = 0.) -> torch.Tensor:
  safe_x = torch.where(x > tol, x, torch.ones_like(x))
  return torch.where(x > tol, torch.sqrt(safe_x), torch.full_like(x, math.sqrt(tol)))


def _arctan2(x: torch.Tensor, y: torch.Tensor, fill_zero: Optional[float] = None) -> torch.Tensor:
  if fill_zero is not None:
    mask = (x == 0.) & (y == 0.)
    res = torch.atan2(x, y)
    return torch.where(mask, torch.tensor(fill_zero, dtype=res.dtype, device=res.device), res)
  return torch.atan2(x, y)


def _elementwise(
    fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    name: str,
    kernel_fn: Optional[LayerKernelFn],
) -> InternalLayer:
  init_fn = lambda rng, input_shape: (input_shape, ())

  def apply_fn(params, inputs, **kwargs):
    if fn is None:
      raise NotImplementedError(fn)
    return fn(inputs)

  @requires(diagonal_spatial=Diagonal(input=Bool.MAYBE, output=Bool.MAYBE))
  def new_kernel_fn(k: Kernel, **kwargs) -> Kernel:
    if kernel_fn is None:
      raise NotImplementedError(kernel_fn)

    # Symbolic assertion to avoid Dynamo guards during tracing
    torch._assert(k.is_gaussian, "The input to the activation function must be Gaussian.")
    k = kernel_fn(k)
    return k.replace(is_gaussian=False)

  init_fn.__name__ = apply_fn.__name__ = new_kernel_fn.__name__ = name
  return init_fn, apply_fn, new_kernel_fn


@layer
@supports_masking(remask_kernel=False)
def Elementwise(*args, **kwargs) -> InternalLayer:
  raise NotImplementedError('Elementwise in Torchtangents is yet to be fully ported.')

def ElementwiseNumerical(*args, **kwargs) -> InternalLayer:
  raise NotImplementedError('ElementwiseNumerical in Torchtangents is yet to be fully ported.')

def _polyval(coeffs, x):
  """Evaluates a polynomial at x. coeffs: [p_n, ..., p_0]"""
  res = coeffs[0]
  for c in coeffs[1:]:
    res = res * x + c
  return res


@layer
@supports_masking(remask_kernel=True)
def Polynomial(coef: Sequence[float]) -> InternalLayer:
  """Polynomials, i.e. `coef[0] + coef[1] * x + … + coef[n] * x**n`.

  Args:
    coef:
      a sequence of coefficients. Follows
      :class:`numpy.polynomial.polynomial.Polynomial` API.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  import numpy as np
  coef = np.array(coef)
  degree = len(coef)

  def fn(x):
    return _polyval(torch.tensor(coef[::-1].copy(), dtype=x.dtype, device=x.device), x)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def r(n: Optional[torch.Tensor], l: int) -> Optional[torch.Tensor]:
      if n is None:
        return None

      coef_dict = {
          2 * i + l: coef[2 * i + l] * _factorial(2 * i + l) / (
              2**i * _factorial(i) * _factorial(l)**0.5)
          for i in range(0, (degree - 1 - l) // 2 + 1)
      }
      coef_l = np.array(
          [coef_dict[i] if i in coef_dict else 0 for i in range(degree)])
      return _polyval(torch.tensor(coef_l[::-1].copy(), dtype=n.dtype, device=n.device), n ** 0.5)

    if degree == 0:
      rs11, rs12, rs22 = [], [], []
    else:
      rs_combined = [
          get_diagonal_outer_prods(r(cov1, l),
                                   r(cov2, l),
                                   k.diagonal_batch,
                                   k.diagonal_spatial,
                                   op.mul)
          for l in range(degree)
      ]
      rs11, rs12, rs22 = list(zip(*rs_combined))

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1,
                                                      cov2,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    def nngp_ntk_fn(
        nngp: torch.Tensor,
        prod: torch.Tensor,
        r_prods: Sequence[torch.Tensor],
        ntk: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
      sqrt_prod = _sqrt(prod)
      ratio = torch.where(sqrt_prod > 0, nngp / sqrt_prod, torch.tensor(0.0, dtype=nngp.dtype, device=nngp.device))

      if ntk is not None:
        t_dot = torch.zeros_like(ntk)
        for l in range(1, degree):
          t_dot = t_dot + l * r_prods[l] * (ratio**(l - 1))
        ntk = ntk * (t_dot / torch.where(sqrt_prod > 0, sqrt_prod, torch.tensor(1e-12, dtype=ntk.dtype, device=ntk.device)))

      nngp_out = torch.zeros_like(nngp)
      for l in range(degree):
        nngp_out = nngp_out + r_prods[l] * (ratio ** l)

      return nngp_out, ntk

    def nngp_fn_diag(nngp: torch.Tensor,
                     r_prods: Sequence[torch.Tensor]) -> torch.Tensor:
      out = torch.zeros_like(nngp)
      for l in range(degree):
        out = out + r_prods[l]
      return out

    nngp, ntk = nngp_ntk_fn(nngp, prod12, rs12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1, rs11)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2, rs22)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11, rs11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22, rs22)

    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return k

  return _elementwise(fn, f'{coef}-polynomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def RectifiedMonomial(degree: int) -> InternalLayer:
  """Rectified monomials, i.e. `(x >= 0) * x^degree`.

  Args:
    degree: a non-negative integer power.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  if degree < 0:
    raise NotImplementedError('`degree` must be a non-negative integer.')

  def fn(x):
    return (x >= 0).to(x.dtype) * x**degree

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1,
                                                      cov2,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    def j_fn(nngp_val: torch.Tensor, sqrt_prod_val: torch.Tensor) -> torch.Tensor:
      nngp_norm = torch.clamp(nngp_val / torch.clamp(sqrt_prod_val, min=1e-12), -1.0, 1.0)
      theta = torch.acos(nngp_norm)

      def f0(t):
        s = torch.sin(t)
        return (math.pi - t) / torch.where(s == 0, torch.tensor(1e-12, dtype=t.dtype, device=t.device), s)

      def diff(f):
        def df(t):
          from torch.func import grad
          return grad(f)(t) / torch.where(torch.sin(t) == 0, torch.tensor(1e-12, dtype=t.dtype, device=t.device), torch.sin(t))
        return df

      f_der = f0
      for _ in range(degree):
        f_der = diff(f_der)

      res = (-1)**degree * (torch.sin(theta))**(2 * degree + 1) * f_der(theta)
      
      if degree > 0:
          res = torch.where(theta < 1e-7, torch.tensor(_double_factorial(2 * degree - 1) * math.pi / (2**degree), dtype=res.dtype, device=res.device), res)
      else:
          res = torch.where(theta < 1e-7, torch.tensor(math.pi, dtype=res.dtype, device=res.device), res)
      return res

    def nngp_ntk_fn(
        nngp: torch.Tensor,
        prod: torch.Tensor,
        ntk: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

      sqrt_prod = _sqrt(prod)
      coeff = (sqrt_prod**degree) / (2 * math.pi)

      from torch.func import vmap, grad
      nngp_flat = nngp.reshape(-1)
      sqrt_prod_flat = sqrt_prod.reshape(-1)
      
      if ntk is not None:
        if degree == 0:
          ntk_out = torch.zeros_like(ntk)
        else:
          j_dot_fn = vmap(grad(j_fn, argnums=0))
          j_dot = j_dot_fn(nngp_flat, sqrt_prod_flat).reshape(nngp.shape)
          ntk_out = ntk * (coeff * j_dot)
      else:
        ntk_out = None

      j_v = vmap(j_fn)(nngp_flat, sqrt_prod_flat).reshape(nngp.shape)
      nngp_new = coeff * j_v
      return nngp_new, ntk_out

    def nngp_fn_diag(nngp: torch.Tensor) -> torch.Tensor:
      return _double_factorial(2 * degree - 1) * (nngp**degree) / 2

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'{degree}-rectified-monomial', kernel_fn)

@layer
@supports_masking(remask_kernel=True)
def ABRelu(
    a: float,
    b: float,
    do_stabilize: bool = False
) -> InternalLayer:
  """ABReLU nonlinearity, i.e. `a * min(x, 0) + b * max(x, 0)`."""
  def fn(x):
    return a * torch.minimum(x, torch.zeros_like(x)) + b * torch.maximum(x, torch.zeros_like(x))

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if do_stabilize:
      factor = torch.maximum(torch.max(torch.abs(nngp)), torch.tensor(1e-12, device=nngp.device))
      nngp = nngp / factor
      cov1 = cov1 / factor
      if cov2 is not None:
        cov2 = cov2 / factor

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1,
                                                      cov2,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    def nngp_ntk_fn(nngp, prod, ntk=None):
      square_root = _sqrt(prod - nngp**2)
      angles = _arctan2(square_root, nngp, fill_zero=math.pi / 2)

      factor = (a - b)**2 / (2 * math.pi)
      dot_sigma = (a**2 + b**2) / 2 - factor * angles
      nngp_out = factor * square_root + dot_sigma * nngp

      if ntk is not None:
        ntk = ntk * dot_sigma

      return nngp_out, ntk

    def nngp_fn_diag(nngp):
      return (a**2 + b**2) / 2 * nngp

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk=ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    if do_stabilize:
      nngp = nngp * factor
      cov1 = cov1 * factor
      if cov2 is not None:
        cov2 = cov2 * factor

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'ABReLU({a}, {b})', kernel_fn)


def ReLU(do_stabilize: bool = False) -> InternalLayer:
  return ABRelu(0, 1, do_stabilize)


def LeakyReLU(alpha: float, do_stabilize: bool = False) -> InternalLayer:
  return ABRelu(alpha, 1, do_stabilize)


def Abs(do_stabilize: bool = False) -> InternalLayer:
  return ABRelu(-1, 1, do_stabilize)


@layer
@supports_masking(remask_kernel=True)
def Erf(
    a: float = 1.,
    b: float = 1.,
    c: float = 0.
) -> InternalLayer:
  """Affine transform of `Erf` nonlinearity, i.e. `a * Erf(b * x) + c`."""
  def fn(x):
    return a * torch.erf(b * x) + c

  def kernel_fn(k: Kernel) -> Kernel:
    # Scale k by b
    cov1_b = k.cov1 * (b ** 2)
    cov2_b = k.cov2 * (b ** 2) if k.cov2 is not None else None
    nngp_b = k.nngp * (b ** 2)
    ntk_b = k.ntk * (b ** 2) if k.ntk is not None else None

    cov1_denom = 1 + 2 * cov1_b
    cov2_denom = None if cov2_b is None else 1 + 2 * cov2_b

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1_denom,
                                                      cov2_denom,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    factor = 2 / math.pi

    def nngp_ntk_fn(
        nngp: torch.Tensor,
        prod: torch.Tensor,
        ntk: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
      square_root = _sqrt(prod - 4 * nngp**2)
      nngp_out = factor * torch.atan2(2 * nngp, square_root)

      if ntk is not None:
        dot_sigma = 2 * factor / square_root
        ntk = ntk * dot_sigma

      return nngp_out, ntk

    def nngp_fn_diag(nngp: torch.Tensor) -> torch.Tensor:
      return factor * torch.atan2(nngp, _sqrt(nngp + 1. / 4))

    nngp_out, ntk_out = nngp_ntk_fn(nngp_b, prod12, ntk_b)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1_out = nngp_fn_diag(cov1_b)
      if cov2_b is not None:
        cov2_out = nngp_fn_diag(cov2_b)
      else:
        cov2_out = None
    else:
      cov1_out, _ = nngp_ntk_fn(cov1_b, prod11)
      if cov2_b is not None:
        cov2_out, _ = nngp_ntk_fn(cov2_b, prod22)
      else:
        cov2_out = None

    # scale by a and shift by c
    cov1_final = cov1_out * (a ** 2)
    if cov2_out is not None:
        cov2_final = cov2_out * (a ** 2)
    else:
        cov2_final = None
    nngp_final = nngp_out * (a ** 2)
    if ntk_out is not None:
        ntk_final = ntk_out * (a ** 2)
    else:
        ntk_final = None
        
    return k.replace(cov1=cov1_final, nngp=nngp_final, cov2=cov2_final, ntk=ntk_final)

  return _elementwise(fn, f'Erf({a}, {b}, {c})', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Sigmoid_like() -> InternalLayer:
  return Erf(a=0.5, b=1/2.4020563531719796, c=0.5)


@layer
@supports_masking(remask_kernel=False)
def Gabor() -> InternalLayer:
  def fn(x):
    return torch.exp(-x ** 2) * torch.sin(x)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    prod11, prod12, prod22 = get_diagonal_outer_prods(
        cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

    sum11, sum12, sum22 = get_diagonal_outer_prods(
        cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.add)

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
        t_dot = exp_left * (lhs + exp_right * rhs) / denom**(5. / 2)
        ntk = ntk * (t_dot / 2)

      nngp_out = exp_left * (exp_right - 1) / (2 * _sqrt(denom))
      return nngp_out, ntk

    def nngp_fn_diag(nngp):
      denom = 1 + 4 * nngp
      return (1 - torch.exp(-2 * nngp / denom)) / (2 * _sqrt(denom))

    nngp, ntk = nngp_ntk_fn(nngp, prod12, sum12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11, sum11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22, sum22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, 'Gabor', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def GeLU(approximate: bool = False) -> InternalLayer:
  def fn(x):
    return torch.nn.functional.gelu(x, approximate='tanh' if approximate else 'none')

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    cov1_plus_1 = cov1 + 1
    cov2_plus_1 = None if cov2 is None else cov2 + 1

    prod11_plus_1, prod12_plus_1, prod22_plus_1 = get_diagonal_outer_prods(
        cov1_plus_1, cov2_plus_1, k.diagonal_batch, k.diagonal_spatial, op.mul)
    prod11, prod12, prod22 = get_diagonal_outer_prods(
        cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

    def nngp_ntk_fn(nngp, prod, prod_plus_1, ntk=None):
      delta_squared = prod_plus_1 - nngp**2
      delta = _sqrt(delta_squared)
      angles = torch.atan2(nngp, delta)
      new_nngp = (nngp**2 + prod * delta_squared) / (prod_plus_1 * delta)
      new_nngp = new_nngp + nngp * angles
      new_nngp = new_nngp / (2 * math.pi)
      new_nngp = new_nngp + 0.25 * nngp

      if ntk is not None:
        second_term = 0.25 + angles / (2 * math.pi)
        first_term = 1 / delta_squared + (1. - prod) / prod_plus_1 + 1.
        first_term = first_term * (nngp / delta / (2. * math.pi))
        dot_sigma = first_term + second_term
        ntk = ntk * dot_sigma
      return new_nngp, ntk

    def nngp_fn_diag(nngp):
      square_root = torch.sqrt(1. + 2. * nngp)
      new_nngp = nngp / ((nngp + 1.) * square_root)
      new_nngp = new_nngp + torch.atan2(nngp, square_root) / 2.
      new_nngp = new_nngp / math.pi
      new_nngp = new_nngp + 0.25
      new_nngp = new_nngp * nngp
      return new_nngp

    nngp, ntk = nngp_ntk_fn(nngp, prod12, prod12_plus_1, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11, prod11_plus_1)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22, prod22_plus_1)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, 'GeLU', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Sin(a: float = 1., b: float = 1., c: float = 0.) -> InternalLayer:
  def fn(x):
    return a * torch.sin(b * x + c)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    sum11, sum12, sum22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.add)
    half_a_square = a**2 / 2.

    def nngp_ntk_fn(nngp, sum_, ntk=None):
      s1 = torch.exp(b ** 2 * (-0.5 * sum_ + nngp))
      s2 = torch.exp(b ** 2 * (-0.5 * sum_ - nngp)) * math.cos(2 * c)
      nngp_out = half_a_square * (s1 - s2)
      if ntk is not None:
        ntk = ntk * (half_a_square * b**2 * (s1 + s2))
      return nngp_out, ntk

    def nngp_fn_diag(nngp):
      return half_a_square * (1. - torch.exp(-2 * b ** 2 * nngp) * math.cos(2 * c))

    nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, sum11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, sum22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'Sin({a}, {b}, {c})', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Cos(a: float = 1., b: float = 1., c: float = 0.) -> InternalLayer:
  return Sin(a=a, b=b, c=c + math.pi / 2)


@layer
@supports_masking(remask_kernel=True)
def Rbf(gamma: float = 1.0) -> InternalLayer:
  def fn(x):
    return math.sqrt(2) * torch.sin(math.sqrt(2 * gamma) * x + math.pi / 4)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    sum11, sum12, sum22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.add)

    def nngp_ntk_fn(nngp, sum_, ntk):
      nngp_out = torch.exp(gamma * (-sum_ + 2 * nngp))
      if ntk is not None:
        ntk = ntk * (2 * gamma * nngp_out)
      return nngp_out, ntk

    def nngp_fn_diag(nngp):
      return torch.ones_like(nngp)

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
def Sign() -> InternalLayer:
  def fn(x):
    return torch.sign(x)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
    if ntk is not None:
      ntk = torch.zeros_like(ntk)
    _, prod12, _ = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)
    angles = _arctan2(_sqrt(prod12 - nngp**2), nngp, fill_zero=math.pi / 2)
    nngp_out = 1 - angles * 2 / math.pi
    cov1_out = torch.where(cov1 == 0., torch.zeros_like(cov1), torch.ones_like(cov1))
    if cov2 is not None:
      cov2_out = torch.where(cov2 == 0., torch.zeros_like(cov2), torch.ones_like(cov2))
    else:
      cov2_out = None
    k = k.replace(cov1=cov1_out, nngp=nngp_out, cov2=cov2_out, ntk=ntk)
    return k

  return _elementwise(fn, 'Sign', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Exp(a: float = 1, b: float = 1) -> InternalLayer:
  def fn(x):
    return a * torch.exp(b * x)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
    sum11, sum12, sum22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.add)

    def nngp_ntk_fn(nngp, sum_, ntk):
      nngp_out = torch.exp(b ** 2 * (sum_ / 2 + nngp))
      if ntk is not None:
        ntk = ntk * (b**2 * nngp_out)
      return nngp_out, ntk

    def nngp_fn_diag(nngp):
      return torch.exp(2 * b ** 2 * nngp)

    nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, sum11, None)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, sum22, None)

    # the ntk/cov will be proportional to a^2 since f(x) * f(y) has a^2
    if cov1 is not None: cov1 = cov1 * (a ** 2)
    if cov2 is not None: cov2 = cov2 * (a ** 2)
    if nngp is not None: nngp = nngp * (a ** 2)
    if ntk is not None: ntk = ntk * (a ** 2)
    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'Exp({a}, {b})', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Gaussian(a: float = 1, b: float = -1) -> InternalLayer:
  def fn(x):
    return a * torch.exp(b * x ** 2)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    cov1_denom = 1 - 2 * b * cov1
    cov2_denom = None if cov2 is None else 1 - 2 * b * cov2

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1_denom, cov2_denom, k.diagonal_batch, k.diagonal_spatial, op.mul)

    factor = 4 * b**2

    def nngp_ntk_fn(nngp, prod, ntk=None):
      det = _sqrt(prod - factor * nngp**2)

      if ntk is not None:
        ntk = ntk * (factor * nngp / det**3)

      nngp_out = 1 / det
      return nngp_out, ntk

    def nngp_fn_diag(nngp):
      return 1 / _sqrt(1 - 4 * b * nngp)

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    if cov1 is not None: cov1 = cov1 * (a ** 2)
    if cov2 is not None: cov2 = cov2 * (a ** 2)
    if nngp is not None: nngp = nngp * (a ** 2)
    if ntk is not None: ntk = ntk * (a ** 2)
    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'Gaussian({a}, {b})', kernel_fn)

@layer
@supports_masking(remask_kernel=True)
def ExpNormalized(gamma: float = 1, shift: float = -1, do_clip: bool = False) -> InternalLayer:
  def kernel_fn(k: Kernel) -> Kernel:
    cov1, cov2, nngp, ntk = k.cov1, k.cov2, k.nngp, k.ntk
    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)
    tol = 1e-30
    prod11 = _sqrt(prod11, tol)
    prod12 = _sqrt(prod12, tol)
    prod22 = _sqrt(prod22, tol) if prod22 is not None else None

    def exp_norm(cov, prod):
      if cov is not None:
        cov = cov / prod
        if do_clip:
          cov = torch.clamp(cov, min=-1., max=1.)
        cov = torch.exp(gamma * (cov + shift))
      return cov

    exp12 = exp_norm(nngp, prod12)

    return k.replace(
        nngp=prod12 * exp12,
        cov1=prod11 * exp_norm(cov1, prod11),
        cov2=None if cov2 is None else prod22 * exp_norm(cov2, prod22),
        ntk=ntk if ntk is None else gamma * ntk * exp12)

  return _elementwise(None, 'ExpNormalized', kernel_fn)


def _factorial(n: int) -> int:
  import functools, operator
  return functools.reduce(operator.mul, range(1, n + 1), 1)

def _double_factorial(n: int) -> int:
  import functools, operator
  if n <= 0: return 1
  return functools.reduce(operator.mul, range(n, 0, -2), 1)

@layer
@supports_masking(remask_kernel=True)
def Hermite(degree: int) -> InternalLayer:
  if degree < 0:
    raise NotImplementedError('`degree` must be a non-negative integer.')
  
  import numpy as np
  p = np.polynomial.hermite_e.herme2poly([0] * degree + [1])[::-1]
  coeff = _factorial(degree)**0.5

  def fn(x):
    res = torch.zeros_like(x)
    for i, c in enumerate(p[::-1]):
      if c != 0:
        res += float(c) * (x ** i)
    return res / float(coeff)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if ntk is not None:
      if degree == 0:
        ntk = torch.zeros_like(ntk)
      else:
        ntk = ntk * (degree * (nngp**(degree - 1)))

    def _power(mat):
      return mat**degree if mat is not None else None

    nngp, cov1, cov2 = map(_power, (nngp, cov1, cov2))
    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return k

  return _elementwise(fn, f'{degree}-Hermite polynomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def Monomial(degree: int) -> InternalLayer:
  if degree not in [0, 1, 2, 3, 4, 5]:
    raise NotImplementedError('The `degree` must be an integer between 0 and 5.')

  def fn(x):
    return x**degree

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1,
                                                      cov2,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    def nngp_ntk_fn(
        nngp: torch.Tensor,
        prod: torch.Tensor,
        ntk: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

      def nngp_fn(nngp: torch.Tensor, degree: int) -> torch.Tensor:
        if degree == -1:
          nngp = torch.zeros_like(nngp)
        elif degree == 0:
          nngp = torch.ones_like(nngp)
        elif degree == 1:
          pass
        elif degree == 2:
          nngp = 2 * nngp ** 2 + prod
        elif degree == 3:
          nngp = 6 * nngp ** 3 + 9 * nngp * prod
        elif degree == 4:
          nngp = 3 * (8 * nngp ** 4 + 3 * prod * (8 * nngp ** 2 + prod))
        elif degree == 5:
          nngp = 15 * nngp * (
              8 * nngp ** 4 + 5 * prod * (8 * nngp ** 2 + 3 * prod))
        else:
          raise NotImplementedError(degree)

        return nngp

      if ntk is not None:
        ntk = ntk * (degree**2 * nngp_fn(nngp, degree - 1))

      nngp = nngp_fn(nngp, degree)
      return nngp, ntk

    def nngp_fn_diag(nngp: torch.Tensor) -> torch.Tensor:
      return _double_factorial(2 * degree - 1) * nngp**degree

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return k

  return _elementwise(fn, f'{degree}-monomial', kernel_fn)
