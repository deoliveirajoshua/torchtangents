"""
Lightweight torch-tangents: core infinite-width kernel transforms for MLPs (Dense + ReLU)

This is a minimal, self-contained subset implementing:
- Kernel object carrying NNGP (nngp) and NTK (ntk) matrices
- Dense layer kernel transform (σ_w^2, σ_b^2) under NTK parameterization
- ReLU analytic kernel (Cho & Saul arccos) and its derivative
- A Sequential-like kernel pipeline and an adapter for torch.nn.Sequential (Linear + ReLU)
- A simple GP posterior inference utility (NNGP posterior mean & cov)

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class Kernel:
	nngp: torch.Tensor
	ntk: torch.Tensor

	def to(self, device=None, dtype=None) -> "Kernel":
		nngp = self.nngp
		ntk = self.ntk
		if device is not None:
			nngp = nngp.to(device)
			ntk = ntk.to(device)
		if dtype is not None:
			nngp = nngp.to(dtype=dtype)
			ntk = ntk.to(dtype=dtype)
		return Kernel(nngp=nngp, ntk=ntk)


def _make_input_kernel(X: torch.Tensor) -> Kernel:
	if X.dim() != 2:
		raise ValueError("X must be 2D: (n_samples, n_features)")
	n, d = X.shape
	K = X @ X.t()
	K = K / float(d)
	return Kernel(nngp=K.clone(), ntk=torch.zeros_like(K))


def _relu_arccos(c: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
	c = torch.clamp(c, -1.0 + eps, 1.0 - eps)
	acos = torch.acos(c)
	sqrt_term = torch.sqrt(torch.clamp(1.0 - c * c, min=0.0))
	C = (1.0 / math.pi) * (sqrt_term + (math.pi - acos) * c)
	Cdot = 0.5 / math.pi * (math.pi - acos)
	return C, Cdot


class Dense:
	def __init__(self, sigma_w2: float = 1.0, sigma_b2: float = 0.0):
		self.sigma_w2 = float(sigma_w2)
		self.sigma_b2 = float(sigma_b2)

	def transform(self, kernel: Kernel) -> Kernel:
		# Linear layer without nonlinearity: analytic linear transform
		Sigma = kernel.nngp
		Theta = kernel.ntk
		Sigma_lin = self.sigma_w2 * Sigma + self.sigma_b2
		# For linear activation, the derivative contribution is just sigma_w2
		Sigma_dot = torch.tensor(self.sigma_w2, dtype=Sigma.dtype, device=Sigma.device)
		Theta_next = Theta * Sigma_dot + Sigma_lin
		return Kernel(nngp=Sigma_lin, ntk=Theta_next)


class DenseReLU:
	def __init__(self, sigma_w2: float = 1.0, sigma_b2: float = 0.0, eps: float = 1e-6):
		self.sigma_w2 = float(sigma_w2)
		self.sigma_b2 = float(sigma_b2)
		self.eps = eps

	def transform(self, kernel: Kernel) -> Kernel:
		Sigma = kernel.nngp
		Theta = kernel.ntk
		diag = torch.diag(Sigma)
		s = torch.sqrt(torch.ger(diag, diag))
		corr = Sigma / (s + self.eps)
		C, Cdot = _relu_arccos(corr, eps=self.eps)
		Sigma_next = self.sigma_w2 * s * C + self.sigma_b2
		Sigma_dot = self.sigma_w2 * Cdot
		Theta_next = Theta * Sigma_dot + Sigma_next
		return Kernel(nngp=Sigma_next, ntk=Theta_next)


class Conv2dReLU:
	"""Convolutional layer (weight sharing) followed by ReLU analytic kernel.

	Minimal implementation: expects Kernel.nngp shape (n, n, H_in, W_in) and
	returns (n, n, H_out, W_out). The conv is implemented as a sum-patch using
	an all-ones kernel to represent weight sharing over channels.
	"""

	def __init__(self, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), sigma_w2: float = 1.0, sigma_b2: float = 0.0, eps: float = 1e-6):
		self.kernel_size = tuple(kernel_size)
		self.stride = tuple(stride)
		self.padding = tuple(padding)
		self.dilation = tuple(dilation)
		self.sigma_w2 = float(sigma_w2)
		self.sigma_b2 = float(sigma_b2)
		self.eps = float(eps)

	def _conv_spatial(self, X: torch.Tensor) -> torch.Tensor:
		"""2D conv over spatial dims for an (n,n,H,W) tensor using kernel of ones.

		Returns convolved tensor of shape (n,n,H_out,W_out).
		"""
		n = X.shape[0]
		H, W = X.shape[-2], X.shape[-1]
		Xflat = X.view(n * n, 1, H, W)
		weight = torch.ones((1, 1, self.kernel_size[0], self.kernel_size[1]), dtype=X.dtype, device=X.device)
		import torch.nn.functional as F

		out = F.conv2d(Xflat, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation)
		H_out, W_out = out.shape[-2], out.shape[-1]
		out = out.view(n, n, H_out, W_out)
		return out

	def transform(self, kernel: Kernel) -> Kernel:
		Sigma_in = kernel.nngp
		Theta_in = kernel.ntk

		Sigma_lin = self.sigma_w2 * self._conv_spatial(Sigma_in) + self.sigma_b2

		# number of samples
		n = Sigma_in.shape[0]

		# per-position variances (diagonal across sample pairs)
		diag = torch.diagonal(Sigma_lin, dim1=0, dim2=1)  # expected (n, H_out, W_out)

		# ensure sample axis is first (robust to possible ordering)
		if diag.shape[0] != n:
			# find axis with length n and move to front
			axes = [i for i, s in enumerate(diag.shape) if s == n]
			if not axes:
				raise RuntimeError("Could not locate sample axis in diagonal tensor")
			axis = axes[0]
			perm = (axis,) + tuple(i for i in range(len(diag.shape)) if i != axis)
			diag = diag.permute(perm)

		diag_flat = diag.contiguous().view(n, -1)

		# flatten spatial positions per-sample: (n, P) where P = H_out * W_out
		diag_flat = diag.contiguous().view(n, -1)

		# clamp small variances to avoid negative/zero under sqrt
		diag_flat_clamped = diag_flat.clamp(min=self.eps)

		# s_flat has shape (n, n, P): sqrt(var_i_p * var_j_p)
		s_flat = torch.sqrt(diag_flat_clamped[:, None, :] * diag_flat_clamped[None, :, :])

		# reshape back to spatial grid (n, n, H_out, W_out)
		s = s_flat.view(n, n, diag.shape[1], diag.shape[2])

		corr = Sigma_lin / (s + self.eps)
		C, Cdot = _relu_arccos(corr, eps=self.eps)

		Sigma_next = s * C
		Sigma_dot = Cdot

		Theta_lin = self.sigma_w2 * self._conv_spatial(Theta_in)
		Theta_next = Theta_lin * Sigma_dot + Sigma_next
		return Kernel(nngp=Sigma_next, ntk=Theta_next)


class Conv2dReLUExact(Conv2dReLU):
	"""Exact convolutional kernel (single-sum over matching patch offsets).

	This follows the common convolutional infinite-width kernel formula where
	the linear pre-activation covariance at an output position is the average
	over matching patch offsets (i.e. a single sum, not a double sum). Given
	that `kernel_from_images` stores per-position covariance averaged over
	channels, the channel dimension cancels and the correct normalization is by
	the kernel area (kH * kW).
	"""

	def transform(self, kernel: Kernel) -> Kernel:
		Sigma_in = kernel.nngp
		Theta_in = kernel.ntk

		kH, kW = self.kernel_size
		kernel_area = float(kH * kW)

		# linear pre-activation covariance: average over matching offsets
		conv_raw = self._conv_spatial(Sigma_in)
		Sigma_lin = (self.sigma_w2 / kernel_area) * conv_raw + self.sigma_b2

		# per-position variances (diagonal across sample pairs)
		n = Sigma_in.shape[0]
		diag = torch.diagonal(Sigma_lin, dim1=0, dim2=1)
		if diag.shape[0] != n:
			axes = [i for i, s in enumerate(diag.shape) if s == n]
			if not axes:
				raise RuntimeError("Could not locate sample axis in diagonal tensor")
			axis = axes[0]
			perm = (axis,) + tuple(i for i in range(len(diag.shape)) if i != axis)
			diag = diag.permute(perm)

		diag_flat = diag.contiguous().view(n, -1)
		diag_flat_clamped = diag_flat.clamp(min=self.eps)
		s_flat = torch.sqrt(diag_flat_clamped[:, None, :] * diag_flat_clamped[None, :, :])
		s = s_flat.view(n, n, diag.shape[1], diag.shape[2])

		corr = Sigma_lin / (s + self.eps)
		C, Cdot = _relu_arccos(corr, eps=self.eps)

		Sigma_next = s * C
		Sigma_dot = Cdot

		Theta_lin = (self.sigma_w2 / kernel_area) * self._conv_spatial(Theta_in)
		Theta_next = Theta_lin * Sigma_dot + Sigma_next
		return Kernel(nngp=Sigma_next, ntk=Theta_next)


class GlobalAvgPool:
	"""Global average pooling over spatial dims. Reduces Kernel.nngp shape (n,n,H,W) -> (n,n)."""

	def transform(self, kernel: Kernel) -> Kernel:
		Sigma = kernel.nngp
		Theta = kernel.ntk
		Sigma_pool = Sigma.mean(dim=(-2, -1))
		Theta_pool = Theta.mean(dim=(-2, -1))
		return Kernel(nngp=Sigma_pool, ntk=Theta_pool)


class KernelSequential:
	def __init__(self, layers: Sequence):
		self.layers = list(layers)

	def apply(self, k: Kernel) -> Kernel:
		cur = k
		for layer in self.layers:
			cur = layer.transform(cur)
		return cur


class Residual:
	"""Generic residual block: output = input + branch(input)

	The branch is any object with a .transform(Kernel) -> Kernel method (for
	example a KernelSequential or a single layer). To avoid double-counting
	parameter contributions in the NTK, the branch is evaluated with an input
	Kernel whose NTK is zeroed; the resulting branch.ntk is then added to the
	input's NTK.

	An optional projection `proj` can be provided to map the input shape to the
	branch output shape (for example a 1x1 conv). If `proj` is None, the code
	assumes shapes already match and uses the identity projection.
	"""

	def __init__(self, branch, proj=None):
		self.branch = branch
		self.proj = proj

	def transform(self, kernel: Kernel) -> Kernel:
		Sigma_in = kernel.nngp
		Theta_in = kernel.ntk

		# prepare branch input kernel with zero NTK to compute branch parameter contribution
		zero_ntk = torch.zeros_like(Sigma_in)
		branch_in = Kernel(nngp=Sigma_in, ntk=zero_ntk)
		# branch may be a single-layer with .transform or a KernelSequential with .apply
		if hasattr(self.branch, "transform"):
			branch_out = self.branch.transform(branch_in)
		elif hasattr(self.branch, "apply"):
			branch_out = self.branch.apply(branch_in)
		else:
			raise TypeError("branch must implement transform(kernel) or apply(kernel)")

		# if a projection is provided, apply it to the input (projection has its own parameters)
		if self.proj is not None:
			if hasattr(self.proj, "transform"):
				proj_out = self.proj.transform(kernel)
			elif hasattr(self.proj, "apply"):
				proj_out = self.proj.apply(kernel)
			else:
				raise TypeError("proj must implement transform(kernel) or apply(kernel)")
			Sigma_proj = proj_out.nngp
			Theta_proj = proj_out.ntk
		else:
			Sigma_proj = Sigma_in
			Theta_proj = Theta_in

		# output is input (possibly projected) plus branch output
		Sigma_out = Sigma_proj + branch_out.nngp
		Theta_out = Theta_proj + branch_out.ntk
		return Kernel(nngp=Sigma_out, ntk=Theta_out)


class LayerNormKernel:
	"""LayerNorm-like transform applied to kernels.

	For vector kernels (n,n) this normalizes each sample by its own variance so
	the resulting per-sample variance equals 1. For spatial kernels
	(n,n,H,W) it normalizes per-sample per-position variances.
	"""

	def __init__(self, eps: float = 1e-6):
		self.eps = float(eps)

	def transform(self, kernel: Kernel) -> Kernel:
		Sigma = kernel.nngp
		Theta = kernel.ntk
		if Sigma.dim() == 2:
			diag = torch.diag(Sigma)
			s = torch.sqrt(torch.ger(diag, diag)).clamp(min=self.eps)
			Sigma_norm = Sigma / s
			Theta_norm = Theta / s
			return Kernel(nngp=Sigma_norm, ntk=Theta_norm)
		elif Sigma.dim() == 4:
			# spatial (n, n, H, W)
			n = Sigma.shape[0]
			diag = torch.diagonal(Sigma, dim1=0, dim2=1)
			if diag.shape[0] != n:
				axes = [i for i, s in enumerate(diag.shape) if s == n]
				if not axes:
					raise RuntimeError("Could not locate sample axis in diagonal tensor")
				axis = axes[0]
				perm = (axis,) + tuple(i for i in range(len(diag.shape)) if i != axis)
				diag = diag.permute(perm)

			diag_flat = diag.contiguous().view(n, -1).clamp(min=self.eps)
			s_flat = torch.sqrt(diag_flat[:, None, :] * diag_flat[None, :, :])
			s = s_flat.view(n, n, diag.shape[1], diag.shape[2])
			Sigma_norm = Sigma / s
			Theta_norm = Theta / s
			return Kernel(nngp=Sigma_norm, ntk=Theta_norm)
		else:
			raise NotImplementedError("LayerNormKernel only supports 2D or 4D kernels")


class BatchNormKernel:
	"""BatchNorm-like transform using batch statistics.

	For vector kernels, divides by the mean variance across samples. For
	spatial kernels, divides per-position by the mean over samples of the
	position variances.
	"""

	def __init__(self, eps: float = 1e-6):
		self.eps = float(eps)

	def transform(self, kernel: Kernel) -> Kernel:
		Sigma = kernel.nngp
		Theta = kernel.ntk
		if Sigma.dim() == 2:
			diag = torch.diag(Sigma)
			mean_var = diag.mean().clamp(min=self.eps)
			Sigma_norm = Sigma / mean_var
			Theta_norm = Theta / mean_var
			return Kernel(nngp=Sigma_norm, ntk=Theta_norm)
		elif Sigma.dim() == 4:
			# per-position mean variance over samples
			diag = torch.diagonal(Sigma, dim1=0, dim2=1)
			if diag.shape[0] != Sigma.shape[0]:
				axes = [i for i, s in enumerate(diag.shape) if s == Sigma.shape[0]]
				if not axes:
					raise RuntimeError("Could not locate sample axis in diagonal tensor")
				axis = axes[0]
				perm = (axis,) + tuple(i for i in range(len(diag.shape)) if i != axis)
				diag = diag.permute(perm)
			# diag: (n, H, W) -> mean over n -> (H, W)
			mean_var = diag.mean(dim=0).clamp(min=self.eps)
			Sigma_norm = Sigma / mean_var
			Theta_norm = Theta / mean_var
			return Kernel(nngp=Sigma_norm, ntk=Theta_norm)
		else:
			raise NotImplementedError("BatchNormKernel only supports 2D or 4D kernels")


class PreActResBlockDense:
	"""Pre-activation ResNet block for dense/vector inputs.

	Structure: Norm -> DenseReLU -> Norm -> DenseReLU, with residual add.
	norm: one of {None, 'layer', 'batch'}
	"""

	def __init__(self, hidden_sigma_w2: float = 1.0, hidden_sigma_b2: float = 0.0, norm: Optional[str] = 'layer', eps: float = 1e-6):
		self.eps = eps
		self.norm = norm
		layers = []
		if norm == 'layer':
			layers.append(LayerNormKernel(eps=eps))
		elif norm == 'batch':
			layers.append(BatchNormKernel(eps=eps))
		# first linear+relu
		layers.append(DenseReLU(sigma_w2=hidden_sigma_w2, sigma_b2=hidden_sigma_b2, eps=eps))
		if norm == 'layer':
			layers.append(LayerNormKernel(eps=eps))
		elif norm == 'batch':
			layers.append(BatchNormKernel(eps=eps))
		layers.append(DenseReLU(sigma_w2=hidden_sigma_w2, sigma_b2=hidden_sigma_b2, eps=eps))
		self.branch = KernelSequential(layers)

	def transform(self, kernel: Kernel) -> Kernel:
		return Residual(self.branch).transform(kernel)


class PreActResBlockConv:
	"""Pre-activation ResNet block for convolutional spatial kernels.

	Structure: Norm -> Conv2dReLUExact -> Norm -> Conv2dReLUExact, with residual add.
	norm: one of {None, 'layer', 'batch'}
	"""

	def __init__(self, kernel_size=(3,3), stride=(1,1), padding=(1,1), hidden_sigma_w2: float = 1.0, hidden_sigma_b2: float = 0.0, norm: Optional[str] = 'layer', eps: float = 1e-6):
		self.eps = eps
		layers = []
		if norm == 'layer':
			layers.append(LayerNormKernel(eps=eps))
		elif norm == 'batch':
			layers.append(BatchNormKernel(eps=eps))
		layers.append(Conv2dReLUExact(kernel_size=kernel_size, stride=stride, padding=padding, sigma_w2=hidden_sigma_w2, sigma_b2=hidden_sigma_b2, eps=eps))
		if norm == 'layer':
			layers.append(LayerNormKernel(eps=eps))
		elif norm == 'batch':
			layers.append(BatchNormKernel(eps=eps))
		layers.append(Conv2dReLUExact(kernel_size=kernel_size, stride=stride, padding=padding, sigma_w2=hidden_sigma_w2, sigma_b2=hidden_sigma_b2, eps=eps))
		self.branch = KernelSequential(layers)

	def transform(self, kernel: Kernel) -> Kernel:
		return Residual(self.branch).transform(kernel)


def kernel_from_inputs(X: torch.Tensor) -> Kernel:
	return _make_input_kernel(X)


def kernel_from_images(X: torch.Tensor) -> Kernel:
	"""Create initial spatial kernel from image batch X.

	X: (n, c, H, W). Returns Kernel with nngp shape (n, n, H, W) where entries are
	per-position inner-products across channels normalized by channel dimension.
	"""
	if X.dim() != 4:
		raise ValueError("X must be 4D: (n_samples, channels, H, W)")
	n, c, H, W = X.shape
	Xflat = X.view(n, c, H * W)
	# compute K_ij_p = sum_c X[i,c,p] * X[j,c,p]
	K_flat = torch.einsum("icp,jcp->ijp", Xflat, Xflat)
	K_flat = K_flat / float(c)
	K = K_flat.view(n, n, H, W)
	ntk = torch.zeros_like(K)
	return Kernel(nngp=K, ntk=ntk)


def kernel_from_sequences(X: torch.Tensor) -> Kernel:
	"""Create initial sequence kernel from input batch X.

	X: (n, L, d) where L is sequence length and d is feature dim.
	Returns Kernel with nngp shape (n, n, L, L) where entries are per-position
	inner-products across features normalized by feature dimension.
	"""
	if X.dim() != 3:
		raise ValueError("X must be 3D: (n_samples, seq_len, features)")
	n, L, d = X.shape
	Xflat = X.view(n, L, d)
	# compute K_ij_pq = sum_f X[i,p,f] * X[j,q,f]
	K_flat = torch.einsum("ipf,jqf->ijpq", Xflat, Xflat)
	K_flat = K_flat / float(d)
	ntk = torch.zeros_like(K_flat)
	return Kernel(nngp=K_flat, ntk=ntk)


def lift_sequential(nnseq) -> KernelSequential:
	import torch.nn as nn

	layers = []
	for m in nnseq:
		if isinstance(m, nn.Linear):
			sigma_w2 = float(getattr(m, "weight_var", 1.0))
			sigma_b2 = float(getattr(m, "bias_var", 0.0)) if m.bias is not None else 0.0
			layers.append(Dense(sigma_w2=sigma_w2, sigma_b2=sigma_b2))
		elif isinstance(m, nn.ReLU):
			if not layers:
				raise ValueError("ReLU cannot be the first layer in lifted Sequential")
			last = layers.pop()
			if isinstance(last, Dense):
				layers.append(DenseReLU(sigma_w2=last.sigma_w2, sigma_b2=last.sigma_b2))
			else:
				raise NotImplementedError("Only Dense -> ReLU pattern is supported in adapter")
		else:
			raise NotImplementedError(f"Module {type(m)} not supported by lift_sequential")

	return KernelSequential(layers)


class AttentionLinear:
	"""A practical attention-like kernel transform.

	This is a pragmatic, analytically tractable approximation useful for
	experimenting with attention-style position mixing in kernel space.

	Behavior and assumptions:
	- Expects input Kernel.nngp of shape (n, n, L, L) (sequence kernel).
	- Two modes are provided:
	  * 'fixed': use a fixed attention matrix `attn` of shape (L, L) that mixes
		positions. If `attn` is None, uses uniform averaging (all entries 1/L).
	  * 'softmax': approximate data-dependent softmax attention by computing a
		mean similarity matrix across samples from the input kernel and applying
		a softmax (row-wise) to obtain an attention matrix. This is an
		approximation (we treat the resulting attention as deterministic and
		then apply a linear mixing of positions).
	- The transform treats the attention weights as fixed (deterministic) and
	  models the learned value projection as a linear transform with variance
	  `sigma_w2` and bias `sigma_b2`. Under this model the analytic update is
	  Sigma_next = sigma_w2 * (A Sigma_in A^T) + sigma_b2 and the NTK uses
	  Sigma_dot = sigma_w2.

	This is not a full treatment of learned Q/K softmax attention in the
	infinite-width limit, but is a useful, testable building block.
	"""

	def __init__(self, sigma_w2: float = 1.0, sigma_b2: float = 0.0, attn_mode: str = 'fixed', attn: Optional[torch.Tensor] = None, eps: float = 1e-6):
		self.sigma_w2 = float(sigma_w2)
		self.sigma_b2 = float(sigma_b2)
		self.attn_mode = attn_mode
		self.attn = attn
		if attn_mode not in ('fixed', 'softmax'):
			raise ValueError("attn_mode must be 'fixed' or 'softmax'")
		self.eps = float(eps)

	def _mix_positions(self, Sigma_in: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
		# Sigma_in: (n, n, L, L), A: (L, L)
		# compute Sigma_lin_{i,j,p,q} = sum_{u,v} A[p,u] * A[q,v] * Sigma_in[i,j,u,v]
		# implement with einsum for clarity
		return torch.einsum('pu,qv,ijuv->ijpq', A, A, Sigma_in)

	def transform(self, kernel: Kernel) -> Kernel:
		Sigma_in = kernel.nngp
		Theta_in = kernel.ntk
		if Sigma_in.dim() != 4:
			raise ValueError('AttentionLinear expects a 4D sequence kernel (n,n,L,L)')
		n, _, L, _ = Sigma_in.shape

		if self.attn_mode == 'fixed':
			if self.attn is None:
				A = torch.full((L, L), 1.0 / float(L), dtype=Sigma_in.dtype, device=Sigma_in.device)
			else:
				A = self.attn.to(device=Sigma_in.device, dtype=Sigma_in.dtype)
		else:
			# softmax-based approximate attention: compute mean within-sample similarities
			# compute per-sample covariance diag-normalized similarity matrix S_i (L,L)
			# average across samples to get S_avg then softmax row-wise
			# Use diagonal elements (variances) for normalization
			S_acc = torch.zeros((L, L), dtype=Sigma_in.dtype, device=Sigma_in.device)
			for i in range(n):
				Si = Sigma_in[i, i]  # shape (L, L)
				var = torch.diag(Si).clamp(min=self.eps)
				denom = torch.sqrt(var[:, None] * var[None, :])
				corr = Si / (denom + self.eps)
				S_acc += corr
			S_avg = S_acc / float(n)
			# temperature scaling by sqrt(d) is omitted here; user can tune via eps
			A = torch.softmax(S_avg, dim=-1)

		Sigma_lin = self._mix_positions(Sigma_in, A)
		Sigma_next = self.sigma_w2 * Sigma_lin + self.sigma_b2
		Sigma_dot = torch.tensor(self.sigma_w2, dtype=Sigma_in.dtype, device=Sigma_in.device)
		Theta_lin = self.sigma_w2 * self._mix_positions(Theta_in, A)
		Theta_next = Theta_lin * Sigma_dot + Sigma_next
		return Kernel(nngp=Sigma_next, ntk=Theta_next)


class AttentionAnalytic:
	"""Analytic attention approximation using moment-matching for softmax.

	This class approximates the expected softmax attention by using a Gaussian
	moment approximation for the exponentiated pre-softmax scores. Assumptions:
	- Q = W_q X and K = W_k X with W_q,W_k having iid Gaussian entries of
	  variance sigma_wq2/d and sigma_wk2/d respectively (zero mean).
	- The pre-softmax score z_{p,q} = Q_p · K_q / sqrt(d_k) has mean ≈ 0 and
	  variance approximated by a simple factor proportional to the product of
	  per-position variances of the input features. This yields a tractable
	  approximation for E[exp(z)] ≈ exp(0.5 Var[z]). We then normalize rows to
	  obtain an attention matrix A per sample.

	While not a complete theoretical derivation of softmax attention in the
	infinite-width limit, this provides a more principled (moment-based)
	attention kernel that depends on the input kernel diagonals and projection
	variances.
	"""

	def __init__(self, sigma_wq2: float = 1.0, sigma_wk2: float = 1.0, sigma_wv2: float = 1.0, sigma_bv2: float = 0.0, eps: float = 1e-6):
		self.sigma_wq2 = float(sigma_wq2)
		self.sigma_wk2 = float(sigma_wk2)
		self.sigma_wv2 = float(sigma_wv2)
		self.sigma_bv2 = float(sigma_bv2)
		self.eps = float(eps)

	def _compute_attention_matrices(self, Sigma_in: torch.Tensor) -> torch.Tensor:
		# Sigma_in: (n, n, L, L)
		n = Sigma_in.shape[0]
		L = Sigma_in.shape[2]


		# per-sample per-position variance: diag_s_p = Sigma_in[s,s,p,p]
		# Use advanced indexing to robustly extract diag[s,p] = Sigma_in[s, s, p, p]
		s_idx = torch.arange(n, device=Sigma_in.device)
		p_idx = torch.arange(L, device=Sigma_in.device)
		S, P = torch.meshgrid(s_idx, p_idx, indexing='ij')
		diag_flat = Sigma_in[S, S, P, P].contiguous().view(n, L).clamp(min=self.eps)

		# simple variance approximation for z_{p,q} in a sample: Var[z] ≈ (sigma_wq2 * sigma_wk2) * (diag_p * diag_q) / d^2
		# Here we normalize by an implicit feature-dim scaling; the kernel `Sigma_in` was already normalized by d.
		# So we use the product of diagonals directly and a scaling factor
		scale = float(self.sigma_wq2 * self.sigma_wk2)

		# compute outer products per sample: (n, L, L)
		# var_s[p,q] = scale * diag_s[p] * diag_s[q]
		var = scale * (diag_flat[:, :, None] * diag_flat[:, None, :])

		# approximate E[exp(z)] per sample and position using log-normal moment
		# E[exp(z)] ≈ exp(0.5 * var)
		exp_m = torch.exp(0.5 * var)

		# normalize rows to form attention matrices A_s (n, L, L)
		A = exp_m / (exp_m.sum(dim=-1, keepdim=True) + self.eps)
		return A

	def transform(self, kernel: Kernel) -> Kernel:
		Sigma_in = kernel.nngp
		Theta_in = kernel.ntk
		if Sigma_in.dim() != 4:
			raise ValueError('AttentionAnalytic expects a 4D sequence kernel (n,n,L,L)')

		# compute per-sample attention matrices A (n, L, L)
		A = self._compute_attention_matrices(Sigma_in)

		# mix positions using sample-dependent A: Sigma_lin_{i,j,p,q} = sum_{u,v} A_i[p,u] * A_j[q,v] * Sigma_in[i,j,u,v]
		Sigma_lin = torch.einsum('ipu,jqv,ijuv->ijpq', A, A, Sigma_in)

		Sigma_next = self.sigma_wv2 * Sigma_lin + self.sigma_bv2
		Sigma_dot = torch.tensor(self.sigma_wv2, dtype=Sigma_in.dtype, device=Sigma_in.device)

		Theta_lin = self.sigma_wv2 * torch.einsum('ipu,jqv,ijuv->ijpq', A, A, Theta_in)
		Theta_next = Theta_lin * Sigma_dot + Sigma_next
		return Kernel(nngp=Sigma_next, ntk=Theta_next)


class AttentionMultiHead:
	"""Multi-head attention kernel composed from multiple AttentionAnalytic heads.

	This class composes `num_heads` independent analytic attention heads and
	averages their NNGP/NTK outputs. Each head may have its own projection
	variances; scalars are broadcast across heads.

	Notes:
	- Under exchangeability and iid projections, the multi-head average reduces
	  to the single-head analytic result when all head params are identical.
	- This provides a practical, testable implementation that integrates with
	  the existing kernel transforms.
	"""

	def __init__(self, num_heads: int = 8, sigma_wq2=1.0, sigma_wk2=1.0, sigma_wv2=1.0, sigma_bv2=0.0, eps: float = 1e-6):
		self.num_heads = int(num_heads)
		# allow scalars or iterables per-head
		def _to_list(x):
			if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
				lst = list(x)
				if len(lst) != self.num_heads:
					raise ValueError('per-head parameter list must have length num_heads')
				return lst
			else:
				return [float(x)] * self.num_heads

		self.sigma_wq2 = _to_list(sigma_wq2)
		self.sigma_wk2 = _to_list(sigma_wk2)
		self.sigma_wv2 = _to_list(sigma_wv2)
		self.sigma_bv2 = _to_list(sigma_bv2)
		self.eps = float(eps)

	def transform(self, kernel: Kernel) -> Kernel:
		# apply each analytic head and average results
		outs_nngp = []
		outs_ntk = []
		for h in range(self.num_heads):
			head = AttentionAnalytic(sigma_wq2=self.sigma_wq2[h], sigma_wk2=self.sigma_wk2[h], sigma_wv2=self.sigma_wv2[h], sigma_bv2=self.sigma_bv2[h], eps=self.eps)
			out = head.transform(kernel)
			outs_nngp.append(out.nngp)
			outs_ntk.append(out.ntk)

		Sigma_avg = torch.stack(outs_nngp, dim=0).mean(dim=0)
		Theta_avg = torch.stack(outs_ntk, dim=0).mean(dim=0)
		return Kernel(nngp=Sigma_avg, ntk=Theta_avg)


def lift_module(mod: "torch.nn.Module", exact_conv: bool = True) -> KernelSequential:
	"""Lift a torch.nn.Module (or nn.Sequential) to a KernelSequential.

	This supports a practical subset of modules commonly used in vision and MLPs:
	- nn.Linear (+ nn.ReLU)
	- nn.Conv2d (+ nn.ReLU) -> Conv2dReLUExact (or Conv2dReLU when exact_conv=False)
	- nn.ReLU
	- nn.LayerNorm -> LayerNormKernel
	- nn.BatchNorm2d -> BatchNormKernel
	- nn.AdaptiveAvgPool2d(output_size=1) -> GlobalAvgPool
	- nn.Identity
	- nested nn.Sequential

	The function attempts to combine Linear/Conv + following ReLU into a single
	kernel transform (DenseReLU or Conv2dReLUExact).
	"""
	import torch.nn as nn

	def _lift_list(mods):
		layers = []
		i = 0
		while i < len(mods):
			m = mods[i]
			# handle nested Sequential
			if isinstance(m, nn.Sequential):
				layers.extend(_lift_list(list(m)))
				i += 1
				continue

			# MultiheadAttention -> AttentionMultiHead
			if isinstance(m, nn.MultiheadAttention):
				# determine num_heads and use default projection variances
				nh = getattr(m, 'num_heads', 1)
				layers.append(AttentionMultiHead(num_heads=nh))
				i += 1
				continue

			# Linear (+ ReLU)
			if isinstance(m, nn.Linear):
				sigma_w2 = float(getattr(m, "weight_var", 1.0))
				sigma_b2 = float(getattr(m, "bias_var", 0.0)) if m.bias is not None else 0.0
				# if next is ReLU, produce DenseReLU and skip next
				if (i + 1) < len(mods) and isinstance(mods[i + 1], nn.ReLU):
					layers.append(DenseReLU(sigma_w2=sigma_w2, sigma_b2=sigma_b2))
					i += 2
				else:
					layers.append(Dense(sigma_w2=sigma_w2, sigma_b2=sigma_b2))
					i += 1
				continue

			# Conv2d (+ ReLU)
			if isinstance(m, nn.Conv2d):
				sigma_w2 = float(getattr(m, "weight_var", 1.0))
				sigma_b2 = float(getattr(m, "bias_var", 0.0)) if m.bias is not None else 0.0
				kernel_size = getattr(m, "kernel_size", (3, 3))
				stride = getattr(m, "stride", (1, 1))
				padding = getattr(m, "padding", (0, 0))
				# if next is ReLU, combine
				use_relu = (i + 1) < len(mods) and isinstance(mods[i + 1], nn.ReLU)
				ConvClass = Conv2dReLUExact if exact_conv else Conv2dReLU
				# always append conv-relu combined transform (skip next if ReLU)
				layers.append(ConvClass(kernel_size=kernel_size, stride=stride, padding=padding, sigma_w2=sigma_w2, sigma_b2=sigma_b2))
				i += 2 if use_relu else 1
				continue

			# ReLU standalone (if not combined with previous)
			if isinstance(m, nn.ReLU):
				# If previous layer is Dense, convert it
				if layers and isinstance(layers[-1], Dense):
					last = layers.pop()
					layers.append(DenseReLU(sigma_w2=last.sigma_w2, sigma_b2=last.sigma_b2))
				else:
					# otherwise ReLU is a no-op at kernel level (nonlinearity only inside transforms)
					pass
				i += 1
				continue

			# LayerNorm -> LayerNormKernel
			if isinstance(m, nn.LayerNorm):
				layers.append(LayerNormKernel())
				i += 1
				continue

			# BatchNorm2d -> BatchNormKernel
			if isinstance(m, nn.BatchNorm2d):
				layers.append(BatchNormKernel())
				i += 1
				continue

			# AdaptiveAvgPool2d(1) -> GlobalAvgPool
			if isinstance(m, nn.AdaptiveAvgPool2d):
				out_size = getattr(m, "output_size", None)
				if out_size == 1 or out_size == (1, 1):
					layers.append(GlobalAvgPool())
					i += 1
					continue

			# Identity -> no-op
			if isinstance(m, nn.Identity):
				i += 1
				continue

			raise NotImplementedError(f"Module {type(m)} not supported by lift_module")

		return layers

	# accept a top-level Sequential or a Module; if Module, try to iterate children
	if isinstance(mod, nn.Sequential):
		layer_list = _lift_list(list(mod))
	else:
		# try to lift as a flat list of children
		children = list(mod.children())
		if children:
			layer_list = _lift_list(children)
		else:
			raise NotImplementedError("lift_module requires a Sequential or a module with child modules")

	return KernelSequential(layer_list)


def gp_posterior(K_xx: torch.Tensor, y: torch.Tensor, K_xs_x: Optional[torch.Tensor] = None, noise: float = 1e-6):
	jitter = 1e-6
	n = K_xx.shape[0]
	device = K_xx.device
	K = K_xx + (noise + jitter) * torch.eye(n, device=device, dtype=K_xx.dtype)
	L = torch.linalg.cholesky(K)
	alpha = torch.cholesky_solve(y, L)
	if K_xs_x is None:
		mean = K_xx @ alpha
		return mean
	else:
		mean = K_xs_x @ alpha
		K_sx = K_xs_x
		v = torch.cholesky_solve(K_sx.t(), L)
		K_ss = K_xs_x @ K_sx.t()
		cov = K_ss - K_xs_x @ v
		return mean, cov




def empirical_ntk(model: torch.nn.Module, inputs: torch.Tensor, device: Optional[torch.device] = None, param_filter=None) -> torch.Tensor:
	"""Compute the empirical (finite-width) NTK for `model` on `inputs`.

	This computes NTK_{ij} = sum_p <grad_theta f(x_i)_p, grad_theta f(x_j)_p>
	where the sum is over output coordinates p (i.e. trace over outputs).

	Notes:
	- Memory heavy: intended for small models / batches used in tests and demos.
	- `param_filter` (callable) can be provided to select parameters.

	Returns a (n, n) NTK matrix.
	"""

	was_training = model.training
	model.eval()

	if device is None:
		device = inputs.device

	# move inputs and model to device
	inputs = inputs.to(device)
	model = model.to(device)

	# forward
	outputs = model(inputs)

	if not torch.is_tensor(outputs):
		if isinstance(outputs, (list, tuple)) and len(outputs) >= 1:
			outputs = outputs[0]
		else:
			raise ValueError("Model output must be a tensor or (tensor, ...) tuple")

	if outputs.dim() == 0:
		raise ValueError("Model produced a scalar output with no batch dimension")

	n = outputs.shape[0]
	outputs_flat = outputs.reshape(n, -1)
	m = outputs_flat.shape[1]

	# collect parameters
	params = [p for p in model.parameters() if p.requires_grad and (param_filter(p) if param_filter is not None else True)]
	if not params:
		raise ValueError("No trainable parameters found for NTK computation")

	# compute per-sample gradient vectors
	grads = []
	for i in range(n):
		# concatenate gradients for all output coordinates for sample i
		pieces = []
		for k in range(m):
			out_ik = outputs_flat[i, k]
			g = torch.autograd.grad(out_ik, params, retain_graph=True, create_graph=False, allow_unused=True)
			flat = []
			for t in g:
				if t is None:
					# zero-length placeholder
					flat.append(torch.zeros(0, device=device))
				else:
					flat.append(t.detach().reshape(-1))
			pieces.append(torch.cat(flat) if flat else torch.tensor([], device=device))
		v = torch.cat(pieces) if pieces else torch.empty((0,), device=device)
		grads.append(v)

	G = torch.stack(grads, dim=0)
	NTK = G @ G.t()

	# restore model training state
	if was_training:
		model.train()

	return NTK


def empirical_ntk_efficient(model: torch.nn.Module, inputs: torch.Tensor, device: Optional[torch.device] = None, param_filter=None, param_chunk_size: Optional[int] = None) -> torch.Tensor:
	"""Memory-efficient exact empirical NTK.

	This implementation computes the NTK by accumulating per-output Jacobian
	blocks. For each output coordinate k it builds the (n, P) Jacobian matrix
	(where P is total number of parameters retained by `param_filter`) and
	accumulates NTK += J_k @ J_k^T. This keeps peak memory at O(n * P) instead
	of O(n * m * P) (where m is number of output coordinates) used by the
	concatenation approach.

	Suitable for moderate P and small batch sizes n. Still exact (not
	approximate).
	"""

	was_training = model.training
	model.eval()

	if device is None:
		device = inputs.device

	inputs = inputs.to(device)
	model = model.to(device)

	outputs = model(inputs)
	if not torch.is_tensor(outputs):
		if isinstance(outputs, (list, tuple)) and len(outputs) >= 1:
			outputs = outputs[0]
		else:
			raise ValueError("Model output must be a tensor or (tensor, ...) tuple")

	if outputs.dim() == 0:
		raise ValueError("Model produced a scalar output with no batch dimension")

	n = outputs.shape[0]
	outputs_flat = outputs.reshape(n, -1)
	m = outputs_flat.shape[1]

	params = [p for p in model.parameters() if p.requires_grad and (param_filter(p) if param_filter is not None else True)]
	if not params:
		raise ValueError("No trainable parameters found for NTK computation")

	# total parameter count
	P = sum(p.numel() for p in params)
	NTK = torch.zeros((n, n), device=device, dtype=outputs_flat.dtype)

	# If chunking requested, partition params into groups whose total numel <= param_chunk_size
	if param_chunk_size is not None and param_chunk_size > 0:
		groups = []
		cur = []
		cur_count = 0
		for p in params:
			sz = p.numel()
			if cur and cur_count + sz > param_chunk_size:
				groups.append(cur)
				cur = [p]
				cur_count = sz
			else:
				cur.append(p)
				cur_count += sz
		if cur:
			groups.append(cur)
	else:
		groups = [params]

	# accumulate per-output Jacobian blocks, chunked over parameter groups
	for k in range(m):
		for grp in groups:
			# build Jk_chunk of shape (n, P_grp)
			P_grp = sum(p.numel() for p in grp)
			Jk_chunk = torch.empty((n, P_grp), device=device, dtype=outputs_flat.dtype)
			for i in range(n):
				g = torch.autograd.grad(outputs_flat[i, k], grp, retain_graph=True, create_graph=False, allow_unused=True)
				flat = []
				for p, t in zip(grp, g):
					if t is None:
						flat.append(torch.zeros(p.numel(), device=device, dtype=outputs_flat.dtype))
					else:
						flat.append(t.detach().reshape(-1))
				Jk_chunk[i] = torch.cat(flat)
			NTK += Jk_chunk @ Jk_chunk.t()

	if was_training:
		model.train()

	return NTK


__all__ = [
	"Kernel",
	"kernel_from_inputs",
	"kernel_from_images",
    "kernel_from_sequences",
	"DenseReLU",
	"Dense",
	"Conv2dReLU",
	"Conv2dReLUExact",
	"LayerNormKernel",
	"BatchNormKernel",
	"PreActResBlockDense",
	"PreActResBlockConv",
	"GlobalAvgPool",
	"Residual",
	"KernelSequential",
	"lift_sequential",
    "gp_posterior",
    "AttentionLinear",
	"AttentionAnalytic",
	"AttentionMultiHead",
	"empirical_ntk",
	"empirical_ntk_efficient",
]



