# Copyright 2019 Google LLC
# Apache-2.0

from __future__ import annotations

from functools import partial
import operator
from typing import Generator, Iterable, Optional, Union, Tuple, Set

import torch
from torch import Tensor

from .batching import batch

from .empirical import _DEFAULT_NTK_FWD
from .empirical import _DEFAULT_NTK_J_RULES
from .empirical import _DEFAULT_NTK_S_RULES
from .empirical import DEFAULT_NTK_IMPLEMENTATION
from .empirical import empirical_kernel_fn
from .empirical import NtkImplementation

from .utils import utils
from .utils.typing import ApplyFn, Axes, EmpiricalGetKernelFn, Get, InitFn
from .utils.typing import MonteCarloKernelFn, NTTree, PyTree, VMapAxes


# ------- RNG utilities (PyTorch equivalent of jax.random.split) -------

def _ensure_generator(key: Union[int, Tensor, torch.Generator]) -> torch.Generator:
    """
    Accepts an int seed, a scalar tensor seed, or a torch.Generator and returns a Generator.
    """
    if isinstance(key, torch.Generator):
        return key
    if isinstance(key, Tensor):
        if key.numel() != 1:
            raise ValueError("Seed tensor must be scalar.")
        seed = int(key.item())
    elif isinstance(key, int):
        seed = key
    else:
        raise TypeError(f"Unsupported RNG key type: {type(key)}")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _split(gen: torch.Generator) -> Tuple[torch.Generator, torch.Generator]:
    """
    Statelessly derive two new generators from `gen` by drawing seeds from it.
    Deterministic given `gen`'s state; suitable replacement for jax.random.split(key, 2).
    """
    # Draw two seeds from the parent generator
    dev = utils._generator_device(gen)
    s1 = int(torch.randint(0, 2**63 - 1, (1,), generator=gen, dtype=torch.int64).item())
    s2 = int(torch.randint(0, 2**63 - 1, (1,), generator=gen, dtype=torch.int64).item())
    g1 = torch.Generator(device=dev); g1.manual_seed(s1)
    g2 = torch.Generator(device=dev); g2.manual_seed(s2)
    return g1, g2


# ------------------------ Single-sample wrapper ------------------------

def _sample_once_kernel_fn(
    kernel_fn: EmpiricalGetKernelFn,
    init_fn: InitFn,
    batch_size: int = 0,
    device_count: int = -1,
    store_on_device: bool = True,
):
    @partial(
        batch,
        batch_size=batch_size,
        device_count=device_count,
        store_on_device=store_on_device,
    )
    def kernel_fn_sample_once(
        x1: NTTree[Tensor],
        x2: Optional[NTTree[Tensor]],
        key: Union[int, Tensor, torch.Generator],
        get: Get,
        **apply_fn_kwargs,
    ):
        rng = _ensure_generator(key)
        init_key, dropout_key = _split(rng)

        # Shapes for init; utils.tree_map assumed to mirror original behavior
        shape = utils.tree_map(lambda x: tuple(x.shape), x1)
        _, params = init_fn(init_key, shape)

        # Empirical kernel: pass dropout RNG (generator) via kwargs
        return kernel_fn(x1, x2, get, params, rng=dropout_key, **apply_fn_kwargs)

    return kernel_fn_sample_once


# ----------------------- Multi-sample aggregator -----------------------

def _sample_many_kernel_fn(
    kernel_fn_sample_once,
    key: Union[int, Tensor, torch.Generator],
    n_samples: Set[int],
    get_generator: bool,
):
    base_gen = _ensure_generator(key)

    def normalize(sample: PyTree, n: int) -> PyTree:
        return utils.tree_map(lambda s: s / n, sample)

    def get_samples(
        x1: NTTree[Tensor],
        x2: Optional[NTTree[Tensor]],
        get: Get,
        **apply_fn_kwargs,
    ):
        gen = base_gen
        ker_sum = None
        max_n = max(n_samples)
        for n in range(1, max_n + 1):
            gen, split = _split(gen)
            one = kernel_fn_sample_once(x1, x2, split, get, **apply_fn_kwargs)
            if ker_sum is None:
                ker_sum = one
            else:
                ker_sum = utils.tree_map(operator.add, ker_sum, one)
            yield n, ker_sum

    if get_generator:
        @utils.get_namedtuple("MonteCarloKernel")
        def get_sampled_kernel(
            x1: NTTree[Tensor],
            x2: Optional[NTTree[Tensor]],
            get: Optional[Get] = None,
            **apply_fn_kwargs,
        ) -> Generator[Union[Tensor, tuple[Tensor, ...]], None, None]:
            for n, sample in get_samples(x1, x2, get, **apply_fn_kwargs):
                if n in n_samples:
                    yield normalize(sample, n)
    else:
        @utils.get_namedtuple("MonteCarloKernel")
        def get_sampled_kernel(
            x1: NTTree[Tensor],
            x2: Optional[NTTree[Tensor]],
            get: Optional[Get] = None,
            **apply_fn_kwargs,
        ) -> Union[Tensor, tuple[Tensor, ...]]:
            last_n = None
            last_sample = None
            for n, sample in get_samples(x1, x2, get, **apply_fn_kwargs):
                last_n, last_sample = n, sample
            return normalize(last_sample, last_n)

    return get_sampled_kernel


# ----------------------------- Public API -----------------------------

def monte_carlo_kernel_fn(
    init_fn: InitFn,
    apply_fn: ApplyFn,
    key: Union[int, Tensor, torch.Generator],
    n_samples: Union[int, Iterable[int]],
    batch_size: int = 0,
    device_count: int = -1,
    store_on_device: bool = True,
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = (),
    vmap_axes: Optional[VMapAxes] = None,
    implementation: Union[int, NtkImplementation] = DEFAULT_NTK_IMPLEMENTATION,
    _j_rules: bool = _DEFAULT_NTK_J_RULES,
    _s_rules: bool = _DEFAULT_NTK_S_RULES,
    _fwd: Optional[bool] = _DEFAULT_NTK_FWD,
) -> MonteCarloKernelFn:
    r"""
    Return a Monte Carlo sampler of NNGP/NTK kernels for an arbitrary function (PyTorch).
    The returned function is already batched/parallelized by `batch(...)`.
    """
    kwargs = dict(
        f=apply_fn,
        trace_axes=trace_axes,
        diagonal_axes=diagonal_axes,
        vmap_axes=vmap_axes,
        implementation=implementation,
        _s_rules=_s_rules,
        _j_rules=_j_rules,
        _fwd=_fwd,
    )

    kernel_fn = empirical_kernel_fn(**kwargs)

    kernel_fn_sample_once = _sample_once_kernel_fn(
        kernel_fn=kernel_fn,
        init_fn=init_fn,
        batch_size=batch_size,
        device_count=device_count,
        store_on_device=store_on_device,
    )

    n_samples_set, get_generator = _canonicalize_n_samples(n_samples)
    kernel_fn_mc = _sample_many_kernel_fn(
        kernel_fn_sample_once=kernel_fn_sample_once,
        key=key,
        n_samples=n_samples_set,
        get_generator=get_generator,
    )
    return kernel_fn_mc


# -------------------------- Helper: n_samples --------------------------

def _canonicalize_n_samples(
    n_samples: Union[int, Iterable[int]]
) -> tuple[Set[int], bool]:
    get_generator = True
    if isinstance(n_samples, int):
        get_generator = False
        n_samples = (n_samples,)

    if hasattr(n_samples, "__iter__"):
        n_samples = set(int(n) for n in n_samples)
        if not all(isinstance(n, int) for n in n_samples):
            raise ValueError(f"`n_samples` must contain only integers, got {n_samples}.")
        if any(n <= 0 for n in n_samples):
            raise ValueError(f"`n_samples` must be positive, got {n_samples}.")
    else:
        raise TypeError(f"`n_samples` must be either an integer or an iterable of integers, got {type(n_samples)}.")
    return n_samples, get_generator
