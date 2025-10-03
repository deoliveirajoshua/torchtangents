# batch_torch.py
# PyTorch port of Neural Tangents batching utilities (serial & parallel).
# Compatible with torch.compile (no graph-breaking Python side-effects in inner paths).

from __future__ import annotations
from functools import partial
from typing import Any, Callable, Iterable, Optional, TypeVar, Union, Tuple
import warnings
import math
import numpy as np
import torch

from .utils import utils
from .kernel import Kernel
from .typing import KernelFn, NTTree   # optional: mirror type aliases

_KernelFn = TypeVar('_KernelFn', bound=KernelFn)

# ------------------------
# helpers
# ------------------------

def _is_tensor_tree(x) -> bool:
    if x is None:
        return False
    return utils.tree_all(utils.tree_map(lambda y: isinstance(y, torch.Tensor), x))

def _scan(
    f: Callable[[Any, Any], Tuple[Any, Any]],
    init: Any,
    xs: Iterable[Any]
) -> Tuple[Any, Any]:
    """Unrolled scan matching the original lax.scan-like contract."""
    carry = init
    ys = []
    flat_xs, tree_def = utils.tree_flatten(xs)
    for flat_x in zip(*flat_xs):
        x = utils.tree_unflatten(tree_def, flat_x)
        carry, y = f(carry, x)
        ys.append(y)
    return carry, utils.tree_map(lambda *y: torch.stack(y, dim=0), *ys)

def _flatten_batch_dimensions(
    k: torch.Tensor,
    is_parallel: bool,
    discard_axis: Optional[int] = None
) -> torch.Tensor:
    """Flatten block-batched kernel layout back to (n1[, n2], ...)."""
    if discard_axis is not None:
        if not is_parallel:
            k = torch.take_along_dim(k, torch.tensor([0], device=k.device), dim=discard_axis)
            # `take_along_dim` keeps the dim; squeeze via reshape for stable compile
            new_shape = list(k.shape)
            new_shape.pop(discard_axis)
            return k.reshape(new_shape)
        # parallel
        if discard_axis == 1:
            return k.reshape(k.shape[0] * k.shape[1], *k.shape[2:])
        return k[0]
    else:
        if is_parallel:
            return k.reshape(k.shape[0] * k.shape[1], *k.shape[2:])
        # serial: blocks are (n1_batches, n2_batches, n1b, n2b, ...)
        perm = (0, 2, 1, 3) + tuple(range(4, k.ndim))
        k = k.permute(perm)
        return k.reshape(k.shape[0]*k.shape[1], k.shape[2]*k.shape[3], *k.shape[4:])

@utils.nt_tree_fn(nargs=1)
def _flatten_kernel_dict(
    k: dict[str, Any],
    x2_is_none: bool,
    is_parallel: bool
) -> dict[str, Any]:
    if 'nngp' in k:
        # infer per-block batch_size for shape1/shape2 update
        batch_size = {'1': k['nngp'].shape[0], '2': k['nngp'].shape[1]}

    if 'diagonal_batch' in k and not k['diagonal_batch']:
        raise NotImplementedError('Batching not implemented for diagonal_batch == False.')

    for key, value in k.items():
        if key == 'cov1':
            k[key] = _flatten_batch_dimensions(value, is_parallel, 1)
        elif key == 'cov2':
            if x2_is_none:
                k[key] = None
            else:
                k[key] = _flatten_batch_dimensions(value, is_parallel, 0)
        elif key == 'x1_is_x2':
            k[key] = value[(0,) * value.ndim]
        elif key == 'mask1':
            k[key] = None if value is None else _flatten_batch_dimensions(value, is_parallel, 1)
        elif key == 'mask2':
            if value is None or x2_is_none:
                k[key] = None
            else:
                k[key] = -_flatten_batch_dimensions(value, is_parallel, 0)
        elif key in ('shape1', 'shape2'):
            if key == 'shape2' and is_parallel:
                continue
            batch_axis = k['batch_axis']
            shape = value
            k[key] = (shape[:batch_axis] +
                      (shape[batch_axis] * batch_size[key[-1]],) +
                      shape[batch_axis + 1:])
        elif isinstance(value, torch.Tensor):
            k[key] = _flatten_batch_dimensions(value, is_parallel)
        else:
            pass
    return k

@utils.nt_tree_fn(nargs=1)
def _flatten_kernel(k: Kernel, x2_is_none: bool, is_parallel: bool) -> Kernel:
    if hasattr(k, '_asdict') and hasattr(k, '_replace'):
        return k._replace(**_flatten_kernel_dict(k._asdict(), x2_is_none, is_parallel))
    elif isinstance(k, Kernel):
        return Kernel(**_flatten_kernel_dict(k.asdict(), x2_is_none, is_parallel))
    elif isinstance(k, torch.Tensor):
        return _flatten_batch_dimensions(k, is_parallel)
    raise TypeError(f'Expected Kernel/tupledict/torch.Tensor, got {type(k)}')

@utils.nt_tree_fn()
def _set_cov2_to_none(k):
    if isinstance(k, Kernel):
        k = k.replace(cov2=None)
    return k

def _get_n_batches_and_batch_sizes(
    n1: int,
    n2: int,
    batch_size: int,
    device_count: int
) -> Tuple[int, int, int, int]:
    # mirror original constraints
    max_serial_batch_size = math.gcd(n1, n2) // max(device_count, 1)
    n2_batch_size = min(batch_size, max_serial_batch_size)
    if n2_batch_size != batch_size:
        warnings.warn(
            f'Batch size reduced from requested {batch_size} to effective {n2_batch_size} to fit the dataset.'
        )
    n1_batch_size = n2_batch_size * max(device_count, 1)

    n1_batches, rag1 = divmod(n1, n1_batch_size)
    if rag1:
        msg = (f'Number of rows must divide batch size. Found n1={n1}, batch_size={n1_batch_size}.')
        if device_count > 1:
            msg += f' Device parallelism expands batch size by {device_count}.'
        raise ValueError(msg)

    n2_batches, rag2 = divmod(n2, n2_batch_size)
    if rag2:
        raise ValueError(f'Number of cols must divide batch size. Found n2={n2}, batch_size={n2_batch_size}.')
    return n1_batches, n1_batch_size, n2_batches, n2_batch_size

# ------------------------
# public entry
# ------------------------

def batch(
    kernel_fn: _KernelFn,
    batch_size: int = 0,
    device_count: int = -1,
    store_on_device: bool = True
) -> _KernelFn:
    """
    Torch version of nt.batch: serial batching with optional multi-GPU row parallelism.
    - kernel_fn: callable(x1, x2, *args, **kwargs) or callable(kernel_in, *args, **kwargs)
    - batch_size: per-device column batch size
    - device_count: -1 => all CUDA devices; 0 => no device parallelism; N => use first N CUDA devices
    - store_on_device: keep outputs on device (True) or move to CPU as they are produced (False)
    """
    use_multidevice = False
    if device_count == -1:
        use_multidevice = torch.cuda.is_available() and torch.cuda.device_count() > 1
    elif device_count > 0:
        use_multidevice = torch.cuda.is_available() and torch.cuda.device_count() >= device_count

    use_serial = bool(batch_size)

    if use_multidevice:
        kernel_fn = _parallel(kernel_fn, use_serial=use_serial, device_count=device_count)
    # else: just use original function

    if not use_serial:
        return kernel_fn

    return _serial(kernel_fn, batch_size=batch_size, store_on_device=store_on_device)

# ------------------------
# serial batching
# ------------------------

def _serial(
    kernel_fn: _KernelFn,
    batch_size: int,
    store_on_device: bool = True
) -> _KernelFn:

    # optionally wrap to move final result to CPU
    if not store_on_device:
        _kernel_fn = kernel_fn
        @utils.wraps(_kernel_fn)
        def kernel_fn(*args, **kwargs):
            out = _kernel_fn(*args, **kwargs)
            return utils.nt_tree_fn(nargs=1)(lambda t: t.to('cpu') if isinstance(t, torch.Tensor) else t)(out)

    flatten = partial(_flatten_kernel, is_parallel=False)

    def _serial_x1(
        x1: NTTree[torch.Tensor],
        x2: Optional[NTTree[Optional[torch.Tensor]]] = None,
        *args,
        **kwargs
    ) -> NTTree[Kernel]:
        x2_is_none = utils.all_none(x2)
        if x2_is_none:
            x2 = x1

        @utils.nt_tree_fn(reduce=lambda x: x[0])
        def _get_n1_n2(x1, x2):
            return x1.shape[0], x2.shape[0]
        n1, n2 = _get_n1_n2(x1, x2)

        device_count = max(getattr(kernel_fn, 'device_count', 1), 1)
        n1_batches, n1b, n2_batches, n2b = _get_n_batches_and_batch_sizes(n1, n2, batch_size, device_count)

        @utils.nt_tree_fn(nargs=1)
        def _batch_input(x: torch.Tensor, bc: int, bs: int):
            return x.reshape(bc, bs, *x.shape[1:])

        x1s = _batch_input(x1, n1_batches, n1b)
        x2s = _batch_input(x2, n2_batches, n2b)

        kwargs_np1, kwargs_np2, kwargs_other = {}, {}, {}
        for k, v in kwargs.items():
            if _is_tensor_tree(v):
                # convention: for tensor-tree kwargs that mirror (x1, x2)
                assert isinstance(v, tuple) and len(v) == 2, f"Tensor-tree kwarg '{k}' must be a (v1, v2) tuple."
                v1 = v[0].reshape(n1_batches, n1b, *v[0].shape[1:])
                v2 = v[1].reshape(n2_batches, n2b, *v[1].shape[1:])
                kwargs_np1[k], kwargs_np2[k] = v1, v2
            else:
                kwargs_other[k] = v

        def row_fn(_, x1_block):
            return _, _scan(col_fn, x1_block, (x2s, kwargs_np2))[1]

        def col_fn(x1_block, x2_block):
            x1_block, kw1 = x1_block
            x2_block, kw2 = x2_block
            kw_merge = {**kwargs_other, **{kk: (kw1[kk], kw2[kk]) for kk in kw1}}
            return (x1_block, kw1), kernel_fn(x1_block, x2_block, *args, **kw_merge)

        _, kernel = _scan(row_fn, 0, (x1s, kwargs_np1))
        return flatten(kernel, x2_is_none)

    def _serial_kernel(k_in: NTTree[Kernel], *args, **kwargs) -> NTTree[Kernel]:
        def _get_n1_n2(k: NTTree[Kernel]) -> Tuple[int, int]:
            if utils.is_list_or_tuple(k):
                return _get_n1_n2(k[0])
            if isinstance(k, Kernel):
                return k.nngp.shape[0], k.nngp.shape[1]
            raise TypeError(type(k), Kernel)

        n1, n2 = _get_n1_n2(k_in)
        device_count = max(getattr(kernel_fn, 'device_count', 1), 1)
        n1_batches, n1b, n2_batches, n2b = _get_n_batches_and_batch_sizes(n1, n2, batch_size, device_count)

        n1s = torch.arange(0, n1, n1b)
        n2s = torch.arange(0, n2, n2b)

        @utils.nt_tree_fn(nargs=1)
        def _slice_kernel(k: Kernel, start1: int, start2: int) -> Kernel:
            return k.slice(slice(start1, start1 + n1b), slice(start2, start2 + n2b))

        kwargs_np1, kwargs_np2, kwargs_other = {}, {}, {}
        for key, v in kwargs.items():
            if _is_tensor_tree(v):
                assert isinstance(v, tuple) and len(v) == 2
                v1 = v[0].reshape(n1_batches, n1b, *v[0].shape[1:])
                v2 = v[1].reshape(n2_batches, n2b, *v[1].shape[1:])
                kwargs_np1[key], kwargs_np2[key] = v1, v2
            else:
                kwargs_other[key] = v

        def row_fn(_, n1_start):
            return _, _scan(col_fn, n1_start, (n2s, kwargs_np2))[1]

        def col_fn(n1_start, n2_start):
            n1_start, kw1 = n1_start
            n2_start, kw2 = n2_start
            kw_merge = {**kwargs_other, **{kk: (kw1[kk], kw2[kk]) for kk in kw1}}
            sub = _slice_kernel(k_in, int(n1_start), int(n2_start))
            return (n1_start, kw1), kernel_fn(sub, *args, **kw_merge)

        cov2_is_none = utils.nt_tree_fn(reduce=lambda acc, x: acc and x)(
            lambda k: k.cov2 is None
        )(k_in, True)
        _, kout = _scan(row_fn, 0, (n1s, kwargs_np1))
        if cov2_is_none:
            kout = _set_cov2_to_none(kout)
        return _flatten_kernel(kout, cov2_is_none, False)

    @utils.wraps(kernel_fn)
    def serial_fn(x1_or_kernel: Union[NTTree[torch.Tensor], NTTree[Kernel]],
                  x2: Optional[NTTree[Optional[torch.Tensor]]] = None,
                  *args, **kwargs) -> NTTree[Kernel]:
        if utils.is_nt_tree_of(x1_or_kernel, torch.Tensor):
            return _serial_x1(x1_or_kernel, x2, *args, **kwargs)
        elif utils.is_nt_tree_of(x1_or_kernel, Kernel):
            if x2 is not None:
                raise ValueError(f'`x2` must be None for compositional calls, got {type(x2)}.')
            return _serial_kernel(x1_or_kernel, *args, **kwargs)
        else:
            raise TypeError(x1_or_kernel, type(x1_or_kernel))

    return serial_fn

# ------------------------
# parallel (multi-GPU row sharding)
# ------------------------

def _parallel(
    kernel_fn: _KernelFn,
    use_serial: bool = True,
    device_count: int = -1,
) -> _KernelFn:
    """Row-parallel batching over CUDA devices (manual scatter/gather)."""
    if device_count == -1:
        device_count = torch.cuda.device_count()

    if device_count <= 1:
        # nothing to parallelize
        return kernel_fn

    devices = [torch.device(f'cuda:{i}') for i in range(device_count)]

    def _get_n_per_device(n1: int) -> Tuple[int, int]:
        n1_per_device, rag = divmod(n1, device_count)
        if n1_per_device and rag:
            raise ValueError(f'Dataset size ({n1}) must divide number of devices ({device_count}).')
        elif not n1_per_device:
            # fewer rows than devices: use only `rag` devices, 1 row each
            return 1, rag
        return n1_per_device, device_count

    def _move_tree(x, device):
        return utils.nt_tree_fn(nargs=1)(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t)(x)

    def parallel_x1(x1, x2=None, *args, **kwargs):
        # Only split over x1 rows; broadcast x2 to every device
        x2_is_none = utils.all_none(x2)
        if x2_is_none:
            x2 = x1

        def _get_n(x):
            if utils.is_list_or_tuple(x):
                return _get_n(x[0])
            return x.shape[0]

        n1 = _get_n(x1)
        n1_per_dev, used_dev = _get_n_per_device(n1)
        used_devices = devices[:used_dev] if used_dev else devices

        # shard x1
        x1_chunks = utils.nt_tree_fn(nargs=1)(
            lambda t: t.chunk(used_dev or device_count, dim=0)  # equal chunks
        )(x1)

        # broadcast x2 to chosen devices
        x2_dev = [_move_tree(x2, dev) for dev in used_devices]

        # split tensor-tree kwargs that mirror (x1, x2)
        kwargs_dev_list = []
        for dev_idx, dev in enumerate(used_devices):
            kd: dict[str, Any] = {}
            for k, v in kwargs.items():
                if _is_tensor_tree(v):
                    assert isinstance(v, tuple) and len(v) == 2
                    v1 = utils.nt_tree_fn(nargs=1)(lambda t: t.chunk(used_dev or device_count, dim=0)[dev_idx])(v[0])
                    v2 = _move_tree(v[1], dev)
                    kd[k] = (v1, v2)
                else:
                    kd[k] = v
            kwargs_dev_list.append(kd)

        # launch per-device calls
        results = []
        for dev_idx, dev in enumerate(used_devices):
            x1_local = _move_tree(utils.nt_tree_fn(nargs=1)(lambda t: t)(  # pick dev_idx chunk
                utils.nt_tree_fn(nargs=1)(lambda t: t[dev_idx])(x1_chunks)
            ), dev)
            res = kernel_fn(x1_local, x2_dev[dev_idx], *args, **kwargs_dev_list[dev_idx])
            results.append(res)

        # concat on rows and flatten parallel structure
        # results is a PyTree of Kernels with leading device axis already eliminated by concat here
        def _cat(a_list):
            if isinstance(a_list[0], torch.Tensor):
                return torch.cat(a_list, dim=0)
            return a_list

        merged = utils.nt_tree_zip(*results)
        merged = utils.nt_tree_fn(reduce=lambda *xs: xs)(
            lambda *xs: _cat(xs)
        )(merged)

        return _flatten_kernel(merged, x2_is_none, True)

    def parallel_kernel(k_in: NTTree[Kernel], *args, **kwargs):
        # split over cov1 rows; cov2 is broadcast
        @utils.nt_tree_fn(reduce=lambda shapes: shapes[0])
        def get_sizes(k: Kernel):
            n1 = k.cov1.shape[0]
            n2 = n1 if k.cov2 is None else k.cov2.shape[0]
            return n1, n2
        n1, _ = get_sizes(k_in)
        n1_per_dev, used_dev = _get_n_per_device(n1)
        used_devices = devices[:used_dev] if used_dev else devices

        # build per-device kernel slices (row shards)
        def _slice_kernel_dev(k: Kernel, dev_idx: int) -> Kernel:
            start = dev_idx * n1_per_dev
            stop = start + n1_per_dev
            return k.slice(slice(start, stop), slice(0, k.nngp.shape[1]))

        k_shards = [utils.nt_tree_fn(nargs=1)(lambda K: _slice_kernel_dev(K, i))(k_in) for i in range(used_dev or device_count)]
        k_shards = [utils.nt_tree_fn(nargs=1)(lambda t: t.to(dev) if isinstance(t, torch.Tensor) else t)(ks)
                    for ks, dev in zip(k_shards, used_devices)]

        # kwargs: mirror (x1,x2) convention
        kwargs_dev_list = []
        for dev_idx, dev in enumerate(used_devices):
            kd: dict[str, Any] = {}
            for k, v in kwargs.items():
                if _is_tensor_tree(v):
                    assert isinstance(v, tuple) and len(v) == 2
                    v1 = utils.nt_tree_fn(nargs=1)(lambda t: t.chunk(used_dev or device_count, dim=0)[dev_idx])(v[0])
                    v2 = _move_tree(v[1], dev)
                    kd[k] = (v1, v2)
                else:
                    kd[k] = v
            kwargs_dev_list.append(kd)

        outs = []
        for ks, dev, kd in zip(k_shards, used_devices, kwargs_dev_list):
            outs.append(kernel_fn(ks, *args, **kd))

        cov2_is_none = utils.nt_tree_fn(reduce=lambda acc, x: acc and x)(
            lambda k: k.cov2 is None
        )(k_in, True)

        # concat on rows (device chunks)
        def _cat_rows(items):
            if isinstance(items[0], torch.Tensor):
                return torch.cat(items, dim=0)
            return items

        merged = utils.nt_tree_zip(*outs)
        merged = utils.nt_tree_fn(reduce=lambda *xs: xs)(
            lambda *xs: _cat_rows(xs)
        )(merged)

        if cov2_is_none:
            merged = _set_cov2_to_none(merged)
        return _flatten_kernel(merged, cov2_is_none, True)

    @utils.wraps(kernel_fn)
    def parallel_fn(x1_or_kernel, x2=None, *args, **kwargs):
        if utils.is_nt_tree_of(x1_or_kernel, torch.Tensor):
            return parallel_x1(x1_or_kernel, x2, *args, **kwargs)
        elif utils.is_nt_tree_of(x1_or_kernel, Kernel):
            if x2 is not None:
                raise ValueError('`x2` must be None for compositional kernel calls.')
            return parallel_kernel(x1_or_kernel, *args, **kwargs)
        else:
            raise NotImplementedError(f'Unsupported input tree: {type(x1_or_kernel)}')

    # expose for serial() to discover effective device_count
    parallel_fn.device_count = device_count
    return parallel_fn