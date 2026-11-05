import torch
from torch.nn.functional import avg_pool1d, avg_pool2d, avg_pool3d, pad
from ..utils import utils
from ..utils.kernel import Kernel
from ..utils.typing import InternalLayerMasked
from .requirements import Bool, Diagonal, layer, requires, supports_masking

from typing import Sequence, Optional
import math

class _Pooling:
  AVG = 'AVG'
  SUM = 'SUM'

def _GlobalPool(
    pool_type: str,
    batch_axis: int,
    channel_axis: int,
) -> InternalLayerMasked:
  if pool_type == _Pooling.AVG:
    # pool_fn expects (x, axis, mask). mean_and_var normally handles it.
    def pool_fn(x, axis, mask=None):
      if mask is not None:
        mask = mask.to(x.dtype)
        mask = mask.expand_as(x)
        # multiply by mask, then sum, then divide by sum of mask
        num = torch.sum(x * mask, dim=axis)
        den = torch.sum(mask, dim=axis)
        # Avoid division by zero
        return num / torch.clamp(den, min=1e-12)
      return torch.mean(x, dim=axis)
  elif pool_type == _Pooling.SUM:
    def pool_fn(x, axis, mask=None):
      if mask is not None:
        mask = mask.to(x.dtype)
        mask = mask.expand_as(x)
        x = x * mask
      return torch.sum(x, dim=axis)
  else:
    raise ValueError(f'Invalid pooling type {pool_type}.')

  def init_fn(rng, input_shape):
    ndim = len(input_shape)
    non_spatial_axes = (batch_axis % ndim, channel_axis % ndim)
    output_shape = tuple(input_shape[i] for i in range(ndim)
                         if i in non_spatial_axes)
    return output_shape, ()

  def apply_fn(params, inputs, mask=None, **kwargs):
    non_spatial_axes = (batch_axis % inputs.ndim, channel_axis % inputs.ndim)
    spatial_axes = tuple(i for i in range(inputs.ndim)
                         if i not in non_spatial_axes)
    out = pool_fn(inputs, spatial_axes, mask)
    return out

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal(input=Bool.MAYBE, output=Bool.YES))
  def kernel_fn(k: Kernel, **kwargs):
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def _pool(mat, batch_ndim, mask=None):
      if mat is None:
        return mat
      spatial_axes = tuple(range(batch_ndim, mat.ndim))
      out = pool_fn(mat, axis=spatial_axes, mask=mask)
      if k.diagonal_spatial and pool_type == _Pooling.AVG:
        out = out / utils.size_at(mat, spatial_axes)
      return out

    mask11, mask12, mask22 = k._get_mask_prods(k.mask1, k.mask2)

    cov1 = _pool(cov1, 1 if k.diagonal_batch else 2, mask11)
    cov2 = None if cov2 is None else (cov1 if (cov2 is cov1 or k.x1_is_x2) else _pool(cov2, 1 if k.diagonal_batch else 2, mask22))
    nngp = _pool(nngp, 2, mask12)
    ntk = _pool(ntk, 2, mask12)

    ndim = len(k.shape1)
    batch_first = batch_axis % ndim < channel_axis % ndim
    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     batch_axis=0 if batch_first else 1,
                     channel_axis=1 if batch_first else 0,
                     diagonal_spatial=False)

  def mask_fn(mask, input_shape):
    non_spatial_axes = (batch_axis % len(input_shape),
                        channel_axis % len(input_shape))
    spatial_axes = tuple(i for i in range(len(input_shape))
                         if i not in non_spatial_axes)
    if mask is not None:
      mask = torch.all(mask, dim=spatial_axes)
    return mask

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def GlobalAvgPool(
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
) -> InternalLayerMasked:
  return _GlobalPool(_Pooling.AVG, batch_axis, channel_axis)


@layer
@supports_masking(remask_kernel=False)
def GlobalSumPool(
    batch_axis: int = 0,
    channel_axis: int = 1, # in pytorch conv, channel is the second dimension, in neural-tangents, it's the last dimension
) -> InternalLayerMasked:
  return _GlobalPool(_Pooling.SUM, batch_axis, channel_axis)

@layer
@supports_masking(remask_kernel=True)
def AvgPool(
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = "VALID",
    normalize_edges: bool = False,
    batch_axis: int = 0,
    channel_axis: int = 1,
) -> InternalLayerMasked:
  return _Pool(_Pooling.AVG, window_shape, strides, padding, normalize_edges, batch_axis, channel_axis)

@layer
@supports_masking(remask_kernel=True)
def SumPool(
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = "VALID",
    batch_axis: int = 0,
    channel_axis: int = 1,
) -> InternalLayerMasked:
  return _Pool(_Pooling.SUM, window_shape, strides, padding, False, batch_axis, channel_axis)

def _Pool(
    pool_type: str,
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]],
    padding: str,
    normalize_edges: bool,
    batch_axis: int,
    channel_axis: int,
) -> InternalLayerMasked:
  strides = strides or (1,) * len(window_shape)

  def _same_pad_for_filter_shape(x, window_shape, strides, spatial_axes, value=0.0):
      padding_tuple = []
      axes_list = list(spatial_axes)
      for i in range(x.ndim-1, -1, -1):
          if i in axes_list:
              sp_idx = axes_list.index(i)
              stride = strides[sp_idx]
              w_shape = window_shape[sp_idx]
              dim_size = x.shape[i]
              pad_tot = max((math.ceil(dim_size / stride) - 1) * stride + w_shape - dim_size, 0)
              pad_front = pad_tot // 2
              pad_back = pad_tot - pad_front
              padding_tuple.extend([pad_front, pad_back])
          else:
              padding_tuple.extend([0, 0])
      return pad(x, tuple(padding_tuple), mode='constant', value=value)

  def init_fn(rng, input_shape):
      ndim = len(input_shape)
      non_sp_axes = (batch_axis % ndim, channel_axis % ndim)
      sp_axes = [i for i in range(ndim) if i not in non_sp_axes]
      out_shape = list(input_shape)
      for i, sp_axis in enumerate(sp_axes):
          w = window_shape[i]
          s = strides[i]
          D = input_shape[sp_axis]
          if padding == 'SAME' or padding == 'CIRCULAR':
              out_shape[sp_axis] = math.ceil(D / s)
          else:
              out_shape[sp_axis] = math.ceil((D - w + 1) / s)
      return tuple(out_shape), ()

  def apply_fn(params, inputs, mask=None, **kwargs):
      non_sp_axes = (batch_axis % inputs.ndim, channel_axis % inputs.ndim)
      sp_axes = tuple(i for i in range(inputs.ndim) if i not in non_sp_axes)
      x = inputs

      if padding == 'SAME':
          x = _same_pad_for_filter_shape(x, window_shape, strides, sp_axes)

      pool_dict = {1: avg_pool1d, 2: avg_pool2d, 3: avg_pool3d}
      
      old_batch = batch_axis % inputs.ndim
      old_channel = channel_axis % inputs.ndim
      
      dims = list(range(inputs.ndim))
      dims.remove(old_batch)
      dims.remove(old_channel)
      
      perm_in = [old_batch, old_channel] + dims
      x = x.permute(*perm_in)
      
      fn = pool_dict[len(sp_axes)]
      divisor = math.prod(window_shape)
      
      res = fn(
          x,
          kernel_size=window_shape,
          stride=strides,
          padding=0,
          divisor_override=divisor if (not normalize_edges and pool_type == _Pooling.AVG) else None
      )
      
      if pool_type == _Pooling.SUM:
          res = res * divisor
          
      inv_perm = [0] * inputs.ndim
      for i, p in enumerate(perm_in):
          inv_perm[p] = i
      res = res.permute(*inv_perm)
      return res

  @requires(batch_axis=batch_axis, channel_axis=channel_axis, diagonal_spatial=Diagonal(input=Bool.MAYBE, output=Bool.MAYBE))
  def kernel_fn(k: Kernel, **kwargs) -> Kernel:
      cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

      def _pool_kernel(mat, b_ndim):
          if mat is None or getattr(mat, "ndim", 0) == 0:
              return mat
          
          ndim = mat.ndim
          if k.diagonal_spatial:
              sp_axes_len = ndim - b_ndim
              pool_dict = {1: torch.nn.functional.avg_pool1d, 2: torch.nn.functional.avg_pool2d, 3: torch.nn.functional.avg_pool3d}
              fn = pool_dict[sp_axes_len]
              
              flat_b = math.prod(mat.shape[:b_ndim])
              x = mat.reshape((flat_b, 1) + mat.shape[b_ndim:])
              if padding == 'SAME':
                  sp_axes = tuple(range(2, 2 + sp_axes_len))
                  x = _same_pad_for_filter_shape(x, window_shape, strides, sp_axes)
              
              divisor = math.prod(window_shape)
              
              if sp_axes_len == 1:
                  # avg_pool1d doesn't support divisor_override
                  res = fn(x, kernel_size=window_shape, stride=strides, padding=0)
                  if not normalize_edges and pool_type == _Pooling.AVG:
                      # Manually override divisor if needed
                      # This is an approximation for normalize_edges=False
                      res = res * (math.prod(window_shape)) # normalize_edges=False behavior in NT
              else:
                  res = fn(x, kernel_size=window_shape, stride=strides, padding=0, 
                           divisor_override=divisor if (not normalize_edges and pool_type == _Pooling.AVG) else None).to(mat.dtype)
              if pool_type == _Pooling.SUM:
                  res = res * divisor
              
              return res.reshape(mat.shape[:b_ndim] + res.shape[2:])
          else:
              out = mat
              is_reversed = getattr(k, 'is_reversed', False)
              cur_window = window_shape[::-1] if is_reversed else window_shape
              cur_strides = strides[::-1] if is_reversed else strides

              for i in range(ndim - 1, b_ndim, -2):
                  sp_i = (i - b_ndim) // 2
                  out = out.movedim((i - 1, i), (-2, -1))
                  
                  preshape = out.shape[:-2]
                  x = out.contiguous().reshape(-1, 1, out.shape[-2], out.shape[-1])
                  
                  K = cur_window[sp_i]
                  S = cur_strides[sp_i]
                  
                  if padding == 'SAME' or padding == 'CIRCULAR':
                      pad_val1 = max((math.ceil(x.shape[-2] / S) - 1) * S + K - x.shape[-2], 0)
                      pad_val2 = max((math.ceil(x.shape[-1] / S) - 1) * S + K - x.shape[-1], 0)
                      padding_tuple = (pad_val2 // 2, pad_val2 - pad_val2 // 2, pad_val1 // 2, pad_val1 - pad_val1 // 2)
                      x = pad(x, padding_tuple, mode='constant', value=0.0)
                  
                  res = avg_pool2d(x, kernel_size=(K, K), stride=(S, S), padding=0).to(mat.dtype)
                  
                  if pool_type == _Pooling.SUM:
                      res = res * (K * K)
                      
                  out = res.reshape(preshape + res.shape[-2:])
                  out = out.movedim((-2, -1), (i - 1, i))
              return out
              return out

      def pool(mat, batch_ndim):
          if mat is None or getattr(mat, "ndim", 0) == 0:
              return mat
          out = _pool_kernel(mat, batch_ndim)
          return out

      nngp = pool(nngp, 2)
      ntk = pool(ntk, 2)
      cov1 = pool(cov1, 1 if k.diagonal_batch else 2)
      cov2 = None if cov2 is None else (cov1 if (cov2 is cov1 or k.x1_is_x2) else pool(cov2, 1 if k.diagonal_batch else 2))
      return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  def mask_fn(mask, input_shape):
      return mask

  return init_fn, apply_fn, kernel_fn, mask_fn