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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class with infinite-width NTK and NNGP as a torch.Tensor subclass."""

import operator as op
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.utils import _pytree as pytree
from . import utils

DotDimensionNumbers = tuple[tuple[Sequence[int], Sequence[int]], tuple[Sequence[int], Sequence[int]]]


def _infer_padding_str(padding: Any) -> str:
    if isinstance(padding, str):
        return padding.upper()
    if isinstance(padding, (tuple, list)):
        return 'SAME' if any(p > 0 for p in padding) else 'VALID'
    if isinstance(padding, int):
        return 'SAME' if padding > 0 else 'VALID'
    return 'VALID'


def _combine_kernels_sum(k1: 'Kernel', k2: 'Kernel') -> 'Kernel':
    from ... import stax as _stax
    from ..stax.branching import _preprocess_kernels_for_fan_in
    try:
        return _stax.FanInSum()([k1, k2])
    except NotImplementedError:
        ks, is_reversed = _preprocess_kernels_for_fan_in([k1, k2])
        a, b = ks[0], ks[1]
        return a.replace(
            cov1=a.cov1 + b.cov1,
            nngp=a.nngp + b.nngp,
            cov2=(a.cov2 + b.cov2) if a.cov2 is not None and b.cov2 is not None else None,
            ntk=(a.ntk + b.ntk) if a.ntk is not None and b.ntk is not None else None,
            is_reversed=is_reversed,
        )


class Kernel:
    """
    Kernel object representing NNGP and NTK matrices.
    The base tensor (self._raw / self.nngp) corresponds to the NNGP matrix.
    """
    
    def __init__(self, nngp, ntk=None, cov1=None, cov2=None, x1_is_x2=None, 
                 is_gaussian=False, is_reversed=False, is_input=False,
                 diagonal_batch=False, diagonal_spatial=False,
                 shape1=None, shape2=None, batch_axis=0, channel_axis=1,
                 is_height_aware=False, mask1=None, mask2=None):
        if isinstance(nngp, Kernel):
            nngp = nngp._raw
        
        # Ensure nngp is a tensor on the correct device
        if not isinstance(nngp, torch.Tensor):
            if cov1 is not None and isinstance(cov1, torch.Tensor):
                nngp = torch.as_tensor(nngp, device=cov1.device)
            elif ntk is not None and isinstance(ntk, torch.Tensor):
                nngp = torch.as_tensor(nngp, device=ntk.device)
            else:
                nngp = torch.as_tensor(nngp)  # Defaults to CPU if nothing else is provided
            
        self._raw = nngp
        self.ntk = ntk
        self.cov1 = cov1 if cov1 is not None else self._raw
        self.cov2 = cov2
        
        # Structural metadata must be Python objects to avoid graph breaks 
        # during structural branching
        self.x1_is_x2 = bool(x1_is_x2.item() if isinstance(x1_is_x2, torch.Tensor) else (True if x1_is_x2 is None else x1_is_x2))
        self.is_gaussian = bool(is_gaussian.item() if isinstance(is_gaussian, torch.Tensor) else is_gaussian)
        self.is_input = bool(is_input.item() if isinstance(is_input, torch.Tensor) else is_input)
        self.diagonal_batch = bool(diagonal_batch)
        self.diagonal_spatial = bool(diagonal_spatial)
        self.is_reversed = bool(is_reversed)
        
        self.shape1 = shape1
        self.shape2 = shape2
        
        self.batch_axis = batch_axis
        self.channel_axis = channel_axis
        self.is_height_aware = bool(is_height_aware)
        
        self.mask1 = mask1
        self.mask2 = mask2

    @property
    def nngp(self) -> torch.Tensor:
        return self._raw

    @property
    def device(self) -> torch.device:
        return self._raw.device

    @property
    def dtype(self) -> torch.dtype:
        return self._raw.dtype

    @property
    def shape(self) -> torch.Size:
        return self._raw.shape

    @property
    def ndim(self) -> int:
        return self._raw.ndim

    def size(self, dim=None):
        if self.shape1 is not None:
            s = torch.Size(self.shape1)
        else:
            s = self._raw.shape
        return s if dim is None else s[dim]

    def dim(self) -> int:
        return len(self.shape1) if self.shape1 is not None else self._raw.ndim

    def numel(self) -> int:
        return math.prod(self.shape1) if self.shape1 is not None else self._raw.numel()

    def __getattr__(self, name: str):
        if name == '_raw' or '_raw' not in self.__dict__:
            raise AttributeError(name)
        try:
            return getattr(self._raw, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def requires_grad(self) -> bool:
        return self._raw.requires_grad

    def __len__(self) -> int:
        return len(self._raw)

    def __repr__(self) -> str:
        return f"Kernel(nngp={self._raw}, ntk={self.ntk}, cov1={self.cov1}, cov2={self.cov2})"

    def to(self, *args, **kwargs) -> 'Kernel':
        return self.replace(
            nngp=self._raw.to(*args, **kwargs),
            ntk=self.ntk.to(*args, **kwargs) if self.ntk is not None else None,
            cov1=self.cov1.to(*args, **kwargs) if self.cov1 is not None else None,
            cov2=self.cov2.to(*args, **kwargs) if self.cov2 is not None else None,
        )

    def cpu(self) -> 'Kernel':
        return self.to('cpu')

    def cuda(self, device=None) -> 'Kernel':
        return self.to(f'cuda:{device}' if device is not None else 'cuda')

    def clone(self) -> 'Kernel':
        return self.replace(
            nngp=self._raw.clone(),
            ntk=self.ntk.clone() if self.ntk is not None else None,
            cov1=self.cov1.clone() if self.cov1 is not None else None,
            cov2=self.cov2.clone() if self.cov2 is not None else None,
        )

    def detach(self) -> 'Kernel':
        return self.replace(
            nngp=self._raw.detach(),
            ntk=self.ntk.detach() if self.ntk is not None else None,
            cov1=self.cov1.detach() if self.cov1 is not None else None,
            cov2=self.cov2.detach() if self.cov2 is not None else None,
        )

    def __add__(self, other):
        if isinstance(other, Kernel):
            return _combine_kernels_sum(self, other)
        return self.replace(nngp=self._raw + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Kernel):
            from ... import stax as _stax
            return _stax.FanInProd()([self, other])
        var = other**2 if isinstance(other, (int, float)) else other
        return self.replace(
            nngp=self._raw * var,
            ntk=self.ntk * var if self.ntk is not None else None,
            cov1=self.cov1 * var,
            cov2=self.cov2 * var if self.cov2 is not None else None
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    @classmethod
    def from_inputs(
        cls,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        compute_ntk: bool = True,
        diagonal_batch: bool = True,
        diagonal_spatial: bool = False,
        batch_axis: int = 0,
        channel_axis: int = 1,
        mask_constant: Optional[float] = None,
        eps: float = 1e-12,
    ) -> 'Kernel':
        """Constructs an initial infinite-width Kernel directly from input tensor(s)."""
        from ..stax.requirements import _inputs_to_kernel
        return _inputs_to_kernel(
            x1,
            x2,
            compute_ntk=compute_ntk,
            diagonal_batch=diagonal_batch,
            diagonal_spatial=diagonal_spatial,
            batch_axis=batch_axis,
            channel_axis=channel_axis,
            mask_constant=mask_constant,
            eps=eps,
        )

    def replace(self, 
                nngp=None, ntk=None, cov1=None, cov2=None, x1_is_x2=None, 
                is_gaussian=None, is_reversed=None, is_input=None,
                diagonal_batch=None, diagonal_spatial=None,
                shape1=None, shape2=None, batch_axis=None, channel_axis=None,
                is_height_aware=None, mask1=None, mask2=None) -> 'Kernel':
        """
        Returns a new Kernel with updated attributes.
        Uses explicit arguments to avoid dynamic kwargs specialization in torch.compile.
        """
        return Kernel(
            nngp=nngp if nngp is not None else self.nngp,
            ntk=ntk if ntk is not None else self.ntk,
            cov1=cov1 if cov1 is not None else self.cov1,
            cov2=cov2 if cov2 is not None else self.cov2,
            x1_is_x2=x1_is_x2 if x1_is_x2 is not None else self.x1_is_x2,
            is_gaussian=is_gaussian if is_gaussian is not None else self.is_gaussian,
            is_reversed=is_reversed if is_reversed is not None else self.is_reversed,
            is_input=is_input if is_input is not None else self.is_input,
            diagonal_batch=diagonal_batch if diagonal_batch is not None else self.diagonal_batch,
            diagonal_spatial=diagonal_spatial if diagonal_spatial is not None else self.diagonal_spatial,
            shape1=shape1 if shape1 is not None else self.shape1,
            shape2=shape2 if shape2 is not None else self.shape2,
            batch_axis=batch_axis if batch_axis is not None else self.batch_axis,
            channel_axis=channel_axis if channel_axis is not None else self.channel_axis,
            is_height_aware=is_height_aware if is_height_aware is not None else self.is_height_aware,
            mask1=mask1 if mask1 is not None else self.mask1,
            mask2=mask2 if mask2 is not None else self.mask2
        )

    def __getitem__(self, idx) -> 'Kernel':
        nngp_raw = self._raw
        idx_can = utils.canonicalize_idx(idx, len(self.shape1))
        
        channel_idx = idx_can[self.channel_axis]
        batch_idx = idx_can[self.batch_axis]

        if channel_idx != slice(None):
            raise NotImplementedError(f'Indexing into the channel axis {self.channel_axis} not supported.')
        if isinstance(batch_idx, int):
            raise NotImplementedError('Indexing an axis with an integer index removes the respective axis. Requires batch axis.')

        spatial_idx = tuple(s for i, s in enumerate(idx_can) if i not in (self.batch_axis, self.channel_axis))
        if self.is_reversed:
            spatial_idx = spatial_idx[::-1]
        
        # Double indices for full spatial if not diagonal
        if not self.diagonal_spatial:
            spatial_idx = tuple(v for s in spatial_idx for v in (s, s))

        nngp_batch_slice = (batch_idx, batch_idx)
        cov_batch_slice = (batch_idx,) if self.diagonal_batch else (batch_idx, batch_idx)
        
        nngp_slice = nngp_batch_slice + spatial_idx
        cov_slice = cov_batch_slice + spatial_idx

        nngp = nngp_raw[nngp_slice]
        ntk = self.ntk[nngp_slice] if self.ntk is not None and self.ntk.ndim > 0 else self.ntk
        cov1 = self.cov1[cov_slice] if self.cov1 is not None else None
        cov2 = self.cov2[cov_slice] if self.cov2 is not None else None

        # Update metadata to reflect dropped axes if any (though we error on int batch_idx)
        new_channel_axis, new_batch_axis = self.channel_axis, self.batch_axis
        for i, s in reversed(list(enumerate(idx_can))):
            if isinstance(s, int):
                if i < new_channel_axis: new_channel_axis -= 1
                if i < new_batch_axis: new_batch_axis -= 1

        return self.replace(
            nngp=nngp, ntk=ntk, cov1=cov1, cov2=cov2, channel_axis=new_channel_axis,
            batch_axis=new_batch_axis, shape1=utils.slice_shape(self.shape1, idx),
            shape2=utils.slice_shape(self.shape2, idx),
            mask1=self.mask1[idx] if self.mask1 is not None else None,
            mask2=self.mask2[idx] if self.mask2 is not None else None,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        from ... import stax as _stax

        # 1. Linear layer dispatch: F.linear(input, weight, bias=None)
        if func is F.linear:
            k = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else kwargs.get('bias', None)
            out_features = weight.shape[0]
            return _stax.Linear(
                out_features,
                W_std=1.0,
                b_std=1.0 if bias is not None else None,
            )(k)

        # 2. Convolution dispatch: F.conv1d / F.conv2d / F.conv3d
        if func in (F.conv1d, F.conv2d, F.conv3d):
            k = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else kwargs.get('bias', None)
            stride = args[3] if len(args) > 3 else kwargs.get('stride', 1)
            padding = args[4] if len(args) > 4 else kwargs.get('padding', 0)
            out_channels = weight.shape[0]
            kernel_size = tuple(weight.shape[2:])
            spatial_ndim = len(kernel_size)
            strides = stride if isinstance(stride, (tuple, list)) else (stride,) * spatial_ndim
            pad_str = _infer_padding_str(padding)
            return _stax.Conv(
                out_channels,
                kernel_size,
                strides=strides,
                padding=pad_str,
                W_std=1.0,
                b_std=1.0 if bias is not None else None,
            )(k)

        # 3. Transposed Convolution dispatch: F.conv_transpose1d / 2d / 3d
        if func in (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d):
            k = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else kwargs.get('bias', None)
            stride = args[3] if len(args) > 3 else kwargs.get('stride', 1)
            padding = args[4] if len(args) > 4 else kwargs.get('padding', 0)
            out_channels = weight.shape[1]
            kernel_size = tuple(weight.shape[2:])
            spatial_ndim = len(kernel_size)
            strides = stride if isinstance(stride, (tuple, list)) else (stride,) * spatial_ndim
            pad_str = _infer_padding_str(padding)
            return _stax.ConvTranspose(
                out_channels,
                kernel_size,
                strides=strides,
                padding=pad_str,
                W_std=1.0,
                b_std=1.0 if bias is not None else None,
            )(k)

        # 4. Elementwise activations
        if func in (F.relu, torch.relu, torch.Tensor.relu, torch.Tensor.relu_):
            return _stax.ReLU()(args[0])

        if func is F.leaky_relu:
            neg_slope = args[1] if len(args) > 1 else kwargs.get('negative_slope', 0.01)
            return _stax.LeakyReLU(neg_slope)(args[0])

        if func is F.gelu:
            approx = args[1] if len(args) > 1 else kwargs.get('approximate', 'none')
            return _stax.GeLU(approximate=approx)(args[0])

        if func in (torch.sigmoid, F.sigmoid, torch.Tensor.sigmoid):
            return _stax.Sigmoid_like()(args[0])

        if func in (torch.erf, torch.Tensor.erf):
            return _stax.Erf()(args[0])

        if func in (torch.sin, torch.Tensor.sin):
            return _stax.Sin()(args[0])

        if func in (torch.cos, torch.Tensor.cos):
            return _stax.Cos()(args[0])

        if func in (torch.exp, torch.Tensor.exp):
            return _stax.Exp()(args[0])

        if func in (torch.abs, torch.Tensor.abs):
            return _stax.Abs()(args[0])

        if func in (torch.sign, torch.Tensor.sign):
            return _stax.Sign()(args[0])

        # 5. Normalization & Dropout
        if func is F.layer_norm:
            k = args[0]
            normalized_shape = args[1] if len(args) > 1 else kwargs.get('normalized_shape')
            eps = args[4] if len(args) > 4 else kwargs.get('eps', 1e-5)
            ndim = len(k.shape1)
            norm_ndim = len(normalized_shape) if isinstance(normalized_shape, (tuple, list, torch.Size)) else 1
            axes = tuple(range(ndim - norm_ndim, ndim))
            if (k.channel_axis % ndim) not in axes:
                axes = (k.channel_axis % ndim,) + axes
            k_in = k if bool(k.is_gaussian) else k.replace(is_gaussian=True)
            return _stax.LayerNorm(axis=axes, eps=eps, batch_axis=k.batch_axis, channel_axis=k.channel_axis)(k_in)

        if func is F.batch_norm:
            return args[0]

        if func in (F.dropout, F.dropout1d, F.dropout2d, F.dropout3d):
            k = args[0]
            p = args[1] if len(args) > 1 else kwargs.get('p', 0.5)
            training = args[2] if len(args) > 2 else kwargs.get('training', False)
            if training and 0.0 < p < 1.0:
                return _stax.Dropout(rate=1.0 - p, mode='train')(k)
            return k

        # 6. Pooling operations
        if func in (F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d):
            return _stax.GlobalAvgPool()(args[0])

        if func in (F.avg_pool1d, F.avg_pool2d, F.avg_pool3d, F.max_pool1d, F.max_pool2d, F.max_pool3d):
            k = args[0]
            kernel_size = args[1] if len(args) > 1 else kwargs.get('kernel_size')
            stride = args[2] if len(args) > 2 else kwargs.get('stride', None)
            padding = args[3] if len(args) > 3 else kwargs.get('padding', 0)
            spatial_ndim = len(k.shape1) - 2
            if spatial_ndim <= 0:
                return k
            ws = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * spatial_ndim
            st = (stride if isinstance(stride, (tuple, list)) else (stride,) * spatial_ndim) if stride is not None else ws
            pad_str = _infer_padding_str(padding)
            return _stax.AvgPool(window_shape=ws, strides=st, padding=pad_str)(k)

        # 7. Shape & Flatten / View / Size / Dim operations
        if func in (torch.Tensor.dim, torch.Tensor.ndimension):
            k = args[0]
            if k.shape1 is not None:
                return len(k.shape1)

        if func is torch.Tensor.size:
            k = args[0]
            if len(args) > 1 and isinstance(args[1], int):
                return k.shape1[args[1]]
            return torch.Size(k.shape1)

        if func in (torch.flatten, torch.Tensor.flatten, torch.Tensor.view, torch.Tensor.reshape):
            k = args[0]
            if k.shape1 is not None and len(k.shape1) > 2:
                return _stax.Flatten()(k)
            return k

        # 8. Branching: Addition (residual connections) & Concatenation & Multiplication
        if func in (torch.add, torch.Tensor.add, torch.Tensor.add_, torch.Tensor.__add__, torch.Tensor.__radd__, torch.Tensor.__iadd__):
            if len(args) >= 2 and isinstance(args[0], Kernel) and isinstance(args[1], Kernel):
                return _combine_kernels_sum(args[0], args[1])

        if func in (torch.mul, torch.Tensor.mul, torch.Tensor.__mul__, torch.Tensor.__rmul__):
            if len(args) >= 2 and isinstance(args[0], Kernel) and isinstance(args[1], Kernel):
                return _stax.FanInProd()([args[0], args[1]])

        if func in (torch.cat, torch.concat):
            tensors = args[0]
            dim = args[1] if len(args) > 1 else kwargs.get('dim', 1)
            if isinstance(tensors, (list, tuple)) and any(isinstance(t, Kernel) for t in tensors):
                return _stax.FanInConcat(axis=dim)(list(tensors))

        if func in (torch.clone, torch.Tensor.clone):
            return args[0].clone()

        if func in (torch.detach, torch.Tensor.detach):
            return args[0].detach()

        def unwrap(x):
            if isinstance(x, Kernel): return x._raw
            if isinstance(x, (list, tuple)): return type(x)([unwrap(i) for i in x])
            if isinstance(x, dict): return {k: unwrap(v) for k, v in x.items()}
            return x

        return func(*unwrap(args), **unwrap(kwargs or {}))

    # Helper methods
    def slice(self, n1_slice, n2_slice): return self.__getitem__((n1_slice, n2_slice))
    
    def reverse(self):
        batch_ndim = 1 if self.diagonal_batch else 2
        return self.replace(
            cov1=utils.reverse_zipped(self.cov1, batch_ndim),
            cov2=utils.reverse_zipped(self.cov2, batch_ndim),
            nngp=utils.reverse_zipped(self.nngp, 2),
            ntk=utils.reverse_zipped(self.ntk, 2),
            is_reversed=not self.is_reversed
        )

    def transpose(self, axes=None):
        if axes is None: axes = tuple(range(len(self.shape1) - 2))
        def perm(mat, b_ndim):
            if mat is None: return None
            _axes = tuple(b_ndim + a for a in axes)
            if not self.diagonal_spatial:
                _axes = tuple(j for a in _axes for j in (2 * a - b_ndim, 2 * a - b_ndim + 1))
            return torch.permute(mat, tuple(range(b_ndim)) + _axes)
        return self.replace(cov1=perm(self.cov1, 1 if self.diagonal_batch else 2),
                            cov2=perm(self.cov2, 1 if self.diagonal_batch else 2),
                            nngp=perm(self.nngp, 2), ntk=perm(self.ntk, 2))

    def mask(self, m1, m2):
        mask11, mask12, mask22 = self._get_mask_prods(m1, m2)
        return self.replace(cov1=utils.mask(self.cov1, mask11), cov2=utils.mask(self.cov2, mask22),
                            nngp=utils.mask(self.nngp, mask12), ntk=utils.mask(self.ntk, mask12),
                            mask1=m1, mask2=m2)

    def _get_mask_prods(self, m1, m2):
        def get_p(m_a, m_b, b_ndim):
            if m_a is None and m_b is None: return None
            def rs(m):
                if m is None: return None
                m = torch.squeeze(torch.movedim(m, (self.batch_axis, self.channel_axis), (0, -1)), -1)
                if self.is_reversed: m = torch.movedim(m, tuple(range(1, m.ndim)), tuple(range(m.ndim-1, 0, -1)))
                return m
            m_a, m_b = rs(m_a), rs(m_b)
            return utils.outer_prod(m_a, m_b, 2 - b_ndim, 1 if self.diagonal_spatial else m_a.ndim, op.or_)
        m11 = get_p(m1, m1, 1 if self.diagonal_batch else 2)
        return m11, get_p(m1, m2, 2), (get_p(m2, m2, 1 if self.diagonal_batch else 2) if m2 is not None else m11)

    def dot_general(self, other1, other2, is_lhs, dimension_numbers):
        if other1 is None and other2 is None: return self

        if other1 is not None and other2 is not None:
            if other1.ndim != other2.ndim:
                raise NotImplementedError(f'Factors with different dimensionality not implemented.')

        if is_lhs:
            (other_cs, input_cs), (other_bs, input_bs) = dimension_numbers
        else:
            (input_cs, other_cs), (input_bs, other_bs) = dimension_numbers

        n_input = len(self.shape1)
        n_other = other1.ndim if other1 is not None else other2.ndim

        input_cs = utils.mod(input_cs, n_input)
        input_bs = utils.mod(input_bs, n_input)
        other_cs = utils.mod(other_cs, n_other)
        other_bs = utils.mod(other_bs, n_other)

        other_dims = tuple(other_bs) + tuple(other_cs)
        input_dims = tuple(input_bs) + tuple(input_cs)

        def to_kernel_dim(i: int, batch_ndim: int, is_left: bool) -> int:
            if i == self.batch_axis:
                i = 0 if (is_left or batch_ndim == 1) else 1
            elif i == self.channel_axis:
                raise ValueError(f'Dimension {i} cannot be equal to `channel_axis`.')
            else:
                i -= int(i > self.batch_axis) + int(i > self.channel_axis)
                i = batch_ndim + (1 if self.diagonal_spatial else 2) * i
                i += not is_left and not self.diagonal_spatial
            return i

        def get_other_dims(batch_ndim: int, is_left: bool) -> list[int]:
            dims = [-i - 1 - (0 if is_left or self.diagonal_spatial else n_other)
                    for i in range(n_other)]
            for i_inputs, i_other in zip(input_dims, other_dims):
                dims[i_other] = to_kernel_dim(i_inputs, batch_ndim, is_left)
            return dims

        def get_mat_non_c_dims(batch_ndim: int) -> list[int]:
            input_non_c_dims = list(input_bs) + [
                i for i in range(n_input)
                if i not in list(input_cs) + list(input_bs) + [self.channel_axis]
            ]
            if self.batch_axis in input_non_c_dims:
                input_non_c_dims.remove(self.batch_axis)
                input_non_c_dims.insert(0, self.batch_axis)
            mat_non_c_dims = []
            for i in input_non_c_dims:
                left = to_kernel_dim(i, batch_ndim, True)
                right = to_kernel_dim(i, batch_ndim, False)
                mat_non_c_dims += [left] if left == right else [left, right]
            return mat_non_c_dims

        def get_other_non_c_dims() -> list[int]:
            other_non_c_dims = [-i - 1 for i in range(n_other) if i not in other_dims]
            if not self.diagonal_spatial:
                other_non_c_dims = list(utils.zip_flat(
                    other_non_c_dims,
                    [-i - 1 - n_other for i in range(n_other) if i not in other_dims]))
            return other_non_c_dims

        def get_out_dims(batch_ndim: int) -> list[int]:
            mat_non_c_dims = get_mat_non_c_dims(batch_ndim)
            other_non_c_dims = get_other_non_c_dims()
            n_b_spatial = len(input_bs) - (1 if self.batch_axis in input_bs else 0)
            n_b = (len(mat_non_c_dims) if not is_lhs else
                   (((0 if self.batch_axis in input_cs else batch_ndim) +
                     (1 if self.diagonal_spatial else 2) * n_b_spatial)))
            return mat_non_c_dims[:n_b] + other_non_c_dims + mat_non_c_dims[n_b:]

        def dot(mat: Optional[torch.Tensor], batch_ndim: int, 
                o1: Optional[torch.Tensor] = None, 
                o2: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
            if mat is None or mat.ndim == 0 or (o1 is None and o2 is None): return mat
            operands_and_dims = []
            if o1 is not None:
                operands_and_dims.extend([o1, get_other_dims(batch_ndim, True)])
            mat_dims = list(range(mat.ndim))
            if self.is_reversed:
                mat_dims = utils.reverse_zipped(mat_dims, batch_ndim)
            operands_and_dims.extend([mat, mat_dims])
            if o2 is not None:
                operands_and_dims.extend([o2, get_other_dims(batch_ndim, False)])
            operands_and_dims.append(get_out_dims(batch_ndim))
            
            counter = [0]
            dim_to_char = {}
            def get_char(dim_idx):
                if dim_idx not in dim_to_char:
                    dim_to_char[dim_idx] = chr(ord('a') + counter[0])
                    counter[0] += 1
                return dim_to_char[dim_idx]
            eq_ps = []
            ts = []
            for i in range(0, len(operands_and_dims) - 1, 2):
                ts.append(operands_and_dims[i])
                eq_ps.append("".join(get_char(d) for d in operands_and_dims[i+1]))
            equation = ",".join(eq_ps) + "->" + "".join(get_char(d) for d in operands_and_dims[-1])
            return torch.einsum(equation, *ts)

        cov1 = dot(self.cov1, 1 if self.diagonal_batch else 2, other1, other1)
        cov2 = dot(self.cov2, 1 if self.diagonal_batch else 2, other2, other2)
        nngp = dot(self.nngp, 2, other1, other2)
        ntk = dot(self.ntk, 2, other1, other2)
        lhs_ndim = n_other if is_lhs else None
        return self.replace(
            cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk, is_reversed=False,
            batch_axis=utils.axis_after_dot(self.batch_axis, input_cs, input_bs, lhs_ndim),
            channel_axis=utils.axis_after_dot(self.channel_axis, input_cs, input_bs, lhs_ndim)
        )

    def asdict(self) -> dict[str, Any]:
        return {
            'nngp': self.nngp, 'ntk': self.ntk, 'cov1': self.cov1, 'cov2': self.cov2,
            'x1_is_x2': self.x1_is_x2, 'is_gaussian': self.is_gaussian, 
            'is_reversed': self.is_reversed, 'is_input': self.is_input,
            'diagonal_batch': self.diagonal_batch, 'diagonal_spatial': self.diagonal_spatial,
            'shape1': self.shape1, 'shape2': self.shape2, 'batch_axis': self.batch_axis,
            'channel_axis': self.channel_axis, 'is_height_aware': self.is_height_aware,
            'mask1': self.mask1, 'mask2': self.mask2
        }

    def astuple(self) -> tuple[Any, ...]:
        return tuple(self.asdict().values())

# Register with PyTorch's PyTree system to support torch.compile
def _kernel_flatten(obj: Kernel):
    """Flattens a Kernel into its Tensor children and metadata context."""
    # Data components (children) that are Tensors
    children = [obj._raw, obj.ntk, obj.cov1, obj.cov2]
    
    # Static metadata (context)
    metadata = {
        'x1_is_x2': obj.x1_is_x2,
        'is_gaussian': obj.is_gaussian,
        'is_input': obj.is_input,
        'diagonal_batch': obj.diagonal_batch,
        'diagonal_spatial': obj.diagonal_spatial,
        'is_reversed': obj.is_reversed,
        'shape1': obj.shape1,
        'shape2': obj.shape2,
        'batch_axis': obj.batch_axis,
        'channel_axis': obj.channel_axis,
        'is_height_aware': obj.is_height_aware,
        'mask1': obj.mask1,
        'mask2': obj.mask2,
    }
    return children, metadata

def _kernel_unflatten(children: list[torch.Tensor], metadata: dict) -> Kernel:
    """Reconstructs a Kernel from flattened children and context."""
    nngp, ntk, cov1, cov2 = children[:4]
    return Kernel(nngp=nngp, ntk=ntk, cov1=cov1, cov2=cov2, **metadata)

pytree.register_pytree_node(Kernel, _kernel_flatten, _kernel_unflatten)
