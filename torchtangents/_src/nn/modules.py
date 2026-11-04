import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.conv import _ConvNd
from torch.utils import _pytree as tree
from typing import Callable, Any, Optional, Tuple, Sequence
from ..utils.kernel import Kernel


def compute_empirical_ntk(model: nn.Module, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the empirical (finite-width) NTK for any PyTorch nn.Module from its current parameters.
    
    Args:
        model: Any PyTorch nn.Module.
        x1: Training inputs (N_train x Features).
        x2: Testing inputs (N_test x Features). If None, computes self-NTK (x1, x1).
        
    Returns:
        Empirical NTK tensor of shape (N_train, N_test, Out_dim, Out_dim).
    """
    import torch.func as func
    if x2 is None:
        x2 = x1
        
    params = dict(model.named_parameters())
    
    def fnet_single(p, x):
        return func.functional_call(model, p, x.unsqueeze(0)).squeeze(0)
        
    jacobian_fn = func.jacrev(fnet_single, argnums=0)
    
    was_training = getattr(model, 'training', False)
    if hasattr(model, 'eval'):
        model.eval()
    try:
        J1_dict = func.vmap(jacobian_fn, in_dims=(None, 0))(params, x1)
        J2_dict = func.vmap(jacobian_fn, in_dims=(None, 0))(params, x2)
    finally:
        if hasattr(model, 'train'):
            model.train(was_training)
    
    def flatten_jacs(j_dict):
        flat_jacs = []
        for k, v in j_dict.items():
            flat_jacs.append(v.view(v.shape[0], v.shape[1], -1))
        return torch.cat(flat_jacs, dim=-1)
        
    J1 = flatten_jacs(J1_dict)  # (N_train, Out_dim, P)
    J2 = flatten_jacs(J2_dict)  # (N_test, Out_dim, P)
    
    return torch.einsum('nop,mqp->nmoq', J1, J2)


class StaxModuleMixin:
    """Mixin that adds StaxModule capabilities to any nn.Module."""
    kernel_fn: Any
    _init_fn_raw: Any
    _apply_fn_raw: Any
    input_shape: Optional[tuple]
    out_shape: Optional[tuple]
    params_spec: Any
    params_list: nn.ParameterList
    _flat_param_types: list
    
    def setup_stax_module(
        self, 
        stax_obj: Any,
        input_shape: Optional[tuple] = None,
        generator: Optional[torch.Generator] = None,
        register_params: bool = True
    ):
        """
        Manually initializes the StaxModule traits on an existing instance.
        """
        self._setup_stax(stax_obj)
        self.input_shape = input_shape
        
        if register_params and input_shape is not None:
            if generator is None:
                generator = torch.default_generator
                
            out_shape, params = self._init_fn_raw(generator, input_shape)
            self.out_shape = out_shape
            
            # Flatten the parameters tree
            flat_params, self.params_spec = tree.tree_flatten(params)
            
            # Register those that are tensors as parameters, keep track of None or empty
            if not hasattr(self, 'params_list'):
                self.params_list = nn.ParameterList()
            self._flat_param_types = []
            
            for p in flat_params:
                if isinstance(p, torch.Tensor):
                    self.params_list.append(nn.Parameter(p))
                    self._flat_param_types.append('tensor')
                else:
                    self._flat_param_types.append(p)

    def _setup_stax(self, stax_obj: Any):
        """Internal helper to extract stax functions from various formats."""
        if hasattr(stax_obj, 'kernel_fn'):
            self.kernel_fn = stax_obj.kernel_fn
            self._init_fn_raw = getattr(stax_obj.kernel_fn, '_init_fn', None)
            self._apply_fn_raw = getattr(stax_obj.kernel_fn, '_apply_fn', None)
        elif isinstance(stax_obj, (tuple, list)):
            self._init_fn_raw, self._apply_fn_raw, self.kernel_fn = stax_obj[:3]
            self.kernel_fn._init_fn = self._init_fn_raw
            self.kernel_fn._apply_fn = self._apply_fn_raw
        else:
            self.kernel_fn = stax_obj
            self._init_fn_raw = getattr(stax_obj, '_init_fn', None)
            self._apply_fn_raw = getattr(stax_obj, '_apply_fn', None)

        if self._init_fn_raw is None or self._apply_fn_raw is None:
            raise ValueError("Provided stax object is missing init_fn or apply_fn.")

    @property
    def init_fn(self): return self._init_fn_raw
    @property
    def apply_fn(self): return self._apply_fn_raw

    def _reconstruct_params(self):
        if not hasattr(self, 'params_list'):
             raise AttributeError("This module does not have registered functional parameters.")
        flat_params = []
        param_idx = 0
        for p_type in self._flat_param_types:
            if p_type == 'tensor':
                flat_params.append(self.params_list[param_idx])
                param_idx += 1
            else:
                flat_params.append(p_type)
        return tree.tree_unflatten(flat_params, self.params_spec)

    def kernel(self, x1, x2=None, get=('nngp', 'ntk'), **kwargs):
        return self.inf_kernel(x1, x2, get=get, **kwargs)

    def kernels(self, x1, x2=None, get=('nngp', 'ntk'), **kwargs):
        return self.inf_kernel(x1, x2, get=get, **kwargs)

    def inf_kernel(self, x1, x2=None, get=None, **kwargs):
        return self.kernel_fn(x1, x2, get=get, **kwargs)

    def empirical_ntk(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the empirical (finite-width) NTK from the current network parameters.
        """
        return compute_empirical_ntk(self, x1, x2)

    def ntk_gradient(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, mode: str = 'all_pairs', **kwargs) -> torch.Tensor:
        """
        Computes the gradient of the NTK matrix with respect to the input x1.
        """
        import torch.func as func
        
        if mode == 'gram':
            def gram_sum(x_input):
                k_out = self.inf_kernel(x_input, x2, get=('ntk',), **kwargs)
                return k_out.ntk.sum()
                
            return func.grad(gram_sum)(x1)
            
        elif mode == 'all_pairs':
            def ntk_single_row(x1_row):
                out = self.inf_kernel(x1_row.unsqueeze(0), x2, get=('ntk',), **kwargs)
                return out.ntk.squeeze(0)
                
            return func.vmap(func.jacrev(ntk_single_row), randomness='same')(x1)
            
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'all_pairs' or 'gram'.")

    def ntk_hessian(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, mode: str = 'all_pairs', **kwargs) -> torch.Tensor:
        """
        Computes the Hessian of the NTK matrix with respect to the input x1.
        """
        import torch.func as func
        
        if mode == 'gram':
            def gram_sum(x_input):
                k_out = self.inf_kernel(x_input, x2, get=('ntk',), **kwargs)
                return k_out.ntk.sum()
                
            return torch.autograd.functional.hessian(gram_sum, x1)
            
        elif mode == 'all_pairs':
            def ntk_single_row(x1_row):
                out = self.inf_kernel(x1_row.unsqueeze(0), x2, get=('ntk',), **kwargs)
                return out.ntk.squeeze(0)
                
            hess_fn = func.jacrev(func.jacrev(ntk_single_row))
            return func.vmap(hess_fn, randomness='same')(x1)
            
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'all_pairs' or 'gram'.")


class StaxModule(nn.Module, StaxModuleMixin):
    """
    A PyTorch nn.Module wrapper for torchtangents.stax layers.
    """
    def __init__(
        self, 
        stax_obj: Any,
        input_shape: tuple,
        generator: Optional[torch.Generator] = None
    ):
        super().__init__()
        self.setup_stax_module(stax_obj, input_shape, generator, register_params=True)

    def forward(self, x, **kwargs):
        if isinstance(x, Kernel):
            return self.inf_kernel(x, **kwargs)
        params = self._reconstruct_params()
        return self.apply_fn(params, x, **kwargs)
                

class Sequential(nn.Sequential, StaxModuleMixin):
    def __init__(self, *modules: Any, input_shape: Optional[tuple] = None):
        nn.Sequential.__init__(self, *modules)
        if input_shape is None and len(modules) > 0 and hasattr(modules[0], 'input_shape'):
            input_shape = modules[0].input_shape
            
        from torchtangents import stax
        from ..stax.requirements import StaxKernelModule
        stax_layers = []
        for m in self:
            if hasattr(m, 'kernel_fn'):
                stax_layers.append(StaxKernelModule(m.kernel_fn))
            else:
                stax_layers.append(m)
        
        self.setup_stax_module(stax.Sequential(*stax_layers), input_shape=input_shape, register_params=False)

    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return super().forward(x)

class Parallel(nn.ModuleList, StaxModuleMixin):
    def __init__(self, *modules: Any, input_shape: Optional[tuple] = None):
        nn.ModuleList.__init__(self, modules)
        if input_shape is None:
            shapes = []
            for m in self:
                if hasattr(m, 'input_shape'): shapes.append(m.input_shape)
            if len(shapes) == len(modules): input_shape = tuple(shapes)

        from torchtangents import stax
        from ..stax.requirements import StaxKernelModule
        stax_layers = []
        for m in self:
            if hasattr(m, 'kernel_fn'): stax_layers.append(StaxKernelModule(m.kernel_fn))
            else: stax_layers.append(m)
                
        self.setup_stax_module(stax.parallel(*stax_layers), input_shape=input_shape, register_params=False)

    def forward(self, inputs, **kwargs):
        if isinstance(inputs, (Kernel, list, tuple)) and any(isinstance(i, Kernel) for i in (inputs if isinstance(inputs, (list, tuple)) else [inputs])):
             return self.inf_kernel(inputs, **kwargs)
        return tuple(m(x, **kwargs) for m, x in zip(self, inputs))

class Repeat(nn.ModuleList, StaxModuleMixin):
    def __init__(self, module: Any, num_repeats: int, input_shape: Optional[tuple] = None):
        import copy
        super().__init__([copy.deepcopy(module) for _ in range(num_repeats)])
        if input_shape is None and hasattr(module, 'input_shape'):
            input_shape = module.input_shape
            
        from torchtangents import stax
        from ..stax.requirements import StaxKernelModule
        l = StaxKernelModule(module.kernel_fn) if hasattr(module, 'kernel_fn') else module
        self.setup_stax_module(stax.repeat(l, num_repeats), input_shape=input_shape, register_params=False)

    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        for m in self: x = m(x, **kwargs)
        return x

class Linear(nn.Linear, StaxModuleMixin):
    def __init__(self, in_features, out_features, W_std=1.0, b_std=0.0, parameterization='ntk', bias: Optional[bool] = None):
        if bias is False:
            b_std = None
        elif bias is True and b_std is None:
            b_std = 1.0
        nn.Linear.__init__(self, in_features, out_features, bias=(b_std is not None))
        self.W_std, self.b_std, self.parameterization = W_std, b_std, parameterization
        from torchtangents import stax
        self.setup_stax_module(stax.Linear(out_features, W_std=W_std, b_std=b_std, parameterization=parameterization), 
                               input_shape=(in_features,), register_params=False)
        with torch.no_grad():
            self.weight.normal_(0, 1)
            if self.bias is not None: self.bias.normal_(0, 1)

    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        if self.parameterization == 'ntk':
            res = F.linear(x, self.weight, None) * (self.W_std / (self.in_features ** 0.5))
            if self.bias is not None and self.b_std is not None: res = res + self.bias * self.b_std
            return res
        return super().forward(x)

class _ConvBase(_ConvNd, StaxModuleMixin):
    def __init__(self, spatial_ndim, in_channels, out_channels, kernel_size, strides=None, padding='SAME', 
                 W_std=1.0, b_std=0.0, parameterization='ntk', is_transpose=False, stride=None, bias: Optional[bool] = None):
        if bias is False:
            b_std = None
        elif bias is True and b_std is None:
            b_std = 1.0
        if strides is None and stride is not None:
            strides = stride
        _k = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * spatial_ndim
        _s = (strides if isinstance(strides, (tuple, list)) else (strides,) * spatial_ndim) if strides is not None else (1,) * spatial_ndim
        if isinstance(padding, str):
            stax_padding = padding.upper()
            py_padding = (0,) * spatial_ndim if stax_padding == 'VALID' else (tuple(k // 2 for k in _k) if stax_padding == 'SAME' else (0,) * spatial_ndim)
        elif isinstance(padding, (int, tuple, list)):
            pad_tuple = padding if isinstance(padding, (tuple, list)) else (padding,) * spatial_ndim
            py_padding = tuple(pad_tuple)
            stax_padding = 'SAME' if any(p > 0 for p in pad_tuple) else 'VALID'
        else:
            py_padding = (0,) * spatial_ndim
            stax_padding = 'VALID'
        
        _ConvNd.__init__(self, in_channels, out_channels, _k, _s, py_padding, (1,)*spatial_ndim, is_transpose, (0,)*spatial_ndim, 1, b_std is not None, 'zeros')
        self.W_std, self.b_std, self.parameterization = W_std, b_std, parameterization
        
        from torchtangents import stax
        stax_fn = stax.ConvTranspose if is_transpose else stax.Conv
        self.setup_stax_module(stax_fn(out_channels, _k, _s, stax_padding, W_std, b_std, parameterization=parameterization), 
                               input_shape=(1, in_channels) + (1,)*spatial_ndim, register_params=False)
        with torch.no_grad():
            self.weight.normal_(0, 1)
            if self.bias is not None: self.bias.normal_(0, 1)

    def _conv_forward(self, input, weight, bias):
        spatial_ndim = len(self.kernel_size)
        if self.transposed:
            fn = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[spatial_ndim]
            return fn(input, weight, bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[spatial_ndim]
        return fn(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        if self.parameterization == 'ntk':
            fan_in = self.in_channels * torch.tensor(self.kernel_size).prod().item()
            out = self._conv_forward(x, self.weight, None) * (self.W_std / (fan_in ** 0.5))
            if self.bias is not None and self.b_std is not None:
                b_shape = [1] * out.ndim
                b_shape[1] = out.shape[1]
                out = out + self.bias.view(b_shape) * self.b_std
            return out
        return super().forward(x)

class Conv1d(_ConvBase): 
    def __init__(self, in_channels, out_channels, kernel_size, strides=None, padding='SAME', W_std=1.0, b_std=0.0, stride=None, bias=None):
        super().__init__(1, in_channels, out_channels, kernel_size, strides, padding, W_std, b_std, stride=stride, bias=bias)
class Conv2d(_ConvBase): 
    def __init__(self, in_channels, out_channels, kernel_size, strides=None, padding='SAME', W_std=1.0, b_std=0.0, stride=None, bias=None):
        super().__init__(2, in_channels, out_channels, kernel_size, strides, padding, W_std, b_std, stride=stride, bias=bias)
class Conv3d(_ConvBase): 
    def __init__(self, in_channels, out_channels, kernel_size, strides=None, padding='SAME', W_std=1.0, b_std=0.0, stride=None, bias=None):
        super().__init__(3, in_channels, out_channels, kernel_size, strides, padding, W_std, b_std, stride=stride, bias=bias)
class ConvTranspose1d(_ConvBase): 
    def __init__(self, in_channels, out_channels, kernel_size, strides=None, padding='SAME', W_std=1.0, b_std=0.0, stride=None, bias=None):
        super().__init__(1, in_channels, out_channels, kernel_size, strides, padding, W_std, b_std, is_transpose=True, stride=stride, bias=bias)
class ConvTranspose2d(_ConvBase): 
    def __init__(self, in_channels, out_channels, kernel_size, strides=None, padding='SAME', W_std=1.0, b_std=0.0, stride=None, bias=None):
        super().__init__(2, in_channels, out_channels, kernel_size, strides, padding, W_std, b_std, is_transpose=True, stride=stride, bias=bias)
class ConvTranspose3d(_ConvBase): 
    def __init__(self, in_channels, out_channels, kernel_size, strides=None, padding='SAME', W_std=1.0, b_std=0.0, stride=None, bias=None):
        super().__init__(3, in_channels, out_channels, kernel_size, strides, padding, W_std, b_std, is_transpose=True, stride=stride, bias=bias)

class ReLU(nn.ReLU, StaxModuleMixin):
    def __init__(self):
        super().__init__()
        from torchtangents import stax
        self.setup_stax_module(stax.ReLU(), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return super().forward(x)

class Identity(nn.Identity, StaxModuleMixin):
    def __init__(self):
        super().__init__()
        from torchtangents import stax
        self.setup_stax_module(stax.Identity(), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return super().forward(x)

class Dropout(nn.Dropout, StaxModuleMixin):
    def __init__(self, p=0.5):
        super().__init__(p)
        from torchtangents import stax
        self.setup_stax_module(stax.Dropout(p), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return super().forward(x)

class Flatten(nn.Flatten, StaxModuleMixin):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__(start_dim, end_dim)
        from torchtangents import stax
        self.setup_stax_module(stax.Flatten(), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return super().forward(x)

class _GlobalPoolBase(nn.Module, StaxModuleMixin):
    def __init__(self, spatial_ndim: int, is_avg: bool = True):
        super().__init__()
        self.is_avg = is_avg
        from torchtangents import stax
        layer = stax.GlobalAvgPool() if is_avg else stax.GlobalSumPool()
        self.setup_stax_module(layer, input_shape=(1, 1) + (1,) * spatial_ndim, register_params=False)

    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        if hasattr(self, 'is_avg') and not self.is_avg:
             return torch.sum(x, dim=tuple(range(2, x.ndim)))
        return torch.mean(x, dim=tuple(range(2, x.ndim)))

class AdaptiveAvgPool1d(_GlobalPoolBase):
    def __init__(self): super().__init__(1, is_avg=True)
class AdaptiveAvgPool2d(_GlobalPoolBase):
    def __init__(self): super().__init__(2, is_avg=True)
class AdaptiveAvgPool3d(_GlobalPoolBase):
    def __init__(self): super().__init__(3, is_avg=True)
class GlobalSumPool1d(_GlobalPoolBase):
    def __init__(self): super().__init__(1, is_avg=False)
class GlobalSumPool2d(_GlobalPoolBase):
    def __init__(self): super().__init__(2, is_avg=False)
class GlobalSumPool3d(_GlobalPoolBase):
    def __init__(self): super().__init__(3, is_avg=False)

class _PoolBase(nn.Module, StaxModuleMixin):
    def __init__(self, spatial_ndim, kernel_size, stride=None, padding="VALID", is_avg=True, normalize_edges=False):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * spatial_ndim
        self.stride = (stride if isinstance(stride, (tuple, list)) else (stride,) * spatial_ndim) if stride is not None else self.kernel_size
        self.padding = padding
        self.is_avg = is_avg
        
        from torchtangents import stax
        if is_avg:
            layer = stax.AvgPool(window_shape=self.kernel_size, strides=self.stride, padding=padding, normalize_edges=normalize_edges)
        else:
            layer = stax.SumPool(window_shape=self.kernel_size, strides=self.stride, padding=padding)
        self.setup_stax_module(layer, input_shape=(1, 1) + (1,) * spatial_ndim, register_params=False)

    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        p = 0 if self.padding == 'VALID' else tuple(k // 2 for k in self.kernel_size)
        if self.is_avg:
            return F.avg_pool2d(x, self.kernel_size, self.stride, p)
        return F.avg_pool2d(x, self.kernel_size, self.stride, p) * math.prod(self.kernel_size)

class AvgPool1d(_PoolBase):
    def __init__(self, kernel_size, stride=None, padding="VALID", normalize_edges=False):
        super().__init__(1, kernel_size, stride, padding, is_avg=True, normalize_edges=normalize_edges)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        p = 0 if self.padding == 'VALID' else (self.kernel_size[0] // 2,)
        return F.avg_pool1d(x, self.kernel_size, self.stride, p)

class AvgPool2d(_PoolBase):
    def __init__(self, kernel_size, stride=None, padding="VALID", normalize_edges=False):
        super().__init__(2, kernel_size, stride, padding, is_avg=True, normalize_edges=normalize_edges)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        p = 0 if self.padding == 'VALID' else tuple(k // 2 for k in self.kernel_size)
        return F.avg_pool2d(x, self.kernel_size, self.stride, p)

class SumPool1d(_PoolBase):
    def __init__(self, kernel_size, stride=None, padding="VALID"):
        super().__init__(1, kernel_size, stride, padding, is_avg=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        p = 0 if self.padding == 'VALID' else (self.kernel_size[0] // 2,)
        return F.avg_pool1d(x, self.kernel_size, self.stride, p) * self.kernel_size[0]

class SumPool2d(_PoolBase):
    def __init__(self, kernel_size, stride=None, padding="VALID"):
        super().__init__(2, kernel_size, stride, padding, is_avg=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        p = 0 if self.padding == 'VALID' else tuple(k // 2 for k in self.kernel_size)
        return F.avg_pool2d(x, self.kernel_size, self.stride, p) * math.prod(self.kernel_size)

class AvgPool3d(_PoolBase):
    def __init__(self, kernel_size, stride=None, padding="VALID", normalize_edges=False):
        super().__init__(3, kernel_size, stride, padding, is_avg=True, normalize_edges=normalize_edges)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        p = 0 if self.padding == 'VALID' else tuple(k // 2 for k in self.kernel_size)
        return F.avg_pool3d(x, self.kernel_size, self.stride, p)

class SumPool3d(_PoolBase):
    def __init__(self, kernel_size, stride=None, padding="VALID"):
        super().__init__(3, kernel_size, stride, padding, is_avg=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        p = 0 if self.padding == 'VALID' else tuple(k // 2 for k in self.kernel_size)
        return F.avg_pool3d(x, self.kernel_size, self.stride, p) * math.prod(self.kernel_size)

class LayerNorm(nn.LayerNorm, StaxModuleMixin):
    def __init__(self, normalized_shape, eps=1e-12, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
        from torchtangents import stax
        self.setup_stax_module(stax.LayerNorm(eps=eps), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return super().forward(x)

class ABReLU(nn.Module, StaxModuleMixin):
    def __init__(self, a, b, do_stabilize=False):
        super().__init__()
        self.a, self.b = a, b
        from torchtangents import stax
        self.setup_stax_module(stax.ABRelu(a, b, do_stabilize), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return self.a * torch.minimum(x, torch.zeros_like(x)) + self.b * torch.maximum(x, torch.zeros_like(x))

class LeakyReLU(nn.LeakyReLU, StaxModuleMixin):
    def __init__(self, negative_slope=0.01, do_stabilize=False):
        super().__init__(negative_slope)
        from torchtangents import stax
        self.setup_stax_module(stax.LeakyReLU(alpha=negative_slope, do_stabilize=do_stabilize), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return super().forward(x)

class Gabor(nn.Module, StaxModuleMixin):
    def __init__(self):
        super().__init__()
        from torchtangents import stax
        self.setup_stax_module(stax.Gabor(), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return torch.exp(-x ** 2) * torch.sin(x)

class GELU(nn.GELU, StaxModuleMixin):
    def __init__(self, approximate='none'):
        super().__init__(approximate=approximate)
        from torchtangents import stax
        self.setup_stax_module(stax.GeLU(approximate=(approximate == 'tanh')), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return super().forward(x)

class Erf(nn.Module, StaxModuleMixin):
    def __init__(self, a=1., b=1., c=0.):
        super().__init__()
        self.a, self.b, self.c = a, b, c
        from torchtangents import stax
        self.setup_stax_module(stax.Erf(a, b, c), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return self.a * torch.erf(self.b * x) + self.c

class Sin(nn.Module, StaxModuleMixin):
    def __init__(self, a=1., b=1., c=0.):
        super().__init__()
        self.a, self.b, self.c = a, b, c
        from torchtangents import stax
        self.setup_stax_module(stax.Sin(a, b, c), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return self.a * torch.sin(self.b * x + self.c)

class Cos(Sin):
    def __init__(self, a=1., b=1., c=0.):
        super().__init__(a, b, c + math.pi / 2)

class Exp(nn.Module, StaxModuleMixin):
    def __init__(self, a=1., b=1.):
        super().__init__()
        self.a, self.b = a, b
        from torchtangents import stax
        self.setup_stax_module(stax.Exp(a, b), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return self.a * torch.exp(self.b * x)

class RBF(nn.Module, StaxModuleMixin):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        from torchtangents import stax
        self.setup_stax_module(stax.Rbf(gamma=gamma), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return math.sqrt(2) * torch.sin(math.sqrt(2 * self.gamma) * x + math.pi / 4)

class Abs(nn.Module, StaxModuleMixin):
    def __init__(self):
        super().__init__()
        from torchtangents import stax
        self.setup_stax_module(stax.Abs(), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return torch.abs(x)

class SigmoidLike(nn.Module, StaxModuleMixin):
    def __init__(self):
        super().__init__()
        from torchtangents import stax
        self.setup_stax_module(stax.Sigmoid_like(), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return torch.sigmoid(x)

class Sign(nn.Module, StaxModuleMixin):
    def __init__(self):
        super().__init__()
        from torchtangents import stax
        self.setup_stax_module(stax.Sign(), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return torch.sign(x)

class Gaussian(nn.Module, StaxModuleMixin):
    def __init__(self, a=1., b=-1.):
        super().__init__()
        self.a, self.b = a, b
        from torchtangents import stax
        self.setup_stax_module(stax.Gaussian(a, b), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return self.a * torch.exp(self.b * x ** 2)

class ExpNormalized(nn.Module, StaxModuleMixin):
    def __init__(self, gamma=1., shift=-1., do_clip=False):
        super().__init__()
        self.gamma, self.shift, self.do_clip = gamma, shift, do_clip
        from torchtangents import stax
        self.setup_stax_module(stax.ExpNormalized(gamma, shift, do_clip), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return torch.exp(self.gamma * (x + self.shift))

class Hermite(nn.Module, StaxModuleMixin):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        from torchtangents import stax
        self.setup_stax_module(stax.Hermite(degree), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return x

class Monomial(nn.Module, StaxModuleMixin):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        from torchtangents import stax
        self.setup_stax_module(stax.Monomial(degree), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return x ** self.degree

class FanOut(nn.Module, StaxModuleMixin):
    def __init__(self, num):
        super().__init__()
        self.num = num
        from torchtangents import stax
        self.setup_stax_module(stax.FanOut(num), register_params=False)
    def forward(self, x, **kwargs):
        if isinstance(x, Kernel): return self.inf_kernel(x, **kwargs)
        return (x,) * self.num

class FanInSum(nn.Module, StaxModuleMixin):
    def __init__(self):
        super().__init__()
        from torchtangents import stax
        self.setup_stax_module(stax.FanInSum(), register_params=False)
    def forward(self, inputs, **kwargs):
        if any(isinstance(i, Kernel) for i in inputs): return self.inf_kernel(inputs, **kwargs)
        return sum(inputs)
