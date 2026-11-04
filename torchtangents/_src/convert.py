import math
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, Dict, Type, Optional, Union, List, Tuple
from .nn.modules import (
    Sequential,
    Linear,
    ReLU,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    SumPool1d,
    SumPool2d,
    SumPool3d,
    LayerNorm,
    Dropout,
    StaxModule,
    Flatten,
    Identity,
    StaxModuleMixin,
    LeakyReLU,
    GELU,
    Abs,
    SigmoidLike,
    Erf,
    compute_empirical_ntk,
)
from .. import stax as _stax
from .utils.kernel import Kernel, _combine_kernels_sum
from .stax.requirements import StaxKernelModule

# Registry mapping standard PyTorch modules to Tangential modules
_CONVERSION_REGISTRY: Dict[Type[nn.Module], Callable[[Any], nn.Module]] = {}


def register_conversion(torch_layer_type: Type[nn.Module], conversion_fn: Callable[[Any], nn.Module]):
    """Registers a mapping allowing Torchtangents to parse standard modules iteratively."""
    _CONVERSION_REGISTRY[torch_layer_type] = conversion_fn


def _add_kernels(args, kw, get):
    k1, k2 = args[0], args[1]
    if isinstance(k1, Kernel) and isinstance(k2, Kernel):
        res = _combine_kernels_sum(k1, k2)
        return _stax.Identity().kernel_fn(res, get=get) if get is not None else res
    return operator.add(k1, k2)


# Functional mapping for torch.fx call_function nodes operating on Kernel objects
_FUNCTIONAL_KERNEL_MAP: Dict[Any, Callable[..., Any]] = {
    torch.relu: lambda args, kw, get: _stax.ReLU().kernel_fn(args[0], get=get),
    F.relu: lambda args, kw, get: _stax.ReLU().kernel_fn(args[0], get=get),
    F.leaky_relu: lambda args, kw, get: _stax.LeakyReLU(kw.get('negative_slope', args[1] if len(args) > 1 else 0.01)).kernel_fn(args[0], get=get),
    F.gelu: lambda args, kw, get: _stax.GeLU(approximate=kw.get('approximate', 'none')).kernel_fn(args[0], get=get),
    torch.sigmoid: lambda args, kw, get: _stax.Sigmoid_like().kernel_fn(args[0], get=get),
    F.sigmoid: lambda args, kw, get: _stax.Sigmoid_like().kernel_fn(args[0], get=get),
    torch.erf: lambda args, kw, get: _stax.Erf().kernel_fn(args[0], get=get),
    torch.abs: lambda args, kw, get: _stax.Abs().kernel_fn(args[0], get=get),
    torch.sin: lambda args, kw, get: _stax.Sin().kernel_fn(args[0], get=get),
    torch.cos: lambda args, kw, get: _stax.Cos().kernel_fn(args[0], get=get),
    torch.exp: lambda args, kw, get: _stax.Exp().kernel_fn(args[0], get=get),
    torch.sign: lambda args, kw, get: _stax.Sign().kernel_fn(args[0], get=get),
    torch.flatten: lambda args, kw, get: _stax.Flatten().kernel_fn(args[0], get=get),
    operator.add: _add_kernels,
    torch.add: _add_kernels,
    operator.mul: lambda args, kw, get: _stax.FanInProd().kernel_fn([args[0], args[1]], get=get),
    torch.mul: lambda args, kw, get: _stax.FanInProd().kernel_fn([args[0], args[1]], get=get),
    torch.cat: lambda args, kw, get: _stax.FanInConcat(axis=kw.get('dim', args[1] if len(args) > 1 else 1)).kernel_fn(args[0], get=get),
    torch.concat: lambda args, kw, get: _stax.FanInConcat(axis=kw.get('dim', args[1] if len(args) > 1 else 1)).kernel_fn(args[0], get=get),
}


def _copy_linear_params(src: nn.Linear, dst: Linear) -> Linear:
    dst.to(device=src.weight.device, dtype=src.weight.dtype)
    with torch.no_grad():
        if dst.parameterization == 'ntk':
            dst.weight.copy_(src.weight * ((src.in_features ** 0.5) / dst.W_std))
            if src.bias is not None and dst.bias is not None and dst.b_std:
                dst.bias.copy_(src.bias / dst.b_std)
        else:
            dst.weight.copy_(src.weight)
            if src.bias is not None and dst.bias is not None:
                dst.bias.copy_(src.bias)
    return dst


def _copy_conv_params(src: nn.Module, dst: nn.Module) -> Any:
    dst.to(device=src.weight.device, dtype=src.weight.dtype)
    with torch.no_grad():
        if getattr(dst, 'parameterization', 'ntk') == 'ntk':
            fan_in = src.in_channels * math.prod(src.kernel_size)
            dst.weight.copy_(src.weight * ((fan_in ** 0.5) / dst.W_std))
            if src.bias is not None and dst.bias is not None and dst.b_std:
                dst.bias.copy_(src.bias / dst.b_std)
        else:
            dst.weight.copy_(src.weight)
            if src.bias is not None and dst.bias is not None:
                dst.bias.copy_(src.bias)
    return dst


# Core Conversions
def _convert_linear(m: nn.Linear) -> Linear:
    has_bias = m.bias is not None
    out = Linear(
        in_features=m.in_features,
        out_features=m.out_features,
        b_std=1.0 if has_bias else None,
        parameterization='ntk'
    )
    return _copy_linear_params(m, out)


def _convert_relu(m: nn.ReLU) -> ReLU:
    return ReLU()


def _convert_leaky_relu(m: nn.LeakyReLU) -> LeakyReLU:
    return LeakyReLU(negative_slope=m.negative_slope)


def _convert_gelu(m: nn.GELU) -> GELU:
    return GELU(approximate=m.approximate)


def _convert_abs(m: nn.Module) -> Abs:
    return Abs()


def _convert_erf(m: nn.Module) -> Erf:
    a = getattr(m, 'a', 1.0)
    b = getattr(m, 'b', 1.0)
    c = getattr(m, 'c', 0.0)
    return Erf(a=a, b=b, c=c)


def _convert_sigmoid(m: nn.Sigmoid) -> SigmoidLike:
    return SigmoidLike()


def _convert_conv1d(m: nn.Conv1d) -> Conv1d:
    has_bias = m.bias is not None
    padding_val = 'VALID'
    if m.padding == 'same' or (isinstance(m.padding, tuple) and any(p > 0 for p in m.padding)):
        padding_val = 'SAME'
    out = Conv1d(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        kernel_size=m.kernel_size[0],
        strides=m.stride[0],
        padding=padding_val,
        b_std=1.0 if has_bias else None
    )
    return _copy_conv_params(m, out)


def _convert_conv2d(m: nn.Conv2d) -> Conv2d:
    has_bias = m.bias is not None
    padding_val = 'VALID'
    if m.padding == 'same' or (isinstance(m.padding, tuple) and any(p > 0 for p in m.padding)):
        padding_val = 'SAME'
    out = Conv2d(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        kernel_size=m.kernel_size,
        strides=m.stride,
        padding=padding_val,
        b_std=1.0 if has_bias else None
    )
    return _copy_conv_params(m, out)


def _convert_conv3d(m: nn.Conv3d) -> Conv3d:
    has_bias = m.bias is not None
    padding_val = 'VALID'
    if m.padding == 'same' or (isinstance(m.padding, tuple) and any(p > 0 for p in m.padding)):
        padding_val = 'SAME'
    out = Conv3d(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        kernel_size=m.kernel_size,
        strides=m.stride,
        padding=padding_val,
        b_std=1.0 if has_bias else None
    )
    return _copy_conv_params(m, out)


def _convert_conv_transpose1d(m: nn.ConvTranspose1d) -> ConvTranspose1d:
    has_bias = m.bias is not None
    padding_val = 'VALID'
    if m.padding != 0 and m.padding != (0,):
        padding_val = 'SAME'
    out = ConvTranspose1d(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        kernel_size=m.kernel_size[0],
        strides=m.stride[0],
        padding=padding_val,
        b_std=1.0 if has_bias else None
    )
    return _copy_conv_params(m, out)


def _convert_conv_transpose2d(m: nn.ConvTranspose2d) -> ConvTranspose2d:
    has_bias = m.bias is not None
    padding_val = 'VALID'
    if m.padding != 0 and m.padding != (0, 0):
        padding_val = 'SAME'
    out = ConvTranspose2d(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        kernel_size=m.kernel_size,
        strides=m.stride,
        padding=padding_val,
        b_std=1.0 if has_bias else None
    )
    return _copy_conv_params(m, out)


def _convert_conv_transpose3d(m: nn.ConvTranspose3d) -> ConvTranspose3d:
    has_bias = m.bias is not None
    padding_val = 'VALID'
    if m.padding != 0 and m.padding != (0, 0, 0):
        padding_val = 'SAME'
    out = ConvTranspose3d(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        kernel_size=m.kernel_size,
        strides=m.stride,
        padding=padding_val,
        b_std=1.0 if has_bias else None
    )
    return _copy_conv_params(m, out)


def _convert_avgpool1d_adaptive(m: nn.AdaptiveAvgPool1d) -> AdaptiveAvgPool1d:
    return AdaptiveAvgPool1d()


def _convert_avgpool2d_adaptive(m: nn.AdaptiveAvgPool2d) -> AdaptiveAvgPool2d:
    return AdaptiveAvgPool2d()


def _convert_avgpool3d_adaptive(m: nn.AdaptiveAvgPool3d) -> AdaptiveAvgPool3d:
    return AdaptiveAvgPool3d()


def _convert_avgpool1d(m: nn.AvgPool1d) -> AvgPool1d:
    padding_val = 'VALID'
    if m.padding != 0 and m.padding != (0,):
        padding_val = 'SAME'
    return AvgPool1d(kernel_size=m.kernel_size, stride=m.stride, padding=padding_val)


def _convert_avgpool2d(m: nn.AvgPool2d) -> AvgPool2d:
    padding_val = 'VALID'
    if m.padding != 0 and m.padding != (0, 0):
        padding_val = 'SAME'
    return AvgPool2d(kernel_size=m.kernel_size, stride=m.stride, padding=padding_val)


def _convert_avgpool3d(m: nn.AvgPool3d) -> AvgPool3d:
    padding_val = 'VALID'
    if m.padding != 0 and m.padding != (0, 0, 0):
        padding_val = 'SAME'
    return AvgPool3d(kernel_size=m.kernel_size, stride=m.stride, padding=padding_val)


def _convert_layernorm(m: nn.LayerNorm) -> LayerNorm:
    out = LayerNorm(normalized_shape=m.normalized_shape, eps=m.eps, elementwise_affine=m.elementwise_affine)
    if m.elementwise_affine and m.weight is not None:
        out.to(device=m.weight.device, dtype=m.weight.dtype)
        with torch.no_grad():
            out.weight.copy_(m.weight)
            if m.bias is not None and out.bias is not None:
                out.bias.copy_(m.bias)
    return out


def _convert_dropout(m: nn.Dropout) -> Dropout:
    return Dropout(p=m.p)


def _convert_flatten(m: nn.Flatten) -> Flatten:
    return Flatten(start_dim=m.start_dim, end_dim=m.end_dim)


def _convert_identity(m: nn.Identity) -> Identity:
    return Identity()


# Registering
register_conversion(nn.Linear, _convert_linear)
register_conversion(nn.ReLU, _convert_relu)
register_conversion(nn.LeakyReLU, _convert_leaky_relu)
register_conversion(nn.GELU, _convert_gelu)
register_conversion(nn.Sigmoid, _convert_sigmoid)
register_conversion(nn.Conv1d, _convert_conv1d)
register_conversion(nn.Conv2d, _convert_conv2d)
register_conversion(nn.Conv3d, _convert_conv3d)
register_conversion(nn.ConvTranspose1d, _convert_conv_transpose1d)
register_conversion(nn.ConvTranspose2d, _convert_conv_transpose2d)
register_conversion(nn.ConvTranspose3d, _convert_conv_transpose3d)
register_conversion(nn.AdaptiveAvgPool1d, _convert_avgpool1d_adaptive)
register_conversion(nn.AdaptiveAvgPool2d, _convert_avgpool2d_adaptive)
register_conversion(nn.AdaptiveAvgPool3d, _convert_avgpool3d_adaptive)
register_conversion(nn.AvgPool1d, _convert_avgpool1d)
register_conversion(nn.AvgPool2d, _convert_avgpool2d)
register_conversion(nn.AvgPool3d, _convert_avgpool3d)
register_conversion(nn.LayerNorm, _convert_layernorm)
register_conversion(nn.Dropout, _convert_dropout)
register_conversion(nn.Flatten, _convert_flatten)
register_conversion(nn.Identity, _convert_identity)
register_conversion(Abs, _convert_abs)
register_conversion(Erf, _convert_erf)


class LiftedModule(StaxModule):
    final_output: Any

    def __init__(self, original_model: nn.Module, force: bool = False):
        import torch.fx
        import warnings

        def init_fn(rng, input_shape): return input_shape, ()
        def apply_fn(params, x, **kwargs): return self._forward_impl(x, **kwargs)
        def kernel_fn(x1, x2=None, get=None, **kwargs): return self._kernel_impl(x1, x2, get, **kwargs)

        super().__init__((init_fn, apply_fn, kernel_fn), input_shape=(1,))
        self.params_list = nn.ParameterList()
        self.registry = nn.ModuleDict()
        self.steps: List[Tuple[str, str, Any]] = []
        self.arg_map: Dict[str, Tuple[List[Any], Dict[str, Any]]] = {}
        self.input_nodes: List[str] = []
        self.final_output = None
        self._original = original_model
        self._use_direct_dispatch = False

        try:
            traced = torch.fx.symbolic_trace(original_model)
        except Exception:
            # Dynamic control flow (if/else, loops) that torch.fx cannot symbolically trace:
            # fall back to direct Kernel.__torch_function__ dispatch on original_model.
            self._use_direct_dispatch = True
            return

        node_to_var = {}
        for node in traced.graph.nodes:
            if node.op == 'placeholder':
                node_id = node.name
                node_to_var[node] = node_id
                self.input_nodes.append(node_id)
                self.arg_map[node_id] = ([], {})
            elif node.op in ['call_module', 'call_function', 'call_method']:
                node_id = node.name
                if node.op == 'call_module':
                    submodule = original_model.get_submodule(str(node.target))
                    sub_type = type(submodule)
                    if sub_type in _CONVERSION_REGISTRY:
                        stax_mod = _CONVERSION_REGISTRY[sub_type](submodule)
                        self.registry[node_id] = stax_mod
                        target = node_id
                    elif isinstance(submodule, (StaxModule, StaxModuleMixin)):
                        self.registry[node_id] = submodule
                        target = node_id
                    elif force:
                        warnings.warn(f"Module {sub_type} unmapped; retaining standard behavior.")
                        self.registry[node_id] = submodule
                        target = node_id
                    else:
                        raise NotImplementedError(f"No conversion for {sub_type}. Use force=True.")
                else:
                    target = node.target
                self.steps.append((node_id, node.op, target))
                inputs = [node_to_var[arg] if isinstance(arg, torch.fx.Node) else arg for arg in node.args]
                kwargs_dict = {k: node_to_var[v] if isinstance(v, torch.fx.Node) else v for k, v in node.kwargs.items()}
                self.arg_map[node_id] = (inputs, kwargs_dict)
                node_to_var[node] = node_id
            elif node.op == 'output':
                output_node = node.args[0]
                if isinstance(output_node, torch.fx.Node):
                    self.final_output = node_to_var[output_node]
                elif isinstance(output_node, (tuple, list)):
                    self.final_output = tuple(node_to_var[v] if isinstance(v, torch.fx.Node) else v for v in output_node)
                else:
                    self.final_output = output_node

    def visualize(self):
        """Displays the physical execution graph using networkx / trace visualizer"""
        try:
            dg, _ = map_module_to_graph(self._original)
            visualize_graph(dg)
        except ImportError as e:
            raise ImportError(
                "Visualizing execution graphs requires 'networkx' and 'matplotlib'. "
                "Install them via: pip install networkx matplotlib"
            ) from e
        except Exception as e:
            print(f"Could not visualize execution graph explicitly: {e}")

    def _forward_impl(self, x, **kwargs):
        if self._use_direct_dispatch:
            return self._original(x, **kwargs)
        cache = {}
        if len(self.input_nodes) == 1:
            cache[self.input_nodes[0]] = x
        else:
            for i, in_name in enumerate(self.input_nodes):
                cache[in_name] = x[i]
        for node_id, op_type, target in self.steps:
            args_names, call_kwargs = self.arg_map[node_id]
            args = [cache[n] if isinstance(n, str) and n in cache else n for n in args_names]
            res_kwargs = {k: cache[v] if isinstance(v, str) and v in cache else v for k, v in call_kwargs.items()}
            if op_type == 'call_module':
                cache[node_id] = self.registry[target](*args, **res_kwargs)
            elif op_type == 'call_function':
                cache[node_id] = target(*args, **res_kwargs)
            elif op_type == 'call_method':
                cache[node_id] = getattr(args[0], str(target))(*args[1:], **res_kwargs)
        if isinstance(self.final_output, tuple):
            return tuple(cache[v] if isinstance(v, str) and v in cache else v for v in self.final_output)
        return cache[self.final_output] if isinstance(self.final_output, str) and self.final_output in cache else self.final_output

    def _kernel_impl(self, x1, x2=None, get=None, **kwargs):
        if self._use_direct_dispatch:
            k_in = x1 if isinstance(x1, Kernel) else Kernel.from_inputs(
                x1, x2, compute_ntk=(get is None) or ('ntk' in get), **kwargs
            )
            k_out = self._original(k_in)
            return _stax.Identity().kernel_fn(k_out, get=get) if get is not None else k_out

        cache = {}
        if len(self.input_nodes) == 1:
            cache[self.input_nodes[0]] = x1
        else:
            for i, in_name in enumerate(self.input_nodes):
                cache[in_name] = x1[i]
        for i, (node_id, op_type, target) in enumerate(self.steps):
            args_n, call_kw = self.arg_map[node_id]
            args = [cache[n] if isinstance(n, str) and n in cache else n for n in args_n]
            res_kw = {k: cache[v] if isinstance(v, str) and v in cache else v for k, v in call_kw.items()}
            is_last = (i == len(self.steps) - 1)
            layer_get = get if is_last else None
            if op_type == 'call_module':
                stax_m = self.registry[target]
                if hasattr(stax_m, 'kernel_fn'):
                    if isinstance(args[0], Kernel):
                        cache[node_id] = stax_m.kernel_fn(args[0], get=layer_get, **res_kw)
                    else:
                        cache[node_id] = stax_m.kernel_fn(args[0], x2, get=layer_get, **res_kw)
                else:
                    raise NotImplementedError(
                        f"Module {type(stax_m)} does not support infinite-width kernel computation."
                    )
            elif op_type == 'call_function':
                has_kernel_arg = any(
                    isinstance(a, Kernel) or (isinstance(a, (list, tuple)) and any(isinstance(x, Kernel) for x in a))
                    for a in args
                )
                if has_kernel_arg and target in _FUNCTIONAL_KERNEL_MAP:
                    cache[node_id] = _FUNCTIONAL_KERNEL_MAP[target](args, res_kw, layer_get)
                else:
                    cache[node_id] = target(*args, **res_kw)
            elif op_type == 'call_method':
                obj = args[0]
                method_name = str(target)
                if isinstance(obj, Kernel):
                    if method_name == 'size':
                        if len(args) > 1 and isinstance(args[1], int):
                            cache[node_id] = obj.shape1[args[1]]
                        else:
                            cache[node_id] = obj.shape1
                        continue
                    elif method_name in ('view', 'reshape', 'flatten'):
                        if len(obj.shape1) > 2 or layer_get is not None:
                            cache[node_id] = _stax.Flatten().kernel_fn(obj, get=layer_get)
                        else:
                            cache[node_id] = obj
                        continue
                cache[node_id] = getattr(obj, method_name)(*args[1:], **res_kw)
        if isinstance(self.final_output, tuple):
            return tuple(cache[v] if isinstance(v, str) and v in cache else v for v in self.final_output)
        return cache[self.final_output] if isinstance(self.final_output, str) and self.final_output in cache else self.final_output


def inject(module: nn.Module, input_shape: Optional[tuple] = None):
    if type(module) in _CONVERSION_REGISTRY:
        original_cls = type(module)
        class InjectedModule(original_cls, StaxModuleMixin):
            def forward(self, x, **kwargs):
                if isinstance(x, Kernel):
                    return self.inf_kernel(x, **kwargs)
                return original_cls.forward(self, x, **kwargs)
        stax_module = _CONVERSION_REGISTRY[original_cls](module)
        module.__class__ = InjectedModule
        module.setup_stax_module(stax_module, input_shape or getattr(stax_module, 'input_shape', None), register_params=False)
        return module
    for name, child in module.named_children():
        inject(child)
    return module


def make_tangential(module: nn.Module) -> nn.Module:
    if type(module) in _CONVERSION_REGISTRY:
        return _CONVERSION_REGISTRY[type(module)](module)
    if isinstance(module, nn.Sequential):
        return Sequential(*[make_tangential(c) for c in module])
    raise NotImplementedError(f"No conversion for {type(module)}.")


def lift(module: torch.nn.Module, force: bool = False) -> 'StaxModule':
    return LiftedModule(module, force=force)


def infinite_analog(module: torch.nn.Module) -> 'StaxKernelModule':
    """Returns the analytical kernel_fn equivalent of the standard PyTorch module."""
    try:
        tangential = make_tangential(module)
        return StaxKernelModule(tangential.kernel_fn, name=f"Infinite{type(module).__name__}")
    except (NotImplementedError, AttributeError):
        lifting = lift(module)
        return StaxKernelModule(lifting.kernel_fn, name=f"Infinite{type(module).__name__}")


def kernel(
    model: Any,
    x1: Union[torch.Tensor, Kernel],
    x2: Optional[torch.Tensor] = None,
    get: Optional[Any] = ('nngp', 'ntk'),
    **kwargs,
):
    """
    Computes the infinite-width NNGP / NTK kernel for any PyTorch nn.Module,
    tt.nn module, or tt.stax module without requiring model rewrites.
    """
    if hasattr(model, 'inf_kernel'):
        return model.inf_kernel(x1, x2, get=get, **kwargs)
    if hasattr(model, 'kernel_fn'):
        return model.kernel_fn(x1, x2, get=get, **kwargs)
    compute_ntk = (get is None) or (get == 'ntk') or (isinstance(get, (tuple, list)) and 'ntk' in get)
    k_in = x1 if isinstance(x1, Kernel) else Kernel.from_inputs(
        x1, x2, compute_ntk=compute_ntk, **kwargs
    )
    k_out = model(k_in)
    return _stax.Identity().kernel_fn(k_out, get=get) if get is not None else k_out


def empirical_ntk(
    model: nn.Module,
    x1: torch.Tensor,
    x2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the finite-width empirical NTK for any PyTorch nn.Module from its current parameters.
    """
    if hasattr(model, 'empirical_ntk') and callable(model.empirical_ntk):
        return model.empirical_ntk(x1, x2)
    return compute_empirical_ntk(model, x1, x2)


def optimize_model(model: Any, backend: str = "inductor") -> Any:
    """Wraps kernel with torch.compile if supported."""
    import sys
    if sys.platform == "win32":
        return model
    if hasattr(model, 'kernel_fn'):
        model.kernel_fn = torch.compile(model.kernel_fn, backend=backend)
    return model


def map_module_to_graph(model: torch.nn.Module):
    import torch.fx
    import networkx as nx
    traced = torch.fx.symbolic_trace(model)
    dg = nx.DiGraph()
    for node in traced.graph.nodes:
        dg.add_node(node.name, op=node.op, target=str(node.target))
        for input_node in node.all_input_nodes:
            dg.add_edge(input_node.name, node.name)
    return dg, traced


def visualize_graph(dg, title="Model Architecture Graph (Operational Flow)"):
    import networkx as nx
    import matplotlib.pyplot as plt
    import os, sys
    if sys.platform == "win32" or sys.platform == "win64":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(dg, prog='dot', args='-Grankdir=LR')
    except Exception:
        for layer, nodes in enumerate(nx.topological_generations(dg)):
            for node in nodes:
                dg.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(dg, subset_key="layer", align="vertical")

    color_map = []
    labels = {}
    for node in dg.nodes():
        node_data = dg.nodes[node]
        op = node_data.get('op', '')
        if op == 'placeholder':
            color_map.append('lightgreen')
        elif op == 'output':
            color_map.append('lightcoral')
        elif op == 'call_module':
            color_map.append('skyblue')
        else:
            color_map.append('lightgrey')
        target = node_data.get('target', '')
        if op == 'call_module':
            labels[node] = f"{node}"
        elif op == 'placeholder':
            labels[node] = f"INPUT\n{node}"
        elif op == 'output':
            labels[node] = "OUTPUT"
        else:
            labels[node] = str(target) if target and str(target) != "None" else node

    plt.figure(figsize=(10, 8))
    nx.draw_networkx(dg, pos, labels=labels, node_size=3000, node_color=color_map,
                     font_size=9, font_weight="bold", arrows=True, arrowsize=10, edge_color="gray")
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
