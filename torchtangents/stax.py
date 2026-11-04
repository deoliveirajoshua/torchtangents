"""Stax layers ported to PyTorch."""

# Layer combinators, combining multiple layers into a single layer.
from ._src.stax.combinators import (
    parallel,
    Sequential,
    repeat
)

# Elementwise nonlinearities.
from ._src.stax.elementwise import (
    ABRelu,
    Abs,
    Cos,
    Elementwise,           # STUBBED: Gaussian Quadrature / Scipy auto-derivation unsupported
    ElementwiseNumerical,  # STUBBED: Same as above
    Erf,
    Exp,
    ExpNormalized,
    Gabor,
    Gaussian,
    GeLU,
    Hermite,
    LeakyReLU,
    Monomial,
    Polynomial,      
    Rbf,
    RectifiedMonomial, 
    ReLU,
    Sigmoid_like,
    Sign,
    Sin,
)

# Linear layers.
from ._src.stax.linear import (
    Aggregate,
    Conv,
    ConvLocal, 
    ConvTranspose, 
    Linear,
    Dense,
    Identity,
    Index,
    DotGeneral,
    Dropout,
    Flatten,
    GlobalSelfAttention,
    ImageResize,
    LayerNorm,
)

# Pooling Layers (Separated in torchtangents)
from ._src.stax.pooling import (
    AvgPool,
    GlobalAvgPool,
    SumPool,
    GlobalSumPool
)

# Helper object for the `Index` layer.
from ._src.stax.linear import (
    Slice
)

# Branching layers.
from ._src.stax.branching import (
    FanInConcat,
    FanInProd,
    FanInSum,
    FanOut,
)

# Batching operations natively slicing RAM equivalents seamlessly.
from ._src.stax.batching import batch

# Enums to specify layer behavior.
from ._src.stax.linear import (
    AggregateImplementation,
    AttentionMechanism,
    Padding,
    PositionalEmbedding,
)

# Decorators and classes for constructing your own layers.
from ._src.stax.requirements import (
    Bool,
    Diagonal,
    MaskedArray,
    layer,
    requires,
    supports_masking,
    unmask_fn,
)

from ._src.utils.kernel import Kernel
