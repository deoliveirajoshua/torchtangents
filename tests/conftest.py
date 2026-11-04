import os

# Force JAX 0.4.x (and up), when installed, to use strict float32 instead of fast
# bfloat16/tensorfloat32 accumulation so stax_parity tests match PyTorch FP32 precision.
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "float32"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

try:
    import jax
    jax.config.update("jax_default_matmul_precision", "float32")
except ImportError:
    pass

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
