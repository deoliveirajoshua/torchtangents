<p align="center">
<img src="https://raw.githubusercontent.com/deoliveirajoshua/torchtangents/main/imgs/logo.png?raw=true" width="700" alt="TorchTangents" />
</p>

<p align="center">
  <a href="https://pypi.org/project/torchtangents/"><img src="https://img.shields.io/pypi/v/torchtangents.svg?logo=pypi" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/torchtangents/"><img src="https://img.shields.io/pypi/pyversions/torchtangents.svg?logo=python" alt="Python Versions" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg?logo=pytorch" alt="PyTorch 2.2+" /></a>
  <a href="https://github.com/deoliveirajoshua/torchtangents/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License" /></a>
</p>

**TorchTangents** (`torchtangents`) is a **PyTorch-native** port of Google's JAX-native [Neural Tangents](https://github.com/google/neural-tangents) library. It lets you build, lift, and analyze deep neural networks in standard PyTorch while extracting their exact infinite-width **Neural Network Gaussian Process (NNGP)** and **Neural Tangent Kernel (NTK)** in closed form.

> Novak et al., **"Fast and Easy Infinite Neural Networks in Python"** (ICLR 2020) introduced the original Neural Tangents library for JAX. With the original JAX repository archived in May 2025, `torchtangents` brings this mathematical framework natively to PyTorch — adding stateful `torch.nn.Module` wrappers, `torch.fx` model lifting, `torch.func` input Jacobians/Hessians, and `torch.compile` acceleration.

---

## Installation

### From PyPI
```bash
pip install torchtangents
```

### Editable / Developer Installation
```bash
git clone https://github.com/deoliveirajoshua/torchtangents.git
cd torchtangents
pip install -e .
```

For running PyTorch-vs-JAX parity tests or benchmarks, see the **[Developer & Benchmark Installation Guide](DEVELOPER_INSTALL.md)**.

**Requirements:** Python >= 3.9, PyTorch >= 2.2.

---

## 30-Second Quickstart

```python
import torch
import torch.nn as nn
import torchtangents as tt

x_train = torch.randn(16, 10)
y_train = torch.randn(16, 2).to(torch.float64)
x_test  = torch.randn(4, 10)

# 1. Build a dual-mode TorchTangents model (drop-in replacement for torch.nn)
model = tt.nn.Sequential(
    tt.nn.Linear(10, 128),
    tt.nn.ReLU(),
    tt.nn.Linear(128, 2),
)
out   = model(x_test)                              # Finite-width forward pass: (4, 2)
k_inf = model.inf_kernel(x_train, x_test)          # Infinite-width NNGP & NTK: (16, 4)
k_emp = model.empirical_ntk(x_train, x_test)       # Finite-width empirical NTK: (16, 4, 2, 2)

# 2. Or lift ANY existing vanilla PyTorch model in one line via torch.fx
vanilla_model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
lifted_model  = tt.convert.lift(vanilla_model)
k_lifted      = lifted_model.inf_kernel(x_train, x_test, get="ntk")  # Shape: (16, 4)

# 3. Train an infinitely wide network in closed form (zero training loop!)
k_tr_tr = model.inf_kernel(x_train, None, get=("nngp", "ntk"))
k_te_tr = model.inf_kernel(x_test, x_train, get=("nngp", "ntk"))
gp_pred = tt.predict.gp_inference(k_tr_tr, y_train, diag_reg=1e-4)
y_pred  = gp_pred(get="ntk", k_test_train=k_te_tr).mean              # Shape: (4, 2)
```

---

## Package Architecture

`torchtangents` is organized into four core submodules:

| Submodule | Role | When to Use |
| :--- | :--- | :--- |
| **[`torchtangents.nn`](torchtangents/nn.py)** | Stateful `torch.nn.Module` drop-in layers (`Linear`, `Conv2d`, `ReLU`, `Sequential`, etc.) | When you want **both** standard finite-width forward passes (`model(x)`), `.empirical_ntk()`, and exact infinite-width kernels (`model.inf_kernel(x1, x2)`). |
| **[`torchtangents.stax`](torchtangents/stax.py)** | Stateless functional kernel layers (mirrors `neural_tangents.stax`) | When you **only** need analytical infinite-width kernels (`kernel_fn(x1, x2)`) without allocating finite-width layer weights. |
| **[`torchtangents.convert`](torchtangents/convert.py)** | Automatic PyTorch-to-TorchTangents conversion (`lift`, `make_tangential`, `infinite_analog`, `inject`, `optimize_model`) | When you already have a vanilla `torch.nn.Module` (including residual/branched graphs) and want its infinite-width or empirical NTK without rewriting it. |
| **[`torchtangents.predict`](torchtangents/predict.py)** | Closed-form infinite-width training & Bayesian GP inference (`gp_inference`, `gradient_descent_mse`, `gradient_descent`) | When you want exact infinite-width test predictions at $t = \infty$ or any continuous training time $t$ without running an optimizer loop. |

---

## Theory at a Glance

### 1. Finite-Width (Empirical) NTK: Dot Product in Gradient Space
For a parameterized neural network $f(x; \theta) \in \mathbb{R}^K$ with finite width and parameters $\theta \in \mathbb{R}^P$, the **Empirical Neural Tangent Kernel** (`model.empirical_ntk(x1, x2)`) measures how a gradient step at input $x'$ changes the network's output at $x$. Mathematically, it is the inner product of the per-sample parameter Jacobians in gradient space:

$$\hat{\Theta}(x, x') = \left\langle \nabla_\theta f(x; \theta),\, \nabla_\theta f(x'; \theta) \right\rangle = \sum_{p=1}^{P} \frac{\partial f(x; \theta)}{\partial \theta_p} \left(\frac{\partial f(x'; \theta)}{\partial \theta_p}\right)^\top \in \mathbb{R}^{K \times K}$$

At finite width, $\hat{\Theta}(x, x')$ depends on the random initialization of $\theta$ and evolves non-linearly during training.

### 2. Infinite-Width Limit: Layer-Wise Recursive NNGP & NTK
As the hidden layer widths approach infinity ($n_1, \dots, n_{L-1} \to \infty$) under NTK parameterization, two remarkable simplifications occur:
1. **At Initialization (NNGP)**: Pre-activations at each layer $\ell$ converge by the Central Limit Theorem to an exact zero-mean Gaussian Process with covariance kernel $\Sigma^{(\ell)}(x, x')$.
2. **During Training (NTK)**: The empirical kernel $\hat{\Theta}(x, x')$ converges deterministically to a scalar-output multiple $\Theta^{(L)}(x, x') \otimes I_K$ that remains **constant (frozen)** throughout gradient descent.

Both $\Sigma^{(L)}$ and $\Theta^{(L)}$ are computed analytically (`model.inf_kernel(x1, x2)`) via a closed-form layer-by-layer recursion:

- **Base Case (First Linear Layer)**:
  $$\Sigma^{(1)}(x, x') = \frac{\sigma_w^2}{d_{\text{in}}} \langle x, x' \rangle + \sigma_b^2, \qquad \Theta^{(1)}(x, x') = \Sigma^{(1)}(x, x')$$

- **Recursive Step (Layer $\ell \to \ell + 1$ with nonlinearity $\phi$)**:
  $$\Sigma^{(\ell+1)}(x, x') = \sigma_w^2 \, \mathbb{E}_{(u, v) \sim \mathcal{N}\left(0,\, \Lambda^{(\ell)}(x, x')\right)} \Big[\phi(u)\,\phi(v)\Big] + \sigma_b^2$$
  $$\Theta^{(\ell+1)}(x, x') = \Sigma^{(\ell+1)}(x, x') + \sigma_w^2 \, \dot{\Sigma}^{(\ell+1)}(x, x') \, \Theta^{(\ell)}(x, x')$$

where the derivative covariance is

$$\dot{\Sigma}^{(\ell+1)}(x, x') = \mathbb{E}_{(u, v) \sim \mathcal{N}\left(0,\, \Lambda^{(\ell)}(x, x')\right)} \Big[\dot{\phi}(u)\,\dot{\phi}(v)\Big], \qquad \Lambda^{(\ell)}(x, x') = \begin{pmatrix} \Sigma^{(\ell)}(x, x) & \Sigma^{(\ell)}(x, x') \\ \Sigma^{(\ell)}(x', x) & \Sigma^{(\ell)}(x', x') \end{pmatrix}$$

### 3. Closed-Form Infinite-Width Training
Because $\Theta^{(L)}$ is constant during training in the infinite-width limit, training a deep network with continuous-time gradient descent under MSE loss reduces to linear ODE dynamics with an exact closed-form solution at $t \to \infty$:

$$\mu(x^{\star}) = \Theta^{(L)}(x^{\star}, X) \Big(\Theta^{(L)}(X, X) + \lambda I\Big)^{-1} Y$$

where $(X, Y)$ is the training dataset, $x^{\star}$ is the test input, and $\lambda$ is ridge regularization.

---

## Core Usage Recipes

All examples below have standalone runnable scripts in [`examples/`](examples/).

### 1. Stateful (`tt.nn`) vs. Stateless (`tt.stax`) Construction
See [`examples/1_model_construction.py`](examples/1_model_construction.py):
```python
import torch
import torchtangents as tt

x1, x2 = torch.randn(4, 10), torch.randn(2, 10)

# Option A: Stateful PyTorch-style module (finite forward + infinite kernel)
model = tt.nn.Sequential(tt.nn.Linear(10, 128), tt.nn.ReLU(), tt.nn.Linear(128, 5))
out  = model(x1)                   # Finite-width forward pass: (4, 5)
k_nn = model.inf_kernel(x1, x2)    # Infinite-width NNGP & NTK: (4, 2)

# Option B: Stateless functional stax (infinite kernel only, zero weights allocated)
kernel_fn = tt.stax.Sequential(tt.stax.Linear(128), tt.stax.ReLU(), tt.stax.Linear(5))
k_stax    = kernel_fn(x1, x2)      # Identical NNGP & NTK:      (4, 2)
```

### 2. Kernel Utilities: Empirical NTK, Input Gradients & Hessians
See [`examples/2_kernel_utilities.py`](examples/2_kernel_utilities.py):
```python
x1 = torch.randn(2, 10, requires_grad=True)
x2 = torch.randn(3, 10)

# Analytical infinite-width kernel vs. finite-width empirical NTK
k_inf = model.inf_kernel(x1, x2)     # k_inf.nngp: (2, 3), k_inf.ntk: (2, 3)
k_emp = model.empirical_ntk(x1, x2)  # Shape: (2, 3, 5, 5) -> (N1, N2, Out, Out)

# First-order Jacobians and second-order Hessians of the infinite-width NTK w.r.t. x1
grad_pairs = model.ntk_gradient(x1, x2, mode="all_pairs")  # Shape: (2, 3, 10)
grad_gram  = model.ntk_gradient(x1, x2, mode="gram")       # Shape: (2, 10)
hess_pairs = model.ntk_hessian(x1, x2, mode="all_pairs")   # Shape: (2, 3, 10, 10)
hess_gram  = model.ntk_hessian(x1, x2, mode="gram")        # Shape: (2, 10, 2, 10)
```

### 3. Lifting Existing Vanilla PyTorch Models
See [`examples/3_model_lifting.py`](examples/3_model_lifting.py) and [`examples/example_auto_converter.py`](examples/example_auto_converter.py):
```python
import torch.nn as nn

class ResMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(10, 16)
        self.fc2  = nn.Linear(16, 16)
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h)) + h  # Residual skip connection traces automatically
        return self.head(h)

# Lift into a dual-mode StaxModule
lifted = tt.convert.lift(ResMLP())
ntk    = lifted.inf_kernel(x1, x2, get="ntk")  # Shape: (2, 3)

# Or extract a lightweight stateless kernel function
kernel_fn = tt.convert.infinite_analog(ResMLP())
ntk       = kernel_fn(x1, x2, get="ntk")       # Shape: (2, 3)
```

### 4. Closed-Form Infinite-Width Training & GP Inference
See [`examples/5_infinite_width_prediction.py`](examples/5_infinite_width_prediction.py):
```python
x_train = torch.randn(16, 8)
y_train = torch.sin(x_train[:, :1]).to(torch.float64)
x_test  = torch.randn(4, 8)

kernel_fn     = tt.stax.Sequential(tt.stax.Linear(256), tt.stax.ReLU(), tt.stax.Linear(1))
k_train_train = kernel_fn(x_train, None, get=("nngp", "ntk"))
k_test_train  = kernel_fn(x_test, x_train, get=("nngp", "ntk"))

# Exact Bayesian NNGP & NTK posterior mean
gp_predict = tt.predict.gp_inference(k_train_train, y_train, diag_reg=1e-4)
ntk_mean   = gp_predict(get="ntk", k_test_train=k_test_train).mean   # Shape: (4, 1)

# Exact continuous-time gradient descent trajectory at finite time t=1.0
gd_predict = tt.predict.gradient_descent_mse(k_train_train, y_train, diag_reg=1e-4)
fx_tr, fx_te = gd_predict(
    t=1.0,
    fx_train_0=torch.zeros_like(y_train),
    fx_test_0=torch.zeros(len(x_test), 1, dtype=torch.float64),
    k_test_train=k_test_train,
)
```

### 5. Mini-Batching & `torch.compile` Acceleration
See [`examples/6_batched_and_compiled_kernels.py`](examples/6_batched_and_compiled_kernels.py):
```python
# Compute exact kernels in mini-batches to fit large datasets in GPU memory
batched_kernel_fn = tt.stax.batch(kernel_fn, batch_size=8)
k_batched         = batched_kernel_fn(x_train, x_test, get=("nngp", "ntk"))

# Fuse and accelerate kernel evaluation with torch.compile (Linux / WSL2)
compiled_kernel_fn = tt.convert.optimize_model(kernel_fn)
k_compiled         = compiled_kernel_fn(x_train, x_test, get=("nngp", "ntk"))
```

---

## Platform Support

- **OS:** Windows, Windows + WSL2, Linux
- **Hardware:** CPU, or CPU + NVIDIA GPUs (CUDA 12.x or later)
- *Note: `torch.compile` (`inductor` backend) is supported on Linux and Windows + WSL2.*

---

## Performance

TorchTangents vs. Neural Tangents (JAX): time to calculate the infinite-width NTK of a WideResNet model.

```python
import torchtangents.stax as tt_stax

def WideResnetBlock_tt(channels, strides=(1, 1), channel_mismatch=False):
    Main = tt_stax.serial(
        tt_stax.ReLU(), tt_stax.Conv(channels, (3, 3), strides, padding='SAME'),
        tt_stax.ReLU(), tt_stax.Conv(channels, (3, 3), padding='SAME'))
    Shortcut = tt_stax.Identity() if not channel_mismatch else tt_stax.Conv(
        channels, (3, 3), strides, padding='SAME')
    return tt_stax.serial(
        tt_stax.FanOut(2),
        tt_stax.parallel(Main, Shortcut),
        tt_stax.FanInSum())

def WideResnetGroup_tt(n, channels, strides=(1, 1)):
    blocks = [WideResnetBlock_tt(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks.append(WideResnetBlock_tt(channels, (1, 1)))
    return tt_stax.serial(*blocks)

def WideResnet_tt(block_size, k, num_classes):
    return tt_stax.serial(
        tt_stax.Conv(16, (3, 3), padding='SAME'),
        WideResnetGroup_tt(block_size, int(16 * k)),
        WideResnetGroup_tt(block_size, int(32 * k), (2, 2)),
        WideResnetGroup_tt(block_size, int(64 * k), (2, 2)),
        tt_stax.AvgPool((8, 8)),
        tt_stax.Flatten(),
        tt_stax.Dense(num_classes, 1., 0.))
```

> The speed % column is the proportion faster or slower `torchtangents` is compared to `neural-tangents` (higher is better).
> E.g., `1.0` = same speed, `1.2` = `torchtangents` is 20% faster.

| Platform | Precision | ms / NTK entry | Speed % of neural-tangents | Max Batch Size ($N \times N$) |
| :--- | :--- | :--- | :--- | :--- |
| **CPU** | FP 32 | | | |
| | FP 64 | | | |
| **NVIDIA P100** | FP 16 | | | |
| | FP 32 | | | |
| | FP 64 | | | |
| **NVIDIA V100** | FP 16 | | | |
| | FP 32 | | | |
| | FP 64 | | | |
| **NVIDIA L40S** | TF 16 | | | |
| | FP 16 | | | |
| | TF 32 | | | |
| | FP 32 | | | |
| | TF 64 | | | |
| | FP 64 | | | |
| **NVIDIA A100** | TF 16 | | | |
| | FP 16 | | | |
| | TF 32 | | | |
| | FP 32 | | | |
| | TF 64 | | | |
| | FP 64 | | | |
| **NVIDIA H100** | TF 16 | | | |
| | FP 16 | | | |
| | TF 32 | | | |
| | FP 32 | | | |
| | TF 64 | | | |
| | FP 64 | | | |
| **NVIDIA H200** | TF 16 | | | |
| | FP 16 | | | |
| | TF 32 | | | |
| | FP 32 | | | |
| | TF 64 | | | |
| | FP 64 | | | |

---

## Citation

If you use `torchtangents` in academic work, please cite:

```bibtex
@software{torchtangents2025,
  title        = {torchtangents: Infinite-width neural networks and NTKs in PyTorch},
  author       = {Joshua DeOliveira},
  year         = {2025},
  url          = {https://github.com/deoliveirajoshua/torchtangents}
}
```

Additionally, cite when appropriate these foundational works that `torchtangents` mirrors in PyTorch:

```bibtex
# Infinite width NTK/NNGP:
@inproceedings{neuraltangents2020,
    title={Neural Tangents: Fast and Easy Infinite Neural Networks in Python},
    author={Roman Novak and Lechao Xiao and Jiri Hron and Jaehoon Lee and Alexander A. Alemi and Jascha Sohl-Dickstein and Samuel S. Schoenholz},
    booktitle={International Conference on Learning Representations},
    year={2020},
    pdf={https://arxiv.org/abs/1912.02803},
    url={https://github.com/google/neural-tangents}
}
```

```bibtex
# Finite width, empirical NTK/NNGP:
@inproceedings{novak2022fast,
    title={Fast Finite Width Neural Tangent Kernel},
    author={Roman Novak and Jascha Sohl-Dickstein and Samuel S. Schoenholz},
    booktitle={International Conference on Machine Learning},
    year={2022},
    pdf={https://arxiv.org/abs/2206.08720},
    url={https://github.com/google/neural-tangents}
}
```

```bibtex
# Attention and variable-length inputs:
@inproceedings{hron2020infinite,
    title={Infinite attention: NNGP and NTK for deep attention networks},
    author={Jiri Hron and Yasaman Bahri and Jascha Sohl-Dickstein and Roman Novak},
    booktitle={International Conference on Machine Learning},
    year={2020},
    pdf={https://arxiv.org/abs/2006.10540},
    url={https://github.com/google/neural-tangents}
}
```

```bibtex
# Infinite-width "standard" parameterization:
@misc{sohl2020on,
    title={On the infinite width limit of neural networks with a standard parameterization},
    author={Jascha Sohl-Dickstein and Roman Novak and Samuel S. Schoenholz and Jaehoon Lee},
    publisher = {arXiv},
    year={2020},
    pdf={https://arxiv.org/abs/2001.07301},
    url={https://github.com/google/neural-tangents}
}
```

```bibtex
# Elementwise nonlinearities and sketching:
@inproceedings{han2022fast,
    title={Fast Neural Kernel Embeddings for General Activations},
    author={Insu Han and Amir Zandieh and Jaehoon Lee and Roman Novak and Lechao Xiao and Amin Karbasi},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2022},
    pdf={https://arxiv.org/abs/2209.04121},
    url={https://github.com/google/neural-tangents}
}
```

---

## License & Acknowledgments

**TorchTangents** operates under the open-source **[Apache License 2.0](LICENSE)**.

Because TorchTangents' `stax` sub-module (`torchtangents.stax`) mirrors and ports the core mathematical abstractions of Google's original [Neural Tangents](https://github.com/google/neural-tangents) (JAX), we proudly acknowledge and extend the original computational logic developed by the Google team. For legal attributions, see the `NOTICE` and `LICENSE` files.
