# Developer & Benchmark Installation Instructions

## Developer & Testing Installation

The testing suite compares TorchTangents against the original JAX-based Neural Tangents library. Because specific older versions of `jax` are required for `neural-tangents`, **you must run the test suite within a Linux environment (like WSL or a Docker container) for reproducibility.**

1. Create and activate a fresh Python environment on your Linux machine (e.g., a conda environment).
```bash
conda create -y -n tt_test_env python=3.11
conda activate tt_test_env
```

2. Download the repo, extract the downloaded zip, and navigate into the main directory:
```bash
unzip torchtangents-0.1.1.zip
cd torchtangents-0.1.1
```

3. Install the dependencies:
- **Pure PyTorch development, testing & benchmarks** (works on Windows, macOS, and Linux):
```bash
pip install -e ".[dev]"
```
- **Full JAX-parity comparison suite** (requires Linux/WSL with CUDA for JAX pinning):
```bash
pip install -e ".[parity]"
```
- **Install literally everything** (tests, benchmarks, and JAX parity):
```bash
pip install -e ".[all]"
```

4. To execute the test suite:
- **Pure PyTorch unit & integration tests** (35 tests, ~1.5s, no JAX required):
```bash
pytest tests/nn/
```
- **Full JAX-parity testbed** (101 tests, requires WSL/Linux with JAX):
```bash
pytest tests/stax_parity/
```
> Note: On Windows or if encountering DLL relocation errors with `pytest.exe`, invoke pytest directly via python:
> ```bash
> python -m pytest tests/nn/
> ```

5. To check code coverage:
Ensure `pytest-cov` is installed (included in `[test]`, or run `pip install pytest-cov`):
- **For pure PyTorch tests:**
```bash
pytest --cov=torchtangents --cov-report=html tests/nn/
```
- **For full repository coverage** (within WSL/Linux environment):
```bash
pytest --cov=torchtangents --cov-report=html tests/
```
This generates an interactive HTML report at `htmlcov/index.html` detailing line-by-line coverage.

## Benchmark Installation & Usage

To install the benchmarking dependencies (including `tqdm` for interactive progress bars), run:

```bash
pip install -e ".[benchmark]"
```

You can then run the benchmarking suite to evaluate wall-clock latency ($\text{Mean} \pm \text{Std Dev}$) and peak memory usage across precision formats (FP16, BF16, FP32, TF32, FP64):

```bash
cd benchmarking
python benchmark.py --model mlp --batch_size 128 --input_dim 64
```

### Command-Line Flags

The script supports the following command-line arguments:

| Flag | Type / Choices | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `mlp`, `cnn`, `resnet` | `mlp` | Network architecture to benchmark. |
| `--batch_size` | `int` | Auto (128 for `mlp`, 8 for `cnn`, 4 for `resnet`) | Batch size for $N_{\text{train}}$ and $N_{\text{test}}$. |
| `--input_dim` | `int` | Auto (64 for `mlp`, 16 for `cnn`, 32 for `resnet`) | Feature dimension (for `mlp`) or spatial image resolution $H=W$ (for `cnn` / `resnet`). |
| `-n`, `--num_iters`, `--iters` | `int` | Auto (20 for GPU / MLP; 5 for CPU on CNN/ResNet) | Number of timed benchmark iterations used to compute $\text{Mean} \pm \text{Std Dev}$. |
| `--warmup_iters` | `int` | Auto (5 for GPU / MLP; 2 for CPU on CNN/ResNet) | Number of untimed warmup iterations to initialize CUDA contexts and JIT caches. |
| `--compile` | `flag` | `False` | Compares standard PyTorch against `torch.compile(mode='max-autotune')`. |
| `--skip_cpu` | `flag` | `False` | Excludes CPU benchmarks (recommended for fast GPU evaluation or spatial models). |

### Recommended Benchmark Recipes

**1. Fast GPU benchmark on 10-layer MLP with `torch.compile`:**
```bash
python benchmark.py --model mlp -n 10 --skip_cpu --compile
```

**2. 5-layer CNN benchmark on GPU (uses auto batch size 8 and $16 \times 16$ spatial resolution):**
```bash
python benchmark.py --model cnn -n 5 --skip_cpu
```

**3. WideResNet benchmark on GPU (with 3 timed iterations):**
```bash
python benchmark.py --model resnet -n 3 --skip_cpu
```

