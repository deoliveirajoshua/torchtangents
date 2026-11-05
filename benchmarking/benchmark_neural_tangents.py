import os
import sys
import time
import csv
import logging
import warnings
import numpy as np

# Set stdout encoding to utf-8 for proper +/- symbol rendering across OSes
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Suppress TF, tf2jax, and Neural Tangents warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import jax
import jax.numpy as jnp
import neural_tangents.stax as jax_stax
import argparse
from tqdm import tqdm


def get_device_name(dev):
    """Return a human-readable platform name for a JAX device."""
    if dev.platform == 'cpu':
        return 'CPU'
    if hasattr(dev, 'device_kind') and dev.device_kind:
        return dev.device_kind
    return f"{dev.platform.upper()} ({dev.id})"


def get_device_peak_mem_gb(dev):
    """Return peak memory allocated on device in GB, or 0.0 if unsupported."""
    if hasattr(dev, 'memory_stats'):
        try:
            stats = dev.memory_stats()
            if stats:
                if 'peak_bytes_in_use' in stats:
                    return stats['peak_bytes_in_use'] / (1024 ** 3)
                elif 'bytes_in_use' in stats:
                    return stats['bytes_in_use'] / (1024 ** 3)
        except Exception:
            pass
    return 0.0


def reset_device_peak_mem(dev):
    """Reset device peak memory stats if supported."""
    if hasattr(dev, 'reset_peak_memory_stats'):
        try:
            dev.reset_peak_memory_stats()
        except Exception:
            pass


def save_results_to_csv(out_path, results, framework, model, batch_size, input_dim):
    """Export benchmark telemetry and configuration to CSV."""
    parent_dir = os.path.dirname(out_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fieldnames = [
        'Framework',
        'Platform',
        'Model',
        'BatchSize',
        'InputDim',
        'Format',
        'Precision',
        'TimeMean_ms',
        'TimeStd_ms',
        'MemoryMean_GB',
        'MemoryStd_GB',
        'TimeCompiledMean_ms',
        'TimeCompiledStd_ms',
        'MemoryCompiledMean_GB',
        'MemoryCompiledStd_GB',
        'Speedup',
    ]

    def parse_stat(val):
        if isinstance(val, (tuple, list)):
            mean, std = val
            if isinstance(mean, (float, int)):
                return mean, std
            return mean, ""
        elif isinstance(val, (float, int)):
            return val, ""
        return val, ""

    rows = []
    for r in results:
        t_mean, t_std = parse_stat(r.get('Time'))
        m_mean, m_std = parse_stat(r.get('Memory'))

        tc_mean, tc_std = "", ""
        mc_mean, mc_std = "", ""
        speedup = ""

        if r.get('TimeCompiled') is not None:
            tc_mean, tc_std = parse_stat(r.get('TimeCompiled'))
            mc_mean, mc_std = parse_stat(r.get('MemoryCompiled'))
            if isinstance(t_mean, (float, int)) and isinstance(tc_mean, (float, int)) and tc_mean > 0:
                speedup = f"{t_mean / tc_mean:.2f}x"

        row = {
            'Framework': framework,
            'Platform': r['Platform'],
            'Model': model,
            'BatchSize': batch_size,
            'InputDim': input_dim,
            'Format': r['Format'],
            'Precision': r['Precision'],
            'TimeMean_ms': t_mean,
            'TimeStd_ms': t_std,
            'MemoryMean_GB': m_mean,
            'MemoryStd_GB': m_std,
            'TimeCompiledMean_ms': tc_mean,
            'TimeCompiledStd_ms': tc_std,
            'MemoryCompiledMean_GB': mc_mean,
            'MemoryCompiledStd_GB': mc_std,
            'Speedup': speedup,
        }
        rows.append(row)

    with open(out_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] Benchmark results saved to: {out_path}")


def run_benchmark(device, dtype, matmul_precision='default', batch_size=128, input_dim=64,
                  model_type='mlp', use_compile=False, num_iters=None, warmup_iters=None,
                  pbar=None, format_label=""):
    times = []
    mems = []
    try:
        if matmul_precision != 'default':
            jax.config.update("jax_default_matmul_precision", matmul_precision)
        else:
            jax.config.update("jax_default_matmul_precision", None)

        with jax.default_device(device):
            key1 = jax.random.PRNGKey(101)
            key2 = jax.random.PRNGKey(202)

            if model_type == 'mlp':
                layers = []
                # Deep MLP - 10 layers
                for _ in range(10):
                    layers.append(jax_stax.Dense(512))
                    layers.append(jax_stax.Relu())
                layers.append(jax_stax.Dense(10))
                _, _, kernel_fn = jax_stax.serial(*layers)

                x_1 = jax.random.normal(key1, (batch_size, input_dim), dtype=dtype)
                x_2 = jax.random.normal(key2, (batch_size, input_dim), dtype=dtype)

            elif model_type == 'cnn':
                spatial = input_dim
                _, _, kernel_fn = jax_stax.serial(
                    jax_stax.Conv(32, (3, 3), padding='SAME'),
                    jax_stax.Relu(),
                    jax_stax.Conv(64, (3, 3), padding='SAME'),
                    jax_stax.Relu(),
                    jax_stax.Conv(128, (3, 3), padding='SAME'),
                    jax_stax.Relu(),
                    jax_stax.AvgPool((2, 2)),
                    jax_stax.Flatten(),
                    jax_stax.Dense(10)
                )
                # NHWC format expected by Neural Tangents
                x_1 = jax.random.normal(key1, (batch_size, spatial, spatial, 3), dtype=dtype)
                x_2 = jax.random.normal(key2, (batch_size, spatial, spatial, 3), dtype=dtype)

            elif model_type == 'resnet':
                def WideResnetBlock_jax(channels, strides=(1, 1), channel_mismatch=False):
                    Main = jax_stax.serial(
                        jax_stax.Relu(), jax_stax.Conv(channels, (3, 3), strides, padding='SAME'),
                        jax_stax.Relu(), jax_stax.Conv(channels, (3, 3), padding='SAME'))
                    Shortcut = jax_stax.Identity() if not channel_mismatch else jax_stax.Conv(
                        channels, (3, 3), strides, padding='SAME')
                    return jax_stax.serial(jax_stax.FanOut(2),
                                           jax_stax.parallel(Main, Shortcut),
                                           jax_stax.FanInSum())

                def WideResnetGroup_jax(n, channels, strides=(1, 1)):
                    blocks = []
                    blocks += [WideResnetBlock_jax(channels, strides, channel_mismatch=True)]
                    for _ in range(n - 1):
                        blocks += [WideResnetBlock_jax(channels, (1, 1))]
                    return jax_stax.serial(*blocks)

                def WideResnet_jax(block_size, k, num_classes):
                    return jax_stax.serial(
                        jax_stax.Conv(16, (3, 3), padding='SAME'),
                        WideResnetGroup_jax(block_size, int(16 * k)),
                        WideResnetGroup_jax(block_size, int(32 * k), (2, 2)),
                        WideResnetGroup_jax(block_size, int(64 * k), (2, 2)),
                        jax_stax.AvgPool((8, 8)),
                        jax_stax.Flatten(),
                        jax_stax.Dense(num_classes, 1., 0.))

                _, _, kernel_fn = WideResnet_jax(block_size=1, k=1, num_classes=10)

                spatial = input_dim
                x_1 = jax.random.normal(key1, (batch_size, spatial, spatial, 3), dtype=dtype)
                x_2 = jax.random.normal(key2, (batch_size, spatial, spatial, 3), dtype=dtype)

            def _raw_get_ntk(a, b):
                res = kernel_fn(a, b, get='ntk')
                if hasattr(res, 'ntk'):
                    return res.ntk
                return res

            if use_compile:
                try:
                    _get_ntk = jax.jit(_raw_get_ntk)
                except Exception as e:
                    if pbar is not None:
                        eff_i = num_iters if num_iters is not None else 20
                        pbar.update(eff_i)
                    return (f"Compilation Error: {e}", 0.0), (0.0, 0.0)
            else:
                _get_ntk = _raw_get_ntk

            # Warmup
            if warmup_iters is None:
                warmup_iters = 1 if (device.platform == 'cpu' and model_type in ('cnn', 'resnet')) else 3
            if num_iters is None:
                num_iters = 5 if (device.platform == 'cpu' and model_type in ('cnn', 'resnet')) else 20

            for _ in range(warmup_iters):
                out = _get_ntk(x_1, x_2)
                if hasattr(out, 'block_until_ready'):
                    out.block_until_ready()

            reset_device_peak_mem(device)

            # Benchmark per-iteration timing & memory
            times = []
            mems = []
            entries = batch_size * batch_size

            for _ in range(num_iters):
                reset_device_peak_mem(device)

                t0 = time.perf_counter()
                out = _get_ntk(x_1, x_2)
                if hasattr(out, 'block_until_ready'):
                    out.block_until_ready()
                t1 = time.perf_counter()

                mem_gb = get_device_peak_mem_gb(device)
                mems.append(mem_gb)

                time_entry_ms = ((t1 - t0) * 1000.0) / entries
                times.append(time_entry_ms)

                if pbar is not None:
                    pbar.update(1)
                    postfix = {"prec": format_label, "ms/entry": f"{time_entry_ms:.5f}"}
                    if device.platform != 'cpu':
                        postfix["mem_gb"] = f"{mem_gb:.3f}"
                    pbar.set_postfix(postfix)

            time_mean = float(np.mean(times))
            time_std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0

            mem_mean = float(np.mean(mems)) if mems else 0.0
            mem_std = float(np.std(mems, ddof=1)) if len(mems) > 1 else 0.0

            return (time_mean, time_std), (mem_mean, mem_std)

    except RuntimeError as e:
        if pbar is not None:
            eff_i = num_iters if num_iters is not None else 20
            remaining = eff_i - len(times)
            if remaining > 0:
                pbar.update(remaining)
        err_lower = str(e).lower()
        if "out of memory" in err_lower or "resourceexhausted" in err_lower:
            return ("OOM", 0.0), (0.0, 0.0)
        else:
            return (f"Error: {e}", 0.0), (0.0, 0.0)
    except Exception as e:
        if pbar is not None:
            eff_i = num_iters if num_iters is not None else 20
            remaining = eff_i - len(times)
            if remaining > 0:
                pbar.update(remaining)
        err_lower = str(e).lower()
        if "out of memory" in err_lower or "resourceexhausted" in err_lower:
            return ("OOM", 0.0), (0.0, 0.0)
        return (f"Error: {e}", 0.0), (0.0, 0.0)


def main():
    parser = argparse.ArgumentParser(description='Run Google Neural Tangents (JAX) benchmarks.')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for N_train and N_test')
    parser.add_argument('--input_dim', type=int, default=None, help='Input feature dimension or spatial size for CNNs')
    parser.add_argument('--model', type=str, choices=['mlp', 'cnn', 'resnet'], default='mlp', help='Model architecture to benchmark')
    parser.add_argument('-n', '--num_iters', '--iters', type=int, default=None, dest='num_iters', help='Number of benchmark iterations')
    parser.add_argument('--warmup_iters', type=int, default=None, help='Number of warmup iterations')
    parser.add_argument('--skip_cpu', action='store_true', help='Skip CPU benchmarking')
    parser.add_argument('--compile', '--jit', action='store_true', dest='compile', help='Also run and compare with jax.jit compilation')
    parser.add_argument('-o', '--out', type=str, default=None, help='Path to output CSV file to save benchmark results')
    args = parser.parse_args()

    # Model-specific defaults if not explicitly provided
    if args.batch_size is None:
        args.batch_size = 128 if args.model == 'mlp' else (8 if args.model == 'cnn' else 4)
    if args.input_dim is None:
        args.input_dim = 64 if args.model == 'mlp' else (16 if args.model == 'cnn' else 32)

    cpu_device = jax.devices('cpu')[0]

    gpu_device = None
    try:
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            gpu_device = gpu_devices[0]
            print(f"JAX GPU device is available: {get_device_name(gpu_device)}")
    except Exception:
        pass

    if gpu_device is None:
        print("JAX GPU device is not available (running with CPU backend).")

    configs = [
        {'format': 'FP', 'precision': 16, 'dtype': jnp.float16, 'matmul_precision': 'default'},
        {'format': 'FP', 'precision': 32, 'dtype': jnp.float32, 'matmul_precision': 'default'},
        {'format': 'FP', 'precision': 64, 'dtype': jnp.float64, 'matmul_precision': 'default'},
    ]

    gpu_configs = [
        {'format': 'FP', 'precision': 16, 'dtype': jnp.float16, 'matmul_precision': 'default'},
        {'format': 'BF', 'precision': 16, 'dtype': jnp.bfloat16, 'matmul_precision': 'default'},
        {'format': 'FP', 'precision': 32, 'dtype': jnp.float32, 'matmul_precision': 'highest'},
        {'format': 'TF', 'precision': 32, 'dtype': jnp.float32, 'matmul_precision': 'tensorfloat32'},
        {'format': 'FP', 'precision': 64, 'dtype': jnp.float64, 'matmul_precision': 'default'},
    ]

    results = []

    if not args.skip_cpu:
        eff_iters = args.num_iters if args.num_iters is not None else (5 if args.model in ('cnn', 'resnet') else 20)
        runs_per_config = 2 if args.compile else 1
        total_cpu_iters = len(configs) * eff_iters * runs_per_config
        format_info = f"{eff_iters} iters x 2 (uncompiled + jax.jit)" if args.compile else f"{eff_iters} iters"
        print(f"Running benchmarks for CPU... (model={args.model}, batch_size={args.batch_size}, input_dim={args.input_dim}, {format_info})")
        with tqdm(total=total_cpu_iters, desc="Benchmarking CPU", unit="iter") as pbar:
            for cfg in configs:
                label = f"{cfg['format']}{cfg['precision']}"
                uncomp_desc = f"CPU ({label} uncompiled)" if args.compile else f"CPU ({label})"
                pbar.set_description(uncomp_desc)

                res, mem = run_benchmark(
                    cpu_device, cfg['dtype'], cfg['matmul_precision'],
                    args.batch_size, args.input_dim, args.model,
                    use_compile=False, num_iters=args.num_iters, warmup_iters=args.warmup_iters,
                    pbar=pbar, format_label=label
                )

                res_compiled, mem_compiled = None, None
                if args.compile:
                    pbar.set_description(f"CPU ({label} jax.jit)")
                    res_compiled, mem_compiled = run_benchmark(
                        cpu_device, cfg['dtype'], cfg['matmul_precision'],
                        args.batch_size, args.input_dim, args.model,
                        use_compile=True, num_iters=args.num_iters, warmup_iters=args.warmup_iters,
                        pbar=pbar, format_label=f"{label}-jit"
                    )

                results.append({
                    'Platform': 'CPU',
                    'Format': cfg['format'],
                    'Precision': cfg['precision'],
                    'Time': res,
                    'Memory': mem,
                    'TimeCompiled': res_compiled,
                    'MemoryCompiled': mem_compiled
                })

    if gpu_device is not None:
        gpu_name = get_device_name(gpu_device)
        eff_iters = args.num_iters if args.num_iters is not None else 20
        runs_per_config = 2 if args.compile else 1
        total_gpu_iters = len(gpu_configs) * eff_iters * runs_per_config
        format_info = f"{eff_iters} iters x 2 (uncompiled + jax.jit)" if args.compile else f"{eff_iters} iters"
        print(f"Running benchmarks for {gpu_name}... (model={args.model}, batch_size={args.batch_size}, input_dim={args.input_dim}, {format_info})")
        with tqdm(total=total_gpu_iters, desc=f"Benchmarking {gpu_name}", unit="iter") as pbar:
            for cfg in gpu_configs:
                label = f"{cfg['format']}{cfg['precision']}"

                # Uncompiled benchmark
                uncomp_desc = f"{gpu_name} ({label} uncompiled)" if args.compile else f"{gpu_name} ({label})"
                pbar.set_description(uncomp_desc)
                res, mem = run_benchmark(
                    gpu_device, cfg['dtype'], cfg['matmul_precision'],
                    args.batch_size, args.input_dim, args.model,
                    use_compile=False, num_iters=args.num_iters, warmup_iters=args.warmup_iters,
                    pbar=pbar, format_label=label
                )

                # Compiled benchmark if requested
                res_compiled, mem_compiled = None, None
                if args.compile:
                    pbar.set_description(f"{gpu_name} ({label} jax.jit)")
                    res_compiled, mem_compiled = run_benchmark(
                        gpu_device, cfg['dtype'], cfg['matmul_precision'],
                        args.batch_size, args.input_dim, args.model,
                        use_compile=True, num_iters=args.num_iters, warmup_iters=args.warmup_iters,
                        pbar=pbar, format_label=f"{label}-jit"
                    )

                results.append({
                    'Platform': gpu_name,
                    'Format': cfg['format'],
                    'Precision': cfg['precision'],
                    'Time': res,
                    'Memory': mem,
                    'TimeCompiled': res_compiled,
                    'MemoryCompiled': mem_compiled
                })

    if not results:
        print("No benchmarks were run.")
        return

    # Calculate size of Platform name for table formatting
    platform_width = max(20, max(len(r['Platform']) for r in results))

    def format_stat(val, prec=6):
        if isinstance(val, (tuple, list)):
            mean, std = val
            if isinstance(mean, (float, int)) and isinstance(std, (float, int)):
                return f"{mean:.{prec}f} ± {std:.{prec}f}"
            return str(mean)
        if isinstance(val, (float, int)):
            return f"{val:.{prec}f}"
        return str(val)

    def get_mean(val):
        if isinstance(val, (tuple, list)):
            return val[0]
        return val

    print(f"\n--- Neural Tangents Performance Table ({args.model.upper()}) ---")

    if args.compile:
        # Grouped Header with Mean ± Std
        print(f"| {'Platform':<{platform_width}} | {'Format':<6} | {'Prec':<4} | {'Uncompiled (ms)':<26} | {'(GB)':<16} | {'Compiled (ms)':<26} | {'(GB)':<16} | {'Speedup':<8} |")
        print(f"| :{'-'*(platform_width-1)} | :{'-'*5} | :{'-'*3} | :{'-'*25} | :{'-'*15} | :{'-'*25} | :{'-'*15} | :{'-'*7} |")
    else:
        # Standard Header with Mean ± Std
        print(f"| {'Platform':<{platform_width}} | {'Format':<6} | {'Prec':<4} | {'Milliseconds / NTK entry':<28} | {'Memory (GB)':<16} |")
        print(f"| :{'-'*(platform_width-1)} | :{'-'*5} | :{'-'*3} | :{'-'*27} | :{'-'*15} |")

    current_platform = None
    for r in results:
        # Platform display optimization
        plat_display = r['Platform'] if r['Platform'] != current_platform else ""
        if r['Platform'] != current_platform and current_platform is not None:
            if args.compile:
                print(f"| {'-'*platform_width} | {'-'*6} | {'-'*4} | {'-'*26} | {'-'*16} | {'-'*26} | {'-'*16} | {'-'*8} |")
            else:
                print(f"| {'-'*platform_width} | {'-'*6} | {'-'*4} | {'-'*28} | {'-'*16} |")

        time_str = format_stat(r['Time'], prec=6)
        mem_str = format_stat(r['Memory'], prec=3)

        if args.compile:
            if r.get('TimeCompiled') is not None:
                time_comp_str = format_stat(r['TimeCompiled'], prec=6)
                mem_comp_str = format_stat(r['MemoryCompiled'], prec=3)
                speedup = "--"
                t_uncomp = get_mean(r['Time'])
                t_comp = get_mean(r['TimeCompiled'])
                if isinstance(t_uncomp, (float, int)) and isinstance(t_comp, (float, int)) and t_comp > 0:
                    speedup = f"{t_uncomp / t_comp:.2f}x"

                print(f"| {plat_display:<{platform_width}} | {r['Format']:<6} | {r['Precision']:<4} | {time_str:<26} | {mem_str:<16} | {time_comp_str:<26} | {mem_comp_str:<16} | {speedup:<8} |")
            else:
                print(f"| {plat_display:<{platform_width}} | {r['Format']:<6} | {r['Precision']:<4} | {time_str:<26} | {mem_str:<16} | {'N/A':<26} | {'N/A':<16} | {'--':<8} |")
        else:
            print(f"| {plat_display:<{platform_width}} | {r['Format']:<6} | {r['Precision']:<4} | {time_str:<28} | {mem_str:<16} |")

        current_platform = r['Platform']

    if args.out:
        save_results_to_csv(args.out, results, 'NeuralTangents', args.model, args.batch_size, args.input_dim)


if __name__ == "__main__":
    main()

