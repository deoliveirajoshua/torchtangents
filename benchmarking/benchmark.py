import os
import sys
import gc
import time
import csv
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torchtangents as tt
from torchtangents import stax
import argparse
from tqdm import tqdm


# Set stdout encoding to utf-8 for proper +/- symbol rendering across OSes
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Suppress internal PyTorch Inductor / compiler deprecation warnings and sympy range fallbacks
warnings.filterwarnings('ignore', category=FutureWarning, module=r'.*torch\._inductor.*')
warnings.filterwarnings('ignore', category=FutureWarning, message=r'.*torch\._prims_common\.check.*')
logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
logging.getLogger("torch.utils._sympy").setLevel(logging.ERROR)

torch._dynamo.config.cache_size_limit = 64  # Allow more metadata variants

def clear_cuda_cache(device=None):
    """Thoroughly free Python garbage, flush CUDA memory allocator cache, and reset peak stats."""
    gc.collect()
    if torch.cuda.is_available():
        if device is not None and getattr(device, 'type', None) == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats(device)
        else:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

def run_benchmark(device, dtype, use_tf32=False, batch_size=128, input_dim=64, model_type='mlp', use_compile=False, num_iters=None, warmup_iters=None, pbar=None, format_label=""):
    clear_cuda_cache(device)
    times = []
    mems = []
    try:
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
            torch.backends.cudnn.allow_tf32 = use_tf32
        
        if model_type == 'mlp':
            layers = []
            # Deep MLP - 10 layers
            for _ in range(10):
                layers.append(stax.Linear(512))
                layers.append(stax.ReLU())
            layers.append(stax.Linear(10))
            tt_model = stax.Sequential(*layers)
            
            x_1 = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
            x_2 = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
            
        elif model_type == 'cnn':
            # Deep CNN
            spatial = input_dim
            tt_model = stax.Sequential(
                stax.Conv(32, (3, 3), padding='SAME'),
                stax.ReLU(),
                stax.Conv(64, (3, 3), padding='SAME'),
                stax.ReLU(),
                stax.Conv(128, (3, 3), padding='SAME'),
                stax.ReLU(),
                stax.AvgPool((2, 2)),
                stax.Flatten(),
                stax.Linear(10)
            )
            x_1 = torch.randn(batch_size, 3, spatial, spatial, device=device, dtype=dtype)
            x_2 = torch.randn(batch_size, 3, spatial, spatial, device=device, dtype=dtype)
            
        elif model_type == 'resnet':
            def WideResnetBlock_tt(channels, strides=(1, 1), channel_mismatch=False):
                Main = stax.Sequential(
                    stax.ReLU(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
                    stax.ReLU(), stax.Conv(channels, (3, 3), padding='SAME'))
                Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
                    channels, (3, 3), strides, padding='SAME')
                return stax.Sequential(stax.FanOut(2),
                                      stax.parallel(Main, Shortcut),
                                      stax.FanInSum())

            def WideResnetGroup_tt(n, channels, strides=(1, 1)):
                blocks = []
                blocks += [WideResnetBlock_tt(channels, strides, channel_mismatch=True)]
                for _ in range(n - 1):
                    blocks += [WideResnetBlock_tt(channels, (1, 1))]
                return stax.Sequential(*blocks)

            def WideResnet_tt(block_size, k, num_classes):
                return stax.Sequential(
                    stax.Conv(16, (3, 3), padding='SAME'),
                    WideResnetGroup_tt(block_size, int(16 * k)),
                    WideResnetGroup_tt(block_size, int(32 * k), (2, 2)),
                    WideResnetGroup_tt(block_size, int(64 * k), (2, 2)),
                    stax.AvgPool((8, 8)),
                    stax.Flatten(),
                    stax.Linear(num_classes, 1., 0.))

            tt_model = WideResnet_tt(block_size=1, k=1, num_classes=10)
            
            spatial = input_dim
            x_1 = torch.randn(batch_size, 3, spatial, spatial, device=device, dtype=dtype)
            x_2 = torch.randn(batch_size, 3, spatial, spatial, device=device, dtype=dtype)
        
        def _get_ntk(a, b):
            # In TorchTangents stax, the module is directly callable to compute kernels
            return tt_model(a, b, get=('ntk',)).ntk
            
        if use_compile:
            try:
                # 'max-autotune' requires Triton, which is often missing on Windows
                # We catch the error to provide a clear explanation.
                _get_ntk = torch.compile(_get_ntk, mode="max-autotune", dynamic=False)
            except Exception as e:
                if pbar is not None:
                    eff_i = num_iters if num_iters is not None else 20
                    pbar.update(eff_i)
                return (f"Compilation Error: {e}", 0.0), (0.0, 0.0)
            
        # Warmup
        if warmup_iters is None:
            warmup_iters = 1 if (device.type == 'cpu' and model_type in ('cnn', 'resnet')) else 3
        if num_iters is None:
            num_iters = 5 if (device.type == 'cpu' and model_type in ('cnn', 'resnet')) else 20
        for _ in range(warmup_iters):
            _ = _get_ntk(x_1, x_2)
            
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            
        # Benchmark per-iteration timing & memory
        times = []
        mems = []
        entries = batch_size * batch_size

        for _ in range(num_iters):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                
            t0 = time.perf_counter()
            _ = _get_ntk(x_1, x_2)
            
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                mems.append(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
            else:
                mems.append(0.0)
                
            t1 = time.perf_counter()
            time_entry_ms = ((t1 - t0) * 1000.0) / entries
            times.append(time_entry_ms)

            if pbar is not None:
                pbar.update(1)
                postfix = {"prec": format_label, "ms/entry": f"{time_entry_ms:.5f}"}
                if device.type == 'cuda':
                    postfix["mem_gb"] = f"{mems[-1]:.3f}"
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
        if "out of memory" in str(e).lower():
            return ("OOM", 0.0), (0.0, 0.0)
        else:
            return (f"Error: {e}", 0.0), (0.0, 0.0)
    except Exception as e:
        if pbar is not None:
            eff_i = num_iters if num_iters is not None else 20
            remaining = eff_i - len(times)
            if remaining > 0:
                pbar.update(remaining)
        return (f"Error: {e}", 0.0), (0.0, 0.0)
    finally:
        clear_cuda_cache(device)

def save_results_to_csv(out_path, results, framework, model, batch_size, input_dim):
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


def main():
    parser = argparse.ArgumentParser(description='Run TorchTangents benchmarks.')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for N_train and N_test')
    parser.add_argument('--input_dim', type=int, default=None, help='Input feature dimension or spatial size for CNNs')
    parser.add_argument('--model', type=str, choices=['mlp', 'cnn', 'resnet'], default='mlp', help='Model architecture to benchmark')
    parser.add_argument('-n', '--num_iters', '--iters', type=int, default=None, dest='num_iters', help='Number of benchmark iterations')
    parser.add_argument('--warmup_iters', type=int, default=None, help='Number of warmup iterations')
    parser.add_argument('--skip_cpu', action='store_true', help='Skip CPU benchmarking')
    parser.add_argument('--compile', action='store_true', help='Also run and compare with torch.compile using max-autotune (can be extremely slow to compile startup)')
    parser.add_argument('-o', '--out', type=str, default=None, help='Path to output CSV file to save benchmark results')
    args = parser.parse_args()
    
    # Model-specific defaults if not explicitly provided
    if args.batch_size is None:
        args.batch_size = 128 if args.model == 'mlp' else (8 if args.model == 'cnn' else 4)
    if args.input_dim is None:
        args.input_dim = 64 if args.model == 'mlp' else (16 if args.model == 'cnn' else 32)
    
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")
        
    configs = [
        {'format': 'FP', 'precision': 16, 'dtype': torch.float16, 'tf32': False},
        {'format': 'FP', 'precision': 32, 'dtype': torch.float32, 'tf32': False},
        {'format': 'FP', 'precision': 64, 'dtype': torch.float64, 'tf32': False},
    ]
    
    cuda_configs = [
        {'format': 'FP', 'precision': 16, 'dtype': torch.float16, 'tf32': False},
        {'format': 'BF', 'precision': 16, 'dtype': torch.bfloat16, 'tf32': False},
        {'format': 'FP', 'precision': 32, 'dtype': torch.float32, 'tf32': False},
        {'format': 'TF', 'precision': 32, 'dtype': torch.float32, 'tf32': True},
        {'format': 'FP', 'precision': 64, 'dtype': torch.float64, 'tf32': False},
    ]
    
    results = []
    
    if not args.skip_cpu:
        eff_iters = args.num_iters if args.num_iters is not None else (5 if args.model in ('cnn', 'resnet') else 20)
        total_cpu_iters = len(configs) * eff_iters
        print(f"Running benchmarks for CPU... (model={args.model}, batch_size={args.batch_size}, input_dim={args.input_dim}, iters={eff_iters})")
        with tqdm(total=total_cpu_iters, desc="Benchmarking CPU", unit="iter") as pbar:
            for cfg in configs:
                label = f"{cfg['format']}{cfg['precision']}"
                pbar.set_description(f"CPU ({label})")
                res, mem = run_benchmark(
                    torch.device('cpu'), cfg['dtype'], cfg['tf32'],
                    args.batch_size, args.input_dim, args.model,
                    num_iters=args.num_iters, warmup_iters=args.warmup_iters,
                    pbar=pbar, format_label=label
                )
                results.append({
                    'Platform': 'CPU',
                    'Format': cfg['format'],
                    'Precision': cfg['precision'],
                    'Time': res,
                    'Memory': mem
                })
        
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        eff_iters = args.num_iters if args.num_iters is not None else 20
        valid_cuda_configs = [
            cfg for cfg in cuda_configs
            if not (cfg['dtype'] == torch.bfloat16 and not torch.cuda.is_bf16_supported())
        ]
        runs_per_config = 2 if args.compile else 1
        total_cuda_iters = len(valid_cuda_configs) * eff_iters * runs_per_config
        format_info = f"{eff_iters} iters x 2 (uncompiled + compiled)" if args.compile else f"{eff_iters} iters"
        print(f"Running benchmarks for {gpu_name}... (model={args.model}, batch_size={args.batch_size}, input_dim={args.input_dim}, {format_info})")
        with tqdm(total=total_cuda_iters, desc=f"Benchmarking {gpu_name}", unit="iter") as pbar:
            for cfg in valid_cuda_configs:
                label = f"{cfg['format']}{cfg['precision']}"
                
                # Uncompiled benchmark
                clear_cuda_cache(torch.device('cuda'))
                uncomp_desc = f"{gpu_name} ({label} uncompiled)" if args.compile else f"{gpu_name} ({label})"
                pbar.set_description(uncomp_desc)
                res, mem = run_benchmark(
                    torch.device('cuda'), cfg['dtype'], cfg['tf32'],
                    args.batch_size, args.input_dim, args.model,
                    use_compile=False, num_iters=args.num_iters, warmup_iters=args.warmup_iters,
                    pbar=pbar, format_label=label
                )
                clear_cuda_cache(torch.device('cuda'))
                
                # Compiled benchmark if requested
                res_compiled, mem_compiled = None, None
                if args.compile:
                    clear_cuda_cache(torch.device('cuda'))
                    pbar.set_description(f"{gpu_name} ({label} compiled)")
                    res_compiled, mem_compiled = run_benchmark(
                        torch.device('cuda'), cfg['dtype'], cfg['tf32'],
                        args.batch_size, args.input_dim, args.model,
                        use_compile=True, num_iters=args.num_iters, warmup_iters=args.warmup_iters,
                        pbar=pbar, format_label=f"{label}-comp"
                    )
                    clear_cuda_cache(torch.device('cuda'))
                
                results.append({
                    'Platform': gpu_name,
                    'Format': cfg['format'],
                    'Precision': cfg['precision'],
                    'Time': res,
                    'Memory': mem,
                    'TimeCompiled': res_compiled,
                    'MemoryCompiled': mem_compiled
                })
    

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

    print(f"\n--- Performance Table ({args.model.upper()}) ---")
    
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
                # Fallback for CPU or cases where compiled failed/was skipped
                print(f"| {plat_display:<{platform_width}} | {r['Format']:<6} | {r['Precision']:<4} | {time_str:<26} | {mem_str:<16} | {'N/A':<26} | {'N/A':<16} | {'--':<8} |")
        else:
            print(f"| {plat_display:<{platform_width}} | {r['Format']:<6} | {r['Precision']:<4} | {time_str:<28} | {mem_str:<16} |")
            
        current_platform = r['Platform']

    if args.out:
        save_results_to_csv(args.out, results, 'TorchTangents', args.model, args.batch_size, args.input_dim)

if __name__ == "__main__":
    main()
