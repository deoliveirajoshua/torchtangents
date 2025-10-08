#!/usr/bin/env python3
"""
TorchTangents Benchmarking Suite

Comprehensive benchmarking tools for measuring computation time and memory usage
of various TorchTangents operations including NTK/NNGP computation, batching,
and kernel functions.
"""

import time
import psutil
import torch
import torch.nn as nn
import gc
import os
import json
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import tracemalloc

import torchtangents as tt


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation: str
    config: Dict[str, Any]
    time_seconds: float
    memory_mb: float
    peak_memory_mb: float
    gpu_memory_mb: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MemoryTracker:
    """Context manager for tracking memory usage."""
    
    def __init__(self, track_gpu: bool = True):
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.start_memory = 0
        self.peak_memory = 0
        self.start_gpu_memory = 0
        self.peak_gpu_memory = 0
        
    def __enter__(self):
        # Clear cache and force garbage collection
        gc.collect()
        if self.track_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Start tracking
        tracemalloc.start()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        if self.track_gpu:
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        self.peak_memory = peak / 1024 / 1024  # MB
        tracemalloc.stop()
        
        if self.track_gpu:
            self.peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    def get_memory_usage(self) -> Tuple[float, float, Optional[float], Optional[float]]:
        """Get memory usage: (current_cpu, peak_cpu, current_gpu, peak_gpu) in MB."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024 - self.start_memory
        current_gpu = None
        
        if self.track_gpu:
            current_gpu = torch.cuda.memory_allocated() / 1024 / 1024 - self.start_gpu_memory
            
        return current_memory, self.peak_memory, current_gpu, self.peak_gpu_memory


@contextmanager
def timer():
    """Simple context manager for timing operations."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    return end - start


class TorchTangentsBenchmark:
    """Main benchmarking class for TorchTangents operations."""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._select_device(device)
        self.results: List[BenchmarkResult] = []
        
    def _select_device(self, device: str) -> torch.device:
        """Select appropriate device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def benchmark_ntk_computation(self, 
                                 input_dims: List[int] = [10, 50, 100],
                                 hidden_dims: List[int] = [64, 128, 256],
                                 batch_sizes: List[int] = [8, 16, 32, 64],
                                 output_dim: int = 10,
                                 num_runs: int = 3) -> List[BenchmarkResult]:
        """Benchmark NTK computation across different configurations."""
        
        results = []
        
        for input_dim in input_dims:
            for hidden_dim in hidden_dims:
                for batch_size in batch_sizes:
                    config = {
                        'input_dim': input_dim,
                        'hidden_dim': hidden_dim, 
                        'batch_size': batch_size,
                        'output_dim': output_dim,
                        'device': str(self.device)
                    }
                    
                    print(f"Benchmarking NTK: {config}")
                    
                    # Create model and data
                    model = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, output_dim)
                    ).to(self.device)
                    
                    x = torch.randn(batch_size, input_dim, device=self.device)
                    
                    # Run benchmark
                    result = self._benchmark_operation(
                        operation_name="ntk_computation",
                        operation_func=self._compute_ntk_stax,
                        config=config,
                        model=model,
                        x=x,
                        num_runs=num_runs
                    )
                    
                    results.append(result)
                    self.results.append(result)
        
        return results
    
    def benchmark_nngp_computation(self,
                                  input_dims: List[int] = [10, 50, 100],
                                  hidden_dims: List[int] = [64, 128, 256], 
                                  batch_sizes: List[int] = [8, 16, 32, 64],
                                  output_dim: int = 10,
                                  num_runs: int = 3) -> List[BenchmarkResult]:
        """Benchmark NNGP computation across different configurations."""
        
        results = []
        
        for input_dim in input_dims:
            for hidden_dim in hidden_dims:
                for batch_size in batch_sizes:
                    config = {
                        'input_dim': input_dim,
                        'hidden_dim': hidden_dim,
                        'batch_size': batch_size, 
                        'output_dim': output_dim,
                        'device': str(self.device)
                    }
                    
                    print(f"Benchmarking NNGP: {config}")
                    
                    # Create model and data
                    model = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim)
                    ).to(self.device)
                    
                    x = torch.randn(batch_size, input_dim, device=self.device)
                    
                    # Run benchmark
                    result = self._benchmark_operation(
                        operation_name="nngp_computation",
                        operation_func=self._compute_nngp_stax,
                        config=config,
                        model=model,
                        x=x,
                        num_runs=num_runs
                    )
                    
                    results.append(result)
                    self.results.append(result)
        
        return results
    
    def benchmark_batching_performance(self,
                                     batch_sizes: List[int] = [1, 2, 4, 8, 16],
                                     dataset_size: int = 128,
                                     input_dim: int = 50,
                                     hidden_dim: int = 128,
                                     num_runs: int = 3) -> List[BenchmarkResult]:
        """Benchmark batching performance for large datasets."""
        
        results = []
        
        # Create model and full dataset
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        ).to(self.device)
        
        x_full = torch.randn(dataset_size, input_dim, device=self.device)
        
        for batch_size in batch_sizes:
            config = {
                'batch_size': batch_size,
                'dataset_size': dataset_size,
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'device': str(self.device)
            }
            
            print(f"Benchmarking batching: batch_size={batch_size}")
            
            # Run benchmark
            result = self._benchmark_operation(
                operation_name="batching_performance",
                operation_func=self._compute_batched_ntk,
                config=config,
                model=model,
                x=x_full,
                batch_size=batch_size,
                num_runs=num_runs
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def benchmark_empirical_vs_infinite(self,
                                       widths: List[int] = [64, 128, 256, 512, 1024],
                                       input_dim: int = 20,
                                       batch_size: int = 16,
                                       num_runs: int = 3) -> List[BenchmarkResult]:
        """Benchmark empirical vs infinite width kernel computation."""
        
        results = []
        
        for width in widths:
            # Empirical kernel
            config_emp = {
                'width': width,
                'input_dim': input_dim,
                'batch_size': batch_size,
                'kernel_type': 'empirical',
                'device': str(self.device)
            }
            
            model = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.ReLU(),
                nn.Linear(width, 1)
            ).to(self.device)
            
            x = torch.randn(batch_size, input_dim, device=self.device)
            
            print(f"Benchmarking empirical kernel: width={width}")
            
            result_emp = self._benchmark_operation(
                operation_name="empirical_kernel",
                operation_func=self._compute_empirical_ntk,
                config=config_emp,
                model=model,
                x=x,
                num_runs=num_runs
            )
            
            results.append(result_emp)
            self.results.append(result_emp)
            
            # Infinite width kernel (using stax)
            config_inf = config_emp.copy()
            config_inf['kernel_type'] = 'infinite'
            
            print(f"Benchmarking infinite kernel: width={width}")
            
            result_inf = self._benchmark_operation(
                operation_name="infinite_kernel", 
                operation_func=self._compute_infinite_ntk,
                config=config_inf,
                width=width,
                input_dim=input_dim,
                x=x,
                num_runs=num_runs
            )
            
            results.append(result_inf)
            self.results.append(result_inf)
        
        return results
    
    def _benchmark_operation(self, 
                           operation_name: str,
                           operation_func: Callable,
                           config: Dict[str, Any],
                           num_runs: int = 3,
                           **kwargs) -> BenchmarkResult:
        """Benchmark a single operation with multiple runs."""
        
        times = []
        memory_usages = []
        peak_memory_usages = []
        gpu_memory_usages = []
        peak_gpu_memory_usages = []
        
        success = True
        error_message = None
        
        try:
            for run in range(num_runs):
                # Warm up (first run)
                if run == 0:
                    try:
                        with MemoryTracker(track_gpu=torch.cuda.is_available()) as tracker:
                            operation_func(**kwargs)
                    except Exception:
                        pass  # Ignore warm-up errors
                
                # Actual benchmark run
                with MemoryTracker(track_gpu=torch.cuda.is_available()) as tracker:
                    start_time = time.perf_counter()
                    operation_func(**kwargs)
                    end_time = time.perf_counter()
                    
                    run_time = end_time - start_time
                    mem_usage, peak_mem, gpu_mem, peak_gpu = tracker.get_memory_usage()
                    
                    times.append(run_time)
                    memory_usages.append(mem_usage)
                    peak_memory_usages.append(peak_mem)
                    
                    if gpu_mem is not None:
                        gpu_memory_usages.append(gpu_mem)
                        peak_gpu_memory_usages.append(peak_gpu)
        
        except Exception as e:
            success = False
            error_message = str(e)
            times = [float('inf')]
            memory_usages = [float('inf')]
            peak_memory_usages = [float('inf')]
        
        # Calculate averages
        avg_time = sum(times) / len(times) if times else float('inf')
        avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else float('inf')
        avg_peak_memory = sum(peak_memory_usages) / len(peak_memory_usages) if peak_memory_usages else float('inf')
        
        avg_gpu_memory = None
        avg_peak_gpu_memory = None
        if gpu_memory_usages:
            avg_gpu_memory = sum(gpu_memory_usages) / len(gpu_memory_usages)
            avg_peak_gpu_memory = sum(peak_gpu_memory_usages) / len(peak_gpu_memory_usages)
        
        return BenchmarkResult(
            operation=operation_name,
            config=config,
            time_seconds=avg_time,
            memory_mb=avg_memory,
            peak_memory_mb=avg_peak_memory,
            gpu_memory_mb=avg_gpu_memory,
            peak_gpu_memory_mb=avg_peak_gpu_memory,
            success=success,
            error_message=error_message
        )
    
    def _compute_ntk_stax(self, model, x, **kwargs):
        """Compute NTK using stax interface."""
        # Convert model to stax-like representation
        init_fn, apply_fn, kernel_fn = self._model_to_stax(model)
        
        # Initialize parameters
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (-1,) + x.shape[1:]
        _, params = init_fn(rng, input_shape)
        
        # Compute kernel
        kernel = kernel_fn(x, None, 'ntk')
        return kernel
    
    def _compute_nngp_stax(self, model, x, **kwargs):
        """Compute NNGP using stax interface."""
        init_fn, apply_fn, kernel_fn = self._model_to_stax(model)
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (-1,) + x.shape[1:]
        _, params = init_fn(rng, input_shape)
        
        kernel = kernel_fn(x, None, 'nngp')
        return kernel
    
    def _compute_batched_ntk(self, model, x, batch_size, **kwargs):
        """Compute NTK with batching."""
        # Use batching functionality
        init_fn, apply_fn, kernel_fn = self._model_to_stax(model)
        
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (-1,) + x.shape[1:]
        _, params = init_fn(rng, input_shape)
        
        # Create batched kernel function
        batched_kernel_fn = tt.batch(
            lambda x1, x2: kernel_fn(x1, x2, 'ntk'),
            batch_size=batch_size,
            device_count=1
        )
        
        return batched_kernel_fn(x, None)
    
    def _compute_empirical_ntk(self, model, x, **kwargs):
        """Compute empirical NTK directly."""
        def model_fn(params, inputs):
            return torch.func.functional_call(model, params, inputs)
        
        params = dict(model.named_parameters())
        ntk_fn = tt.empirical_ntk_fn(model_fn, trace_axes=(), vmap_axes=0)
        
        return ntk_fn(x, None, params)
    
    def _compute_infinite_ntk(self, width, input_dim, x, **kwargs):
        """Compute infinite width NTK using stax."""
        init_fn, apply_fn, kernel_fn = tt.stax.serial(
            tt.stax.Dense(width),
            tt.stax.Relu(),
            tt.stax.Dense(1)
        )
        
        kernel = kernel_fn(x, None, 'ntk')
        return kernel
    
    def _model_to_stax(self, model):
        """Convert PyTorch model to stax representation (simplified)."""
        layers = []
        
        for module in model:
            if isinstance(module, nn.Linear):
                layers.append(tt.stax.Dense(module.out_features))
            elif isinstance(module, nn.ReLU):
                layers.append(tt.stax.Relu())
            elif isinstance(module, nn.Tanh):
                layers.append(tt.stax.Tanh())
        
        return tt.stax.serial(*layers)
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON file."""
        results_dict = [result.to_dict() for result in self.results]
        
        with open(filepath, 'w') as f:
            json.dump({
                'device': str(self.device),
                'timestamp': time.time(),
                'results': results_dict
            }, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.results:
            print("No benchmark results to display.")
            return
        
        print("\n" + "="*80)
        print("TORCHTANGENTS BENCHMARK SUMMARY")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Total operations benchmarked: {len(self.results)}")
        
        # Group by operation type
        by_operation = {}
        for result in self.results:
            op = result.operation
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(result)
        
        for operation, results in by_operation.items():
            print(f"\n{operation.upper()}:")
            print("-" * 40)
            
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            if successful_results:
                avg_time = sum(r.time_seconds for r in successful_results) / len(successful_results)
                avg_memory = sum(r.memory_mb for r in successful_results) / len(successful_results)
                min_time = min(r.time_seconds for r in successful_results)
                max_time = max(r.time_seconds for r in successful_results)
                
                print(f"  Successful runs: {len(successful_results)}")
                print(f"  Average time: {avg_time:.4f}s")
                print(f"  Time range: {min_time:.4f}s - {max_time:.4f}s")
                print(f"  Average memory: {avg_memory:.2f} MB")
                
                if any(r.gpu_memory_mb is not None for r in successful_results):
                    gpu_results = [r for r in successful_results if r.gpu_memory_mb is not None]
                    avg_gpu_memory = sum(r.gpu_memory_mb for r in gpu_results) / len(gpu_results)
                    print(f"  Average GPU memory: {avg_gpu_memory:.2f} MB")
            
            if failed_results:
                print(f"  Failed runs: {len(failed_results)}")
                for result in failed_results[:3]:  # Show first 3 errors
                    print(f"    Error: {result.error_message}")


def main():
    """Run comprehensive benchmarks."""
    print("Starting TorchTangents Benchmark Suite...")
    
    # Initialize benchmark
    benchmark = TorchTangentsBenchmark(device='auto')
    
    print(f"Using device: {benchmark.device}")
    
    # Run different benchmark categories
    print("\n1. NTK Computation Benchmark")
    benchmark.benchmark_ntk_computation(
        input_dims=[10, 20],
        hidden_dims=[64, 128], 
        batch_sizes=[8, 16],
        num_runs=2
    )
    
    print("\n2. NNGP Computation Benchmark")
    benchmark.benchmark_nngp_computation(
        input_dims=[10, 20],
        hidden_dims=[64, 128],
        batch_sizes=[8, 16], 
        num_runs=2
    )
    
    print("\n3. Batching Performance Benchmark")
    benchmark.benchmark_batching_performance(
        batch_sizes=[1, 2, 4],
        dataset_size=32,
        num_runs=2
    )
    
    print("\n4. Empirical vs Infinite Width Benchmark")
    benchmark.benchmark_empirical_vs_infinite(
        widths=[64, 128],
        batch_size=8,
        num_runs=2
    )
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    timestamp = int(time.time())
    benchmark.save_results(f'benchmark_results_{timestamp}.json')


if __name__ == "__main__":
    main()