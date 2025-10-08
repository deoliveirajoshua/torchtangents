#!/usr/bin/env python3
"""
Timing Benchmark for TorchTangents

Focused performance benchmarking tool that measures execution times
for various TorchTangents operations across different configurations.
"""

import time
import torch
import torch.nn as nn
import statistics
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import json
import os

import torchtangents as tt


@dataclass
class TimingResult:
    """Container for timing benchmark results."""
    operation: str
    config: Dict
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    samples: int
    throughput_ops_per_sec: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'operation': self.operation,
            'config': self.config,
            'mean_time': self.mean_time,
            'median_time': self.median_time,
            'std_dev': self.std_dev,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'samples': self.samples,
            'throughput_ops_per_sec': self.throughput_ops_per_sec
        }


class PrecisionTimer:
    """High-precision timer with warm-up and multiple samples."""
    
    def __init__(self, warmup_runs: int = 2, timing_runs: int = 5):
        self.warmup_runs = warmup_runs
        self.timing_runs = timing_runs
    
    def time_operation(self, 
                      operation: Callable,
                      operation_name: str,
                      config: Dict,
                      *args, **kwargs) -> TimingResult:
        """Time an operation with multiple runs and statistics."""
        
        # Warm-up runs
        for _ in range(self.warmup_runs):
            try:
                operation(*args, **kwargs)
            except Exception:
                pass  # Ignore warm-up errors
        
        # Actual timing runs
        times = []
        for _ in range(self.timing_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            start_time = time.perf_counter()
            operation(*args, **kwargs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)
        
        # Calculate throughput if applicable
        throughput = None
        if 'batch_size' in config:
            throughput = config['batch_size'] / mean_time
        
        return TimingResult(
            operation=operation_name,
            config=config,
            mean_time=mean_time,
            median_time=median_time,
            std_dev=std_dev,
            min_time=min_time,
            max_time=max_time,
            samples=len(times),
            throughput_ops_per_sec=throughput
        )


class TorchTangentsTimingBenchmark:
    """Comprehensive timing benchmark for TorchTangents."""
    
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.timer = PrecisionTimer()
        self.results: List[TimingResult] = []
        
        print(f"Timing benchmark initialized on device: {self.device}")
    
    def benchmark_ntk_computation_scaling(self,
                                         batch_sizes: List[int] = [4, 8, 16, 32, 64, 128],
                                         input_dims: List[int] = [10, 20, 50, 100],
                                         hidden_dim: int = 128,
                                         output_dim: int = 10) -> List[TimingResult]:
        """Benchmark NTK computation scaling with batch size and input dimension."""
        
        results = []
        
        for input_dim in input_dims:
            for batch_size in batch_sizes:
                config = {
                    'operation_type': 'ntk_computation',
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'output_dim': output_dim,
                    'batch_size': batch_size,
                    'device': str(self.device)
                }
                
                print(f"Timing NTK: input_dim={input_dim}, batch_size={batch_size}")
                
                try:
                    # Create model and data
                    model = self._create_standard_model(input_dim, hidden_dim, output_dim)
                    x = torch.randn(batch_size, input_dim, device=self.device)
                    
                    # Time the operation
                    result = self.timer.time_operation(
                        operation=self._compute_ntk_stax,
                        operation_name='ntk_computation',
                        config=config,
                        model=model,
                        x=x
                    )
                    
                    results.append(result)
                    self.results.append(result)
                    
                    print(f"  Time: {result.mean_time:.4f}s ± {result.std_dev:.4f}s")
                    
                except Exception as e:
                    print(f"  Error: {e}")
        
        return results
    
    def benchmark_nngp_computation_scaling(self,
                                          batch_sizes: List[int] = [4, 8, 16, 32, 64],
                                          input_dims: List[int] = [10, 20, 50],
                                          hidden_dim: int = 128) -> List[TimingResult]:
        """Benchmark NNGP computation scaling."""
        
        results = []
        
        for input_dim in input_dims:
            for batch_size in batch_sizes:
                config = {
                    'operation_type': 'nngp_computation',
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'batch_size': batch_size,
                    'device': str(self.device)
                }
                
                print(f"Timing NNGP: input_dim={input_dim}, batch_size={batch_size}")
                
                try:
                    model = self._create_standard_model(input_dim, hidden_dim, 10)
                    x = torch.randn(batch_size, input_dim, device=self.device)
                    
                    result = self.timer.time_operation(
                        operation=self._compute_nngp_stax,
                        operation_name='nngp_computation',
                        config=config,
                        model=model,
                        x=x
                    )
                    
                    results.append(result)
                    self.results.append(result)
                    
                    print(f"  Time: {result.mean_time:.4f}s ± {result.std_dev:.4f}s")
                    
                except Exception as e:
                    print(f"  Error: {e}")
        
        return results
    
    def benchmark_width_scaling(self,
                               widths: List[int] = [32, 64, 128, 256, 512],
                               batch_size: int = 16,
                               input_dim: int = 50) -> List[TimingResult]:
        """Benchmark how computation time scales with network width."""
        
        results = []
        
        for width in widths:
            config = {
                'operation_type': 'width_scaling',
                'width': width,
                'input_dim': input_dim,
                'batch_size': batch_size,
                'device': str(self.device)
            }
            
            print(f"Timing width scaling: width={width}")
            
            try:
                # Create wider model
                model = nn.Sequential(
                    nn.Linear(input_dim, width),
                    nn.ReLU(),
                    nn.Linear(width, width),
                    nn.ReLU(),
                    nn.Linear(width, 10)
                ).to(self.device)
                
                x = torch.randn(batch_size, input_dim, device=self.device)
                
                result = self.timer.time_operation(
                    operation=self._compute_ntk_stax,
                    operation_name='width_scaling_ntk',
                    config=config,
                    model=model,
                    x=x
                )
                
                results.append(result)
                self.results.append(result)
                
                num_params = sum(p.numel() for p in model.parameters())
                print(f"  Width {width}: {result.mean_time:.4f}s ({num_params:,} params)")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        return results
    
    def benchmark_depth_scaling(self,
                               depths: List[int] = [2, 3, 4, 5, 6],
                               width: int = 128,
                               batch_size: int = 16,
                               input_dim: int = 50) -> List[TimingResult]:
        """Benchmark how computation time scales with network depth."""
        
        results = []
        
        for depth in depths:
            config = {
                'operation_type': 'depth_scaling',
                'depth': depth,
                'width': width,
                'input_dim': input_dim,
                'batch_size': batch_size,
                'device': str(self.device)
            }
            
            print(f"Timing depth scaling: depth={depth}")
            
            try:
                # Create deeper model
                layers = [nn.Linear(input_dim, width), nn.ReLU()]
                for _ in range(depth - 2):
                    layers.extend([nn.Linear(width, width), nn.ReLU()])
                layers.append(nn.Linear(width, 10))
                
                model = nn.Sequential(*layers).to(self.device)
                x = torch.randn(batch_size, input_dim, device=self.device)
                
                result = self.timer.time_operation(
                    operation=self._compute_ntk_stax,
                    operation_name='depth_scaling_ntk',
                    config=config,
                    model=model,
                    x=x
                )
                
                results.append(result)
                self.results.append(result)
                
                num_params = sum(p.numel() for p in model.parameters())
                print(f"  Depth {depth}: {result.mean_time:.4f}s ({num_params:,} params)")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        return results
    
    def benchmark_batching_efficiency(self,
                                    dataset_sizes: List[int] = [64, 128, 256],
                                    batch_sizes: List[int] = [1, 2, 4, 8, 16],
                                    input_dim: int = 50,
                                    hidden_dim: int = 128) -> List[TimingResult]:
        """Benchmark batching efficiency for large datasets."""
        
        results = []
        
        for dataset_size in dataset_sizes:
            for batch_size in batch_sizes:
                if batch_size > dataset_size:
                    continue
                
                config = {
                    'operation_type': 'batching_efficiency',
                    'dataset_size': dataset_size,
                    'batch_size': batch_size,
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'device': str(self.device)
                }
                
                print(f"Timing batching: dataset={dataset_size}, batch={batch_size}")
                
                try:
                    model = self._create_standard_model(input_dim, hidden_dim, 10)
                    x = torch.randn(dataset_size, input_dim, device=self.device)
                    
                    result = self.timer.time_operation(
                        operation=self._compute_batched_ntk,
                        operation_name='batching_efficiency',
                        config=config,
                        model=model,
                        x=x,
                        batch_size=batch_size
                    )
                    
                    results.append(result)
                    self.results.append(result)
                    
                    # Calculate efficiency metric
                    efficiency = dataset_size / (result.mean_time * batch_size)
                    print(f"  Time: {result.mean_time:.4f}s, Efficiency: {efficiency:.2f}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
        
        return results
    
    def benchmark_empirical_vs_infinite(self,
                                       widths: List[int] = [64, 128, 256],
                                       batch_size: int = 16,
                                       input_dim: int = 50) -> List[TimingResult]:
        """Compare timing of empirical vs infinite width kernels."""
        
        results = []
        
        for width in widths:
            # Empirical kernel timing
            config_emp = {
                'operation_type': 'empirical_kernel',
                'width': width,
                'batch_size': batch_size,
                'input_dim': input_dim,
                'device': str(self.device)
            }
            
            print(f"Timing empirical kernel: width={width}")
            
            try:
                model = nn.Sequential(
                    nn.Linear(input_dim, width),
                    nn.ReLU(),
                    nn.Linear(width, 1)
                ).to(self.device)
                
                x = torch.randn(batch_size, input_dim, device=self.device)
                
                result_emp = self.timer.time_operation(
                    operation=self._compute_empirical_ntk,
                    operation_name='empirical_kernel',
                    config=config_emp,
                    model=model,
                    x=x
                )
                
                results.append(result_emp)
                self.results.append(result_emp)
                
                print(f"  Empirical: {result_emp.mean_time:.4f}s")
                
            except Exception as e:
                print(f"  Empirical error: {e}")
            
            # Infinite kernel timing
            config_inf = config_emp.copy()
            config_inf['operation_type'] = 'infinite_kernel'
            
            print(f"Timing infinite kernel: width={width}")
            
            try:
                result_inf = self.timer.time_operation(
                    operation=self._compute_infinite_ntk,
                    operation_name='infinite_kernel',
                    config=config_inf,
                    width=width,
                    input_dim=input_dim,
                    x=x
                )
                
                results.append(result_inf)
                self.results.append(result_inf)
                
                print(f"  Infinite: {result_inf.mean_time:.4f}s")
                
                # Compare
                speedup = result_emp.mean_time / result_inf.mean_time
                print(f"  Speedup (inf/emp): {speedup:.2f}x")
                
            except Exception as e:
                print(f"  Infinite error: {e}")
        
        return results
    
    def _create_standard_model(self, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        """Create a standard model for benchmarking."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        ).to(self.device)
    
    def _model_to_stax(self, model: nn.Module):
        """Convert PyTorch model to stax."""
        layers = []
        for module in model:
            if isinstance(module, nn.Linear):
                layers.append(tt.stax.Dense(module.out_features))
            elif isinstance(module, nn.ReLU):
                layers.append(tt.stax.Relu())
            elif isinstance(module, nn.Tanh):
                layers.append(tt.stax.Tanh())
        return tt.stax.serial(*layers)
    
    def _compute_ntk_stax(self, model, x):
        """Compute NTK using stax interface."""
        init_fn, apply_fn, kernel_fn = self._model_to_stax(model)
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (-1,) + x.shape[1:]
        _, params = init_fn(rng, input_shape)
        kernel = kernel_fn(x, None, 'ntk')
        return kernel
    
    def _compute_nngp_stax(self, model, x):
        """Compute NNGP using stax interface."""
        init_fn, apply_fn, kernel_fn = self._model_to_stax(model)
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (-1,) + x.shape[1:]
        _, params = init_fn(rng, input_shape)
        kernel = kernel_fn(x, None, 'nngp')
        return kernel
    
    def _compute_batched_ntk(self, model, x, batch_size):
        """Compute NTK with batching."""
        init_fn, apply_fn, kernel_fn = self._model_to_stax(model)
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (-1,) + x.shape[1:]
        _, params = init_fn(rng, input_shape)
        
        batched_kernel_fn = tt.batch(
            lambda x1, x2: kernel_fn(x1, x2, 'ntk'),
            batch_size=batch_size,
            device_count=1
        )
        
        return batched_kernel_fn(x, None)
    
    def _compute_empirical_ntk(self, model, x):
        """Compute empirical NTK."""
        def model_fn(params, inputs):
            return torch.func.functional_call(model, params, inputs)
        
        params = dict(model.named_parameters())
        ntk_fn = tt.empirical_ntk_fn(model_fn, trace_axes=(), vmap_axes=0)
        return ntk_fn(x, None, params)
    
    def _compute_infinite_ntk(self, width, input_dim, x):
        """Compute infinite width NTK."""
        init_fn, apply_fn, kernel_fn = tt.stax.serial(
            tt.stax.Dense(width),
            tt.stax.Relu(),
            tt.stax.Dense(1)
        )
        kernel = kernel_fn(x, None, 'ntk')
        return kernel
    
    def save_results(self, filepath: str):
        """Save timing results to JSON file."""
        data = {
            'device': str(self.device),
            'timestamp': time.time(),
            'total_benchmarks': len(self.results),
            'results': [result.to_dict() for result in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Timing results saved to {filepath}")
    
    def print_performance_summary(self):
        """Print a summary of timing results."""
        if not self.results:
            print("No timing results available.")
            return
        
        print("\n" + "="*80)
        print("TORCHTANGENTS TIMING BENCHMARK SUMMARY")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Total benchmarks: {len(self.results)}")
        
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
            
            times = [r.mean_time for r in results]
            print(f"  Benchmarks: {len(results)}")
            print(f"  Time range: {min(times):.4f}s - {max(times):.4f}s")
            print(f"  Average time: {sum(times)/len(times):.4f}s")
            
            # Show fastest and slowest configurations
            fastest = min(results, key=lambda x: x.mean_time)
            slowest = max(results, key=lambda x: x.mean_time)
            
            print(f"  Fastest config: {fastest.config} ({fastest.mean_time:.4f}s)")
            print(f"  Slowest config: {slowest.config} ({slowest.mean_time:.4f}s)")
            
            if fastest.throughput_ops_per_sec and slowest.throughput_ops_per_sec:
                print(f"  Throughput range: {slowest.throughput_ops_per_sec:.1f} - {fastest.throughput_ops_per_sec:.1f} ops/sec")


def run_comprehensive_timing_benchmark():
    """Run a comprehensive timing benchmark suite."""
    
    print("TorchTangents Comprehensive Timing Benchmark")
    print("=" * 50)
    
    benchmark = TorchTangentsTimingBenchmark()
    
    # Run different benchmark categories
    print("\n1. NTK Computation Scaling")
    benchmark.benchmark_ntk_computation_scaling(
        batch_sizes=[4, 8, 16, 32],
        input_dims=[10, 20, 50],
        hidden_dim=64
    )
    
    print("\n2. NNGP Computation Scaling")
    benchmark.benchmark_nngp_computation_scaling(
        batch_sizes=[4, 8, 16],
        input_dims=[10, 20],
        hidden_dim=64
    )
    
    print("\n3. Width Scaling")
    benchmark.benchmark_width_scaling(
        widths=[32, 64, 128, 256],
        batch_size=16
    )
    
    print("\n4. Depth Scaling")
    benchmark.benchmark_depth_scaling(
        depths=[2, 3, 4, 5],
        width=64,
        batch_size=16
    )
    
    print("\n5. Batching Efficiency")
    benchmark.benchmark_batching_efficiency(
        dataset_sizes=[64, 128],
        batch_sizes=[4, 8, 16],
        hidden_dim=64
    )
    
    print("\n6. Empirical vs Infinite Width")
    benchmark.benchmark_empirical_vs_infinite(
        widths=[64, 128],
        batch_size=16
    )
    
    # Print summary and save results
    benchmark.print_performance_summary()
    
    timestamp = int(time.time())
    benchmark.save_results(f'timing_benchmark_{timestamp}.json')


if __name__ == "__main__":
    run_comprehensive_timing_benchmark()