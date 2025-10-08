#!/usr/bin/env python3
"""
Memory Profiler for TorchTangents Operations

Detailed memory analysis tool for understanding memory usage patterns
in NTK and NNGP computations.
"""

import torch
import torch.nn as nn
import gc
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys

import torchtangents as tt


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    operation: str
    cpu_memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    
    
class SimpleMemoryTracker:
    """Simple memory tracker that doesn't require external dependencies."""
    
    def __init__(self, track_gpu: bool = True):
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_cpu = 0
        self.baseline_gpu = 0
        
    def start_tracking(self):
        """Start memory tracking."""
        gc.collect()
        if self.track_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        # Get baseline memory usage
        self.baseline_cpu = self._get_cpu_memory()
        if self.track_gpu:
            self.baseline_gpu = torch.cuda.memory_allocated() / 1024 / 1024
            
        self.take_snapshot("baseline")
    
    def take_snapshot(self, operation: str):
        """Take a memory snapshot."""
        cpu_mem = self._get_cpu_memory() - self.baseline_cpu
        gpu_mem = None
        gpu_reserved = None
        
        if self.track_gpu:
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 - self.baseline_gpu
            gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            operation=operation,
            cpu_memory_mb=cpu_mem,
            gpu_memory_mb=gpu_mem,
            gpu_reserved_mb=gpu_reserved
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def _get_cpu_memory(self) -> float:
        """Get current CPU memory usage in MB."""
        try:
            # Try using psutil if available
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback to approximation using sys.getsizeof
            # This is less accurate but doesn't require external dependencies
            return sys.getsizeof(gc.get_objects()) / 1024 / 1024
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage."""
        if not self.snapshots:
            return {"cpu_peak": 0, "gpu_peak": 0}
            
        cpu_peak = max(s.cpu_memory_mb for s in self.snapshots)
        gpu_peak = 0
        
        if self.track_gpu:
            gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 - self.baseline_gpu
            
        return {"cpu_peak": cpu_peak, "gpu_peak": gpu_peak}
    
    def print_summary(self):
        """Print memory usage summary."""
        if not self.snapshots:
            print("No memory snapshots taken.")
            return
            
        print("\nMEMORY USAGE SUMMARY")
        print("=" * 50)
        
        for snapshot in self.snapshots:
            print(f"{snapshot.operation:20} | CPU: {snapshot.cpu_memory_mb:8.2f} MB", end="")
            if snapshot.gpu_memory_mb is not None:
                print(f" | GPU: {snapshot.gpu_memory_mb:8.2f} MB (Reserved: {snapshot.gpu_reserved_mb:8.2f} MB)")
            else:
                print()
        
        peaks = self.get_peak_memory()
        print("-" * 50)
        print(f"Peak CPU Memory: {peaks['cpu_peak']:.2f} MB")
        if self.track_gpu:
            print(f"Peak GPU Memory: {peaks['gpu_peak']:.2f} MB")


class NTKMemoryProfiler:
    """Specialized memory profiler for NTK computations."""
    
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.tracker = SimpleMemoryTracker(track_gpu=(self.device.type == 'cuda'))
        
    def profile_ntk_scaling(self, 
                           batch_sizes: List[int] = [4, 8, 16, 32, 64],
                           input_dim: int = 50,
                           hidden_dim: int = 128) -> Dict[int, Dict[str, float]]:
        """Profile how NTK memory usage scales with batch size."""
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nProfiling batch size: {batch_size}")
            
            # Create model and data
            model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 10)
            ).to(self.device)
            
            x = torch.randn(batch_size, input_dim, device=self.device)
            
            # Start tracking
            self.tracker.start_tracking()
            
            try:
                # Convert to stax for NTK computation
                self.tracker.take_snapshot("model_created")
                
                init_fn, apply_fn, kernel_fn = self._model_to_stax(model)
                self.tracker.take_snapshot("stax_conversion")
                
                # Initialize parameters
                rng = torch.Generator()
                rng.manual_seed(42)
                input_shape = (-1, input_dim)
                _, params = init_fn(rng, input_shape)
                self.tracker.take_snapshot("params_initialized")
                
                # Compute NTK
                kernel = kernel_fn(x, None, 'ntk')
                self.tracker.take_snapshot("ntk_computed")
                
                # Get final memory stats
                peaks = self.tracker.get_peak_memory()
                final_snapshot = self.tracker.snapshots[-1]
                
                results[batch_size] = {
                    'final_cpu_mb': final_snapshot.cpu_memory_mb,
                    'final_gpu_mb': final_snapshot.gpu_memory_mb or 0,
                    'peak_cpu_mb': peaks['cpu_peak'],
                    'peak_gpu_mb': peaks['gpu_peak']
                }
                
                print(f"  Final memory: CPU {final_snapshot.cpu_memory_mb:.2f} MB, "
                      f"GPU {final_snapshot.gpu_memory_mb or 0:.2f} MB")
                print(f"  Peak memory: CPU {peaks['cpu_peak']:.2f} MB, "
                      f"GPU {peaks['gpu_peak']:.2f} MB")
                
            except Exception as e:
                print(f"  Error: {e}")
                results[batch_size] = {
                    'final_cpu_mb': float('inf'),
                    'final_gpu_mb': float('inf'),
                    'peak_cpu_mb': float('inf'),
                    'peak_gpu_mb': float('inf')
                }
            
            # Clean up
            del model, x
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return results
    
    def profile_width_scaling(self,
                             widths: List[int] = [64, 128, 256, 512],
                             batch_size: int = 16,
                             input_dim: int = 50) -> Dict[int, Dict[str, float]]:
        """Profile how memory usage scales with network width."""
        
        results = {}
        
        for width in widths:
            print(f"\nProfiling width: {width}")
            
            # Create model
            model = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, 10)
            ).to(self.device)
            
            x = torch.randn(batch_size, input_dim, device=self.device)
            
            # Track memory
            self.tracker.start_tracking()
            
            try:
                # Compute NTK
                init_fn, apply_fn, kernel_fn = self._model_to_stax(model)
                self.tracker.take_snapshot("stax_converted")
                
                rng = torch.Generator()
                rng.manual_seed(42) 
                _, params = init_fn(rng, (-1, input_dim))
                self.tracker.take_snapshot("params_init")
                
                kernel = kernel_fn(x, None, 'ntk')
                self.tracker.take_snapshot("ntk_computed")
                
                peaks = self.tracker.get_peak_memory()
                final = self.tracker.snapshots[-1]
                
                results[width] = {
                    'final_cpu_mb': final.cpu_memory_mb,
                    'final_gpu_mb': final.gpu_memory_mb or 0,
                    'peak_cpu_mb': peaks['cpu_peak'],
                    'peak_gpu_mb': peaks['gpu_peak'],
                    'num_parameters': sum(p.numel() for p in model.parameters())
                }
                
                print(f"  Parameters: {results[width]['num_parameters']:,}")
                print(f"  Peak memory: CPU {peaks['cpu_peak']:.2f} MB, "
                      f"GPU {peaks['gpu_peak']:.2f} MB")
                
            except Exception as e:
                print(f"  Error: {e}")
                results[width] = {
                    'final_cpu_mb': float('inf'),
                    'final_gpu_mb': float('inf'), 
                    'peak_cpu_mb': float('inf'),
                    'peak_gpu_mb': float('inf'),
                    'num_parameters': sum(p.numel() for p in model.parameters())
                }
            
            # Clean up
            del model, x
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return results
    
    def profile_operation_breakdown(self,
                                   batch_size: int = 32,
                                   input_dim: int = 50,
                                   hidden_dim: int = 128) -> Dict[str, float]:
        """Profile memory usage of individual operations."""
        
        print(f"\nProfiling operation breakdown (batch_size={batch_size})")
        
        # Create model and data
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        ).to(self.device)
        
        x = torch.randn(batch_size, input_dim, device=self.device)
        
        self.tracker.start_tracking()
        
        # Step-by-step profiling
        self.tracker.take_snapshot("start")
        
        # Convert to stax
        init_fn, apply_fn, kernel_fn = self._model_to_stax(model)
        self.tracker.take_snapshot("stax_conversion")
        
        # Initialize parameters
        rng = torch.Generator()
        rng.manual_seed(42)
        _, params = init_fn(rng, (-1, input_dim))
        self.tracker.take_snapshot("param_initialization")
        
        # Forward pass
        output = apply_fn(params, x)
        self.tracker.take_snapshot("forward_pass")
        
        # NTK computation
        ntk_kernel = kernel_fn(x, None, 'ntk')
        self.tracker.take_snapshot("ntk_computation")
        
        # NNGP computation
        nngp_kernel = kernel_fn(x, None, 'nngp')
        self.tracker.take_snapshot("nngp_computation")
        
        # Print detailed breakdown
        self.tracker.print_summary()
        
        # Return memory increments for each operation
        increments = {}
        for i in range(1, len(self.tracker.snapshots)):
            current = self.tracker.snapshots[i]
            previous = self.tracker.snapshots[i-1]
            
            cpu_increment = current.cpu_memory_mb - previous.cpu_memory_mb
            gpu_increment = 0
            if current.gpu_memory_mb is not None and previous.gpu_memory_mb is not None:
                gpu_increment = current.gpu_memory_mb - previous.gpu_memory_mb
            
            increments[current.operation] = {
                'cpu_increment_mb': cpu_increment,
                'gpu_increment_mb': gpu_increment
            }
        
        return increments
    
    def _model_to_stax(self, model):
        """Convert PyTorch model to stax (simplified version)."""
        layers = []
        
        for module in model:
            if isinstance(module, nn.Linear):
                layers.append(tt.stax.Dense(module.out_features))
            elif isinstance(module, nn.ReLU):
                layers.append(tt.stax.Relu())
            elif isinstance(module, nn.Tanh):
                layers.append(tt.stax.Tanh())
        
        return tt.stax.serial(*layers)


def run_memory_profiling():
    """Run comprehensive memory profiling."""
    print("TorchTangents Memory Profiling Suite")
    print("=" * 40)
    
    profiler = NTKMemoryProfiler()
    print(f"Using device: {profiler.device}")
    
    # Profile batch size scaling
    print("\n1. Batch Size Scaling Analysis")
    batch_results = profiler.profile_ntk_scaling(
        batch_sizes=[4, 8, 16, 32] if profiler.device.type == 'cuda' else [4, 8, 16],
        input_dim=20,
        hidden_dim=64
    )
    
    # Profile width scaling
    print("\n2. Network Width Scaling Analysis")
    width_results = profiler.profile_width_scaling(
        widths=[32, 64, 128] if profiler.device.type == 'cuda' else [32, 64],
        batch_size=8,
        input_dim=20
    )
    
    # Profile operation breakdown
    print("\n3. Operation Breakdown Analysis")
    breakdown = profiler.profile_operation_breakdown(
        batch_size=16,
        input_dim=20,
        hidden_dim=64
    )
    
    # Print scaling analysis
    print("\n" + "="*60)
    print("SCALING ANALYSIS SUMMARY")
    print("="*60)
    
    print("\nBatch Size Scaling (Peak GPU Memory):")
    for batch_size, stats in batch_results.items():
        gpu_mem = stats['peak_gpu_mb']
        print(f"  Batch {batch_size:2d}: {gpu_mem:8.2f} MB")
    
    print("\nWidth Scaling (Peak GPU Memory):")
    for width, stats in width_results.items():
        gpu_mem = stats['peak_gpu_mb']
        params = stats['num_parameters']
        print(f"  Width {width:3d}: {gpu_mem:8.2f} MB ({params:,} params)")
    
    print("\nOperation Memory Increments:")
    for operation, stats in breakdown.items():
        cpu_inc = stats['cpu_increment_mb']
        gpu_inc = stats['gpu_increment_mb']
        print(f"  {operation:20}: CPU +{cpu_inc:6.2f} MB, GPU +{gpu_inc:6.2f} MB")


if __name__ == "__main__":
    run_memory_profiling()