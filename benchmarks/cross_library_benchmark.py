#!/usr/bin/env python3
"""
Cross-Library Benchmark: TorchTangents vs Neural-Tangents

Benchmark comparing performance and memory usage between TorchTangents (PyTorch)
and the original neural-tangents library (JAX) for equivalent operations.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import warnings
import json

import torchtangents as tt


class CrossLibraryBenchmark:
    """Benchmark TorchTangents against neural-tangents (if available)."""
    
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.has_jax = self._check_jax_availability()
        self.has_nt = self._check_neural_tangents_availability()
        
        print(f"Cross-library benchmark on device: {self.device}")
        print(f"JAX available: {self.has_jax}")
        print(f"Neural-tangents available: {self.has_nt}")
        
        if self.has_jax and self.has_nt:
            self._setup_jax()
    
    def _check_jax_availability(self) -> bool:
        """Check if JAX is available."""
        try:
            import jax
            import jax.numpy as jnp
            return True
        except ImportError:
            return False
    
    def _check_neural_tangents_availability(self) -> bool:
        """Check if neural-tangents is available."""
        try:
            import neural_tangents as nt
            return True
        except ImportError:
            return False
    
    def _setup_jax(self):
        """Setup JAX environment."""
        import jax
        import jax.numpy as jnp
        import neural_tangents as nt
        
        self.jax = jax
        self.jnp = jnp
        self.nt = nt
        
        # Set JAX to use GPU if available
        if torch.cuda.is_available():
            try:
                # Try to use GPU
                test_array = jnp.array([1.0])
                print(f"JAX device: {test_array.device()}")
            except Exception:
                print("JAX running on CPU")
    
    def benchmark_equivalent_networks(self,
                                    input_dims: List[int] = [10, 50],
                                    hidden_dims: List[int] = [64, 128],
                                    batch_sizes: List[int] = [8, 16, 32],
                                    num_runs: int = 3) -> Dict:
        """Benchmark equivalent networks in both libraries."""
        
        results = {
            'torchtangents': [],
            'neural_tangents': [],
            'comparisons': []
        }
        
        for input_dim in input_dims:
            for hidden_dim in hidden_dims:
                for batch_size in batch_sizes:
                    print(f"\nBenchmarking equivalent networks:")
                    print(f"  Input: {input_dim}, Hidden: {hidden_dim}, Batch: {batch_size}")
                    
                    # TorchTangents benchmark
                    tt_result = self._benchmark_torchtangents_network(
                        input_dim, hidden_dim, batch_size, num_runs
                    )
                    results['torchtangents'].append(tt_result)
                    print(f"  TorchTangents: {tt_result['mean_time']:.4f}s")
                    
                    # Neural-tangents benchmark (if available)
                    if self.has_jax and self.has_nt:
                        nt_result = self._benchmark_neural_tangents_network(
                            input_dim, hidden_dim, batch_size, num_runs
                        )
                        results['neural_tangents'].append(nt_result)
                        print(f"  Neural-tangents: {nt_result['mean_time']:.4f}s")
                        
                        # Compare
                        speedup = nt_result['mean_time'] / tt_result['mean_time']
                        memory_ratio = tt_result.get('memory_mb', 0) / max(nt_result.get('memory_mb', 1), 1)
                        
                        comparison = {
                            'config': {
                                'input_dim': input_dim,
                                'hidden_dim': hidden_dim,
                                'batch_size': batch_size
                            },
                            'torchtangents_time': tt_result['mean_time'],
                            'neural_tangents_time': nt_result['mean_time'],
                            'speedup_ratio': speedup,
                            'memory_ratio': memory_ratio
                        }
                        results['comparisons'].append(comparison)
                        
                        print(f"  Speedup (TT vs NT): {speedup:.2f}x")
                        print(f"  Memory ratio (TT/NT): {memory_ratio:.2f}x")
                    else:
                        print("  Neural-tangents not available for comparison")
        
        return results
    
    def _benchmark_torchtangents_network(self,
                                       input_dim: int,
                                       hidden_dim: int,
                                       batch_size: int,
                                       num_runs: int) -> Dict:
        """Benchmark a network using TorchTangents."""
        
        # Create TorchTangents network
        init_fn, apply_fn, kernel_fn = tt.stax.serial(
            tt.stax.Dense(hidden_dim),
            tt.stax.Relu(),
            tt.stax.Dense(hidden_dim // 2),
            tt.stax.Relu(),
            tt.stax.Dense(10)
        )
        
        # Generate data
        x = torch.randn(batch_size, input_dim, device=self.device)
        
        # Initialize parameters
        rng = torch.Generator()
        rng.manual_seed(42)
        input_shape = (-1, input_dim)
        _, params = init_fn(rng, input_shape)
        
        # Benchmark NTK computation
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            start_time = time.perf_counter()
            kernel = kernel_fn(x, None, 'ntk')
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate memory usage (approximation)
        memory_mb = self._estimate_torch_memory()
        
        return {
            'library': 'torchtangents',
            'config': {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'batch_size': batch_size
            },
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'memory_mb': memory_mb,
            'device': str(self.device)
        }
    
    def _benchmark_neural_tangents_network(self,
                                         input_dim: int,
                                         hidden_dim: int,
                                         batch_size: int,
                                         num_runs: int) -> Dict:
        """Benchmark equivalent network using neural-tangents."""
        
        if not (self.has_jax and self.has_nt):
            return {'error': 'JAX/neural-tangents not available'}
        
        # Create equivalent neural-tangents network
        init_fn, apply_fn, kernel_fn = self.nt.stax.serial(
            self.nt.stax.Dense(hidden_dim),
            self.nt.stax.Relu(),
            self.nt.stax.Dense(hidden_dim // 2),
            self.nt.stax.Relu(),
            self.nt.stax.Dense(10)
        )
        
        # Generate data (JAX)
        key = self.jax.random.PRNGKey(42)
        x = self.jax.random.normal(key, (batch_size, input_dim))
        
        # Initialize parameters
        input_shape = (-1, input_dim)
        _, params = init_fn(self.jax.random.PRNGKey(42), input_shape)
        
        # Benchmark NTK computation
        times = []
        for _ in range(num_runs):
            # JAX uses lazy evaluation, so we need to block until completion
            start_time = time.perf_counter()
            kernel = kernel_fn(x, None, 'ntk')
            # Force computation
            _ = kernel.block_until_ready() if hasattr(kernel, 'block_until_ready') else kernel
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        # Estimate memory usage
        memory_mb = self._estimate_jax_memory()
        
        return {
            'library': 'neural_tangents',
            'config': {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'batch_size': batch_size
            },
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'memory_mb': memory_mb,
            'device': 'jax_default'
        }
    
    def benchmark_api_complexity(self) -> Dict:
        """Compare API complexity between libraries."""
        
        print("\nAPI Complexity Comparison:")
        print("-" * 40)
        
        # TorchTangents API example
        tt_code = """
# TorchTangents API
import torchtangents as tt

init_fn, apply_fn, kernel_fn = tt.stax.serial(
    tt.stax.Dense(128), tt.stax.Relu(), tt.stax.Dense(10)
)
rng = torch.Generator()
_, params = init_fn(rng, (-1, 784))
ntk = kernel_fn(x_train, x_test, 'ntk')
"""
        
        # Neural-tangents API example
        nt_code = """
# Neural-tangents API  
import neural_tangents as nt

init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Dense(128), nt.stax.Relu(), nt.stax.Dense(10)
)
_, params = init_fn(jax.random.PRNGKey(0), (-1, 784))
ntk = kernel_fn(x_train, x_test, 'ntk')
"""
        
        # Proposed PyTorch-native API
        proposed_code = """
# Proposed PyTorch-native API
import torchtangents as tt

model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
)
ntk = tt.ntk(model, x_train, x_test)
"""
        
        tt_lines = len([l for l in tt_code.strip().split('\n') if l.strip() and not l.strip().startswith('#')])
        nt_lines = len([l for l in nt_code.strip().split('\n') if l.strip() and not l.strip().startswith('#')])
        proposed_lines = len([l for l in proposed_code.strip().split('\n') if l.strip() and not l.strip().startswith('#')])
        
        print(f"TorchTangents (current): {tt_lines} lines")
        print(f"Neural-tangents: {nt_lines} lines")
        print(f"TorchTangents (proposed): {proposed_lines} lines")
        print(f"Complexity reduction: {(tt_lines - proposed_lines) / tt_lines * 100:.1f}%")
        
        return {
            'torchtangents_current_lines': tt_lines,
            'neural_tangents_lines': nt_lines,
            'torchtangents_proposed_lines': proposed_lines,
            'complexity_reduction_percent': (tt_lines - proposed_lines) / tt_lines * 100
        }
    
    def benchmark_feature_completeness(self) -> Dict:
        """Compare feature completeness between libraries."""
        
        features = [
            ('Basic NTK computation', True, True),
            ('NNGP computation', True, True),
            ('Batching support', True, True),
            ('GPU acceleration', True, True),
            ('Empirical kernels', True, True),
            ('Monte Carlo kernels', True, True),
            ('Prediction utilities', True, True),
            ('PyTorch integration', True, False),
            ('JAX integration', False, True),
            ('Infinite width theory', True, True),
            ('Custom architectures', True, True),
            ('Convolutional layers', False, True),  # TorchTangents limited
            ('Attention layers', False, True),      # TorchTangents limited
        ]
        
        print("\nFeature Completeness Comparison:")
        print("-" * 50)
        print(f"{'Feature':<25} {'TorchTangents':<15} {'Neural-tangents':<15}")
        print("-" * 50)
        
        tt_score = 0
        nt_score = 0
        
        for feature, tt_has, nt_has in features:
            tt_symbol = "✓" if tt_has else "✗"
            nt_symbol = "✓" if nt_has else "✗"
            
            print(f"{feature:<25} {tt_symbol:<15} {nt_symbol:<15}")
            
            if tt_has:
                tt_score += 1
            if nt_has:
                nt_score += 1
        
        print("-" * 50)
        print(f"Total score: TorchTangents {tt_score}/{len(features)}, "
              f"Neural-tangents {nt_score}/{len(features)}")
        
        return {
            'torchtangents_score': tt_score,
            'neural_tangents_score': nt_score,
            'total_features': len(features),
            'features': features
        }
    
    def _estimate_torch_memory(self) -> float:
        """Estimate PyTorch memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0  # CPU memory estimation is complex
    
    def _estimate_jax_memory(self) -> float:
        """Estimate JAX memory usage in MB."""
        # JAX memory estimation is complex, return placeholder
        return 0.0
    
    def run_comprehensive_comparison(self) -> Dict:
        """Run comprehensive comparison between libraries."""
        
        results = {}
        
        # Performance comparison
        print("1. Performance Benchmark")
        results['performance'] = self.benchmark_equivalent_networks(
            input_dims=[10, 20],
            hidden_dims=[64, 128],
            batch_sizes=[8, 16],
            num_runs=2
        )
        
        # API complexity comparison
        print("\n2. API Complexity Analysis")
        results['api_complexity'] = self.benchmark_api_complexity()
        
        # Feature completeness comparison
        print("\n3. Feature Completeness Analysis") 
        results['feature_completeness'] = self.benchmark_feature_completeness()
        
        return results
    
    def save_comparison_results(self, results: Dict, filepath: str):
        """Save comparison results to JSON."""
        
        # Make results JSON serializable
        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif hasattr(obj, 'device'):
                return str(obj)
            return obj
        
        serializable_results = json.loads(json.dumps(results, default=make_serializable))
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'device': str(self.device),
                'libraries_available': {
                    'jax': self.has_jax,
                    'neural_tangents': self.has_nt
                },
                'results': serializable_results
            }, f, indent=2)
        
        print(f"\nComparison results saved to {filepath}")


def main():
    """Run cross-library benchmark."""
    
    print("TorchTangents vs Neural-Tangents Benchmark")
    print("=" * 50)
    
    benchmark = CrossLibraryBenchmark()
    
    if not (benchmark.has_jax and benchmark.has_nt):
        print("\nWarning: JAX and/or neural-tangents not available.")
        print("Install them with: pip install jax neural-tangents")
        print("Some comparisons will be skipped.\n")
    
    # Run comprehensive comparison
    results = benchmark.run_comprehensive_comparison()
    
    # Save results
    timestamp = int(time.time())
    benchmark.save_comparison_results(results, f'cross_library_benchmark_{timestamp}.json')
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-LIBRARY BENCHMARK SUMMARY")
    print("="*60)
    
    if 'performance' in results and results['performance']['comparisons']:
        comparisons = results['performance']['comparisons']
        avg_speedup = sum(c['speedup_ratio'] for c in comparisons) / len(comparisons)
        print(f"Average speedup (TorchTangents vs Neural-tangents): {avg_speedup:.2f}x")
    
    if 'api_complexity' in results:
        complexity = results['api_complexity']
        print(f"API complexity reduction: {complexity['complexity_reduction_percent']:.1f}%")
    
    if 'feature_completeness' in results:
        features = results['feature_completeness']
        tt_percent = features['torchtangents_score'] / features['total_features'] * 100
        nt_percent = features['neural_tangents_score'] / features['total_features'] * 100
        print(f"Feature completeness: TorchTangents {tt_percent:.1f}%, Neural-tangents {nt_percent:.1f}%")


if __name__ == "__main__":
    main()