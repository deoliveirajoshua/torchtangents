#!/usr/bin/env python3
"""
Benchmark Visualization Tools

Command-line tool for visualizing and analyzing TorchTangents benchmark results.
Creates plots, charts, and reports from benchmark data.

Usage:
    python visualization.py <results_file> [options]
    python visualization.py --example [options]

Examples:
    python visualization.py benchmark_results.json
    python visualization.py results.json --output-dir my_report
    python visualization.py --example --plots performance,api_complexity
"""

import json
import time
import numpy as np
import argparse
import sys
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


class BenchmarkVisualizer:
    """Visualize benchmark results with plots and reports."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize visualizer with plotting style."""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def load_benchmark_results(self, filepath: str) -> Dict:
        """Load benchmark results from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_performance_scaling(self, results: Dict, save_path: Optional[str] = None):
        """Plot performance scaling with batch size, width, depth."""
        
        if 'performance' not in results:
            print("No performance data found in results")
            return
        
        perf_data = results['performance']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('TorchTangents Performance Scaling Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        batch_sizes = []
        ntk_times = []
        nngp_times = []
        memory_usage = []
        widths = []
        
        for result in perf_data.get('results', []):
            batch_sizes.append(result['config']['batch_size'])
            ntk_times.append(result['timings']['ntk']['mean'])
            nngp_times.append(result['timings']['nngp']['mean'])
            memory_usage.append(result['memory']['peak_mb'])
            widths.append(result['config']['width'])
        
        # Plot 1: Time vs Batch Size
        ax1 = axes[0, 0]
        batch_unique = sorted(set(batch_sizes))
        ntk_by_batch = [np.mean([ntk_times[i] for i, b in enumerate(batch_sizes) if b == batch]) 
                       for batch in batch_unique]
        nngp_by_batch = [np.mean([nngp_times[i] for i, b in enumerate(batch_sizes) if b == batch]) 
                        for batch in batch_unique]
        
        ax1.plot(batch_unique, ntk_by_batch, 'o-', label='NTK', linewidth=2, markersize=8)
        ax1.plot(batch_unique, nngp_by_batch, 's-', label='NNGP', linewidth=2, markersize=8)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Computation Time vs Batch Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory vs Batch Size
        ax2 = axes[0, 1]
        memory_by_batch = [np.mean([memory_usage[i] for i, b in enumerate(batch_sizes) if b == batch]) 
                          for batch in batch_unique]
        
        ax2.plot(batch_unique, memory_by_batch, 'o-', color=self.colors[2], linewidth=2, markersize=8)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Time vs Width
        ax3 = axes[1, 0]
        width_unique = sorted(set(widths))
        ntk_by_width = [np.mean([ntk_times[i] for i, w in enumerate(widths) if w == width]) 
                       for width in width_unique]
        nngp_by_width = [np.mean([nngp_times[i] for i, w in enumerate(widths) if w == width]) 
                        for width in width_unique]
        
        ax3.plot(width_unique, ntk_by_width, 'o-', label='NTK', linewidth=2, markersize=8)
        ax3.plot(width_unique, nngp_by_width, 's-', label='NNGP', linewidth=2, markersize=8)
        ax3.set_xlabel('Network Width')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Computation Time vs Network Width')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency (Time per sample)
        ax4 = axes[1, 1]
        efficiency = [ntk_times[i] / batch_sizes[i] for i in range(len(ntk_times))]
        efficiency_by_batch = [np.mean([efficiency[i] for i, b in enumerate(batch_sizes) if b == batch]) 
                              for batch in batch_unique]
        
        ax4.plot(batch_unique, efficiency_by_batch, 'o-', color=self.colors[3], linewidth=2, markersize=8)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Time per Sample (seconds)')
        ax4.set_title('Computational Efficiency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance scaling plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_cross_library_comparison(self, results: Dict, save_path: Optional[str] = None):
        """Plot comparison between TorchTangents and neural-tangents."""
        
        if 'performance' not in results or 'comparisons' not in results['performance']:
            print("No cross-library comparison data found")
            return
        
        comparisons = results['performance']['comparisons']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('TorchTangents vs Neural-Tangents Comparison', fontsize=16, fontweight='bold')
        
        # Extract comparison data
        configs = [f"B{c['config']['batch_size']}_H{c['config']['hidden_dim']}" for c in comparisons]
        tt_times = [c['torchtangents_time'] for c in comparisons]
        nt_times = [c['neural_tangents_time'] for c in comparisons]
        speedups = [c['speedup_ratio'] for c in comparisons]
        memory_ratios = [c['memory_ratio'] for c in comparisons]
        
        # Plot 1: Time comparison
        ax1 = axes[0, 0]
        x = np.arange(len(configs))
        width = 0.35
        
        ax1.bar(x - width/2, tt_times, width, label='TorchTangents', alpha=0.8)
        ax1.bar(x + width/2, nt_times, width, label='Neural-tangents', alpha=0.8)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Computation Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup ratios
        ax2 = axes[0, 1]
        bars = ax2.bar(configs, speedups, alpha=0.8, color=self.colors[2])
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal performance')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Speedup Ratio (TT/NT)')
        ax2.set_title('TorchTangents vs Neural-tangents Speedup')
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{speedup:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Memory comparison
        ax3 = axes[1, 0]
        bars = ax3.bar(configs, memory_ratios, alpha=0.8, color=self.colors[3])
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal memory')
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Memory Ratio (TT/NT)')
        ax3.set_title('Memory Usage Comparison')
        ax3.set_xticklabels(configs, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance vs Memory tradeoff
        ax4 = axes[1, 1]
        scatter = ax4.scatter(speedups, memory_ratios, s=100, alpha=0.7, c=range(len(speedups)), 
                             cmap='viridis')
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Speedup Ratio (TT/NT)')
        ax4.set_ylabel('Memory Ratio (TT/NT)')
        ax4.set_title('Performance vs Memory Tradeoff')
        ax4.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax4.text(1.5, 1.5, 'Slower + More Memory', ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
        ax4.text(0.5, 0.5, 'Faster + Less Memory', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cross-library comparison plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_feature_completeness(self, results: Dict, save_path: Optional[str] = None):
        """Plot feature completeness comparison."""
        
        if 'feature_completeness' not in results:
            print("No feature completeness data found")
            return
        
        features_data = results['feature_completeness']
        features = features_data['features']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle('Feature Completeness Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Feature comparison matrix
        feature_names = [f[0] for f in features]
        tt_support = [1 if f[1] else 0 for f in features]
        nt_support = [1 if f[2] else 0 for f in features]
        
        comparison_matrix = np.array([tt_support, nt_support])
        
        im = ax1.imshow(comparison_matrix, aspect='auto', cmap='RdYlGn', alpha=0.8)
        ax1.set_xticks(range(len(feature_names)))
        ax1.set_xticklabels(feature_names, rotation=45, ha='right')
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['TorchTangents', 'Neural-tangents'])
        ax1.set_title('Feature Support Matrix')
        
        # Add text annotations
        for i in range(len(feature_names)):
            for j in range(2):
                text = '✓' if comparison_matrix[j, i] else '✗'
                ax1.text(i, j, text, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Plot 2: Feature completeness scores
        libraries = ['TorchTangents', 'Neural-tangents']
        scores = [features_data['torchtangents_score'], features_data['neural_tangents_score']]
        total = features_data['total_features']
        
        bars = ax2.bar(libraries, scores, alpha=0.8)
        ax2.set_ylabel('Number of Features Supported')
        ax2.set_title(f'Feature Completeness Score (out of {total})')
        ax2.set_ylim(0, total)
        
        # Add percentage labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            percentage = (score / total) * 100
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{score}/{total}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature completeness plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_api_complexity(self, results: Dict, save_path: Optional[str] = None):
        """Plot API complexity comparison."""
        
        if 'api_complexity' not in results:
            print("No API complexity data found")
            return
        
        api_data = results['api_complexity']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('API Complexity Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Lines of code comparison
        apis = ['TorchTangents\n(Current)', 'Neural-tangents', 'TorchTangents\n(Proposed)']
        lines = [
            api_data['torchtangents_current_lines'],
            api_data['neural_tangents_lines'],
            api_data['torchtangents_proposed_lines']
        ]
        
        bars = ax1.bar(apis, lines, alpha=0.8, color=[self.colors[0], self.colors[1], self.colors[2]])
        ax1.set_ylabel('Lines of Code')
        ax1.set_title('API Complexity (Lines of Code)')
        
        # Add value labels
        for bar, line_count in zip(bars, lines):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{line_count}', ha='center', va='bottom', fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Complexity reduction
        reduction = api_data['complexity_reduction_percent']
        
        # Pie chart showing reduction
        sizes = [reduction, 100 - reduction]
        labels = [f'Reduced\n({reduction:.1f}%)', f'Remaining\n({100-reduction:.1f}%)']
        colors = ['lightgreen', 'lightcoral']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('API Complexity Reduction\n(Current → Proposed)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"API complexity plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_benchmark_report(self, results: Dict, output_dir: str = "benchmark_report"):
        """Generate comprehensive benchmark report with all visualizations."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating comprehensive benchmark report in {output_path}")
        
        # Generate all plots
        if 'performance' in results:
            self.plot_performance_scaling(results, str(output_path / "performance_scaling.png"))
        
        if 'performance' in results and 'comparisons' in results['performance']:
            self.plot_cross_library_comparison(results, str(output_path / "cross_library_comparison.png"))
        
        if 'feature_completeness' in results:
            self.plot_feature_completeness(results, str(output_path / "feature_completeness.png"))
        
        if 'api_complexity' in results:
            self.plot_api_complexity(results, str(output_path / "api_complexity.png"))
        
        # Generate summary report
        self._generate_summary_report(results, str(output_path / "benchmark_summary.md"))
        
        print(f"Benchmark report generated successfully!")
        print(f"Open {output_path / 'benchmark_summary.md'} to view the complete report.")
    
    def _generate_summary_report(self, results: Dict, filepath: str):
        """Generate markdown summary report."""
        
        with open(filepath, 'w') as f:
            f.write("# TorchTangents Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            if 'performance' in results and 'comparisons' in results['performance']:
                comparisons = results['performance']['comparisons']
                if comparisons:
                    avg_speedup = sum(c['speedup_ratio'] for c in comparisons) / len(comparisons)
                    f.write(f"- **Average speedup vs Neural-tangents**: {avg_speedup:.2f}x\n")
            
            if 'api_complexity' in results:
                complexity = results['api_complexity']
                f.write(f"- **API complexity reduction**: {complexity['complexity_reduction_percent']:.1f}%\n")
            
            if 'feature_completeness' in results:
                features = results['feature_completeness']
                tt_percent = features['torchtangents_score'] / features['total_features'] * 100
                f.write(f"- **Feature completeness**: {tt_percent:.1f}%\n")
            
            f.write("\n")
            
            # Performance Analysis
            if 'performance' in results:
                f.write("## Performance Analysis\n\n")
                f.write("![Performance Scaling](performance_scaling.png)\n\n")
                
                if 'comparisons' in results['performance']:
                    f.write("![Cross-library Comparison](cross_library_comparison.png)\n\n")
                    f.write("### Performance Comparison Details\n\n")
                    
                    comparisons = results['performance']['comparisons']
                    f.write("| Configuration | TorchTangents (s) | Neural-tangents (s) | Speedup |\n")
                    f.write("|---------------|-------------------|---------------------|----------|\n")
                    
                    for comp in comparisons:
                        config = f"B{comp['config']['batch_size']}_H{comp['config']['hidden_dim']}"
                        f.write(f"| {config} | {comp['torchtangents_time']:.4f} | "
                               f"{comp['neural_tangents_time']:.4f} | {comp['speedup_ratio']:.2f}x |\n")
            
            # Feature Analysis
            if 'feature_completeness' in results:
                f.write("\n## Feature Completeness\n\n")
                f.write("![Feature Completeness](feature_completeness.png)\n\n")
                
                features_data = results['feature_completeness']['features']
                f.write("| Feature | TorchTangents | Neural-tangents |\n")
                f.write("|---------|---------------|------------------|\n")
                
                for feature, tt_has, nt_has in features_data:
                    tt_symbol = "✓" if tt_has else "✗"
                    nt_symbol = "✓" if nt_has else "✗"
                    f.write(f"| {feature} | {tt_symbol} | {nt_symbol} |\n")
            
            # API Analysis
            if 'api_complexity' in results:
                f.write("\n## API Complexity\n\n")
                f.write("![API Complexity](api_complexity.png)\n\n")
                
                api_data = results['api_complexity']
                f.write(f"- Current TorchTangents API: {api_data['torchtangents_current_lines']} lines\n")
                f.write(f"- Neural-tangents API: {api_data['neural_tangents_lines']} lines\n")
                f.write(f"- Proposed TorchTangents API: {api_data['torchtangents_proposed_lines']} lines\n")
                f.write(f"- **Complexity reduction**: {api_data['complexity_reduction_percent']:.1f}%\n")
            
            f.write("\n## Conclusions\n\n")
            f.write("TorchTangents provides a competitive alternative to neural-tangents with:\n")
            f.write("- Strong performance characteristics\n")
            f.write("- Native PyTorch integration\n")
            f.write("- Simplified API design\n")
            f.write("- Comprehensive feature coverage\n")


def print_usage_examples():
    """Print usage examples for the visualization tool."""
    
    examples = """
TorchTangents Benchmark Visualization Tool
==========================================

This tool generates visualizations and reports from TorchTangents benchmark results.

USAGE:
    python visualization.py <results_file> [options]
    python visualization.py --example [options]

EXAMPLES:

    # Basic usage with benchmark results file
    python visualization.py benchmark_results.json
    
    # Custom output directory
    python visualization.py results.json --output-dir my_analysis
    
    # Generate specific plots only
    python visualization.py results.json --plots performance,api_complexity
    
    # Use example data for demonstration
    python visualization.py --example
    
    # Generate all available plots
    python visualization.py results.json --plots all
    
    # Use different matplotlib style
    python visualization.py results.json --style dark_background

INPUT FILE FORMAT:
    The results file should be a JSON file with benchmark data containing:
    - 'performance': Performance benchmarking results
    - 'api_complexity': API complexity analysis  
    - 'feature_completeness': Feature comparison data
    - Cross-library comparison data (optional)

OUTPUT:
    - Individual plot images (PNG format)
    - Comprehensive markdown report (benchmark_summary.md)
    - All files saved to specified output directory

PLOT TYPES:
    - performance: Scaling analysis and timing plots
    - cross_library: TorchTangents vs Neural-tangents comparison
    - features: Feature completeness matrix and scores
    - api_complexity: API complexity and reduction analysis
    - all: Generate all available plots

For more help: python visualization.py --help
"""
    
    print(examples)


def main():
    """Main function to handle command line arguments and generate visualizations."""
    
    parser = argparse.ArgumentParser(
        description='Generate visualizations from TorchTangents benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report from benchmark results file
  python visualization.py results.json
  
  # Generate report with custom output directory
  python visualization.py results.json --output-dir my_report
  
  # Generate only specific plots
  python visualization.py results.json --plots performance,api_complexity
  
  # Use example data for demonstration
  python visualization.py --example
"""
    )
    
    parser.add_argument(
        'results_file',
        nargs='?',
        help='Path to JSON file containing benchmark results'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='benchmark_report',
        help='Output directory for generated report (default: benchmark_report)'
    )
    
    parser.add_argument(
        '--plots',
        help='Comma-separated list of plots to generate (performance,cross_library,features,api_complexity,all)'
    )
    
    parser.add_argument(
        '--example',
        action='store_true',
        help='Use example data instead of loading from file (for demonstration)'
    )
    
    parser.add_argument(
        '--style',
        default='seaborn-v0_8',
        help='Matplotlib style to use for plots (default: seaborn-v0_8)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.example and not args.results_file:
        parser.error("Must provide either a results file or use --example")
    
    if args.example and args.results_file:
        parser.error("Cannot use both --example and provide a results file")
    
    # Initialize visualizer
    visualizer = BenchmarkVisualizer(style=args.style)
    
    # Load results
    if args.example:
        print("Using example benchmark data for demonstration...")
        results = _get_example_results()
    else:
        print(f"Loading benchmark results from {args.results_file}...")
        try:
            results = visualizer.load_benchmark_results(args.results_file)
        except FileNotFoundError:
            print(f"Error: Could not find results file: {args.results_file}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in results file: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading results: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Determine which plots to generate
    if args.plots:
        if args.plots.lower() == 'all':
            plot_types = ['performance', 'cross_library', 'features', 'api_complexity']
        else:
            plot_types = [p.strip() for p in args.plots.split(',')]
            valid_plots = ['performance', 'cross_library', 'features', 'api_complexity']
            invalid_plots = [p for p in plot_types if p not in valid_plots]
            if invalid_plots:
                print(f"Error: Invalid plot types: {invalid_plots}", file=sys.stderr)
                print(f"Valid plot types: {valid_plots}", file=sys.stderr)
                sys.exit(1)
    else:
        # Generate all available plots based on data
        plot_types = []
        if 'performance' in results:
            plot_types.append('performance')
        if 'performance' in results and 'comparisons' in results['performance']:
            plot_types.append('cross_library')
        if 'feature_completeness' in results:
            plot_types.append('features')
        if 'api_complexity' in results:
            plot_types.append('api_complexity')
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Generating reports in: {output_path.absolute()}")
    
    # Generate individual plots
    plot_count = 0
    
    if 'performance' in plot_types and 'performance' in results:
        print("Generating performance scaling plot...")
        visualizer.plot_performance_scaling(results, str(output_path / "performance_scaling.png"))
        plot_count += 1
    
    if 'cross_library' in plot_types and 'performance' in results and 'comparisons' in results['performance']:
        print("Generating cross-library comparison plot...")
        visualizer.plot_cross_library_comparison(results, str(output_path / "cross_library_comparison.png"))
        plot_count += 1
    
    if 'features' in plot_types and 'feature_completeness' in results:
        print("Generating feature completeness plot...")
        visualizer.plot_feature_completeness(results, str(output_path / "feature_completeness.png"))
        plot_count += 1
    
    if 'api_complexity' in plot_types and 'api_complexity' in results:
        print("Generating API complexity plot...")
        visualizer.plot_api_complexity(results, str(output_path / "api_complexity.png"))
        plot_count += 1
    
    # Generate comprehensive report if multiple plots
    if plot_count > 1 or not args.plots:
        print("Generating comprehensive markdown report...")
        visualizer._generate_summary_report(results, str(output_path / "benchmark_summary.md"))
    
    print(f"\n✅ Generated {plot_count} plots successfully!")
    print(f"📁 Report available in: {output_path.absolute()}")
    if plot_count > 1 or not args.plots:
        print(f"📄 Open {output_path / 'benchmark_summary.md'} to view the complete report")


def _get_example_results() -> Dict:
    """Get example benchmark results for demonstration."""
    return {
        'performance': {
            'results': [
                {
                    'config': {'batch_size': 8, 'width': 64, 'depth': 2},
                    'timings': {'ntk': {'mean': 0.1}, 'nngp': {'mean': 0.08}},
                    'memory': {'peak_mb': 250}
                },
                {
                    'config': {'batch_size': 16, 'width': 128, 'depth': 3},
                    'timings': {'ntk': {'mean': 0.2}, 'nngp': {'mean': 0.15}},
                    'memory': {'peak_mb': 500}
                },
                {
                    'config': {'batch_size': 32, 'width': 256, 'depth': 4},
                    'timings': {'ntk': {'mean': 0.4}, 'nngp': {'mean': 0.3}},
                    'memory': {'peak_mb': 1000}
                }
            ],
            'comparisons': [
                {
                    'config': {'batch_size': 8, 'hidden_dim': 64},
                    'torchtangents_time': 0.1,
                    'neural_tangents_time': 0.15,
                    'speedup_ratio': 1.5,
                    'memory_ratio': 0.8
                },
                {
                    'config': {'batch_size': 16, 'hidden_dim': 128},
                    'torchtangents_time': 0.2,
                    'neural_tangents_time': 0.3,
                    'speedup_ratio': 1.5,
                    'memory_ratio': 0.9
                }
            ]
        },
        'api_complexity': {
            'torchtangents_current_lines': 6,
            'neural_tangents_lines': 6,
            'torchtangents_proposed_lines': 3,
            'complexity_reduction_percent': 50.0
        },
        'feature_completeness': {
            'torchtangents_score': 10,
            'neural_tangents_score': 12,
            'total_features': 13,
            'features': [
                ('Basic NTK', True, True),
                ('NNGP', True, True),
                ('Batching support', True, True),
                ('GPU acceleration', True, True),
                ('Empirical kernels', True, True),
                ('Monte Carlo kernels', True, True),
                ('PyTorch integration', True, False),
                ('JAX integration', False, True),
                ('Convolutions', False, True),
                ('Attention layers', False, True)
            ]
        }
    }


if __name__ == "__main__":
    main()