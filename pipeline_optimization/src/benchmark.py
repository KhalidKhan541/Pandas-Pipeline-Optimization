import pandas as pd
import numpy as np
import time
import memory_profiler
import line_profiler
from typing import Dict, Any, Callable
import logging
import json
import os
from datetime import datetime


class PipelineBenchmark:
    """Benchmark and profile pipeline performance."""

    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.results = {}

    def measure_memory(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Measure memory usage of a function."""
        memory_before = memory_profiler.memory_usage()[0]
        result = func(*args, **kwargs)
        memory_after = memory_profiler.memory_usage()[0]

        return {
            'result': result,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_after - memory_before,
        }

    def benchmark_pipeline(self, pipeline_class, pipeline_name: str, n_rows: int = 10_000_000) -> Dict[str, Any]:
        """Benchmark a complete pipeline."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"BENCHMARKING: {pipeline_name}")
        self.logger.info(f"{'='*60}")

        self.logger.info(f"Running with memory profiling...")
        mem_result = self.measure_memory(
            lambda: pipeline_class(n_rows=n_rows).run_full_pipeline()
        )

        timing = mem_result['result']['timing']
        memory_mb = mem_result['result']['memory_mb']

        benchmark = {
            'pipeline': pipeline_name,
            'n_rows': n_rows,
            'timing': timing,
            'total_time_s': timing.get('total', 0),
            'memory_mb': memory_mb,
            'memory_before_mb': mem_result['memory_before_mb'],
            'memory_after_mb': mem_result['memory_after_mb'],
            'memory_delta_mb': mem_result['memory_delta_mb'],
            'rows_per_second': n_rows / timing.get('total', 1),
            'timestamp': datetime.now().isoformat(),
        }

        self.results[pipeline_name] = benchmark

        self.logger.info(f"\nResults for {pipeline_name}:")
        self.logger.info(f"  Total time: {benchmark['total_time_s']:.2f}s")
        self.logger.info(f"  Memory: {benchmark['memory_mb']:.2f} MB")
        self.logger.info(f"  Memory delta: {benchmark['memory_delta_mb']:.2f} MB")
        self.logger.info(f"  Rows/second: {benchmark['rows_per_second']:,.0f}")

        return benchmark

    def profile_line(self, func: Callable, *args, **kwargs) -> str:
        """Profile a function with line_profiler and return results."""
        profiler = line_profiler.LineProfiler()
        profiler.add_function(func)

        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        profiler.print_stats()
        sys.stdout = old_stdout

        profile_output = buffer.getvalue()
        return profile_output

    def compare_results(self) -> pd.DataFrame:
        """Compare benchmark results across pipelines."""
        comparison = []
        for name, result in self.results.items():
            comparison.append({
                'Pipeline': name,
                'Total Time (s)': result['total_time_s'],
                'Memory (MB)': result['memory_mb'],
                'Rows/Second': result['rows_per_second'],
                'Speedup vs Baseline': None,
            })

        df = pd.DataFrame(comparison)

        if 'slow' in self.results:
            baseline_time = self.results['slow']['total_time_s']
            for i, row in df.iterrows():
                df.at[i, 'Speedup vs Baseline'] = baseline_time / row['Total Time (s)']

        return df

    def generate_report(self) -> str:
        """Generate markdown benchmark report."""
        report = []
        report.append("# Pipeline Optimization Benchmark Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        comparison = self.compare_results()
        report.append("## Performance Summary\n")
        report.append(comparison.to_markdown(index=False))

        report.append("\n## Detailed Results\n")
        for name, result in self.results.items():
            report.append(f"\n### {name.title()} Pipeline\n")
            report.append(f"- **Total Time**: {result['total_time_s']:.2f}s")
            report.append(f"- **Memory Usage**: {result['memory_mb']:.2f} MB")
            report.append(f"- **Rows/Second**: {result['rows_per_second']:,.0f}")
            report.append(f"- **Memory Delta**: {result['memory_delta_mb']:.2f} MB")
            report.append("\n**Stage Breakdown:**\n")
            for stage, time_val in result['timing'].items():
                if stage != 'total':
                    report.append(f"- {stage}: {time_val:.2f}s")

        report.append("\n## Optimization Techniques Applied\n")
        report.append("1. **dtype Downcasting**: float64→float32 (50% memory), int64→int8/int16")
        report.append("2. **Category Dtypes**: object→category for low-cardinality strings")
        report.append("3. **Vectorized Operations**: .apply() → NumPy/pandas vectorized")
        report.append("4. **Single-pass Aggregations**: Multiple groupby → single groupby with multiple aggs")
        report.append("5. **Dask Parallelism**: Multi-partition parallel processing")
        report.append("6. **Chunked I/O**: Large file reading in chunks")

        report_text = '\n'.join(report)

        report_path = os.path.join(self.output_dir, 'benchmark_report.md')
        with open(report_path, 'w') as f:
            f.write(report_text)

        self.logger.info(f"Report saved to {report_path}")
        return report_text

    def save_results(self):
        """Save benchmark results to JSON."""
        results_path = os.path.join(self.output_dir, 'benchmark_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {results_path}")

    def run_full_benchmark(self, n_rows: int = 10_000_000) -> pd.DataFrame:
        """Run full benchmark suite."""
        from .slow_pipeline import SlowPipeline
        from .fast_pipeline import FastPipeline
        from .dask_pipeline import DaskPipeline

        self.benchmark_pipeline(SlowPipeline, 'slow', n_rows)
        self.benchmark_pipeline(FastPipeline, 'fast', n_rows)
        self.benchmark_pipeline(DaskPipeline, 'dask', n_rows)

        self.generate_report()
        self.save_results()

        return self.compare_results()
