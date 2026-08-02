"""Source modules for pipeline optimization."""

from pipeline_optimization.src.slow_pipeline import run_slow_pipeline
from pipeline_optimization.src.fast_pipeline import run_fast_pipeline
from pipeline_optimization.src.dask_pipeline import run_dask_pipeline
from pipeline_optimization.src.benchmark import run_benchmark
from pipeline_optimization.src.optimizer import optimize_csv

__all__ = [
    "run_slow_pipeline",
    "run_fast_pipeline",
    "run_dask_pipeline",
    "run_benchmark",
    "optimize_csv",
]
