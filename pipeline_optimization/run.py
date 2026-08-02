#!/usr/bin/env python3
"""CLI entry point for Pandas Pipeline Optimization.

Usage:
    python -m pipeline_optimization.run slow
    python -m pipeline_optimization.run fast
    python -m pipeline_optimization.run dask
    python -m pipeline_optimization.run benchmark
    python -m pipeline_optimization.run optimize --input data.csv --output data_optimized.csv
"""

import argparse
import sys
import time
from pathlib import Path

import yaml


def load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def cmd_slow(args: argparse.Namespace) -> None:
    from pipeline_optimization.src.slow_pipeline import run_slow_pipeline

    config = load_config(args.config)
    print(f"[slow] Running slow pipeline with {config['pipeline']['n_rows']:,} rows...")
    t0 = time.perf_counter()
    run_slow_pipeline(config)
    elapsed = time.perf_counter() - t0
    print(f"[slow] Completed in {elapsed:.2f}s")


def cmd_fast(args: argparse.Namespace) -> None:
    from pipeline_optimization.src.fast_pipeline import run_fast_pipeline

    config = load_config(args.config)
    print(f"[fast] Running optimized pipeline with {config['pipeline']['n_rows']:,} rows...")
    t0 = time.perf_counter()
    run_fast_pipeline(config)
    elapsed = time.perf_counter() - t0
    print(f"[fast] Completed in {elapsed:.2f}s")


def cmd_dask(args: argparse.Namespace) -> None:
    from pipeline_optimization.src.dask_pipeline import run_dask_pipeline

    config = load_config(args.config)
    print(f"[dask] Running Dask pipeline with {config['pipeline']['n_rows']:,} rows...")
    t0 = time.perf_counter()
    run_dask_pipeline(config)
    elapsed = time.perf_counter() - t0
    print(f"[dask] Completed in {elapsed:.2f}s")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from pipeline_optimization.src.benchmark import run_benchmark

    config = load_config(args.config)
    print(f"[benchmark] Running full benchmark comparison...")
    run_benchmark(config)
    print("[benchmark] Benchmark complete. Check outputs/ for results.")


def cmd_optimize(args: argparse.Namespace) -> None:
    from pipeline_optimization.src.optimizer import optimize_csv

    config = load_config(args.config)
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_stem(
        input_path.stem + "_optimized"
    )

    if not input_path.exists():
        print(f"[optimize] Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[optimize] Optimizing {input_path} -> {output_path}...")
    t0 = time.perf_counter()
    optimize_csv(input_path, output_path, config)
    elapsed = time.perf_counter() - t0
    print(f"[optimize] Completed in {elapsed:.2f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline-optimization",
        description="Large-Scale Pandas Pipeline Optimization",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    subparsers.add_parser("slow", help="Run slow pipeline (baseline with anti-patterns)")
    subparsers.add_parser("fast", help="Run optimized pipeline")
    subparsers.add_parser("dask", help="Run Dask parallel pipeline")
    subparsers.add_parser("benchmark", help="Run full benchmark comparison")

    opt_parser = subparsers.add_parser("optimize", help="Optimize a CSV file's dtypes")
    opt_parser.add_argument("--input", required=True, help="Input CSV path")
    opt_parser.add_argument("--output", default=None, help="Output CSV path (default: <input>_optimized.csv)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "slow": cmd_slow,
        "fast": cmd_fast,
        "dask": cmd_dask,
        "benchmark": cmd_benchmark,
        "optimize": cmd_optimize,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
