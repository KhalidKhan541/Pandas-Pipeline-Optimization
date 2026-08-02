# Large-Scale Pandas Pipeline Optimization

Systematic optimization of a deliberately slow 50M-row pipeline: dtype downcasting, vectorization, chunked I/O, and Dask parallelism. Full benchmark report with memory and runtime comparisons.

## Optimization Techniques

### 1. Dtype Downcasting
| Before | After | Savings |
|--------|-------|---------|
| float64 | float32 | 50% |
| int64 | int8/int16 | 75-87% |
| object | category | 60-90% |

### 2. Vectorized Operations
- `.apply(lambda x: ...)` → `np.where()`, `np.clip()`, vectorized math
- Eliminates Python-level loops

### 3. Chunked I/O
- Read large CSVs in configurable chunks
- Parquet format for 2-5x compression

### 4. Dask Parallelism
- Multi-partition parallel processing
- Out-of-core computation for data larger than RAM

## Quick Start

```bash
pip install -r requirements.txt

# Run slow pipeline (baseline)
python -m pipeline_optimization.run slow

# Run fast pipeline (optimized)
python -m pipeline_optimization.run fast

# Run Dask pipeline (parallel)
python -m pipeline_optimization.run dask

# Run full benchmark comparison
python -m pipeline_optimization.run benchmark

# Optimize a CSV file
python -m pipeline_optimization.run optimize --input data.csv --output data_optimized.csv
```

## Benchmark Results (10M rows)

| Pipeline | Time (s) | Memory (MB) | Rows/sec | Speedup |
|----------|----------|-------------|----------|---------|
| Slow | ~120 | ~4000 | ~83K | 1.0x |
| Fast | ~45 | ~2000 | ~222K | 2.7x |
| Dask | ~30 | ~1800 | ~333K | 4.0x |

## Architecture

```
pipeline_optimization/
├── run.py                          # CLI entry point
├── src/
│   ├── slow_pipeline.py            # Baseline (anti-patterns)
│   ├── fast_pipeline.py            # Optimized pandas
│   ├── dask_pipeline.py            # Dask parallel
│   ├── benchmark.py                # Benchmarking & profiling
│   └── optimizer.py                # Dtype & IO utilities
├── configs/
│   └── default.yaml                # Configuration
└── outputs/                        # Reports & results
```

## Dependencies

- pandas, numpy - Core data processing
- dask[complete] - Parallel computing
- pyarrow - Parquet I/O
- memory-profiler - Memory measurement
- line-profiler - Line-by-line profiling
