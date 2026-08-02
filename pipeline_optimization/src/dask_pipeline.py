import dask.dataframe as dd
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any
import os

class DaskPipeline:
    """Dask-based parallel pipeline for large-scale data processing.
    Uses Dask DataFrames for out-of-core and parallel processing.
    """
    
    def __init__(self, n_rows: int = 50_000_000, seed: int = 42, n_partitions: int = 8):
        self.n_rows = n_rows
        self.seed = seed
        self.n_partitions = n_partitions
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        self.timing = {}
    
    def generate_data_dask(self) -> dd.DataFrame:
        """Generate dataset as Dask DataFrame with optimized dtypes."""
        self.logger.info(f"Generating {self.n_rows:,} rows as Dask DataFrame...")
        start = time.time()
        
        # Generate pandas DataFrame first, then convert to Dask
        pdf = pd.DataFrame({
            'value_a': np.random.randn(self.n_rows).astype(np.float32),
            'value_b': np.random.uniform(0, 100, self.n_rows).astype(np.float32),
            'value_c': np.random.exponential(5, self.n_rows).astype(np.float32),
            'value_d': np.random.normal(50, 15, self.n_rows).astype(np.float32),
            'value_e': np.random.uniform(1000, 10000, self.n_rows).astype(np.float32),
            'category_a': np.random.choice(['A', 'B', 'C', 'D', 'E'], self.n_rows),
            'category_b': np.random.choice(['low', 'medium', 'high'], self.n_rows),
            'category_c': np.random.choice(['region_1', 'region_2', 'region_3', 'region_4', 'region_5'], self.n_rows),
            'status': np.random.choice(['active', 'inactive', 'pending'], self.n_rows),
            'quantity': np.random.randint(1, 100, self.n_rows).astype(np.int16),
            'score': np.random.randint(0, 1000, self.n_rows).astype(np.int16),
            'priority': np.random.randint(1, 10, self.n_rows).astype(np.int8),
            'is_flagged': np.random.choice([True, False], self.n_rows),
        })
        
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(pdf, npartitions=self.n_partitions)
        
        # Optimize dtypes
        ddf['category_a'] = ddf['category_a'].astype('category')
        ddf['category_b'] = ddf['category_b'].astype('category')
        ddf['category_c'] = ddf['category_c'].astype('category')
        ddf['status'] = ddf['status'].astype('category')
        
        self.timing['generate_data'] = time.time() - start
        self.logger.info(f"Dask DataFrame created in {self.timing['generate_data']:.2f}s")
        return ddf
    
    def parallel_transforms(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """Parallel transformations using Dask."""
        self.logger.info("Running parallel Dask transformations...")
        start = time.time()
        
        # Vectorized operations (Dask lazily evaluates)
        ddf['computed_a'] = ddf['value_a'] ** 2 + 2 * ddf['value_a'] + 1
        ddf['computed_b'] = dd.from_delayed(
            ddf.map_partitions(lambda df: np.where(df['value_b'] > 50, df['value_b'] * 1.5, df['value_b'] * 0.5))
        )
        ddf['computed_c'] = ddf['value_c'].clip(upper=20)
        
        # Vectorized status flag
        ddf['status_flag'] = (ddf['status'] == 'active').astype(np.int8)
        
        # Rolling operations (Dask handles partition-wise)
        ddf['rolling_mean'] = ddf['value_a'].rolling(window=1000, min_periods=1).mean()
        ddf['rolling_std'] = ddf['value_b'].rolling(window=1000, min_periods=1).std().fillna(1)
        ddf['z_score'] = (ddf['value_a'] - ddf['rolling_mean']) / ddf['rolling_std']
        
        # Groupby transforms (parallel across partitions)
        ddf['value_a_group_mean'] = ddf.groupby('category_a', observed=True)['value_a'].transform('mean')
        ddf['value_b_group_mean'] = ddf.groupby('category_a', observed=True)['value_b'].transform('mean')
        
        # Trigger computation
        ddf = ddf.compute()
        
        self.timing['parallel_transforms'] = time.time() - start
        self.logger.info(f"Parallel transforms completed in {self.timing['parallel_transforms']:.2f}s")
        return ddf
    
    def parallel_aggregations(self, ddf: dd.DataFrame) -> pd.DataFrame:
        """Parallel aggregations using Dask."""
        self.logger.info("Running parallel Dask aggregations...")
        start = time.time()
        
        # Parallel groupby aggregation
        agg = ddf.groupby('category_a', observed=True).agg({
            'value_a': ['mean', 'std', 'min', 'max'],
            'value_b': ['mean', 'std', 'min', 'max'],
            'value_c': ['mean', 'std', 'min', 'max'],
        }).compute()
        
        # Flatten column names
        agg.columns = ['_'.join(col) for col in agg.columns]
        agg = agg.reset_index()
        
        self.timing['parallel_aggregations'] = time.time() - start
        self.logger.info(f"Parallel aggregations completed in {self.timing['parallel_aggregations']:.2f}s")
        
        return agg
    
    def parallel_filter(self, ddf: dd.DataFrame) -> pd.DataFrame:
        """Parallel filtering using Dask."""
        self.logger.info("Running parallel Dask filter...")
        start = time.time()
        
        # Parallel filter
        mean_val = ddf['value_a'].mean().compute()
        filtered = ddf[
            (ddf['value_a'] > mean_val) &
            (ddf['value_b'] > 30) &
            (ddf['category_b'] == 'high') &
            (ddf['is_flagged'] == True)
        ].compute()
        
        self.timing['parallel_filter'] = time.time() - start
        self.logger.info(f"Parallel filter completed in {self.timing['parallel_filter']:.2f}s")
        
        return filtered
    
    def chunked_csv_read(self, filepath: str, chunksize: int = 100_000) -> pd.DataFrame:
        """Read large CSV in chunks and concatenate."""
        self.logger.info(f"Reading CSV in chunks of {chunksize:,}...")
        start = time.time()
        
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        
        self.timing['chunked_read'] = time.time() - start
        self.logger.info(f"Chunked read completed in {self.timing['chunked_read']:.2f}s")
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, filepath: str):
        """Save DataFrame to Parquet for efficient storage."""
        self.logger.info(f"Saving to Parquet: {filepath}")
        df.to_parquet(filepath, index=False, engine='pyarrow')
    
    def read_parquet(self, filepath: str) -> pd.DataFrame:
        """Read Parquet file efficiently."""
        return pd.read_parquet(filepath, engine='pyarrow')
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the full Dask pipeline and return results."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING DASK PARALLEL PIPELINE (50M rows)")
        self.logger.info("=" * 60)
        
        total_start = time.time()
        
        ddf = self.generate_data_dask()
        df = self.parallel_transforms(ddf)
        agg = self.parallel_aggregations(ddf)
        filtered = self.parallel_filter(ddf)
        
        total_time = time.time() - total_start
        self.timing['total'] = total_time
        
        self.logger.info(f"\nTotal Dask pipeline time: {total_time:.2f}s")
        self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        
        return {
            'dataframe': df,
            'aggregations': agg,
            'filtered': filtered,
            'timing': self.timing,
            'memory_mb': df.memory_usage(deep=True).sum() / 1e6,
        }
