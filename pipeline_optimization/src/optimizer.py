import pandas as pd
import numpy as np
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class DtypeOptimizer:
    """Optimize DataFrame dtypes for memory efficiency."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def downcast_floats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast float64 columns to float32."""
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = df[col].astype(np.float32)
        self.logger.info(f"Downcast {len(float_cols)} float columns to float32")
        return df
    
    def downcast_ints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast int64 columns to smallest possible integer type."""
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype(np.int8)
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype(np.int16)
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
        return df
    
    def convert_objects_to_category(self, df: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
        """Convert object columns with low cardinality to category."""
        object_cols = df.select_dtypes(include=['object']).columns
        converted = 0
        for col in object_cols:
            if df[col].nunique() < threshold:
                df[col] = df[col].astype('category')
                converted += 1
        self.logger.info(f"Converted {converted} object columns to category")
        return df
    
    def optimize_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all dtype optimizations."""
        initial_memory = df.memory_usage(deep=True).sum() / 1e6
        
        df = self.downcast_floats(df)
        df = self.downcast_ints(df)
        df = self.convert_objects_to_category(df)
        
        final_memory = df.memory_usage(deep=True).sum() / 1e6
        reduction = (1 - final_memory / initial_memory) * 100
        
        self.logger.info(f"Memory: {initial_memory:.2f} MB → {final_memory:.2f} MB ({reduction:.1f}% reduction)")
        return df
    
    def get_memory_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate memory usage report by column."""
        memory = df.memory_usage(deep=True) / 1e6
        memory = memory.reset_index()
        memory.columns = ['Column', 'Memory (MB)']
        memory['Dtype'] = [df[col].dtype for col in memory['Column']]
        memory['% of Total'] = (memory['Memory (MB)'] / memory['Memory (MB)'].sum() * 100).round(2)
        return memory.sort_values('Memory (MB)', ascending=False).head(20)
    
    def estimate_savings(self, df: pd.DataFrame) -> Dict[str, float]:
        """Estimate memory savings without applying."""
        current = df.memory_usage(deep=True).sum() / 1e6
        
        estimated = 0
        for col in df.columns:
            dtype = df[col].dtype
            if dtype == np.float64:
                estimated += df[col].astype(np.float32).memory_usage(deep=True) / 1e6
            elif dtype == np.int64:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min >= -128 and col_max <= 127:
                    estimated += df[col].astype(np.int8).memory_usage(deep=True) / 1e6
                elif col_min >= -32768 and col_max <= 32767:
                    estimated += df[col].astype(np.int16).memory_usage(deep=True) / 1e6
                else:
                    estimated += df[col].astype(np.int32).memory_usage(deep=True) / 1e6
            elif dtype == object and df[col].nunique() < 100:
                estimated += df[col].astype('category').memory_usage(deep=True) / 1e6
            else:
                estimated += df[col].memory_usage(deep=True) / 1e6
        
        return {
            'current_mb': current,
            'estimated_mb': estimated,
            'savings_mb': current - estimated,
            'savings_percent': (1 - estimated / current) * 100 if current > 0 else 0,
        }


class ChunkedIO:
    """Chunked file I/O for large datasets."""
    
    def __init__(self, chunksize: int = 100_000):
        self.chunksize = chunksize
        self.logger = logging.getLogger(__name__)
    
    def read_csv_chunked(self, filepath: str, dtypes: Dict = None) -> pd.DataFrame:
        """Read large CSV file in chunks."""
        self.logger.info(f"Reading {filepath} in chunks of {self.chunksize:,}...")
        start = time.time()
        
        chunks = []
        for i, chunk in enumerate(pd.read_csv(filepath, dtype=dtypes, chunksize=self.chunksize)):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                self.logger.info(f"  Read {(i + 1) * self.chunksize:,} rows...")
        
        df = pd.concat(chunks, ignore_index=True)
        elapsed = time.time() - start
        self.logger.info(f"Chunked read completed: {len(df):,} rows in {elapsed:.2f}s")
        return df
    
    def write_csv_chunked(self, df: pd.DataFrame, filepath: str):
        """Write DataFrame to CSV in chunks."""
        self.logger.info(f"Writing {len(df):,} rows to {filepath} in chunks...")
        start = time.time()
        
        n_chunks = (len(df) + self.chunksize - 1) // self.chunksize
        for i in range(n_chunks):
            start_idx = i * self.chunksize
            end_idx = min((i + 1) * self.chunksize, len(df))
            chunk = df.iloc[start_idx:end_idx]
            
            if i == 0:
                chunk.to_csv(filepath, index=False, mode='w')
            else:
                chunk.to_csv(filepath, index=False, mode='a', header=False)
        
        elapsed = time.time() - start
        self.logger.info(f"Chunked write completed in {elapsed:.2f}s")
    
    def write_parquet(self, df: pd.DataFrame, filepath: str):
        """Write DataFrame to Parquet for efficient storage."""
        df.to_parquet(filepath, index=False, engine='pyarrow')
    
    def read_parquet(self, filepath: str) -> pd.DataFrame:
        """Read Parquet file efficiently."""
        return pd.read_parquet(filepath, engine='pyarrow')
    
    def benchmark_io(self, df: pd.DataFrame, filepath: str) -> Dict[str, float]:
        """Benchmark CSV vs Parquet I/O."""
        results = {}
        
        # CSV write
        start = time.time()
        df.to_csv(filepath + '.csv', index=False)
        results['csv_write'] = time.time() - start
        
        # CSV read
        start = time.time()
        pd.read_csv(filepath + '.csv')
        results['csv_read'] = time.time() - start
        
        # Parquet write
        start = time.time()
        df.to_parquet(filepath + '.parquet', index=False)
        results['parquet_write'] = time.time() - start
        
        # Parquet read
        start = time.time()
        pd.read_parquet(filepath + '.parquet')
        results['parquet_read'] = time.time() - start
        
        # File sizes
        results['csv_size_mb'] = os.path.getsize(filepath + '.csv') / 1e6
        results['parquet_size_mb'] = os.path.getsize(filepath + '.parquet') / 1e6
        results['compression_ratio'] = results['csv_size_mb'] / results['parquet_size_mb']
        
        return results
