import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any
import os

class FastPipeline:
    """Optimized pipeline with all performance fixes applied.
    Optimizations:
    - dtype downcasting (float64→float32, int64→int8/int16, object→category)
    - Vectorized NumPy operations instead of .apply()
    - Single-pass aggregations
    - Chunked processing where applicable
    """
    
    def __init__(self, n_rows: int = 50_000_000, seed: int = 42):
        self.n_rows = n_rows
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        self.timing = {}
    
    def generate_data_optimized(self) -> pd.DataFrame:
        """Generate dataset with optimal dtypes from the start."""
        self.logger.info(f"Generating {self.n_rows:,} rows with optimized dtypes...")
        start = time.time()
        
        df = pd.DataFrame({
            # Float32 instead of float64 (50% memory savings)
            'value_a': np.random.randn(self.n_rows).astype(np.float32),
            'value_b': np.random.uniform(0, 100, self.n_rows).astype(np.float32),
            'value_c': np.random.exponential(5, self.n_rows).astype(np.float32),
            'value_d': np.random.normal(50, 15, self.n_rows).astype(np.float32),
            'value_e': np.random.uniform(1000, 10000, self.n_rows).astype(np.float32),
            
            # Category dtype instead of object (massive memory savings)
            'category_a': pd.Categorical(np.random.choice(['A', 'B', 'C', 'D', 'E'], self.n_rows)),
            'category_b': pd.Categorical(np.random.choice(['low', 'medium', 'high'], self.n_rows)),
            'category_c': pd.Categorical(np.random.choice(['region_1', 'region_2', 'region_3', 'region_4', 'region_5'], self.n_rows)),
            'status': pd.Categorical(np.random.choice(['active', 'inactive', 'pending'], self.n_rows)),
            
            # int8/int16 instead of int64
            'quantity': np.random.randint(1, 100, self.n_rows).astype(np.int8),
            'score': np.random.randint(0, 1000, self.n_rows).astype(np.int16),
            'priority': np.random.randint(1, 10, self.n_rows).astype(np.int8),
            
            # Boolean as actual boolean
            'is_flagged': np.random.choice([True, False], self.n_rows),
            
            # Datetime instead of object
            'date': pd.date_range('2020-01-01', periods=self.n_rows, freq='s'),
        })
        
        self.timing['generate_data'] = time.time() - start
        self.logger.info(f"Generated in {self.timing['generate_data']:.2f}s")
        return df
    
    def vectorized_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized transformations using vectorized NumPy/pandas operations.
        No .apply() calls - everything is vectorized.
        """
        self.logger.info("Running vectorized transformations...")
        start = time.time()
        
        # Vectorized: no .apply() needed
        df['computed_a'] = df['value_a'] ** 2 + 2 * df['value_a'] + 1
        df['computed_b'] = np.where(df['value_b'] > 50, df['value_b'] * 1.5, df['value_b'] * 0.5)
        df['computed_c'] = np.clip(df['value_c'], None, 20)
        
        # Vectorized string operations using .str accessor
        df['status_flag'] = (df['status'] == 'active').astype(np.int8)
        
        # Single rolling window pass (combine operations)
        rolling = df['value_a'].rolling(window=1000, min_periods=1)
        df['rolling_mean'] = rolling.mean()
        df['rolling_std'] = rolling.std().replace(0, 1)
        df['z_score'] = (df['value_a'] - df['rolling_mean']) / df['rolling_std']
        
        # Single groupby with multiple aggregations
        for col in ['value_a', 'value_b', 'value_c']:
            df[f'{col}_group_mean'] = df.groupby('category_a', observed=True)[col].transform('mean')
        
        # Vectorized lag/diff operations
        df['lag_1'] = df['value_a'].shift(1)
        df['lag_2'] = df['value_a'].shift(2)
        df['diff_1'] = df['value_a'].diff(1)
        df['pct_change'] = df['value_a'].pct_change(1)
        
        self.timing['vectorized_transforms'] = time.time() - start
        self.logger.info(f"Vectorized transforms completed in {self.timing['vectorized_transforms']:.2f}s")
        return df
    
    def vectorized_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized aggregations using single groupby with multiple functions."""
        self.logger.info("Running vectorized aggregations...")
        start = time.time()
        
        # Single groupby with multiple aggregations (not separate groupby calls)
        agg = df.groupby('category_a', observed=True).agg({
            'value_a': ['mean', 'std', 'min', 'max'],
            'value_b': ['mean', 'std', 'min', 'max'],
            'value_c': ['mean', 'std', 'min', 'max'],
        })
        
        # Flatten column names
        agg.columns = ['_'.join(col) for col in agg.columns]
        agg = agg.reset_index()
        
        # Status counts in one operation
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        
        self.timing['vectorized_aggregations'] = time.time() - start
        self.logger.info(f"Vectorized aggregations completed in {self.timing['vectorized_aggregations']:.2f}s")
        
        return agg
    
    def vectorized_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized filtering with single mask operation."""
        self.logger.info("Running vectorized filter...")
        start = time.time()
        
        # Single combined mask (not separate filters)
        mean_val = df['value_a'].mean()
        mask = (
            (df['value_a'] > mean_val) &
            (df['value_b'] > 30) &
            (df['category_b'] == 'high') &
            (df['is_flagged'] == True)
        )
        
        filtered = df.loc[mask]
        
        self.timing['vectorized_filter'] = time.time() - start
        self.logger.info(f"Vectorized filter completed in {self.timing['vectorized_filter']:.2f}s")
        
        return filtered
    
    def downcast_floats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast float64 columns to float32."""
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = df[col].astype(np.float32)
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
            else:
                df[col] = df[col].astype(np.int32)
        return df
    
    def convert_objects_to_category(self, df: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
        """Convert object columns with low cardinality to category."""
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if df[col].nunique() < threshold:
                df[col] = df[col].astype('category')
        return df
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all dtype optimizations."""
        self.logger.info("Optimizing dtypes...")
        start = time.time()
        
        df = self.downcast_floats(df)
        df = self.downcast_ints(df)
        df = self.convert_objects_to_category(df)
        
        self.timing['optimize_dtypes'] = time.time() - start
        return df
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the full optimized pipeline and return results."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING OPTIMIZED PIPELINE (50M rows)")
        self.logger.info("=" * 60)
        
        total_start = time.time()
        
        df = self.generate_data_optimized()
        df = self.vectorized_transforms(df)
        agg = self.vectorized_aggregations(df)
        filtered = self.vectorized_filter(df)
        
        total_time = time.time() - total_start
        self.timing['total'] = total_time
        
        self.logger.info(f"\nTotal pipeline time: {total_time:.2f}s")
        self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        
        return {
            'dataframe': df,
            'aggregations': agg,
            'filtered': filtered,
            'timing': self.timing,
            'memory_mb': df.memory_usage(deep=True).sum() / 1e6,
        }
