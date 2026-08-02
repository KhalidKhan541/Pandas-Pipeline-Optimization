import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any
import os

class SlowPipeline:
    """Deliberately slow pipeline for optimization benchmarking.
    Uses anti-patterns: float64 everywhere, object dtypes, .apply() loops.
    """
    
    def __init__(self, n_rows: int = 50_000_000, seed: int = 42):
        self.n_rows = n_rows
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        self.timing = {}
    
    def generate_data(self) -> pd.DataFrame:
        """Generate large dataset with intentionally bad dtypes.
        Anti-patterns:
        - All floats as float64 (should be float32)
        - Strings stored as object (should be category)
        - Booleans stored as object
        - Dates stored as object
        """
        self.logger.info(f"Generating {self.n_rows:,} rows of data...")
        start = time.time()
        
        df = pd.DataFrame({
            # Float64 columns (should be float32)
            'value_a': np.random.randn(self.n_rows).astype(np.float64),
            'value_b': np.random.uniform(0, 100, self.n_rows).astype(np.float64),
            'value_c': np.random.exponential(5, self.n_rows).astype(np.float64),
            'value_d': np.random.normal(50, 15, self.n_rows).astype(np.float64),
            'value_e': np.random.uniform(1000, 10000, self.n_rows).astype(np.float64),
            
            # Object columns (should be category)
            'category_a': np.random.choice(['A', 'B', 'C', 'D', 'E'], self.n_rows).astype(object),
            'category_b': np.random.choice(['low', 'medium', 'high'], self.n_rows).astype(object),
            'category_c': np.random.choice(['region_1', 'region_2', 'region_3', 'region_4', 'region_5'], self.n_rows).astype(object),
            'status': np.random.choice(['active', 'inactive', 'pending'], self.n_rows).astype(object),
            
            # Integer columns (should be int8/int16 instead of int64)
            'quantity': np.random.randint(1, 100, self.n_rows).astype(np.int64),
            'score': np.random.randint(0, 1000, self.n_rows).astype(np.int64),
            'priority': np.random.randint(1, 10, self.n_rows).astype(np.int64),
            
            # Boolean stored as object
            'is_flagged': np.random.choice(['True', 'False'], self.n_rows).astype(object),
            
            # Date stored as object
            'date_str': pd.date_range('2020-01-01', periods=self.n_rows, freq='s').astype(str).astype(object),
        })
        
        self.timing['generate_data'] = time.time() - start
        self.logger.info(f"Generated in {self.timing['generate_data']:.2f}s")
        return df
    
    def slow_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deliberately slow transformations using .apply() and object operations.
        Anti-patterns:
        - .apply() with lambda functions
        - String operations on object columns
        - Repeated DataFrame copies
        - No vectorization
        """
        self.logger.info("Running slow transformations...")
        start = time.time()
        
        # Anti-pattern: .apply() with lambda (should be vectorized)
        df['computed_a'] = df['value_a'].apply(lambda x: x ** 2 + 2 * x + 1)
        df['computed_b'] = df['value_b'].apply(lambda x: x * 1.5 if x > 50 else x * 0.5)
        df['computed_c'] = df['value_c'].apply(lambda x: min(x, 20))
        
        # Anti-pattern: string operations on object column
        df['category_upper'] = df['category_a'].apply(lambda x: x.upper())
        df['status_flag'] = df['status'].apply(lambda x: 1 if x == 'active' else 0)
        
        # Anti-pattern: multiple passes over data
        df['rolling_mean'] = df['value_a'].rolling(window=1000, min_periods=1).mean()
        df['rolling_std'] = df['value_a'].rolling(window=1000, min_periods=1).std()
        df['z_score'] = (df['value_a'] - df['rolling_mean']) / df['rolling_std'].replace(0, 1)
        
        # Anti-pattern: repeated groupby operations
        for col in ['value_a', 'value_b', 'value_c']:
            df[f'{col}_group_mean'] = df.groupby('category_a')[col].transform('mean')
        
        # Anti-pattern: copy and modify
        df2 = df.copy()
        df2['lag_1'] = df2['value_a'].shift(1)
        df2['lag_2'] = df2['value_a'].shift(2)
        df2['diff_1'] = df2['value_a'].diff(1)
        df2['pct_change'] = df2['value_a'].pct_change(1)
        
        self.timing['slow_transforms'] = time.time() - start
        self.logger.info(f"Slow transforms completed in {self.timing['slow_transforms']:.2f}s")
        return df2
    
    def slow_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deliberately slow aggregations."""
        self.logger.info("Running slow aggregations...")
        start = time.time()
        
        # Anti-pattern: multiple separate groupby operations
        agg1 = df.groupby('category_a')['value_a'].agg(['mean', 'std', 'min', 'max'])
        agg2 = df.groupby('category_b')['value_b'].agg(['mean', 'std', 'min', 'max'])
        agg3 = df.groupby('category_c')['value_c'].agg(['mean', 'std', 'min', 'max'])
        agg4 = df.groupby('status')['value_a'].count()
        
        # Anti-pattern: apply on groups
        def custom_agg(group):
            return pd.Series({
                'mean': group['value_a'].mean(),
                'std': group['value_a'].std(),
                'range': group['value_a'].max() - group['value_a'].min(),
            })
        
        agg5 = df.groupby('category_a').apply(custom_agg)
        
        self.timing['slow_aggregations'] = time.time() - start
        self.logger.info(f"Slow aggregations completed in {self.timing['slow_aggregations']:.2f}s")
        
        return agg5
    
    def slow_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deliberately slow filtering with multiple passes."""
        self.logger.info("Running slow filter...")
        start = time.time()
        
        # Anti-pattern: multiple separate filters instead of combined
        mask1 = df['value_a'] > df['value_a'].mean()
        mask2 = df['value_b'] > 30
        mask3 = df['category_b'] == 'high'
        mask4 = df['is_flagged'] == 'True'
        
        filtered = df[mask1 & mask2 & mask3 & mask4]
        
        self.timing['slow_filter'] = time.time() - start
        self.logger.info(f"Slow filter completed in {self.timing['slow_filter']:.2f}s")
        
        return filtered
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the full slow pipeline and return results."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING SLOW PIPELINE (50M rows)")
        self.logger.info("=" * 60)
        
        total_start = time.time()
        
        df = self.generate_data()
        df = self.slow_transforms(df)
        agg = self.slow_aggregations(df)
        filtered = self.slow_filter(df)
        
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
