"""
Data loading utilities for AgroClimate project
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_climate_data(self, filename: str) -> pd.DataFrame:
        """Load climate data from CSV file"""
        filepath = os.path.join(self.data_path, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            # Generate synthetic climate data for demonstration
            return self._generate_synthetic_climate_data()
    
    def load_crop_yield_data(self, filename: str) -> pd.DataFrame:
        """Load crop yield data from CSV file"""
        filepath = os.path.join(self.data_path, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            # Generate synthetic crop yield data for demonstration
            return self._generate_synthetic_yield_data()
    
    def _generate_synthetic_climate_data(self) -> pd.DataFrame:
        """Generate synthetic climate data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'year': np.random.randint(2000, 2024, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples),
            'temperature_avg': np.random.normal(25, 8, n_samples),
            'temperature_max': np.random.normal(32, 10, n_samples),
            'temperature_min': np.random.normal(18, 6, n_samples),
            'rainfall': np.random.exponential(50, n_samples),
            'humidity': np.random.normal(65, 15, n_samples),
            'wind_speed': np.random.exponential(10, n_samples),
            'solar_radiation': np.random.normal(200, 50, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_yield_data(self) -> pd.DataFrame:
        """Generate synthetic crop yield data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        crops = ['wheat', 'rice', 'maize', 'soybeans', 'barley']
        
        data = {
            'year': np.random.randint(2000, 2024, n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples),
            'crop_type': np.random.choice(crops, n_samples),
            'yield_tons_per_hectare': np.random.normal(3.5, 1.2, n_samples),
            'area_harvested': np.random.exponential(1000, n_samples),
            'production_tons': np.random.exponential(3500, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def merge_climate_yield_data(self, climate_df: pd.DataFrame, yield_df: pd.DataFrame) -> pd.DataFrame:
        """Merge climate and yield data on year and region"""
        # Group climate data by year and region (average monthly values)
        climate_grouped = climate_df.groupby(['year', 'region']).agg({
            'temperature_avg': 'mean',
            'temperature_max': 'max',
            'temperature_min': 'min',
            'rainfall': 'sum',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'solar_radiation': 'mean'
        }).reset_index()
        
        # Merge with yield data
        merged_df = pd.merge(yield_df, climate_grouped, on=['year', 'region'], how='inner')
        
        return merged_df