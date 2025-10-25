"""
Data preprocessing module for AgroClimate project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import List
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.config import *
from src.utils.data_loader import DataLoader

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._create_features(df)
        
        # Encode categorical variables
        df = self._encode_categorical(df)
        
        # Create target variables
        df = self._create_targets(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill numerical columns with mean
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for climate risk assessment"""
        # Temperature stress indicators
        df['temp_stress'] = np.where(df['temperature_max'] > 35, 1, 0)
        df['cold_stress'] = np.where(df['temperature_min'] < 5, 1, 0)
        
        # Rainfall indicators
        df['drought_risk'] = np.where(df['rainfall'] < 200, 1, 0)
        df['flood_risk'] = np.where(df['rainfall'] > 1000, 1, 0)
        
        # Temperature range
        df['temp_range'] = df['temperature_max'] - df['temperature_min']
        
        # Growing degree days (simplified)
        df['growing_degree_days'] = np.maximum(0, df['temperature_avg'] - 10)
        
        # Humidity stress
        df['humidity_stress'] = np.where((df['humidity'] < 30) | (df['humidity'] > 90), 1, 0)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = ['region', 'crop_type']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        return df
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction"""
        # Yield efficiency (yield relative to area)
        df['yield_efficiency'] = df['yield_tons_per_hectare']
        
        # Climate risk score (0-1 scale)
        risk_factors = ['temp_stress', 'cold_stress', 'drought_risk', 'flood_risk', 'humidity_stress']
        df['climate_risk_score'] = df[risk_factors].mean(axis=1)
        
        # Yield impact (binary: 1 if yield is below median, 0 otherwise)
        median_yield = df['yield_tons_per_hectare'].median()
        df['yield_impact'] = np.where(df['yield_tons_per_hectare'] < median_yield, 1, 0)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df_scaled

def main():
    """Main preprocessing function"""
    # Initialize data loader
    loader = DataLoader(DATA_RAW_PATH)
    
    # Load data
    print("Loading climate data...")
    climate_df = loader.load_climate_data('climate_data.csv')
    
    print("Loading crop yield data...")
    yield_df = loader.load_crop_yield_data('crop_yield_data.csv')
    
    # Merge datasets
    print("Merging datasets...")
    merged_df = loader.merge_climate_yield_data(climate_df, yield_df)
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess_data(merged_df)
    
    # Save processed data
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    processed_df.to_csv(os.path.join(DATA_PROCESSED_PATH, 'processed_data.csv'), index=False)
    
    print(f"Processed data saved to {DATA_PROCESSED_PATH}")
    print(f"Dataset shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")

if __name__ == "__main__":
    main()