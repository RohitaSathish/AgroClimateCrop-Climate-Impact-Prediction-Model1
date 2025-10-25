"""
Climate Risk Prediction Model for AgroClimate project
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
# import xgboost as xgb
# import lightgbm as lgb
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.config import *

class ClimateRiskModel:
    def __init__(self):
        self.risk_model = None
        self.yield_model = None
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for model training"""
        # Select feature columns
        feature_cols = [
            'temperature_avg', 'temperature_max', 'temperature_min',
            'rainfall', 'humidity', 'wind_speed', 'solar_radiation',
            'temp_stress', 'cold_stress', 'drought_risk', 'flood_risk',
            'temp_range', 'growing_degree_days', 'humidity_stress',
            'region_encoded', 'crop_type_encoded'
        ]
        
        # Filter existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols
        
        X = df[available_cols]
        y_risk = df['climate_risk_score']
        y_yield = df['yield_tons_per_hectare']
        
        return X, y_risk, y_yield
    
    def train_risk_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train climate risk prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Train RandomForest model
        self.risk_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        self.risk_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.risk_model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation
        cv_scores = cross_val_score(self.risk_model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        return {
            'model_type': 'Climate Risk Prediction',
            'rmse': rmse,
            'cv_rmse': cv_rmse,
            'feature_importance': dict(zip(X.columns, self.risk_model.feature_importances_))
        }
    
    def train_yield_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train crop yield prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Train RandomForest model
        self.yield_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        self.yield_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.yield_model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation
        cv_scores = cross_val_score(self.yield_model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        return {
            'model_type': 'Crop Yield Prediction',
            'rmse': rmse,
            'cv_rmse': cv_rmse,
            'feature_importance': dict(zip(X.columns, self.yield_model.feature_importances_))
        }
    
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict climate risk scores"""
        if self.risk_model is None:
            raise ValueError("Risk model not trained yet")
        return self.risk_model.predict(X[self.feature_columns])
    
    def predict_yield(self, X: pd.DataFrame) -> np.ndarray:
        """Predict crop yields"""
        if self.yield_model is None:
            raise ValueError("Yield model not trained yet")
        return self.yield_model.predict(X[self.feature_columns])
    
    def get_risk_category(self, risk_scores: np.ndarray) -> np.ndarray:
        """Convert risk scores to categories"""
        categories = np.full(len(risk_scores), 'low')
        categories[risk_scores >= RISK_THRESHOLDS['medium']] = 'medium'
        categories[risk_scores >= RISK_THRESHOLDS['high']] = 'high'
        return categories
    
    def save_models(self, model_path: str):
        """Save trained models"""
        os.makedirs(model_path, exist_ok=True)
        
        if self.risk_model:
            joblib.dump(self.risk_model, os.path.join(model_path, 'risk_model.pkl'))
        
        if self.yield_model:
            joblib.dump(self.yield_model, os.path.join(model_path, 'yield_model.pkl'))
        
        # Save feature columns
        joblib.dump(self.feature_columns, os.path.join(model_path, 'feature_columns.pkl'))
    
    def load_models(self, model_path: str):
        """Load trained models"""
        risk_model_path = os.path.join(model_path, 'risk_model.pkl')
        yield_model_path = os.path.join(model_path, 'yield_model.pkl')
        features_path = os.path.join(model_path, 'feature_columns.pkl')
        
        if os.path.exists(risk_model_path):
            self.risk_model = joblib.load(risk_model_path)
        
        if os.path.exists(yield_model_path):
            self.yield_model = joblib.load(yield_model_path)
        
        if os.path.exists(features_path):
            self.feature_columns = joblib.load(features_path)