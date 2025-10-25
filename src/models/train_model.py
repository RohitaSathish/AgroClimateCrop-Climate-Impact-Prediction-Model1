"""
Model training script for AgroClimate project
"""

import pandas as pd
import numpy as np
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.config import *
from src.models.climate_risk_model import ClimateRiskModel

def train_models():
    """Train climate risk and yield prediction models"""
    
    # Load processed data
    data_path = os.path.join(DATA_PROCESSED_PATH, 'processed_data.csv')
    
    if not os.path.exists(data_path):
        print("Processed data not found. Please run preprocessing first.")
        return
    
    print("Loading processed data...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize model
    model = ClimateRiskModel()
    
    # Prepare features
    print("Preparing features...")
    X, y_risk, y_yield = model.prepare_features(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Train risk prediction model
    print("\nTraining climate risk prediction model...")
    risk_metrics = model.train_risk_model(X, y_risk)
    
    print("Risk Model Results:")
    print(f"RMSE: {risk_metrics['rmse']:.4f}")
    print(f"CV RMSE: {risk_metrics['cv_rmse']:.4f}")
    
    # Train yield prediction model
    print("\nTraining crop yield prediction model...")
    yield_metrics = model.train_yield_model(X, y_yield)
    
    print("Yield Model Results:")
    print(f"RMSE: {yield_metrics['rmse']:.4f}")
    print(f"CV RMSE: {yield_metrics['cv_rmse']:.4f}")
    
    # Save models
    print("\nSaving models...")
    os.makedirs(MODELS_PATH, exist_ok=True)
    model.save_models(MODELS_PATH)
    
    # Save metrics
    metrics = {
        'risk_model': risk_metrics,
        'yield_model': yield_metrics
    }
    
    with open(os.path.join(MODELS_PATH, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Models saved to {MODELS_PATH}")
    
    # Feature importance analysis
    print("\nTop 10 Important Features for Risk Prediction:")
    risk_importance = sorted(risk_metrics['feature_importance'].items(), 
                           key=lambda x: x[1], reverse=True)[:10]
    for feature, importance in risk_importance:
        print(f"{feature}: {importance:.4f}")
    
    print("\nTop 10 Important Features for Yield Prediction:")
    yield_importance = sorted(yield_metrics['feature_importance'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
    for feature, importance in yield_importance:
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    train_models()