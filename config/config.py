"""
Configuration settings for AgroClimate project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, 'outputs')

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Climate risk thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8
}

# Crop categories
CROP_TYPES = [
    'wheat', 'rice', 'maize', 'soybeans', 'barley', 
    'cotton', 'sugarcane', 'potatoes', 'tomatoes'
]

# Climate features
CLIMATE_FEATURES = [
    'temperature_avg', 'temperature_max', 'temperature_min',
    'rainfall', 'humidity', 'wind_speed', 'solar_radiation'
]

# Model hyperparameters
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    }
}