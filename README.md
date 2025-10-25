# AgroClimate - Crop Climate Impact Prediction Model

## Overview
AgroClimate is a machine learning project that predicts the impact of climate variations on different crops using historical weather and yield data. This project contributes to SDG 2 (Zero Hunger) and SDG 13 (Climate Action).

## Problem Statement
Build a machine learning model that predicts the impact of climate variations (rainfall deficit, temperature spikes, drought risk) on different crops using historical weather and yield data.

## Dataset Sources
- FAO Climate and Agriculture Data
- Global Land Temperature Dataset (Kaggle)

## Expected Output
- Climate risk scores for different crops
- Adaptive crop recommendations based on climate predictions

## Project Structure
```
AgroClimate/
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed datasets
├── src/
│   ├── data_processing/  # Data preprocessing modules
│   ├── models/          # ML model implementations
│   └── utils/           # Utility functions
├── models/              # Trained model files
├── notebooks/           # Jupyter notebooks for analysis
├── config/             # Configuration files
├── outputs/
│   ├── predictions/    # Model predictions
│   └── visualizations/ # Charts and plots
└── requirements.txt    # Dependencies

```

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python src/main.py`

## Usage
1. Place your datasets in the `data/raw/` directory
2. Run data preprocessing: `python src/data_processing/preprocess.py`
3. Train the model: `python src/models/train_model.py`
4. Generate predictions: `python src/models/predict.py`

## Features
- Climate risk assessment
- Crop yield prediction
- Adaptive crop recommendations
- Visualization dashboards
- Model performance metrics