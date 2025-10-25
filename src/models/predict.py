"""
Prediction script for AgroClimate project
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
from src.models.crop_recommender import CropRecommender

def make_predictions(input_data: dict = None):
    """Make climate risk predictions and crop recommendations"""
    
    # Load trained models
    model = ClimateRiskModel()
    model.load_models(MODELS_PATH)
    
    # Initialize crop recommender
    recommender = CropRecommender()
    
    # Use sample data if no input provided
    if input_data is None:
        input_data = {
            'temperature_avg': 28.5,
            'temperature_max': 35.2,
            'temperature_min': 22.1,
            'rainfall': 450.0,
            'humidity': 68.5,
            'wind_speed': 12.3,
            'solar_radiation': 220.5,
            'region_encoded': 2,
            'crop_type_encoded': 1
        }
    
    print("Input Climate Data:")
    for key, value in input_data.items():
        print(f"{key}: {value}")
    
    # Create DataFrame for prediction
    df_input = pd.DataFrame([input_data])
    
    # Add engineered features
    df_input['temp_stress'] = np.where(df_input['temperature_max'] > 35, 1, 0)
    df_input['cold_stress'] = np.where(df_input['temperature_min'] < 5, 1, 0)
    df_input['drought_risk'] = np.where(df_input['rainfall'] < 200, 1, 0)
    df_input['flood_risk'] = np.where(df_input['rainfall'] > 1000, 1, 0)
    df_input['temp_range'] = df_input['temperature_max'] - df_input['temperature_min']
    df_input['growing_degree_days'] = np.maximum(0, df_input['temperature_avg'] - 10)
    df_input['humidity_stress'] = np.where((df_input['humidity'] < 30) | (df_input['humidity'] > 90), 1, 0)
    
    # Make predictions
    print("\nMaking predictions...")
    
    # Climate risk prediction
    risk_score = model.predict_risk_score(df_input)[0]
    risk_category = model.get_risk_category([risk_score])[0]
    
    # Yield prediction
    predicted_yield = model.predict_yield(df_input)[0]
    
    print(f"\nPrediction Results:")
    print(f"Climate Risk Score: {risk_score:.3f}")
    print(f"Risk Category: {risk_category}")
    print(f"Predicted Yield: {predicted_yield:.2f} tons/hectare")
    
    # Crop recommendations
    climate_data_for_recommender = {
        'temperature_avg': input_data['temperature_avg'],
        'rainfall': input_data['rainfall'],
        'humidity': input_data['humidity'],
        'drought_risk': df_input['drought_risk'].iloc[0],
        'temp_stress': df_input['temp_stress'].iloc[0],
        'cold_stress': df_input['cold_stress'].iloc[0],
        'flood_risk': df_input['flood_risk'].iloc[0],
        'humidity_stress': df_input['humidity_stress'].iloc[0]
    }
    
    # Generate recommendation report
    recommendation_report = recommender.generate_recommendation_report(climate_data_for_recommender)
    
    print(f"\nClimate Summary:")
    print(f"Overall Risk Score: {recommendation_report['climate_summary']['overall_risk_score']:.3f}")
    print(f"Risk Category: {recommendation_report['climate_summary']['risk_category']}")
    
    print(f"\nTop 3 Recommended Crops:")
    for i, crop_rec in enumerate(recommendation_report['recommended_crops'][:3], 1):
        print(f"{i}. {crop_rec['crop'].title()}: {crop_rec['suitability_percentage']} suitable")
    
    print(f"\nClimate Risks Detected:")
    for risk, value in recommendation_report['climate_risks'].items():
        if value > 0:
            print(f"- {risk.replace('_', ' ').title()}: {'Yes' if value else 'No'}")
    
    # Save predictions
    results = {
        'input_data': input_data,
        'predictions': {
            'climate_risk_score': float(risk_score),
            'risk_category': risk_category,
            'predicted_yield': float(predicted_yield)
        },
        'recommendations': recommendation_report
    }
    
    # Save to file
    os.makedirs(os.path.join(OUTPUTS_PATH, 'predictions'), exist_ok=True)
    output_file = os.path.join(OUTPUTS_PATH, 'predictions', 'latest_prediction.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

def batch_predictions(input_file: str):
    """Make predictions for multiple inputs from CSV file"""
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return
    
    # Load input data
    df = pd.read_csv(input_file)
    
    # Load models
    model = ClimateRiskModel()
    model.load_models(MODELS_PATH)
    
    recommender = CropRecommender()
    
    results = []
    
    for idx, row in df.iterrows():
        input_data = row.to_dict()
        
        # Create DataFrame for prediction
        df_input = pd.DataFrame([input_data])
        
        # Add engineered features
        df_input['temp_stress'] = np.where(df_input['temperature_max'] > 35, 1, 0)
        df_input['cold_stress'] = np.where(df_input['temperature_min'] < 5, 1, 0)
        df_input['drought_risk'] = np.where(df_input['rainfall'] < 200, 1, 0)
        df_input['flood_risk'] = np.where(df_input['rainfall'] > 1000, 1, 0)
        df_input['temp_range'] = df_input['temperature_max'] - df_input['temperature_min']
        df_input['growing_degree_days'] = np.maximum(0, df_input['temperature_avg'] - 10)
        df_input['humidity_stress'] = np.where((df_input['humidity'] < 30) | (df_input['humidity'] > 90), 1, 0)
        
        # Make predictions
        risk_score = model.predict_risk_score(df_input)[0]
        risk_category = model.get_risk_category([risk_score])[0]
        predicted_yield = model.predict_yield(df_input)[0]
        
        # Get top crop recommendation
        climate_data = {
            'temperature_avg': input_data['temperature_avg'],
            'rainfall': input_data['rainfall'],
            'humidity': input_data['humidity'],
            'drought_risk': df_input['drought_risk'].iloc[0],
            'temp_stress': df_input['temp_stress'].iloc[0],
            'cold_stress': df_input['cold_stress'].iloc[0],
            'flood_risk': df_input['flood_risk'].iloc[0],
            'humidity_stress': df_input['humidity_stress'].iloc[0]
        }
        
        top_crops = recommender.recommend_crops(climate_data, top_n=3)
        
        result = {
            'index': idx,
            'climate_risk_score': float(risk_score),
            'risk_category': risk_category,
            'predicted_yield': float(predicted_yield),
            'top_recommended_crop': top_crops[0][0] if top_crops else 'unknown',
            'crop_suitability_score': float(top_crops[0][1]) if top_crops else 0.0
        }
        
        results.append(result)
    
    # Save batch results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUTS_PATH, 'predictions', 'batch_predictions.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"Batch predictions saved to: {output_file}")
    print(f"Processed {len(results)} records")
    
    return results_df

if __name__ == "__main__":
    # Make single prediction with sample data
    make_predictions()