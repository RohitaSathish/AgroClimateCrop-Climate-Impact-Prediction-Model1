"""
Flask web application for AgroClimate project
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config.config import *
from src.models.climate_risk_model import ClimateRiskModel
from src.models.crop_recommender import AdaptiveCropRecommender

app = Flask(__name__)

# Load models
model = ClimateRiskModel()
recommender = AdaptiveCropRecommender()
if os.path.exists(MODELS_PATH):
    model.load_models(MODELS_PATH)

@app.route('/')
def index():
    return render_template('terminal_output.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.json
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            'temperature_avg': float(data['temperature_avg']),
            'temperature_max': float(data['temperature_max']),
            'temperature_min': float(data['temperature_min']),
            'rainfall': float(data['rainfall']),
            'humidity': float(data['humidity']),
            'wind_speed': float(data['wind_speed']),
            'solar_radiation': float(data['solar_radiation']),
            'temp_stress': 1 if float(data['temperature_max']) > 35 else 0,
            'cold_stress': 1 if float(data['temperature_min']) < 5 else 0,
            'drought_risk': 1 if float(data['rainfall']) < 200 else 0,
            'flood_risk': 1 if float(data['rainfall']) > 1000 else 0,
            'temp_range': float(data['temperature_max']) - float(data['temperature_min']),
            'growing_degree_days': max(0, float(data['temperature_avg']) - 10),
            'humidity_stress': 1 if float(data['humidity']) < 30 or float(data['humidity']) > 90 else 0,
            'region_encoded': int(data['region']),
            'crop_type_encoded': int(data['crop_type'])
        }])
        
        # Make predictions
        risk_score = model.predict_risk_score(input_data)[0]
        predicted_yield = model.predict_yield(input_data)[0]
        risk_category = model.get_risk_category([risk_score])[0]
        
        # Crop recommendations
        crops = ['wheat', 'rice', 'maize', 'soybeans', 'barley']
        recommendations = []
        for i, crop in enumerate(crops):
            input_data['crop_type_encoded'] = i
            crop_yield = model.predict_yield(input_data)[0]
            suitability = max(0, min(100, (crop_yield / 5.0) * 100))
            recommendations.append({
                'crop': crop,
                'predicted_yield': round(crop_yield, 2),
                'suitability_percentage': f"{suitability:.0f}%"
            })
        
        recommendations.sort(key=lambda x: x['predicted_yield'], reverse=True)
        
        return jsonify({
            'success': True,
            'predictions': {
                'climate_risk_score': round(risk_score, 3),
                'risk_category': risk_category,
                'predicted_yield': round(predicted_yield, 2)
            },
            'recommendations': {
                'recommended_crops': recommendations[:3]
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/terminal')
def terminal():
    return render_template('terminal_output.html')

@app.route('/terminal_predict', methods=['POST'])
def terminal_predict():
    try:
        data = request.json
        
        # Calculate risk and recommendations
        risk_score = recommender.calculate_climate_risk(data)
        risk_category = recommender.get_risk_category(risk_score)
        recommendations = recommender.predict_crop_suitability(data)
        strategies = recommender.get_adaptation_strategies(data, recommendations[:3])
        
        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'recommendations': recommendations,
            'strategies': strategies
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/metrics')
def metrics():
    try:
        metrics_path = os.path.join(MODELS_PATH, 'model_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Metrics not found'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)