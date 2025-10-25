"""
Adaptive Crop Recommendation System
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import *

class AdaptiveCropRecommender:
    def __init__(self):
        self.risk_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.crop_models = {}
        self.scaler = StandardScaler()
        self.crops = ['wheat', 'rice', 'maize', 'soybeans', 'barley']
        
    def calculate_climate_risk(self, climate_data):
        """Calculate climate risk score (0-1)"""
        risk_factors = []
        
        # Temperature stress
        temp_stress = 1 if climate_data['temperature_max'] > 35 else 0
        cold_stress = 1 if climate_data['temperature_min'] < 5 else 0
        
        # Rainfall stress
        drought_risk = 1 if climate_data['rainfall'] < 200 else 0
        flood_risk = 1 if climate_data['rainfall'] > 1000 else 0
        
        # Humidity stress
        humidity_stress = 1 if climate_data['humidity'] < 30 or climate_data['humidity'] > 90 else 0
        
        risk_factors = [temp_stress, cold_stress, drought_risk, flood_risk, humidity_stress]
        return np.mean(risk_factors)
    
    def get_risk_category(self, risk_score):
        """Convert risk score to category"""
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def predict_crop_suitability(self, climate_data):
        """Predict suitability for each crop"""
        recommendations = []
        
        for i, crop in enumerate(self.crops):
            # Calculate base yield potential
            base_yield = self._calculate_base_yield(climate_data, crop)
            
            # Apply climate risk adjustment
            risk_score = self.calculate_climate_risk(climate_data)
            adjusted_yield = base_yield * (1 - risk_score * 0.5)
            
            # Calculate suitability percentage
            suitability = min(100, max(0, (adjusted_yield / 5.0) * 100))
            
            recommendations.append({
                'crop': crop,
                'predicted_yield': round(adjusted_yield, 2),
                'suitability_percentage': f"{suitability:.0f}%",
                'risk_score': round(risk_score, 3),
                'risk_category': self.get_risk_category(risk_score)
            })
        
        # Sort by predicted yield
        recommendations.sort(key=lambda x: x['predicted_yield'], reverse=True)
        return recommendations
    
    def _calculate_base_yield(self, climate_data, crop):
        """Calculate base yield for specific crop"""
        # Crop-specific optimal conditions
        crop_preferences = {
            'wheat': {'temp_opt': 20, 'rain_opt': 400, 'humidity_opt': 60},
            'rice': {'temp_opt': 28, 'rain_opt': 800, 'humidity_opt': 80},
            'maize': {'temp_opt': 25, 'rain_opt': 600, 'humidity_opt': 65},
            'soybeans': {'temp_opt': 24, 'rain_opt': 500, 'humidity_opt': 70},
            'barley': {'temp_opt': 18, 'rain_opt': 350, 'humidity_opt': 55}
        }
        
        prefs = crop_preferences[crop]
        
        # Calculate yield based on how close conditions are to optimal
        temp_factor = 1 - abs(climate_data['temperature_avg'] - prefs['temp_opt']) / 20
        rain_factor = 1 - abs(climate_data['rainfall'] - prefs['rain_opt']) / 500
        humidity_factor = 1 - abs(climate_data['humidity'] - prefs['humidity_opt']) / 30
        
        # Ensure factors are between 0 and 1
        temp_factor = max(0, min(1, temp_factor))
        rain_factor = max(0, min(1, rain_factor))
        humidity_factor = max(0, min(1, humidity_factor))
        
        # Base yield calculation
        base_yield = 3.5 * (temp_factor * 0.4 + rain_factor * 0.4 + humidity_factor * 0.2)
        return max(0.5, base_yield)  # Minimum yield of 0.5
    
    def get_adaptation_strategies(self, climate_data, top_crops):
        """Provide adaptation strategies based on climate conditions"""
        strategies = []
        risk_score = self.calculate_climate_risk(climate_data)
        
        if climate_data['rainfall'] < 200:
            strategies.append("Implement drought-resistant irrigation systems")
            strategies.append("Use drought-tolerant crop varieties")
        
        if climate_data['temperature_max'] > 35:
            strategies.append("Plant heat-resistant crop varieties")
            strategies.append("Use shade nets during extreme heat")
        
        if climate_data['temperature_min'] < 5:
            strategies.append("Use frost protection methods")
            strategies.append("Consider greenhouse cultivation")
        
        if risk_score > 0.6:
            strategies.append("Diversify crop portfolio to reduce risk")
            strategies.append("Implement climate monitoring systems")
        
        return strategies

def main():
    """Test the crop recommender"""
    recommender = AdaptiveCropRecommender()
    
    # Sample climate data
    test_climate = {
        'temperature_avg': 25.0,
        'temperature_max': 32.0,
        'temperature_min': 18.0,
        'rainfall': 450.0,
        'humidity': 65.0,
        'wind_speed': 12.0,
        'solar_radiation': 200.0
    }
    
    print("Climate Risk and Crop Recommendation System")
    print("=" * 50)
    
    # Calculate climate risk
    risk_score = recommender.calculate_climate_risk(test_climate)
    risk_category = recommender.get_risk_category(risk_score)
    
    print(f"Climate Risk Score: {risk_score:.3f}")
    print(f"Risk Category: {risk_category.upper()}")
    print()
    
    # Get crop recommendations
    recommendations = recommender.predict_crop_suitability(test_climate)
    
    print("Top 3 Recommended Crops:")
    print("-" * 30)
    for i, crop in enumerate(recommendations[:3]):
        print(f"{i+1}. {crop['crop'].upper()}")
        print(f"   Predicted Yield: {crop['predicted_yield']} tons/hectare")
        print(f"   Suitability: {crop['suitability_percentage']}")
        print(f"   Risk Level: {crop['risk_category']}")
        print()
    
    # Get adaptation strategies
    strategies = recommender.get_adaptation_strategies(test_climate, recommendations[:3])
    
    print("Recommended Adaptation Strategies:")
    print("-" * 35)
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy}")

if __name__ == "__main__":
    main()