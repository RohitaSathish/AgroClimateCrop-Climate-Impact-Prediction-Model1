"""
Climate Risk Prediction and Adaptive Crop Recommendation Script
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.models.crop_recommender import AdaptiveCropRecommender

def get_user_input():
    """Get climate data from user"""
    print("Enter Climate Data for Prediction:")
    print("-" * 35)
    
    try:
        climate_data = {
            'temperature_avg': float(input("Average Temperature (°C): ")),
            'temperature_max': float(input("Maximum Temperature (°C): ")),
            'temperature_min': float(input("Minimum Temperature (°C): ")),
            'rainfall': float(input("Annual Rainfall (mm): ")),
            'humidity': float(input("Average Humidity (%): ")),
            'wind_speed': float(input("Wind Speed (km/h): ")),
            'solar_radiation': float(input("Solar Radiation (W/m²): "))
        }
        return climate_data
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None

def display_results(recommender, climate_data):
    """Display prediction results"""
    print("\n" + "=" * 60)
    print("CLIMATE RISK ASSESSMENT & CROP RECOMMENDATIONS")
    print("=" * 60)
    
    # Climate Risk Assessment
    risk_score = recommender.calculate_climate_risk(climate_data)
    risk_category = recommender.get_risk_category(risk_score)
    
    print(f"\nCLIMATE RISK ANALYSIS")
    print("-" * 30)
    print(f"Risk Score: {risk_score:.3f} (0 = No Risk, 1 = High Risk)")
    print(f"Risk Level: {risk_category.upper()}")
    
    # Risk level interpretation
    if risk_category == 'low':
        print("[LOW RISK] Favorable conditions for most crops")
    elif risk_category == 'medium':
        print("[MEDIUM RISK] Moderate risk - consider adaptive strategies")
    else:
        print("[HIGH RISK] High risk - implement risk mitigation measures")
    
    # Crop Recommendations
    recommendations = recommender.predict_crop_suitability(climate_data)
    
    print(f"\nTOP CROP RECOMMENDATIONS")
    print("-" * 30)
    
    for i, crop in enumerate(recommendations[:3], 1):
        print(f"{i}. {crop['crop'].upper()}")
        print(f"   • Predicted Yield: {crop['predicted_yield']} tons/hectare")
        print(f"   • Suitability: {crop['suitability_percentage']}")
        print(f"   • Risk Level: {crop['risk_category']}")
        print()
    
    # All crops comparison
    print("ALL CROPS COMPARISON")
    print("-" * 25)
    print(f"{'Crop':<10} {'Yield':<8} {'Suitability':<12} {'Risk'}")
    print("-" * 40)
    
    for crop in recommendations:
        print(f"{crop['crop']:<10} {crop['predicted_yield']:<8} {crop['suitability_percentage']:<12} {crop['risk_category']}")
    
    # Adaptation Strategies
    strategies = recommender.get_adaptation_strategies(climate_data, recommendations[:3])
    
    if strategies:
        print(f"\nADAPTATION STRATEGIES")
        print("-" * 25)
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy}")
    
    print("\n" + "=" * 60)

def run_sample_predictions():
    """Run predictions with sample data"""
    recommender = AdaptiveCropRecommender()
    
    # Sample scenarios
    scenarios = [
        {
            'name': 'Optimal Conditions',
            'data': {'temperature_avg': 24, 'temperature_max': 30, 'temperature_min': 18, 
                    'rainfall': 500, 'humidity': 65, 'wind_speed': 10, 'solar_radiation': 200}
        },
        {
            'name': 'Drought Conditions',
            'data': {'temperature_avg': 32, 'temperature_max': 40, 'temperature_min': 24, 
                    'rainfall': 150, 'humidity': 35, 'wind_speed': 15, 'solar_radiation': 250}
        },
        {
            'name': 'Cold & Wet Conditions',
            'data': {'temperature_avg': 12, 'temperature_max': 18, 'temperature_min': 6, 
                    'rainfall': 800, 'humidity': 85, 'wind_speed': 8, 'solar_radiation': 120}
        }
    ]
    
    for scenario in scenarios:
        print(f"\nSCENARIO: {scenario['name']}")
        display_results(recommender, scenario['data'])
        input("\nPress Enter to continue to next scenario...")

def main():
    """Main function"""
    recommender = AdaptiveCropRecommender()
    
    print("AgroClimate - Climate Risk & Crop Recommendation System")
    print("=" * 60)
    
    while True:
        print("\nChoose an option:")
        print("1. Enter your climate data")
        print("2. Run sample predictions")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            climate_data = get_user_input()
            if climate_data:
                display_results(recommender, climate_data)
        
        elif choice == '2':
            run_sample_predictions()
        
        elif choice == '3':
            print("Thank you for using AgroClimate!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()