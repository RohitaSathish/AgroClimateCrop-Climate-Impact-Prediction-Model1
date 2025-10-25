"""
Main execution script for AgroClimate project
"""

import os
import sys
import argparse
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.config import *
from src.data_processing.preprocess import main as preprocess_main
from src.models.train_model import train_models
from src.models.predict import make_predictions, batch_predictions
from src.utils.visualization import generate_all_visualizations
from src.utils.data_loader import DataLoader

def run_full_pipeline():
    """Run the complete AgroClimate pipeline"""
    
    print("=" * 60)
    print("AgroClimate - Crop Climate Impact Prediction Model")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    print("\n1. Data Preprocessing...")
    try:
        preprocess_main()
        print("[SUCCESS] Data preprocessing completed successfully")
    except Exception as e:
        print(f"[ERROR] Data preprocessing failed: {e}")
        return
    
    # Step 2: Model Training
    print("\n2. Model Training...")
    try:
        train_models()
        print("[SUCCESS] Model training completed successfully")
    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        return
    
    # Step 3: Generate Predictions
    print("\n3. Generating Predictions...")
    try:
        results = make_predictions()
        print("[SUCCESS] Predictions generated successfully")
    except Exception as e:
        print(f"[ERROR] Prediction generation failed: {e}")
        return
    
    # Step 4: Generate Visualizations
    print("\n4. Generating Visualizations...")
    try:
        # Load processed data for visualization
        processed_data_path = os.path.join(DATA_PROCESSED_PATH, 'processed_data.csv')
        if os.path.exists(processed_data_path):
            df = pd.read_csv(processed_data_path)
            generate_all_visualizations(df)
            print("[SUCCESS] Visualizations generated successfully")
        else:
            print("[WARNING] Processed data not found, skipping visualizations")
    except Exception as e:
        print(f"[ERROR] Visualization generation failed: {e}")
    
    print("\n" + "=" * 60)
    print("AgroClimate Pipeline Execution Completed!")
    print("=" * 60)
    
    # Display summary
    print(f"\nProject Structure:")
    print(f"├── Data: {DATA_PROCESSED_PATH}")
    print(f"├── Models: {MODELS_PATH}")
    print(f"├── Predictions: {os.path.join(OUTPUTS_PATH, 'predictions')}")
    print(f"└── Visualizations: {os.path.join(OUTPUTS_PATH, 'visualizations')}")
    
    if 'results' in locals():
        print(f"\nLatest Prediction Summary:")
        print(f"Climate Risk Score: {results['predictions']['climate_risk_score']:.3f}")
        print(f"Risk Category: {results['predictions']['risk_category']}")
        print(f"Predicted Yield: {results['predictions']['predicted_yield']:.2f} tons/hectare")
        
        top_crop = results['recommendations']['recommended_crops'][0]
        print(f"Top Recommended Crop: {top_crop['crop'].title()} ({top_crop['suitability_percentage']} suitable)")

def run_preprocessing_only():
    """Run only data preprocessing"""
    print("Running data preprocessing...")
    preprocess_main()
    print("Data preprocessing completed!")

def run_training_only():
    """Run only model training"""
    print("Running model training...")
    train_models()
    print("Model training completed!")

def run_prediction_only():
    """Run only prediction"""
    print("Running predictions...")
    results = make_predictions()
    print("Predictions completed!")
    return results

def run_visualization_only():
    """Run only visualization generation"""
    print("Generating visualizations...")
    processed_data_path = os.path.join(DATA_PROCESSED_PATH, 'processed_data.csv')
    if os.path.exists(processed_data_path):
        df = pd.read_csv(processed_data_path)
        generate_all_visualizations(df)
        print("Visualizations completed!")
    else:
        print("Processed data not found. Please run preprocessing first.")

def run_custom_prediction():
    """Run prediction with custom input"""
    print("Enter climate data for prediction:")
    
    try:
        temp_avg = float(input("Average Temperature (°C): "))
        temp_max = float(input("Maximum Temperature (°C): "))
        temp_min = float(input("Minimum Temperature (°C): "))
        rainfall = float(input("Rainfall (mm): "))
        humidity = float(input("Humidity (%): "))
        wind_speed = float(input("Wind Speed (km/h): "))
        solar_radiation = float(input("Solar Radiation (W/m²): "))
        
        print("\nRegion options: 0=North, 1=South, 2=East, 3=West, 4=Central")
        region_encoded = int(input("Region (0-4): "))
        
        print("\nCrop options: 0=wheat, 1=rice, 2=maize, 3=soybeans, 4=barley")
        crop_encoded = int(input("Crop Type (0-4): "))
        
        custom_input = {
            'temperature_avg': temp_avg,
            'temperature_max': temp_max,
            'temperature_min': temp_min,
            'rainfall': rainfall,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'solar_radiation': solar_radiation,
            'region_encoded': region_encoded,
            'crop_type_encoded': crop_encoded
        }
        
        results = make_predictions(custom_input)
        print("\nCustom prediction completed!")
        return results
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='AgroClimate - Crop Climate Impact Prediction Model')
    parser.add_argument('--mode', choices=['full', 'preprocess', 'train', 'predict', 'visualize', 'custom'], 
                       default='full', help='Execution mode')
    parser.add_argument('--batch', type=str, help='Path to CSV file for batch predictions')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(DATA_RAW_PATH, exist_ok=True)
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    
    if args.batch:
        print(f"Running batch predictions on {args.batch}")
        batch_predictions(args.batch)
    elif args.mode == 'full':
        run_full_pipeline()
    elif args.mode == 'preprocess':
        run_preprocessing_only()
    elif args.mode == 'train':
        run_training_only()
    elif args.mode == 'predict':
        run_prediction_only()
    elif args.mode == 'visualize':
        run_visualization_only()
    elif args.mode == 'custom':
        run_custom_prediction()

if __name__ == "__main__":
    main()