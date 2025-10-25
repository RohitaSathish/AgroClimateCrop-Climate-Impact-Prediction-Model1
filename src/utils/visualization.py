"""
Visualization utilities for AgroClimate project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.config import OUTPUTS_PATH

class AgroClimateVisualizer:
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create output directory
        self.viz_path = os.path.join(OUTPUTS_PATH, 'visualizations')
        os.makedirs(self.viz_path, exist_ok=True)
    
    def plot_climate_trends(self, df: pd.DataFrame, save_plot: bool = True):
        """Plot climate trends over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Climate Trends Analysis', fontsize=16, fontweight='bold')
        
        # Temperature trends
        axes[0, 0].plot(df.groupby('year')['temperature_avg'].mean(), marker='o', linewidth=2)
        axes[0, 0].set_title('Average Temperature Trend')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rainfall trends
        axes[0, 1].plot(df.groupby('year')['rainfall'].mean(), marker='s', color='blue', linewidth=2)
        axes[0, 1].set_title('Average Rainfall Trend')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Rainfall (mm)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Humidity trends
        axes[1, 0].plot(df.groupby('year')['humidity'].mean(), marker='^', color='green', linewidth=2)
        axes[1, 0].set_title('Average Humidity Trend')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Humidity (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Climate risk trends
        if 'climate_risk_score' in df.columns:
            axes[1, 1].plot(df.groupby('year')['climate_risk_score'].mean(), marker='d', color='red', linewidth=2)
            axes[1, 1].set_title('Climate Risk Score Trend')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Risk Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.viz_path, 'climate_trends.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_crop_yield_analysis(self, df: pd.DataFrame, save_plot: bool = True):
        """Plot crop yield analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Crop Yield Analysis', fontsize=16, fontweight='bold')
        
        # Yield by crop type
        if 'crop_type' in df.columns:
            crop_yields = df.groupby('crop_type')['yield_tons_per_hectare'].mean().sort_values(ascending=False)
            axes[0, 0].bar(crop_yields.index, crop_yields.values, color='skyblue', edgecolor='navy')
            axes[0, 0].set_title('Average Yield by Crop Type')
            axes[0, 0].set_xlabel('Crop Type')
            axes[0, 0].set_ylabel('Yield (tons/hectare)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Yield distribution
        axes[0, 1].hist(df['yield_tons_per_hectare'], bins=30, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        axes[0, 1].set_title('Yield Distribution')
        axes[0, 1].set_xlabel('Yield (tons/hectare)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Yield vs Temperature
        axes[1, 0].scatter(df['temperature_avg'], df['yield_tons_per_hectare'], alpha=0.6, color='orange')
        axes[1, 0].set_title('Yield vs Average Temperature')
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Yield (tons/hectare)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Yield vs Rainfall
        axes[1, 1].scatter(df['rainfall'], df['yield_tons_per_hectare'], alpha=0.6, color='purple')
        axes[1, 1].set_title('Yield vs Rainfall')
        axes[1, 1].set_xlabel('Rainfall (mm)')
        axes[1, 1].set_ylabel('Yield (tons/hectare)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.viz_path, 'crop_yield_analysis.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_risk_assessment(self, df: pd.DataFrame, save_plot: bool = True):
        """Plot climate risk assessment"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Climate Risk Assessment', fontsize=16, fontweight='bold')
        
        # Risk score distribution
        if 'climate_risk_score' in df.columns:
            axes[0, 0].hist(df['climate_risk_score'], bins=20, color='red', alpha=0.7, edgecolor='darkred')
            axes[0, 0].set_title('Climate Risk Score Distribution')
            axes[0, 0].set_xlabel('Risk Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Risk factors
        risk_factors = ['temp_stress', 'cold_stress', 'drought_risk', 'flood_risk', 'humidity_stress']
        available_factors = [col for col in risk_factors if col in df.columns]
        
        if available_factors:
            risk_counts = df[available_factors].sum()
            axes[0, 1].bar(range(len(risk_counts)), risk_counts.values, color='coral')
            axes[0, 1].set_title('Climate Risk Factors Frequency')
            axes[0, 1].set_xlabel('Risk Factors')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_xticks(range(len(risk_counts)))
            axes[0, 1].set_xticklabels([f.replace('_', ' ').title() for f in risk_counts.index], rotation=45)
        
        # Risk by region
        if 'region' in df.columns and 'climate_risk_score' in df.columns:
            region_risk = df.groupby('region')['climate_risk_score'].mean()
            axes[1, 0].bar(region_risk.index, region_risk.values, color='orange')
            axes[1, 0].set_title('Average Risk Score by Region')
            axes[1, 0].set_xlabel('Region')
            axes[1, 0].set_ylabel('Average Risk Score')
        
        # Correlation heatmap
        if len(available_factors) > 1:
            corr_matrix = df[available_factors + ['climate_risk_score']].corr() if 'climate_risk_score' in df.columns else df[available_factors].corr()
            im = axes[1, 1].imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto')
            axes[1, 1].set_title('Risk Factors Correlation')
            axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
            axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
            axes[1, 1].set_xticklabels([col.replace('_', ' ').title() for col in corr_matrix.columns], rotation=45)
            axes[1, 1].set_yticklabels([col.replace('_', ' ').title() for col in corr_matrix.columns])
            
            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                   ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.viz_path, 'risk_assessment.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame, save_plot: bool = True):
        """Create interactive dashboard using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature vs Yield', 'Rainfall vs Yield', 
                          'Risk Score Distribution', 'Yield by Crop Type'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Temperature vs Yield scatter
        fig.add_trace(
            go.Scatter(x=df['temperature_avg'], y=df['yield_tons_per_hectare'],
                      mode='markers', name='Temp vs Yield',
                      marker=dict(color='red', opacity=0.6)),
            row=1, col=1
        )
        
        # Rainfall vs Yield scatter
        fig.add_trace(
            go.Scatter(x=df['rainfall'], y=df['yield_tons_per_hectare'],
                      mode='markers', name='Rainfall vs Yield',
                      marker=dict(color='blue', opacity=0.6)),
            row=1, col=2
        )
        
        # Risk score histogram
        if 'climate_risk_score' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['climate_risk_score'], name='Risk Distribution',
                           marker=dict(color='orange')),
                row=2, col=1
            )
        
        # Yield by crop type
        if 'crop_type' in df.columns:
            crop_yields = df.groupby('crop_type')['yield_tons_per_hectare'].mean()
            fig.add_trace(
                go.Bar(x=crop_yields.index, y=crop_yields.values,
                      name='Avg Yield by Crop', marker=dict(color='green')),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="AgroClimate Interactive Dashboard",
            showlegend=False,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Yield (tons/ha)", row=1, col=1)
        fig.update_xaxes(title_text="Rainfall (mm)", row=1, col=2)
        fig.update_yaxes(title_text="Yield (tons/ha)", row=1, col=2)
        fig.update_xaxes(title_text="Risk Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Crop Type", row=2, col=2)
        fig.update_yaxes(title_text="Avg Yield (tons/ha)", row=2, col=2)
        
        if save_plot:
            fig.write_html(os.path.join(self.viz_path, 'interactive_dashboard.html'))
        
        fig.show()
    
    def plot_feature_importance(self, feature_importance: dict, model_name: str, save_plot: bool = True):
        """Plot feature importance"""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importance, color='steelblue')
        plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(os.path.join(self.viz_path, filename), dpi=300, bbox_inches='tight')
        
        plt.show()

def generate_all_visualizations(df: pd.DataFrame):
    """Generate all visualizations for the dataset"""
    visualizer = AgroClimateVisualizer()
    
    print("Generating climate trends visualization...")
    visualizer.plot_climate_trends(df)
    
    print("Generating crop yield analysis...")
    visualizer.plot_crop_yield_analysis(df)
    
    print("Generating risk assessment visualization...")
    visualizer.plot_risk_assessment(df)
    
    print("Generating interactive dashboard...")
    visualizer.create_interactive_dashboard(df)
    
    print(f"All visualizations saved to: {visualizer.viz_path}")

if __name__ == "__main__":
    # Example usage with sample data
    np.random.seed(42)
    sample_data = {
        'year': np.random.randint(2000, 2024, 100),
        'temperature_avg': np.random.normal(25, 5, 100),
        'rainfall': np.random.exponential(500, 100),
        'humidity': np.random.normal(65, 15, 100),
        'yield_tons_per_hectare': np.random.normal(3.5, 1, 100),
        'climate_risk_score': np.random.beta(2, 5, 100),
        'crop_type': np.random.choice(['wheat', 'rice', 'maize'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    generate_all_visualizations(df)