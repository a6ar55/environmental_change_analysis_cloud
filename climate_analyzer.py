#!/usr/bin/env python3
"""
Climate Analyzer Service
Loads trained climate models from Google Cloud Storage and provides predictions and analysis.
"""

import os
import pandas as pd
import numpy as np
import joblib
import tempfile
import io
from datetime import datetime, timedelta
from google.cloud import storage
from config import (
    PROJECT_ID, 
    LOCATION, 
    BUCKET_NAME
)

# Initialize Google Cloud (uses Application Default Credentials in Cloud Run)
storage_client = storage.Client()

class ClimateAnalyzer:
    """Climate analyzer using trained models from Google Cloud Storage."""
    
    def __init__(self):
        self.bucket = storage_client.bucket(BUCKET_NAME)
        self.model = None
        self.scaler = None
        self.predictions_data = None
        self.country_data = None
        self.city_data = None
        self.model_loaded = False
        
    def load_model_from_gcs(self):
        """Load trained climate model from Google Cloud Storage."""
        if self.model_loaded:
            return True
            
        try:
            print("Loading climate model from Google Cloud Storage...")
            
            # Load model
            with tempfile.NamedTemporaryFile() as temp_file:
                model_blob = self.bucket.blob("models/climate-model/climate_model.pkl")
                model_blob.download_to_filename(temp_file.name)
                self.model = joblib.load(temp_file.name)
            
            # Load scaler
            with tempfile.NamedTemporaryFile() as temp_file:
                scaler_blob = self.bucket.blob("models/climate-model/scaler.pkl")
                scaler_blob.download_to_filename(temp_file.name)
                self.scaler = joblib.load(temp_file.name)
            
            self.model_loaded = True
            print("Climate model loaded successfully from GCS")
            return True
            
        except Exception as e:
            print(f"Error loading climate model from GCS: {e}")
            return False
    
    def load_data_from_gcs(self):
        """Load climate data from Google Cloud Storage."""
        try:
            print("Loading climate data from Google Cloud Storage...")
            
            # Load predictions dataset
            predictions_blob = self.bucket.blob("climate_data/predictions/climate_predictions.csv")
            predictions_content = predictions_blob.download_as_text()
            self.predictions_data = pd.read_csv(io.StringIO(predictions_content))
            
            # Convert date column
            self.predictions_data['dt'] = pd.to_datetime(self.predictions_data['dt'])
            
            # Load country data
            try:
                country_blob = self.bucket.blob("climate_data/processed/climate_country.csv")
                country_content = country_blob.download_as_text()
                self.country_data = pd.read_csv(io.StringIO(country_content))
                self.country_data['dt'] = pd.to_datetime(self.country_data['dt'])
            except Exception:
                print("Country data not available")
                self.country_data = None
            
            # Load city data
            try:
                city_blob = self.bucket.blob("climate_data/processed/climate_city.csv")
                city_content = city_blob.download_as_text()
                self.city_data = pd.read_csv(io.StringIO(city_content))
                self.city_data['dt'] = pd.to_datetime(self.city_data['dt'])
            except Exception:
                print("City data not available")
                self.city_data = None
            
            print("Climate data loaded successfully from GCS")
            return True
            
        except Exception as e:
            print(f"Error loading climate data from GCS: {e}")
            return False
    
    def get_global_temperature_trends(self):
        """Get global temperature trends analysis."""
        if self.predictions_data is None:
            return None
        
        # Group by year for annual trends
        annual_data = self.predictions_data.groupby('Year').agg({
            'LandAverageTemperature': 'mean',
            'Predicted_Temperature': 'mean',
            'Temperature_Anomaly': 'mean'
        }).reset_index()
        
        # Calculate decade averages
        decade_data = self.predictions_data.groupby('Decade').agg({
            'LandAverageTemperature': 'mean',
            'Temperature_Anomaly': 'mean'
        }).reset_index()
        
        # Calculate recent trends (last 30 years)
        recent_data = annual_data[annual_data['Year'] >= annual_data['Year'].max() - 30]
        
        # Calculate temperature change rate
        if len(recent_data) > 1:
            temp_change_rate = np.polyfit(recent_data['Year'], recent_data['LandAverageTemperature'], 1)[0]
        else:
            temp_change_rate = 0
        
        return {
            'annual_data': annual_data.to_dict('records'),
            'decade_data': decade_data.to_dict('records'),
            'recent_trend': recent_data.to_dict('records'),
            'temperature_change_rate': float(temp_change_rate),
            'current_anomaly': float(annual_data.iloc[-1]['Temperature_Anomaly']) if len(annual_data) > 0 else 0,
            'max_temperature': float(annual_data['LandAverageTemperature'].max()) if len(annual_data) > 0 else 0,
            'min_temperature': float(annual_data['LandAverageTemperature'].min()) if len(annual_data) > 0 else 0
        }
    
    def get_country_analysis(self, country_name=None):
        """Get country-specific temperature analysis."""
        if self.country_data is None:
            return None
        
        if country_name:
            country_data = self.country_data[
                self.country_data['Country'].str.contains(country_name, case=False, na=False)
            ]
        else:
            # Get top 10 countries by recent temperature change
            recent_years = self.country_data[self.country_data['Year'] >= 2000]
            country_trends = []
            
            for country in recent_years['Country'].unique():
                if pd.isna(country):
                    continue
                    
                country_subset = recent_years[recent_years['Country'] == country]
                if len(country_subset) > 5:  # Need enough data points
                    trend = np.polyfit(country_subset['Year'], country_subset['AverageTemperature'], 1)[0]
                    avg_temp = country_subset['AverageTemperature'].mean()
                    country_trends.append({
                        'country': country,
                        'trend': trend,
                        'avg_temp': avg_temp,
                        'data_points': len(country_subset)
                    })
            
            # Sort by trend (most warming)
            country_trends = sorted(country_trends, key=lambda x: x['trend'], reverse=True)[:10]
            
            return {
                'top_warming_countries': country_trends,
                'total_countries': len(self.country_data['Country'].unique())
            }
        
        if len(country_data) == 0:
            return None
        
        # Analyze specific country
        annual_data = country_data.groupby('Year')['AverageTemperature'].mean().reset_index()
        
        # Calculate trend
        if len(annual_data) > 1:
            trend = np.polyfit(annual_data['Year'], annual_data['AverageTemperature'], 1)[0]
        else:
            trend = 0
        
        return {
            'country_name': country_name,
            'annual_data': annual_data.to_dict('records'),
            'temperature_trend': float(trend),
            'avg_temperature': float(country_data['AverageTemperature'].mean()),
            'data_points': len(country_data)
        }
    
    def get_seasonal_analysis(self):
        """Get seasonal temperature analysis."""
        if self.predictions_data is None:
            return None
        
        # Group by month for seasonal patterns
        seasonal_data = self.predictions_data.groupby('Month').agg({
            'LandAverageTemperature': ['mean', 'std'],
            'Temperature_Anomaly': 'mean'
        }).reset_index()
        
        # Flatten column names
        seasonal_data.columns = ['Month', 'Avg_Temperature', 'Temp_Std', 'Avg_Anomaly']
        
        # Add month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal_data['Month_Name'] = seasonal_data['Month'].apply(lambda x: month_names[x-1])
        
        return seasonal_data.to_dict('records')
    
    def predict_future_temperature(self, years_ahead=5):
        """Predict future temperatures using the trained model."""
        if not self.model_loaded or self.predictions_data is None:
            return None
        
        try:
            # Get the last known data point
            last_data = self.predictions_data.iloc[-1]
            last_year = int(last_data['Year'])
            last_month = int(last_data['Month'])
            
            predictions = []
            
            for year_offset in range(years_ahead):
                for month in range(1, 13):
                    future_year = last_year + year_offset + 1
                    
                    # Create features for prediction
                    year_since_start = future_year - self.predictions_data['Year'].min()
                    month_sin = np.sin(2 * np.pi * month / 12)
                    month_cos = np.cos(2 * np.pi * month / 12)
                    
                    # Use last known temperatures as lag features
                    land_temp_lag1 = last_data['LandAverageTemperature']
                    land_temp_lag12 = last_data['LandAverageTemperature']
                    land_temp_ma3 = last_data['LandAverageTemperature']
                    land_temp_ma12 = last_data['LandAverageTemperature']
                    temp_change_1month = 0.0
                    temp_change_1year = 0.0
                    
                    # Create feature vector
                    features = np.array([[
                        year_since_start, month_sin, month_cos,
                        land_temp_lag1, land_temp_lag12,
                        land_temp_ma3, land_temp_ma12,
                        temp_change_1month, temp_change_1year
                    ]])
                    
                    # Scale features
                    features_scaled = self.scaler.transform(features)
                    
                    # Make prediction
                    predicted_temp = self.model.predict(features_scaled)[0]
                    
                    predictions.append({
                        'year': future_year,
                        'month': month,
                        'predicted_temperature': float(predicted_temp),
                        'date': f"{future_year}-{month:02d}-01"
                    })
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting future temperatures: {e}")
            return None
    
    def get_climate_summary(self):
        """Get overall climate analysis summary."""
        try:
            trends = self.get_global_temperature_trends()
            seasonal = self.get_seasonal_analysis()
            future = self.predict_future_temperature(years_ahead=5)
            
            if not trends:
                return None
            
            # Calculate key statistics
            current_year = datetime.now().year
            recent_warming = trends['temperature_change_rate'] * 10  # Per decade
            
            # Determine climate status
            if recent_warming > 0.2:
                climate_status = "Rapid Warming"
                status_color = "danger"
            elif recent_warming > 0.1:
                climate_status = "Moderate Warming"
                status_color = "warning"
            elif recent_warming > 0:
                climate_status = "Slight Warming"
                status_color = "info"
            else:
                climate_status = "Stable/Cooling"
                status_color = "success"
            
            # Get hottest and coldest months
            if seasonal:
                hottest_month = max(seasonal, key=lambda x: x['Avg_Temperature'])
                coldest_month = min(seasonal, key=lambda x: x['Avg_Temperature'])
            else:
                hottest_month = coldest_month = None
            
            return {
                'climate_status': climate_status,
                'status_color': status_color,
                'warming_rate_per_decade': round(recent_warming, 3),
                'current_anomaly': round(trends['current_anomaly'], 2),
                'temperature_range': {
                    'max': round(trends['max_temperature'], 2),
                    'min': round(trends['min_temperature'], 2)
                },
                'hottest_month': hottest_month['Month_Name'] if hottest_month else 'N/A',
                'coldest_month': coldest_month['Month_Name'] if coldest_month else 'N/A',
                'data_years': len(trends['annual_data']),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'future_predictions_available': future is not None and len(future) > 0
            }
            
        except Exception as e:
            print(f"Error generating climate summary: {e}")
            return None

# Global instance
climate_analyzer = ClimateAnalyzer() 