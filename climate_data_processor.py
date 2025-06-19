#!/usr/bin/env python3
"""
Climate Data Processor for Environmental Change Analysis
Downloads climate data from Kaggle and uploads to Google Cloud Storage for Vertex AI processing.
"""

import os
import kagglehub
import pandas as pd
import numpy as np
from google.cloud import storage
from google.cloud import aiplatform
import tempfile
import shutil
from config import (
    PROJECT_ID, 
    LOCATION, 
    BUCKET_NAME, 
    SERVICE_ACCOUNT_PATH
)

# Initialize Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
storage_client = storage.Client()
aiplatform.init(project=PROJECT_ID, location=LOCATION)

def ensure_bucket_exists():
    """Ensure that the GCS bucket exists."""
    try:
        bucket = storage_client.get_bucket(BUCKET_NAME)
        print(f"Using existing bucket: {BUCKET_NAME}")
    except Exception:
        try:
            bucket = storage_client.create_bucket(BUCKET_NAME, location=LOCATION)
            print(f"Created new bucket: {BUCKET_NAME}")
        except Exception as e:
            raise Exception(f"Failed to access or create bucket: {e}")
    return bucket

def download_climate_data():
    """Download climate change data from Kaggle."""
    print("Downloading climate change dataset from Kaggle...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("berkeleyearth/climate-change-earth-surface-temperature-data")
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def process_climate_data(data_path):
    """Process and clean climate data for analysis."""
    print("Processing climate data...")
    
    processed_data = {}
    
    # Process GlobalLandTemperaturesByCountry.csv
    try:
        country_file = os.path.join(data_path, "GlobalLandTemperaturesByCountry.csv")
        if os.path.exists(country_file):
            df_country = pd.read_csv(country_file)
            
            # Clean data
            df_country = df_country.dropna(subset=['AverageTemperature'])
            df_country['dt'] = pd.to_datetime(df_country['dt'])
            df_country['Year'] = df_country['dt'].dt.year
            df_country['Month'] = df_country['dt'].dt.month
            
            # Filter data from 1900 onwards for better quality
            df_country = df_country[df_country['Year'] >= 1900]
            
            processed_data['country'] = df_country
            print(f"Processed country data: {len(df_country)} records")
        
        # Process GlobalLandTemperaturesByCity.csv
        city_file = os.path.join(data_path, "GlobalLandTemperaturesByCity.csv")
        if os.path.exists(city_file):
            # Sample the data to make it manageable
            df_city = pd.read_csv(city_file, nrows=100000)  # Sample first 100k rows
            
            # Clean data
            df_city = df_city.dropna(subset=['AverageTemperature'])
            df_city['dt'] = pd.to_datetime(df_city['dt'])
            df_city['Year'] = df_city['dt'].dt.year
            df_city['Month'] = df_city['dt'].dt.month
            
            # Filter data from 1900 onwards
            df_city = df_city[df_city['Year'] >= 1900]
            
            processed_data['city'] = df_city
            print(f"Processed city data: {len(df_city)} records")
            
        # Process GlobalTemperatures.csv
        global_file = os.path.join(data_path, "GlobalTemperatures.csv")
        if os.path.exists(global_file):
            df_global = pd.read_csv(global_file)
            
            # Clean data
            df_global = df_global.dropna(subset=['LandAverageTemperature'])
            df_global['dt'] = pd.to_datetime(df_global['dt'])
            df_global['Year'] = df_global['dt'].dt.year
            df_global['Month'] = df_global['dt'].dt.month
            
            # Filter data from 1900 onwards
            df_global = df_global[df_global['Year'] >= 1900]
            
            processed_data['global'] = df_global
            print(f"Processed global data: {len(df_global)} records")
        
        return processed_data
        
    except Exception as e:
        print(f"Error processing climate data: {e}")
        raise

def create_training_features(processed_data):
    """Create features for machine learning model."""
    print("Creating training features...")
    
    # Use global temperature data for trend analysis
    df_global = processed_data.get('global')
    if df_global is None:
        raise ValueError("Global temperature data not available")
    
    # Create features for time series prediction
    df_features = df_global.copy()
    
    # Add time-based features
    df_features['YearSinceStart'] = df_features['Year'] - df_features['Year'].min()
    df_features['MonthSin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
    df_features['MonthCos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
    
    # Add lag features (previous temperatures)
    df_features = df_features.sort_values(['Year', 'Month'])
    df_features['LandTemp_Lag1'] = df_features['LandAverageTemperature'].shift(1)
    df_features['LandTemp_Lag12'] = df_features['LandAverageTemperature'].shift(12)  # Same month previous year
    
    # Add moving averages
    df_features['LandTemp_MA3'] = df_features['LandAverageTemperature'].rolling(window=3).mean()
    df_features['LandTemp_MA12'] = df_features['LandAverageTemperature'].rolling(window=12).mean()
    
    # Add temperature change features
    df_features['TempChange_1Month'] = df_features['LandAverageTemperature'] - df_features['LandTemp_Lag1']
    df_features['TempChange_1Year'] = df_features['LandAverageTemperature'] - df_features['LandTemp_Lag12']
    
    # Remove rows with NaN values created by lag features
    df_features = df_features.dropna()
    
    # Select features for training
    feature_columns = [
        'YearSinceStart', 'MonthSin', 'MonthCos',
        'LandTemp_Lag1', 'LandTemp_Lag12',
        'LandTemp_MA3', 'LandTemp_MA12',
        'TempChange_1Month', 'TempChange_1Year'
    ]
    
    target_column = 'LandAverageTemperature'
    
    X = df_features[feature_columns]
    y = df_features[target_column]
    
    print(f"Created training features: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, df_features

def upload_data_to_gcs(processed_data, features_data, bucket):
    """Upload processed data to Google Cloud Storage."""
    print("Uploading data to Google Cloud Storage...")
    
    # Create temporary directory for processed files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save processed datasets
        for data_type, df in processed_data.items():
            csv_path = os.path.join(temp_dir, f"climate_{data_type}.csv")
            df.to_csv(csv_path, index=False)
            
            # Upload to GCS
            blob_name = f"climate_data/processed/climate_{data_type}.csv"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(csv_path)
            print(f"Uploaded {data_type} data to gs://{BUCKET_NAME}/{blob_name}")
        
        # Save features data
        X, y, df_features = features_data
        
        # Save training features
        features_path = os.path.join(temp_dir, "training_features.csv")
        X.to_csv(features_path, index=False)
        blob = bucket.blob("climate_data/training/features.csv")
        blob.upload_from_filename(features_path)
        
        # Save target values
        target_path = os.path.join(temp_dir, "training_target.csv")
        y.to_csv(target_path, index=False)
        blob = bucket.blob("climate_data/training/target.csv")
        blob.upload_from_filename(target_path)
        
        # Save full features dataset
        full_features_path = os.path.join(temp_dir, "full_features.csv")
        df_features.to_csv(full_features_path, index=False)
        blob = bucket.blob("climate_data/training/full_features.csv")
        blob.upload_from_filename(full_features_path)
        
        print("All data uploaded to Google Cloud Storage successfully")

def main():
    """Main function to process climate data."""
    print("=" * 60)
    print("CLIMATE DATA PROCESSOR")
    print("=" * 60)
    
    try:
        # Ensure bucket exists
        bucket = ensure_bucket_exists()
        
        # Download climate data
        data_path = download_climate_data()
        
        # Process the data
        processed_data = process_climate_data(data_path)
        
        # Create training features
        features_data = create_training_features(processed_data)
        
        # Upload to Google Cloud Storage
        upload_data_to_gcs(processed_data, features_data, bucket)
        
        print("=" * 60)
        print("SUCCESS: Climate data processing completed!")
        print("=" * 60)
        print("Data is now ready for Vertex AI training.")
        
    except Exception as e:
        print(f"Error in climate data processing: {e}")
        raise

if __name__ == "__main__":
    main() 