#!/usr/bin/env python3
"""
Climate Model Training on Vertex AI
Trains climate prediction models entirely on Google Cloud using Vertex AI.
"""

import os
import pandas as pd
import numpy as np
import joblib
import tempfile
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from google.cloud import storage
from google.cloud import aiplatform
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

class ClimateModelTrainer:
    """Climate model trainer for Vertex AI."""
    
    def __init__(self):
        self.bucket = storage_client.bucket(BUCKET_NAME)
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        
    def load_data_from_gcs(self):
        """Load training data from Google Cloud Storage."""
        print("Loading training data from Google Cloud Storage...")
        
        try:
            # Download features
            features_blob = self.bucket.blob("climate_data/training/features.csv")
            features_content = features_blob.download_as_text()
            X = pd.read_csv(pd.io.common.StringIO(features_content))
            
            # Download target
            target_blob = self.bucket.blob("climate_data/training/target.csv")
            target_content = target_blob.download_as_text()
            y = pd.read_csv(pd.io.common.StringIO(target_content))
            y = y.iloc[:, 0]  # Extract series from DataFrame
            
            # Download full features for analysis
            full_blob = self.bucket.blob("climate_data/training/full_features.csv")
            full_content = full_blob.download_as_text()
            df_full = pd.read_csv(pd.io.common.StringIO(full_content))
            
            print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, df_full
            
        except Exception as e:
            print(f"Error loading data from GCS: {e}")
            raise
    
    def prepare_data(self, X, y):
        """Prepare data for training."""
        print("Preparing data for training...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Keep time series order
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple climate prediction models."""
        print("Training climate prediction models...")
        
        # Define models to train
        models_config = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
        # Train each model
        for model_name, model in models_config.items():
            print(f"\nTraining {model_name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Store model and metrics
            self.models[model_name] = model
            self.metrics[model_name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"{model_name} - Test R²: {test_r2:.4f}, Test RMSE: {np.sqrt(test_mse):.4f}")
    
    def create_predictions_dataset(self, df_full, X, y):
        """Create predictions dataset for dashboard."""
        print("Creating predictions dataset...")
        
        # Use the best model (highest test R²)
        best_model_name = max(self.metrics.keys(), 
                            key=lambda k: self.metrics[k]['test_r2'])
        best_model = self.models[best_model_name]
        
        print(f"Using best model: {best_model_name} (R² = {self.metrics[best_model_name]['test_r2']:.4f})")
        
        # Scale all features
        X_scaled = self.scalers['main'].transform(X)
        
        # Make predictions
        predictions = best_model.predict(X_scaled)
        
        # Create predictions dataset
        df_predictions = df_full.copy()
        df_predictions = df_predictions.iloc[:len(predictions)]  # Match predictions length
        df_predictions['Predicted_Temperature'] = predictions
        df_predictions['Temperature_Difference'] = (
            df_predictions['Predicted_Temperature'] - df_predictions['LandAverageTemperature']
        )
        
        # Add trend analysis
        df_predictions['Decade'] = (df_predictions['Year'] // 10) * 10
        
        # Calculate anomalies (difference from long-term average)
        long_term_avg = df_predictions['LandAverageTemperature'].mean()
        df_predictions['Temperature_Anomaly'] = df_predictions['LandAverageTemperature'] - long_term_avg
        
        return df_predictions, best_model_name
    
    def upload_models_to_gcs(self, df_predictions, best_model_name):
        """Upload trained models and predictions to Google Cloud Storage."""
        print("Uploading models and predictions to Google Cloud Storage...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save best model
            model_path = os.path.join(temp_dir, "climate_model.pkl")
            joblib.dump(self.models[best_model_name], model_path)
            
            blob = self.bucket.blob("models/climate-model/climate_model.pkl")
            blob.upload_from_filename(model_path)
            print(f"Uploaded best model ({best_model_name}) to GCS")
            
            # Save scaler
            scaler_path = os.path.join(temp_dir, "climate_scaler.pkl")
            joblib.dump(self.scalers['main'], scaler_path)
            
            blob = self.bucket.blob("models/climate-model/scaler.pkl")
            blob.upload_from_filename(scaler_path)
            print("Uploaded scaler to GCS")
            
            # Save all models for comparison
            for model_name, model in self.models.items():
                model_path = os.path.join(temp_dir, f"{model_name}_model.pkl")
                joblib.dump(model, model_path)
                
                blob = self.bucket.blob(f"models/climate-model/{model_name}_model.pkl")
                blob.upload_from_filename(model_path)
            
            # Save metrics
            metrics_path = os.path.join(temp_dir, "model_metrics.csv")
            metrics_df = pd.DataFrame(self.metrics).T
            metrics_df.to_csv(metrics_path)
            
            blob = self.bucket.blob("models/climate-model/metrics.csv")
            blob.upload_from_filename(metrics_path)
            print("Uploaded model metrics to GCS")
            
            # Save predictions dataset
            predictions_path = os.path.join(temp_dir, "climate_predictions.csv")
            df_predictions.to_csv(predictions_path, index=False)
            
            blob = self.bucket.blob("climate_data/predictions/climate_predictions.csv")
            blob.upload_from_filename(predictions_path)
            print("Uploaded predictions dataset to GCS")
            
            # Save model info
            model_info = {
                'best_model': best_model_name,
                'training_date': datetime.now().isoformat(),
                'test_r2': self.metrics[best_model_name]['test_r2'],
                'test_rmse': np.sqrt(self.metrics[best_model_name]['test_mse']),
                'features_count': len(df_predictions.columns),
                'samples_count': len(df_predictions)
            }
            
            info_path = os.path.join(temp_dir, "model_info.txt")
            with open(info_path, 'w') as f:
                for key, value in model_info.items():
                    f.write(f"{key}: {value}\n")
            
            blob = self.bucket.blob("models/climate-model/model_info.txt")
            blob.upload_from_filename(info_path)
            print("Uploaded model info to GCS")
    
    def print_training_summary(self):
        """Print training summary."""
        print("\n" + "=" * 60)
        print("CLIMATE MODEL TRAINING SUMMARY")
        print("=" * 60)
        
        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Test R²: {metrics['test_r2']:.4f}")
            print(f"  Test RMSE: {np.sqrt(metrics['test_mse']):.4f}")
            print(f"  Test MAE: {metrics['test_mae']:.4f}")
            print(f"  CV R² (mean ± std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        
        # Find best model
        best_model = max(self.metrics.keys(), 
                        key=lambda k: self.metrics[k]['test_r2'])
        
        print(f"\nBEST MODEL: {best_model}")
        print(f"Test R²: {self.metrics[best_model]['test_r2']:.4f}")
        print(f"Test RMSE: {np.sqrt(self.metrics[best_model]['test_mse']):.4f}")
        
        print("\n" + "=" * 60)
        print("Models uploaded to Google Cloud Storage successfully!")
        print("Ready for climate dashboard deployment.")
        print("=" * 60)

def main():
    """Main training function."""
    print("=" * 60)
    print("CLIMATE MODEL TRAINING ON VERTEX AI")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = ClimateModelTrainer()
        
        # Load data from GCS
        X, y, df_full = trainer.load_data_from_gcs()
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
        
        # Train models
        trainer.train_models(X_train, y_train, X_test, y_test)
        
        # Create predictions dataset
        df_predictions, best_model_name = trainer.create_predictions_dataset(df_full, X, y)
        
        # Upload everything to GCS
        trainer.upload_models_to_gcs(df_predictions, best_model_name)
        
        # Print summary
        trainer.print_training_summary()
        
    except Exception as e:
        print(f"Error in climate model training: {e}")
        raise

if __name__ == "__main__":
    main() 