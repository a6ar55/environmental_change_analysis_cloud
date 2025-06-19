
import os
import pickle
import numpy as np
import cv2
from google.cloud import storage
import json

class ForestSegmentationPredictor:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from GCS"""
        try:
            client = storage.Client()
            bucket = client.bucket("deforestation-detection-bucket")
            blob = bucket.blob("models/deforestation-detection-model/model.pkl")
            
            # Download model to local file
            blob.download_to_filename("model.pkl")
            
            # Load the model
            with open("model.pkl", "rb") as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def extract_features(self, image):
        """Extract features from image for prediction"""
        height, width = image.shape[:2]
        features = []
        
        for y in range(height):
            for x in range(width):
                # RGB values + normalized position
                pixel_features = list(image[y, x]) + [y/height, x/width]
                features.append(pixel_features)
        
        return np.array(features)
    
    def predict(self, instances):
        """Make predictions on input instances"""
        predictions = []
        
        for instance in instances:
            # Decode image from base64 or process image array
            if "image" in instance:
                image_data = np.array(instance["image"], dtype=np.uint8)
                
                # Resize to consistent size
                image = cv2.resize(image_data, (128, 128))
                
                # Extract features
                features = self.extract_features(image)
                
                # Make predictions
                pixel_predictions = self.model.predict(features)
                
                # Reshape to image dimensions
                segmentation_mask = pixel_predictions.reshape(128, 128)
                
                predictions.append(segmentation_mask.tolist())
        
        return {"predictions": predictions}

# Initialize predictor
predictor = ForestSegmentationPredictor()

def predict(request):
    """Cloud Function entry point"""
    instances = request.get_json()["instances"]
    return predictor.predict(instances)
