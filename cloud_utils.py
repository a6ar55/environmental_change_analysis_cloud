import os
import uuid
import pickle
import tempfile
import numpy as np
import cv2
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.aiplatform import Model, Endpoint
from config import (
    PROJECT_ID, 
    LOCATION, 
    BUCKET_NAME, 
    UPLOAD_FOLDER, 
    RESULTS_FOLDER,
    FOREST_COLOR,
    DEFORESTED_COLOR,
    MODEL_NAME,
    MODEL_VERSION
)
storage_client = storage.Client()
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Global model cache
_loaded_model = None

def ensure_bucket_exists():
    """Ensure that the GCS bucket exists, create it if it doesn't."""
    try:
        bucket = storage_client.get_bucket(BUCKET_NAME)
        print(f"Using existing bucket: {BUCKET_NAME}")
    except Exception:
        try:
            bucket = storage_client.create_bucket(BUCKET_NAME, location=LOCATION)
            print(f"Created new bucket: {BUCKET_NAME}")
            
            # Set bucket to allow public read access for objects
            try:
                policy = bucket.get_iam_policy()
                policy.bindings.append({
                    "role": "roles/storage.objectViewer",
                    "members": {"allUsers"}
                })
                bucket.set_iam_policy(policy)
                print(f"Set public read access for bucket: {BUCKET_NAME}")
            except Exception as e:
                print(f"Warning: Could not set public access for bucket: {e}")
                
        except Exception as e:
            raise Exception(f"Failed to access or create bucket: {e}. Please ensure the service account has Storage Admin permissions.")
    return bucket

def ensure_local_dirs():
    """Ensure that local directories for uploads and results exist."""
    upload_dir = os.path.join('static', UPLOAD_FOLDER)
    results_dir = os.path.join('static', RESULTS_FOLDER)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return upload_dir, results_dir

def upload_to_gcs(file_path, destination_blob_name=None):
    """
    Upload a file to Google Cloud Storage.
    """
    if destination_blob_name is None:
        # Generate a unique filename
        file_extension = os.path.splitext(file_path)[1]
        destination_blob_name = f"{UPLOAD_FOLDER}/{str(uuid.uuid4())}{file_extension}"
    
    # Upload to GCS
    bucket = ensure_bucket_exists()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    
    # In Cloud Run, use public URL instead of signed URL to avoid private key requirement
    try:
        # Try to generate signed URL first (works if proper service account is configured)
        url = blob.generate_signed_url(
            version="v4",
            expiration=3600,  # URL expires in 1 hour
            method="GET"
        )
    except Exception:
        # Fallback to public URL (requires bucket to be publicly readable)
        # Make blob publicly readable
        blob.make_public()
        url = blob.public_url
    
    return destination_blob_name, url

def download_from_gcs(blob_name, local_path):
    """
    Download a file from Google Cloud Storage.
    """
    bucket = ensure_bucket_exists()
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path

def get_signed_url(blob_name):
    """
    Generate a signed URL for a blob.
    """
    bucket = ensure_bucket_exists()
    blob = bucket.blob(blob_name)
    
    try:
        # Try to generate signed URL first (works if proper service account is configured)
        url = blob.generate_signed_url(
            version="v4",
            expiration=3600,  # URL expires in 1 hour
            method="GET"
        )
    except Exception:
        # Fallback to public URL (requires bucket to be publicly readable)
        # Make blob publicly readable
        blob.make_public()
        url = blob.public_url
    
    return url

def find_deployed_model():
    """
    Find a deployed model endpoint for deforestation detection.
    """
    try:
        # List all endpoints
        endpoints = Endpoint.list()
        
        for endpoint in endpoints:
            if MODEL_NAME.lower() in endpoint.display_name.lower():
                print(f"Found deployed model endpoint: {endpoint.display_name}")
                return endpoint
        
        # If no endpoint found, look for models
        models = Model.list()
        for model in models:
            if MODEL_NAME.lower() in model.display_name.lower():
                print(f"Found model (not deployed): {model.display_name}")
                return model
                
        return None
    except Exception as e:
        print(f"Error finding deployed model: {e}")
        return None

def load_model_from_gcs():
    """
    Load the trained model from Google Cloud Storage.
    """
    global _loaded_model
    
    if _loaded_model is not None:
        return _loaded_model
    
    try:
        print("Loading model from Google Cloud Storage...")
        bucket = ensure_bucket_exists()
        
        # Download model from GCS
        model_blob_name = f"models/{MODEL_NAME}/model.pkl"
        blob = bucket.blob(model_blob_name)
        
        # Create a temporary file to download the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            blob.download_to_filename(temp_file.name)
            
            # Load the model
            with open(temp_file.name, 'rb') as f:
                _loaded_model = pickle.load(f)
            
            # Clean up temporary file
            os.unlink(temp_file.name)
        
        print("Model loaded successfully from GCS")
        return _loaded_model
        
    except Exception as e:
        raise Exception(f"Failed to load model from GCS: {e}. Make sure the model has been trained and uploaded.")

def extract_features_for_prediction(image):
    """
    Extract features from image for model prediction.
    This should match the feature extraction used during training.
    """
    height, width = image.shape[:2]
    features = []
    
    for y in range(height):
        for x in range(width):
            # RGB values + normalized position (same as training)
            pixel_features = list(image[y, x]) + [y/height, x/width]
            features.append(pixel_features)
    
    return np.array(features)

def run_semantic_segmentation(image_path):
    """
    Run semantic segmentation using the trained Random Forest model from GCS.
    """
    print(f"Running forest segmentation on {image_path}")
    
    try:
        # Load the model
        model = load_model_from_gcs()
        
        # Run inference
        result = run_model_inference(image_path, model)
        return result
        
    except Exception as e:
        raise Exception(f"Model inference failed: {e}")

def run_model_inference(image_path, model):
    """
    Run inference using the trained Random Forest model.
    
    Args:
        image_path: Path to the image file
        model: Trained scikit-learn model
    
    Returns:
        Dictionary with segmentation results
    """
    print(f"Running inference on {image_path}")
    
    # Read and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Store original dimensions
    original_height, original_width = image.shape[:2]
    
    # Resize image to consistent size for feature extraction (same as training)
    processing_size = (128, 128)
    resized_image = cv2.resize(image, processing_size)
    
    # Extract features
    print("Extracting features...")
    features = extract_features_for_prediction(resized_image)
    
    # Make predictions
    print("Making predictions...")
    pixel_predictions = model.predict(features)
    
    # Reshape predictions to image dimensions
    segmentation_mask = pixel_predictions.reshape(processing_size)
    
    # Resize mask back to original image size
    segmentation_mask = cv2.resize(
        segmentation_mask.astype(np.uint8), 
        (original_width, original_height), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create colored segmentation mask
    colored_mask = np.zeros_like(image)
    colored_mask[segmentation_mask == 0] = DEFORESTED_COLOR  # Red for deforested
    colored_mask[segmentation_mask == 1] = FOREST_COLOR      # Green for forest
    
    # Save the segmentation mask
    result_filename = f"segmentation_{str(uuid.uuid4())}.jpeg"
    result_path = os.path.join(os.path.dirname(image_path), result_filename)
    cv2.imwrite(result_path, colored_mask)
    
    # Create an overlay image
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    overlay_filename = f"overlay_{str(uuid.uuid4())}.jpeg"
    overlay_path = os.path.join(os.path.dirname(image_path), overlay_filename)
    cv2.imwrite(overlay_path, overlay)
    
    # Calculate statistics
    total_pixels = segmentation_mask.size
    forest_pixels = np.sum(segmentation_mask == 1)
    deforested_pixels = np.sum(segmentation_mask == 0)
    
    forest_percentage = round((forest_pixels / total_pixels) * 100, 2)
    deforested_percentage = round((deforested_pixels / total_pixels) * 100, 2)
    
    print(f"Segmentation complete: Forest: {forest_percentage}%, Deforested: {deforested_percentage}%")
    
    return {
        "segmentation_path": result_path,
        "overlay_path": overlay_path,
        "forest_percentage": forest_percentage,
        "deforested_percentage": deforested_percentage
    }

def encode_image(image_path):
    """
    Encode an image as base64 for API requests.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Base64-encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_vertex_ai_model():
    """
    Create a Vertex AI model for semantic segmentation.
    This is a placeholder function for the actual model creation process.
    """
    # This would be implemented based on your specific model architecture and training data
    pass 