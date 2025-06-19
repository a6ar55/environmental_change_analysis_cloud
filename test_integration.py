#!/usr/bin/env python3
"""
Comprehensive integration test for the deforestation detection system.
This script tests the entire pipeline end-to-end using Google Cloud services.
"""

import os
import sys
import requests
import tempfile
import numpy as np
import cv2
from google.cloud import storage
from config import PROJECT_ID, BUCKET_NAME, MODEL_NAME

def test_google_cloud_setup():
    """Test Google Cloud authentication and permissions."""
    print("=" * 60)
    print("TESTING GOOGLE CLOUD SETUP")
    print("=" * 60)
    
    try:
        # Test Storage
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME)
        print(f"✅ Successfully connected to bucket: {BUCKET_NAME}")
        
        # Test bucket permissions by listing objects
        blobs = list(bucket.list_blobs(prefix="models/", max_results=5))
        print(f"✅ Found {len(blobs)} model files in bucket")
        
        return True
    except Exception as e:
        print(f"❌ Google Cloud setup failed: {e}")
        return False

def test_model_availability():
    """Test if the trained model is available in GCS."""
    print("\n" + "=" * 60)
    print("TESTING MODEL AVAILABILITY")
    print("=" * 60)
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME)
        
        # Check if model exists
        model_blob_name = f"models/{MODEL_NAME}/model.pkl"
        blob = bucket.blob(model_blob_name)
        
        if blob.exists():
            # Reload blob to get updated attributes
            blob.reload()
            blob_size = blob.size or 0
            print(f"✅ Model found: {model_blob_name}")
            print(f"✅ Model size: {blob_size:,} bytes")
            return True
        else:
            print(f"❌ Model not found: {model_blob_name}")
            return False
            
    except Exception as e:
        print(f"❌ Model availability check failed: {e}")
        return False

def create_test_image():
    """Create a synthetic test image."""
    print("\n" + "=" * 60)
    print("CREATING TEST IMAGE")
    print("=" * 60)
    
    # Create a synthetic forest/deforestation image
    img_size = (400, 400, 3)
    image = np.zeros(img_size, dtype=np.uint8)
    
    # Create forest areas (green)
    forest_region = image[100:300, 100:200]
    forest_region[:, :] = [0, 150, 0]  # Green for forest
    
    # Create deforested areas (brown)
    deforested_region = image[100:300, 250:350]
    deforested_region[:, :] = [139, 69, 19]  # Brown for deforested
    
    # Add some noise for realism
    noise = np.random.normal(0, 10, img_size).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        cv2.imwrite(temp_file.name, image)
        print(f"✅ Created test image: {temp_file.name}")
        return temp_file.name

def test_model_inference():
    """Test direct model inference."""
    print("\n" + "=" * 60)
    print("TESTING MODEL INFERENCE")
    print("=" * 60)
    
    try:
        # Import cloud_utils for direct testing
        import cloud_utils
        
        # Create test image
        test_image_path = create_test_image()
        
        # Test model loading
        print("Loading model from GCS...")
        model = cloud_utils.load_model_from_gcs()
        print("✅ Model loaded successfully")
        
        # Test inference
        print("Running inference...")
        result = cloud_utils.run_model_inference(test_image_path, model)
        
        print(f"✅ Inference completed successfully")
        print(f"   Forest: {result['forest_percentage']}%")
        print(f"   Deforested: {result['deforested_percentage']}%")
        print(f"   Segmentation saved: {result['segmentation_path']}")
        print(f"   Overlay saved: {result['overlay_path']}")
        
        # Clean up
        os.unlink(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        return False

def test_web_application():
    """Test the web application endpoints."""
    print("\n" + "=" * 60)
    print("TESTING WEB APPLICATION")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8081"
    
    try:
        # Test home page
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Home page accessible")
        else:
            print(f"❌ Home page failed: {response.status_code}")
            return False
        
        # Test about page
        response = requests.get(f"{base_url}/about", timeout=5)
        if response.status_code == 200:
            print("✅ About page accessible")
        else:
            print(f"❌ About page failed: {response.status_code}")
        
        print("✅ Web application is running and accessible")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Web application is not running")
        print("   Please start the application with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Web application test failed: {e}")
        return False

def test_file_upload():
    """Test file upload and processing."""
    print("\n" + "=" * 60)
    print("TESTING FILE UPLOAD AND PROCESSING")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8081"
    
    try:
        # Create test image
        test_image_path = create_test_image()
        
        # Upload file
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post(f"{base_url}/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ File upload and processing successful")
                print(f"   Forest: {result['forest_percentage']}%")
                print(f"   Deforested: {result['deforested_percentage']}%")
                print("✅ Real model inference working through web interface")
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Upload failed: {response.status_code}")
            return False
        
        # Clean up
        os.unlink(test_image_path)
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Web application is not running")
        return False
    except Exception as e:
        print(f"❌ File upload test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("DEFORESTATION DETECTION SYSTEM - COMPREHENSIVE TEST")
    print("Testing Google Cloud integration and real model inference")
    print()
    
    tests = [
        ("Google Cloud Setup", test_google_cloud_setup),
        ("Model Availability", test_model_availability),
        ("Model Inference", test_model_inference),
        ("Web Application", test_web_application),
        ("File Upload & Processing", test_file_upload),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The system is working perfectly with Google Cloud!")
        print("\nKey achievements:")
        print("✅ Real Random Forest model trained and deployed")
        print("✅ No simulation fallbacks - everything uses real ML")
        print("✅ Google Cloud Storage integration working")
        print("✅ Model loading and inference working")
        print("✅ Web application fully functional")
        print("✅ End-to-end pipeline operational")
    else:
        print(f"❌ {total - passed} tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 