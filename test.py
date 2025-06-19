#!/usr/bin/env python3
"""
Test script for the Deforestation Detection application.
This script tests the basic functionality of the application components.
"""

import os
import sys
import cv2
import numpy as np
from config import FOREST_COLOR, DEFORESTED_COLOR
import cloud_utils

def create_test_image():
    """Create a test image with forest and deforested areas."""
    print("Creating test image...")
    
    # Create a 500x500 image with forest (green) and deforested (brown) areas
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Create a forest area (green) on the left side
    img[:, :250] = (34, 139, 34)  # Forest green in BGR
    
    # Create a deforested area (brown) on the right side
    img[:, 250:] = (42, 42, 165)  # Brown in BGR
    
    # Add some random noise to make it look more natural
    noise = np.random.randint(0, 30, img.shape, dtype=np.int32)
    img = np.clip(img.astype(np.int32) + noise - 15, 0, 255).astype(np.uint8)
    
    # Create directory if it doesn't exist
    os.makedirs('static/test', exist_ok=True)
    
    # Save the test image
    test_image_path = 'static/test/forest_test.jpg'
    cv2.imwrite(test_image_path, img)
    
    print(f"Test image created at: {test_image_path}")
    return test_image_path

def test_segmentation(image_path):
    """Test the semantic segmentation functionality."""
    print("\nTesting semantic segmentation...")
    
    try:
        # Run segmentation
        results = cloud_utils.run_semantic_segmentation(image_path)
        
        print("Segmentation completed successfully!")
        print(f"Forest percentage: {results['forest_percentage']}%")
        print(f"Deforested percentage: {results['deforested_percentage']}%")
        print(f"Segmentation mask saved at: {results['segmentation_path']}")
        print(f"Overlay image saved at: {results['overlay_path']}")
        
        return True
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return False

def test_cloud_storage():
    """Test Google Cloud Storage functionality."""
    print("\nTesting Google Cloud Storage...")
    
    try:
        # Ensure bucket exists
        bucket = cloud_utils.ensure_bucket_exists()
        print(f"Bucket exists: {bucket.name}")
        return True
    except Exception as e:
        print(f"Error with Google Cloud Storage: {e}")
        return False

def main():
    """Main test function."""
    print("Running tests for Deforestation Detection application...\n")
    
    # Create test image
    test_image_path = create_test_image()
    
    # Test cloud storage if credentials are available
    if os.path.exists('service_account.json'):
        cloud_storage_ok = test_cloud_storage()
    else:
        print("\nSkipping Google Cloud Storage test (service_account.json not found)")
        cloud_storage_ok = False
    
    # Test segmentation
    segmentation_ok = test_segmentation(test_image_path)
    
    # Print summary
    print("\nTest Summary:")
    print(f"- Image Creation: Success")
    print(f"- Cloud Storage: {'Success' if cloud_storage_ok else 'Skipped/Failed'}")
    print(f"- Segmentation: {'Success' if segmentation_ok else 'Failed'}")
    
    if segmentation_ok:
        print("\nAll critical tests passed! The application should work correctly.")
    else:
        print("\nSome tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 