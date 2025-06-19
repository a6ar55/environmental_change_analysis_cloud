#!/usr/bin/env python3
"""
Download forest images from Google Drive.
This script is used to download sample forest images for the deforestation detection system.
"""

import os
import sys
import shutil

def download_from_drive(folder_id=None, output_dir="data/images"):
    """
    Download images from Google Drive.
    
    Args:
        folder_id: Google Drive folder ID
        output_dir: Directory to save the downloaded images
    """
    print("Attempting to download images from Google Drive...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Try to import gdown
        try:
            import gdown
        except ImportError:
            print("gdown package not installed. Please install it with 'pip install gdown'")
            print("Using sample test image instead.")
            use_sample_image(output_dir)
            return
        
        # If no folder ID is provided, use sample image
        if not folder_id:
            print("No Google Drive folder ID provided.")
            use_sample_image(output_dir)
            return
        
        # Download all files from the folder
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
        
        # Check if any files were downloaded
        files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if not files:
            print("No image files found in the Google Drive folder.")
            use_sample_image(output_dir)
        else:
            print(f"Downloaded {len(files)} images to {output_dir}")
            
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        print("Using sample test image instead.")
        use_sample_image(output_dir)

def use_sample_image(output_dir):
    """
    Copy the sample test image to the output directory.
    
    Args:
        output_dir: Directory to save the sample image
    """
    test_image = os.path.join("static", "test", "forest_test.jpg")
    if os.path.exists(test_image):
        dest_path = os.path.join(output_dir, "forest_test.jpg")
        shutil.copy2(test_image, dest_path)
        print(f"Copied sample test image to {dest_path}")
    else:
        print("Sample test image not found at static/test/forest_test.jpg")
        print("Please add images manually to the data/images directory.")

if __name__ == "__main__":
    # Parse command line arguments
    folder_id = None
    output_dir = "data/images"
    
    if len(sys.argv) > 1:
        folder_id = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    download_from_drive(folder_id, output_dir) 