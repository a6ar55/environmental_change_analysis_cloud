#!/usr/bin/env python3
"""
Setup script for the Deforestation Detection System.
This script creates the necessary directories and checks for dependencies.
"""

import os
import sys
import importlib.util
import shutil

def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    return path

def check_dependency(module_name):
    """Check if a Python module is installed."""
    if importlib.util.find_spec(module_name) is None:
        print(f"Missing dependency: No module named '{module_name}'")
        return False
    return True

def setup():
    """Set up the project structure and check dependencies."""
    print("Setting up Deforestation Detection project...")
    
    # Create directory structure
    static_dir = create_directory("static")
    create_directory(os.path.join(static_dir, "css"))
    create_directory(os.path.join(static_dir, "js"))
    create_directory(os.path.join(static_dir, "uploads"))
    create_directory(os.path.join(static_dir, "results"))
    create_directory(os.path.join(static_dir, "test"))
    
    create_directory("templates")
    create_directory("data")
    create_directory(os.path.join("data", "images"))
    create_directory(os.path.join("data", "masks"))
    
    # Copy sample test image if it doesn't exist
    test_dir = os.path.join(static_dir, "test")
    test_image = os.path.join(test_dir, "forest_test.jpg")
    
    if not os.path.exists(test_image):
        # Check if we have a sample image to copy
        sample_sources = [
            "sample_images/forest_test.jpg",
            "data/images/forest_test.jpg"
        ]
        
        copied = False
        for source in sample_sources:
            if os.path.exists(source):
                shutil.copy2(source, test_image)
                print(f"Copied sample test image from {source} to {test_dir}")
                copied = True
                break
        
        if not copied:
            print("No sample test image found. Please add one manually to static/test/forest_test.jpg")
    
    # Check dependencies
    dependencies = [
        "flask",
        "numpy",
        "cv2",  # OpenCV
        "dotenv",
        "PIL"  # Pillow
    ]
    
    missing_deps = False
    for dep in dependencies:
        if not check_dependency(dep):
            missing_deps = True
    
    if missing_deps:
        print("\nPlease run: pip install -r requirements.txt")
    
    print("\nSetup completed.")

if __name__ == "__main__":
    setup() 