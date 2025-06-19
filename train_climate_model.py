#!/usr/bin/env python3
"""
Master Climate Model Training Script
Runs the complete pipeline: download data, process, train models, and deploy to Google Cloud.
"""

import sys
import subprocess
import os
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error Output:")
            print(e.stderr)
        return False

def check_prerequisites():
    """Check if required environment is set up."""
    print("Checking prerequisites...")
    
    # Check if in virtual environment
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Warning: Not in a virtual environment")
    
    # Check if Google Cloud credentials are set
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set")
        return False
    
    # Check if required files exist
    required_files = ['config.py', 'climate_data_processor.py', 'climate_vertex_trainer.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file {file} not found")
            return False
    
    print("✅ Prerequisites check passed")
    return True

def main():
    """Main training pipeline."""
    print("🌍 CLIMATE MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Please fix the issues and try again.")
        return 1
    
    # Step 1: Activate virtual environment and install dependencies
    if not run_command(
        "source venv/bin/activate && pip install -r requirements.txt",
        "STEP 1: Installing required packages"
    ):
        print("Failed to install packages")
        return 1
    
    # Step 2: Download and process climate data
    if not run_command(
        "source venv/bin/activate && python climate_data_processor.py",
        "STEP 2: Download and process climate data"
    ):
        print("Failed to process climate data")
        return 1
    
    # Step 3: Train climate models on Vertex AI
    if not run_command(
        "source venv/bin/activate && python climate_vertex_trainer.py",
        "STEP 3: Train climate models on Vertex AI"
    ):
        print("Failed to train climate models")
        return 1
    
    # Step 4: Test the web application
    print(f"\n{'='*60}")
    print("STEP 4: Testing web application")
    print(f"{'='*60}")
    print("Starting Flask application for testing...")
    print("You can now:")
    print("1. Visit http://localhost:8081 for deforestation detection")
    print("2. Visit http://localhost:8081/climate for climate analysis")
    print("3. Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(
            "source venv/bin/activate && python app.py",
            shell=True,
            check=True
        )
    except KeyboardInterrupt:
        print("\n✅ Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed: {e}")
        return 1
    
    print(f"\n{'='*60}")
    print("🎉 CLIMATE MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("Your climate analysis system is now ready!")
    print("✅ Data processed and uploaded to Google Cloud Storage")
    print("✅ Models trained and deployed on Vertex AI")
    print("✅ Web dashboard ready for interactive analysis")
    print("✅ All processing runs entirely on Google Cloud")
    print(f"Completed at: {datetime.now()}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 