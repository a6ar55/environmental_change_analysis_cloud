#!/bin/bash

# Set error handling
set -e

# Print header
echo "====================================================="
echo "  Deforestation Detection - Training and Deployment  "
echo "====================================================="
echo ""

# Check if service account file exists
if [ ! -f "service_account.json" ]; then
    echo "Error: service_account.json not found!"
    echo "Please place your Google Cloud service account key file in the project root directory."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check Python installation
echo "Checking Python installation..."
python --version

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Verify Google Cloud SDK installation
echo "Verifying Google Cloud SDK installation..."
if ! command -v gcloud &> /dev/null; then
    echo "Google Cloud SDK not found. Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/service_account.json"
echo "Using service account: $GOOGLE_APPLICATION_CREDENTIALS"

# Verify permissions
echo "Verifying Google Cloud permissions..."
python -c "
from google.cloud import storage
from google.cloud import aiplatform
import sys
import os

try:
    # Test Storage permissions
    storage_client = storage.Client()
    print('✓ Successfully authenticated with Google Cloud Storage')
    
    # Test Vertex AI permissions
    aiplatform.init(project=os.environ.get('PROJECT_ID', 'air-pollution-platform'), location='us-central1')
    print('✓ Successfully authenticated with Vertex AI')
    
except Exception as e:
    print(f'Error: {e}')
    print('Please ensure your service account has the necessary permissions:')
    print('- Storage Admin')
    print('- Storage Object Admin')
    print('- Vertex AI User')
    print('- Vertex AI Admin')
    sys.exit(1)
"

# If the permission check failed, the script will have exited
echo "Permissions verified successfully!"

# Run the model creation script
echo "Starting model training and deployment..."
python create_vertex_model.py

echo "====================================================="
echo "  Training and deployment process completed!         "
echo "=====================================================" 