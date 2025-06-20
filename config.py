import os
import secrets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if running in Cloud Run (Application Default Credentials)
RUNNING_IN_CLOUD = os.environ.get('K_SERVICE') is not None

# Set Google Application Credentials
if not RUNNING_IN_CLOUD:
    # Local development - use service account file
    SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), "service_account.json")
    if os.path.exists(SERVICE_ACCOUNT_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
    else:
        print("Warning: service_account.json not found. Using Application Default Credentials.")

# Google Cloud configuration
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'air-pollution-platform')
os.environ["PROJECT_ID"] = PROJECT_ID
LOCATION = 'us-central1'
BUCKET_NAME = 'deforestation-detection-bucket'

# Model configuration
MODEL_NAME = 'deforestation-detection-model'
MODEL_VERSION = 'v1'

# Flask configuration
SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(16))
DEBUG = os.environ.get('FLASK_ENV', 'development') != 'production'

# Server configuration
PORT = int(os.environ.get('PORT', 8080))
HOST = '0.0.0.0' if RUNNING_IN_CLOUD else '127.0.0.1'

# File storage paths
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

# Segmentation colors (BGR format for OpenCV)
FOREST_COLOR = (0, 255, 0)  # Green
DEFORESTED_COLOR = (0, 0, 255)  # Red 