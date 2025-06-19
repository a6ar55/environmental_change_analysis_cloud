import os
import secrets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Google Application Credentials
SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), "service_account.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH

# Google Cloud configuration
PROJECT_ID = 'air-pollution-platform'  # Hardcoding the project ID from service account
os.environ["PROJECT_ID"] = PROJECT_ID
LOCATION = 'us-central1'
BUCKET_NAME = 'deforestation-detection-bucket'

# Model configuration
MODEL_NAME = 'deforestation-detection-model'
MODEL_VERSION = 'v1'

# Flask configuration
SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(16))
DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'

# File storage paths
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

# Segmentation colors (BGR format for OpenCV)
FOREST_COLOR = (0, 255, 0)  # Green
DEFORESTED_COLOR = (0, 0, 255)  # Red 