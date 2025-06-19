# Environmental Change Analysis Platform

A comprehensive platform for environmental monitoring combining deforestation detection and climate analysis using AI and Google Cloud.

## Features

### 🌲 Deforestation Detection
- Upload satellite/aerial images for analysis
- AI-powered semantic segmentation using Random Forest
- Real-time forest coverage percentage calculation
- Visual overlay showing forest vs deforested areas
- Powered by Google Cloud Vertex AI

### 🌡️ Climate Analysis Dashboard
- Interactive temperature trends visualization
- Seasonal climate patterns analysis
- Future temperature predictions (1-10 years)
- Country-specific climate data exploration
- Real-time climate status monitoring

## Data Management

### Large Files Handling
This project uses large datasets and trained models that are **NOT stored in Git** for performance reasons:

**Excluded from Git (stored locally/cloud only):**
- `climate_data/*.csv` (508MB+ of historical temperature data)
- `climate_models.pkl` (217MB trained ML models)
- `*.pkl` files (processed datasets and models)

**Why these files are large:**
- **Climate data**: 270+ years of global temperature records (1743-2013)
- **City data**: Temperature records for thousands of cities worldwide
- **ML models**: Trained Random Forest, Gradient Boosting, and Linear Regression models

**Data Sources:**
- Climate data: Berkeley Earth temperature dataset via Kaggle API
- Models: Trained on Google Cloud Vertex AI
- Storage: Google Cloud Storage for production data

**To get the data:**
1. Run the climate data processor: `python climate_data_processor.py`
2. Train models: `python climate_vertex_trainer.py`
3. Data will be downloaded/generated automatically

## Google Cloud Setup

### Prerequisites
1. Google Cloud Project with billing enabled
2. Service Account with permissions:
   - Storage Admin
   - Storage Object Admin
   - Vertex AI User
   - Vertex AI Admin

### Installation
```bash
# Clone the repository
git clone https://github.com/a6ar55/environmental_change_analysis_cloud
cd environmental_change_analysis_cloud

# Install dependencies
pip install -r requirements.txt

# Add your service account key
# Place service_account.json in the project root

# Train and deploy models
./train_and_deploy.sh

# Run the application
python app.py
```

## Architecture

### Deforestation Detection
- **Frontend**: HTML5, Bootstrap, JavaScript with drag-drop upload
- **Backend**: Flask with Google Cloud integration
- **ML Model**: Random Forest classifier for pixel-level segmentation
- **Storage**: Google Cloud Storage for images and results
- **Processing**: Real-time inference on uploaded images

### Climate Analysis
- **Data Processing**: Kaggle API integration for Berkeley Earth dataset
- **ML Pipeline**: Multiple models (Random Forest, Gradient Boosting, Linear Regression)
- **Visualization**: Interactive Chart.js dashboards with dark theme
- **Predictions**: AI-powered future temperature forecasting
- **API**: RESTful endpoints for climate data access

### Cloud Infrastructure
- **Google Cloud Storage**: File storage and model artifacts
- **Vertex AI**: Model training and deployment
- **Service Account**: Secure authentication
- **Signed URLs**: Temporary access to generated results

## Technical Features

### Performance Optimizations
- Model caching for faster inference
- Efficient feature extraction for image processing
- Lazy loading of large datasets
- Signed URLs for direct cloud access

### Security
- Secure file upload validation
- Google Cloud IAM integration
- Environment variable configuration
- Error handling with user guidance

### User Experience
- Modern glassmorphism dark theme
- Real-time progress indicators
- Interactive visualizations
- Responsive design for all devices
- Comprehensive error messages

## License
MIT License - see LICENSE file for details

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support
For issues related to:
- **Google Cloud setup**: Check service account permissions
- **Large file errors**: Files are intentionally excluded from Git
- **Model training**: Ensure sufficient Cloud Storage quota
- **API errors**: Verify Vertex AI permissions 