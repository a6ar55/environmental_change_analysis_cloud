# 🚀 **Environmental Analysis Platform - Cloud Deployment Guide**

## **Overview**
This guide will help you deploy the Environmental Change Analysis Platform to Google Cloud Run. The platform combines deforestation detection and climate analysis using real machine learning models deployed on Google Cloud.

## **🔧 Prerequisites**

### **1. Google Cloud Setup**
- Google Cloud Project with billing enabled
- Project ID: `air-pollution-platform` (or update in `config.py`)
- Required APIs enabled:
  - Cloud Run API
  - Cloud Build API
  - Container Registry API
  - Cloud Storage API
  - Vertex AI API

### **2. Local Development Environment**
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
- Docker installed (for local testing)
- Python 3.9+ installed

### **3. Authentication**
```bash
# Authenticate with Google Cloud
gcloud auth login

# Set your default project
gcloud config set project air-pollution-platform

# Enable Application Default Credentials
gcloud auth application-default login
```

## **📦 Deployment Methods**

### **Method 1: One-Click Deployment (Recommended)**

Simply run the deployment script:

```bash
chmod +x deploy.sh
./deploy.sh
```

This script will:
- ✅ Check authentication and dependencies
- 🔧 Enable required Google Cloud APIs
- 🏗️ Build the container image using Cloud Build
- 🚀 Deploy to Cloud Run with optimized settings
- 🌐 Provide you with the live application URL

### **Method 2: Manual Deployment**

#### **Step 1: Build and Push Container**
```bash
# Build the container image
gcloud builds submit --tag gcr.io/air-pollution-platform/environmental-analysis

# Or build locally and push
docker build -t gcr.io/air-pollution-platform/environmental-analysis .
docker push gcr.io/air-pollution-platform/environmental-analysis
```

#### **Step 2: Deploy to Cloud Run**
```bash
gcloud run deploy environmental-analysis-app \
    --image gcr.io/air-pollution-platform/environmental-analysis \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 900 \
    --max-instances 10 \
    --set-env-vars GOOGLE_CLOUD_PROJECT=air-pollution-platform,FLASK_ENV=production \
    --port 8080
```

### **Method 3: CI/CD with Cloud Build**

Commit your changes and push to trigger automatic deployment:

```bash
git add .
git commit -m "Deploy to Cloud Run"
git push origin main
```

The `cloudbuild.yaml` file will automatically:
- Build the container
- Push to Container Registry
- Deploy to Cloud Run

## **🔒 Security & Configuration**

### **Environment Variables**
The following environment variables are automatically set:

- `GOOGLE_CLOUD_PROJECT`: Your project ID
- `FLASK_ENV`: Set to `production`
- `PORT`: Set to `8080` (Cloud Run standard)
- `K_SERVICE`: Automatically set by Cloud Run (indicates cloud environment)

### **Service Account Permissions**
Cloud Run uses the default compute service account which has these permissions:
- Cloud Storage access
- Vertex AI access
- Container Registry access

For production, create a custom service account with minimal permissions.

### **Resource Allocation**
Optimized for ML workloads:
- **Memory**: 2GB (for image processing and ML models)
- **CPU**: 2 vCPUs (for parallel processing)
- **Timeout**: 15 minutes (for complex analysis)
- **Max Instances**: 10 (cost optimization)

## **🌐 Features Available After Deployment**

### **🌲 Deforestation Detection**
- Real-time satellite image analysis
- Pixel-level forest classification
- Deforestation percentage calculations
- Interactive result visualizations

### **🌡️ Climate Analysis**
- Global temperature trend analysis
- Seasonal pattern recognition
- Country-specific climate data
- Future temperature predictions (1-10 years)
- Interactive dashboard with dark theme

### **🔗 API Endpoints**
- `/api/climate/summary` - Climate summary statistics
- `/api/climate/trends` - Global temperature trends
- `/api/climate/seasonal` - Seasonal analysis
- `/api/climate/predictions` - Future predictions
- `/api/climate/countries` - Country-specific data

## **📊 Monitoring & Maintenance**

### **View Logs**
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=environmental-analysis-app" --limit 50
```

### **Update Deployment**
```bash
# After making changes to your code
./deploy.sh
```

### **Scale Resources**
```bash
gcloud run services update environmental-analysis-app \
    --region us-central1 \
    --memory 4Gi \
    --cpu 4
```

### **Delete Service**
```bash
gcloud run services delete environmental-analysis-app --region us-central1
```

## **🚨 Troubleshooting**

### **Common Issues**

1. **Build Fails**
   - Check Docker file syntax
   - Ensure all dependencies are in requirements.txt
   - Verify Python version compatibility

2. **Deployment Timeout**
   - Increase timeout: `--timeout 900`
   - Check resource allocation
   - Monitor cold start performance

3. **Memory Issues**
   - Increase memory: `--memory 4Gi`
   - Optimize model loading
   - Implement lazy loading

4. **Authentication Errors**
   - Verify service account permissions
   - Check Application Default Credentials
   - Ensure APIs are enabled

### **Performance Optimization**

1. **Cold Start Reduction**
   - Keep one instance warm: `--min-instances 1`
   - Optimize import statements
   - Use lazy loading for ML models

2. **Memory Optimization**
   - Stream large files instead of loading entirely
   - Clear unused variables
   - Use efficient data structures

## **💰 Cost Optimization**

- **CPU**: Allocated only during request processing
- **Memory**: Charged per GB-second of usage
- **Requests**: Free tier includes 2 million requests/month
- **Always Free**: 180,000 vCPU-seconds, 360,000 GiB-seconds

**Estimated Monthly Cost** (moderate usage):
- ~$5-15/month for typical workloads
- Scales to zero when not in use

## **🔄 Updates & Maintenance**

### **Regular Updates**
1. Update Python dependencies monthly
2. Monitor security vulnerabilities
3. Update base Docker image
4. Refresh ML models with new data

### **Backup Strategy**
- Models and data stored in Google Cloud Storage
- Source code in Git repository
- Automated daily backups of GCS buckets

## **📞 Support**

For issues or questions:
1. Check Cloud Run logs first
2. Verify all prerequisites are met
3. Test locally with Docker before deploying
4. Review Google Cloud documentation

---

**🎉 Your Environmental Analysis Platform is now live on Google Cloud!**

Access your deployed application at the URL provided after deployment completion. 