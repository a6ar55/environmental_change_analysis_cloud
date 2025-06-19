@echo off
echo === Deforestation Detection System Setup ===
echo.

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist static\uploads mkdir static\uploads
if not exist static\results mkdir static\results
if not exist data\images mkdir data\images
if not exist data\masks mkdir data\masks

REM Step 1: Download images from Google Drive (if available)
echo === Step 1: Download images from Google Drive ===
python download_drive_images.py

REM Step 2: Create segmentation masks
echo === Step 2: Create segmentation masks ===
python create_vertex_model.py

REM Step 3: Run the application
echo === Step 3: Starting the application ===
echo The application will be available at http://localhost:8081
python app.py 