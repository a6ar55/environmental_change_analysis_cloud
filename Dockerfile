FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and Google Cloud SDK
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p static/uploads static/results static/data climate_data \
    && chmod 755 static/uploads static/results static/data climate_data

# Set environment variables for Cloud Run
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV GOOGLE_CLOUD_PROJECT=air-pollution-platform
ENV FLASK_ENV=production

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose the port that Cloud Run expects
EXPOSE 8080

# Cloud Run handles health checks automatically

# Run the application with gunicorn (production WSGI server)
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app 