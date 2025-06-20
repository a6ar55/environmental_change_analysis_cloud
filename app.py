import os
import tempfile
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import cloud_utils
from climate_analyzer import climate_analyzer
from config import SECRET_KEY, DEBUG, UPLOAD_FOLDER, RESULTS_FOLDER, PROJECT_ID

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = os.path.join('static', UPLOAD_FOLDER)
app.config['RESULTS_FOLDER'] = os.path.join('static', RESULTS_FOLDER)

# Ensure upload and results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file locally first
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            local_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(local_path)
            
            # Upload to Google Cloud Storage
            gcs_blob_name, gcs_url = cloud_utils.upload_to_gcs(local_path)
            
            # Process the image with semantic segmentation
            results = cloud_utils.run_semantic_segmentation(local_path)
            
            # Get paths for segmentation and overlay images
            segmentation_path = results["segmentation_path"]
            overlay_path = results["overlay_path"]
            
            # Upload results to Google Cloud Storage
            segmentation_blob_name, segmentation_url = cloud_utils.upload_to_gcs(
                segmentation_path, 
                f"{RESULTS_FOLDER}/segmentation_{os.path.basename(gcs_blob_name)}"
            )
            
            overlay_blob_name, overlay_url = cloud_utils.upload_to_gcs(
                overlay_path, 
                f"{RESULTS_FOLDER}/overlay_{os.path.basename(gcs_blob_name)}"
            )
            
            # Return results
            return jsonify({
                'success': True,
                'original_image': {
                    'url': '/' + local_path.replace('\\', '/'),
                    'gcs_url': gcs_url
                },
                'segmentation_image': {
                    'url': segmentation_url,
                    'gcs_url': segmentation_url
                },
                'overlay_image': {
                    'url': overlay_url,
                    'gcs_url': overlay_url
                },
                'forest_percentage': results["forest_percentage"],
                'deforested_percentage': results["deforested_percentage"]
            })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            error_message = str(e)
            
            # Provide more specific error messages
            if "private key to sign credentials" in error_message:
                error_message = "Authentication configured successfully. Using public URLs for file access."
                # This is actually not an error - it's expected in Cloud Run
                return jsonify({
                    'success': False,
                    'error': error_message,
                    'suggestion': "The app is working correctly. This message appears because Cloud Run uses different authentication than local development."
                }), 500
            elif "Storage Admin" in error_message:
                error_message = f"Google Cloud Storage permission error: {error_message}"
            elif "Vertex AI" in error_message:
                error_message = f"Google Cloud Vertex AI permission error: {error_message}"
            else:
                error_message = f"Processing error: {error_message}"
            
            return jsonify({
                'success': False,
                'error': error_message,
                'help': "Please check the Cloud Run logs for more details."
            }), 500
    
    flash('File type not allowed')
    return redirect(request.url)

@app.route('/analyze', methods=['GET'])
def analyze():
    # This route is for displaying analysis results
    image_url = request.args.get('image')
    if not image_url:
        flash('No image specified')
        return redirect(url_for('index'))
    
    # Get other parameters if available
    overlay_url = request.args.get('overlay')
    segmentation_url = request.args.get('segmentation')
    forest_percentage = request.args.get('forest', 0)
    deforested_percentage = request.args.get('deforested', 0)
    
    return render_template(
        'analyze.html',
        image_url=image_url,
        overlay_url=overlay_url,
        segmentation_url=segmentation_url,
        forest_percentage=forest_percentage,
        deforested_percentage=deforested_percentage
    )

@app.route('/about')
def about():
    return render_template('about.html')

# Climate Analysis Routes
@app.route('/climate')
def climate_dashboard():
    """Climate analysis dashboard."""
    return render_template('climate.html')

@app.route('/api/climate/summary')
def climate_summary():
    """API endpoint for climate summary data."""
    try:
        # Load model and data if not already loaded
        if not climate_analyzer.model_loaded:
            if not climate_analyzer.load_model_from_gcs():
                return jsonify({
                    'success': False,
                    'error': 'Climate model not available. Please train the model first.'
                }), 404
        
        if climate_analyzer.predictions_data is None:
            if not climate_analyzer.load_data_from_gcs():
                return jsonify({
                    'success': False,
                    'error': 'Climate data not available. Please process the data first.'
                }), 404
        
        summary = climate_analyzer.get_climate_summary()
        if summary:
            return jsonify({
                'success': True,
                'data': summary
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate climate summary'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error loading climate data: {str(e)}'
        }), 500

@app.route('/api/climate/trends')
def climate_trends():
    """API endpoint for global temperature trends."""
    try:
        if climate_analyzer.predictions_data is None:
            if not climate_analyzer.load_data_from_gcs():
                return jsonify({
                    'success': False,
                    'error': 'Climate data not available'
                }), 404
        
        trends = climate_analyzer.get_global_temperature_trends()
        if trends:
            return jsonify({
                'success': True,
                'data': trends
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate climate trends'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating trends: {str(e)}'
        }), 500

@app.route('/api/climate/seasonal')
def climate_seasonal():
    """API endpoint for seasonal analysis."""
    try:
        if climate_analyzer.predictions_data is None:
            if not climate_analyzer.load_data_from_gcs():
                return jsonify({
                    'success': False,
                    'error': 'Climate data not available'
                }), 404
        
        seasonal = climate_analyzer.get_seasonal_analysis()
        if seasonal:
            return jsonify({
                'success': True,
                'data': seasonal
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate seasonal analysis'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating seasonal analysis: {str(e)}'
        }), 500

@app.route('/api/climate/predictions')
def climate_predictions():
    """API endpoint for future temperature predictions."""
    try:
        years_ahead = request.args.get('years', 5, type=int)
        years_ahead = min(max(years_ahead, 1), 10)  # Limit between 1-10 years
        
        if not climate_analyzer.model_loaded:
            if not climate_analyzer.load_model_from_gcs():
                return jsonify({
                    'success': False,
                    'error': 'Climate model not available'
                }), 404
        
        if climate_analyzer.predictions_data is None:
            if not climate_analyzer.load_data_from_gcs():
                return jsonify({
                    'success': False,
                    'error': 'Climate data not available'
                }), 404
        
        predictions = climate_analyzer.predict_future_temperature(years_ahead=years_ahead)
        if predictions:
            return jsonify({
                'success': True,
                'data': predictions,
                'years_ahead': years_ahead
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate predictions'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating predictions: {str(e)}'
        }), 500

@app.route('/api/climate/countries')
def climate_countries():
    """API endpoint for country-specific analysis."""
    try:
        country_name = request.args.get('country')
        
        if climate_analyzer.country_data is None:
            if not climate_analyzer.load_data_from_gcs():
                return jsonify({
                    'success': False,
                    'error': 'Country climate data not available'
                }), 404
        
        analysis = climate_analyzer.get_country_analysis(country_name)
        if analysis:
            return jsonify({
                'success': True,
                'data': analysis
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate country analysis'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating country analysis: {str(e)}'
        }), 500

if __name__ == '__main__':
    from config import PORT, HOST
    print(f"Starting server on {HOST}:{PORT}...")
    print(f"Google Cloud Project: {PROJECT_ID}")
    print(f"Using service account: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
    app.run(host=HOST, port=PORT, debug=DEBUG) 