// Main JavaScript file for the Deforestation Detection App

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const uploadButton = document.getElementById('upload-button');
    const resultsContainer = document.getElementById('results-container');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    
    // Preview the selected image
    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.style.display = 'block';
                uploadButton.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            showError('Please select an image to upload.');
            return;
        }
        
        // Show loading spinner
        loadingSpinner.style.display = 'block';
        resultsContainer.style.display = 'none';
        errorContainer.style.display = 'none';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send the request
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            if (data.success) {
                // Display results
                displayResults(data);
            } else {
                // Show error message
                showError(data.error || 'An error occurred during processing. Please check Google Cloud permissions.');
            }
        })
        .catch(error => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            // Show error message
            showError('An error occurred: ' + error.message + '. Please check Google Cloud permissions.');
        });
    });
    
    // Display the results
    function displayResults(data) {
        // Show results container
        resultsContainer.style.display = 'block';
        
        // Update original image
        document.getElementById('original-image').src = data.original_image.url;
        
        // Update segmentation image
        document.getElementById('segmentation-image').src = data.segmentation_image.url;
        
        // Update overlay image
        document.getElementById('overlay-image').src = data.overlay_image.url;
        
        // Update statistics
        document.getElementById('forest-percentage').textContent = data.forest_percentage.toFixed(2) + '%';
        document.getElementById('deforested-percentage').textContent = data.deforested_percentage.toFixed(2) + '%';
        
        // Update forest percentage bar
        const forestBar = document.getElementById('forest-bar');
        forestBar.style.width = data.forest_percentage + '%';
        
        // Update deforested percentage bar
        const deforestedBar = document.getElementById('deforested-bar');
        deforestedBar.style.width = data.deforested_percentage + '%';
    }
    
    // Show error message
    function showError(message) {
        errorContainer.style.display = 'block';
        errorMessage.textContent = message;
        
        // Add Google Cloud permissions guidance
        if (message.includes('403') || message.includes('permission') || message.includes('Permission')) {
            const permissionsHelp = document.createElement('div');
            permissionsHelp.innerHTML = `
                <p class="mt-2">This error is related to Google Cloud permissions. Please ensure your service account has:</p>
                <ul class="list-disc ml-5">
                    <li>Storage Admin</li>
                    <li>Storage Object Admin</li>
                    <li>Vertex AI User</li>
                    <li>Vertex AI Admin</li>
                </ul>
                <p class="mt-2">Check the README.md file for detailed setup instructions.</p>
            `;
            errorMessage.appendChild(permissionsHelp);
        }
    }
    
    // Add drag and drop functionality for file upload
    const uploadContainer = document.querySelector('.upload-container');
    
    if (uploadContainer) {
        const fileInput = document.getElementById('file');
        
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadContainer.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadContainer.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadContainer.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            uploadContainer.classList.add('border-primary');
        }
        
        function unhighlight() {
            uploadContainer.classList.remove('border-primary');
        }
        
        // Handle dropped files
        uploadContainer.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                // Show file name
                const fileName = files[0].name;
                const fileLabel = document.querySelector('.form-label');
                if (fileLabel) {
                    fileLabel.textContent = `Selected file: ${fileName}`;
                }
            }
        }
    }
}); 