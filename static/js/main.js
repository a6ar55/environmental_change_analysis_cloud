// Enhanced JavaScript for Environmental Analysis Platform

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const fileDropZone = document.getElementById('file-drop-zone');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const uploadButton = document.getElementById('upload-button');
    const removeButton = document.getElementById('remove-image');
    const resultsContainer = document.getElementById('results-container');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');

    // Initialize drag and drop functionality
    initializeDragAndDrop();
    
    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                handleFileSelection(file);
            }
        });
    }

    // File drop zone click handler
    if (fileDropZone) {
        fileDropZone.addEventListener('click', function() {
            fileInput.click();
        });
    }

    // Remove image button
    if (removeButton) {
        removeButton.addEventListener('click', function() {
            clearFileSelection();
        });
    }
    
    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const file = fileInput.files[0];
            if (!file) {
                showError('Please select an image to upload.');
                return;
            }
            
            // Show loading spinner
            showLoading();
            hideError();
            hideResults();
            
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
                hideLoading();
                
                if (data.success) {
                    displayResults(data);
                } else {
                    showError(data.error || 'An error occurred during processing. Please check Google Cloud permissions.');
                }
            })
            .catch(error => {
                hideLoading();
                showError('Network error: ' + error.message + '. Please check your connection and try again.');
            });
        });
    }

    // Initialize drag and drop functionality
    function initializeDragAndDrop() {
        if (!fileDropZone) return;

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileDropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            fileDropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            fileDropZone.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            fileDropZone.classList.add('dragover');
        }

        function unhighlight() {
            fileDropZone.classList.remove('dragover');
        }

        // Handle dropped files
        fileDropZone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;

            if (files.length > 0) {
                const file = files[0];
                if (isValidImageFile(file)) {
                    fileInput.files = files;
                    handleFileSelection(file);
                } else {
                    showError('Please select a valid image file (JPG, PNG, or TIFF).');
                }
            }
        }
    }

    // Handle file selection
    function handleFileSelection(file) {
        if (!isValidImageFile(file)) {
            showError('Please select a valid image file (JPG, PNG, or TIFF).');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            if (imagePreview) {
                imagePreview.src = e.target.result;
                showPreview();
                enableUploadButton();
            }
        };
        reader.readAsDataURL(file);
    }

    // Validate image file
    function isValidImageFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff'];
        return validTypes.includes(file.type.toLowerCase());
    }

    // Clear file selection
    function clearFileSelection() {
        if (fileInput) {
            fileInput.value = '';
        }
        hidePreview();
        disableUploadButton();
    }

    // Show/hide preview
    function showPreview() {
        if (previewContainer) {
            previewContainer.style.display = 'block';
            previewContainer.style.opacity = '0';
            setTimeout(() => {
                previewContainer.style.opacity = '1';
            }, 100);
        }
    }

    function hidePreview() {
        if (previewContainer) {
            previewContainer.style.opacity = '0';
            setTimeout(() => {
                previewContainer.style.display = 'none';
            }, 300);
        }
    }

    // Enable/disable upload button
    function enableUploadButton() {
        if (uploadButton) {
            uploadButton.disabled = false;
            uploadButton.classList.add('enabled');
        }
    }

    function disableUploadButton() {
        if (uploadButton) {
            uploadButton.disabled = true;
            uploadButton.classList.remove('enabled');
        }
    }

    // Show/hide loading
    function showLoading() {
        if (loadingSpinner) {
            loadingSpinner.style.display = 'flex';
            loadingSpinner.style.opacity = '0';
            setTimeout(() => {
                loadingSpinner.style.opacity = '1';
            }, 100);
        }
    }

    function hideLoading() {
        if (loadingSpinner) {
            loadingSpinner.style.opacity = '0';
            setTimeout(() => {
                loadingSpinner.style.display = 'none';
            }, 300);
        }
    }

    // Show/hide error
    function showError(message) {
        if (errorMessage) {
            errorMessage.innerHTML = '';
            errorMessage.textContent = message;
        }
        if (errorContainer) {
            errorContainer.style.display = 'flex';
            errorContainer.style.opacity = '0';
            setTimeout(() => {
                errorContainer.style.opacity = '1';
            }, 100);
        }

        // Add Google Cloud permissions guidance for specific errors
        if (message.includes('403') || message.includes('permission') || message.includes('Permission')) {
            const helpText = document.createElement('div');
            helpText.innerHTML = `
                <div style="margin-top: 1rem; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 12px;">
                    <p><strong>Google Cloud Setup Required:</strong></p>
                    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                        <li>Storage Admin permissions</li>
                        <li>Storage Object Admin permissions</li>
                        <li>Vertex AI User permissions</li>
                        <li>Vertex AI Admin permissions</li>
                    </ul>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem; color: var(--text-tertiary);">
                        Check the deployment documentation for setup instructions.
                    </p>
                </div>
            `;
            if (errorMessage) {
                errorMessage.appendChild(helpText);
            }
        }
    }

    function hideError() {
        if (errorContainer) {
            errorContainer.style.opacity = '0';
            setTimeout(() => {
                errorContainer.style.display = 'none';
            }, 300);
        }
    }

    // Show/hide results
    function showResults() {
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
            resultsContainer.style.opacity = '0';
            setTimeout(() => {
                resultsContainer.style.opacity = '1';
            }, 100);
        }
    }

    function hideResults() {
        if (resultsContainer) {
            resultsContainer.style.opacity = '0';
            setTimeout(() => {
                resultsContainer.style.display = 'none';
            }, 300);
        }
    }

    // Display the results with animations
    function displayResults(data) {
        // Show results container
        showResults();
        
        // Update images
        updateResultImages(data);
        
        // Update statistics with animation
        updateStatistics(data);
        
        // Scroll to results
        setTimeout(() => {
            if (resultsContainer) {
                resultsContainer.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        }, 500);
        
        // Add image click handlers
        setTimeout(addImageClickHandlers, 1000);
    }

    // Update result images
    function updateResultImages(data) {
        const originalImg = document.getElementById('original-image');
        const segmentationImg = document.getElementById('segmentation-image');
        const overlayImg = document.getElementById('overlay-image');

        if (originalImg && data.original_image) {
            originalImg.src = data.original_image.gcs_url || data.original_image.url;
        }
        
        if (segmentationImg && data.segmentation_image) {
            segmentationImg.src = data.segmentation_image.gcs_url || data.segmentation_image.url;
        }
        
        if (overlayImg && data.overlay_image) {
            overlayImg.src = data.overlay_image.gcs_url || data.overlay_image.url;
        }
    }

    // Update statistics with animations
    function updateStatistics(data) {
        const forestPercentage = parseFloat(data.forest_percentage) || 0;
        const deforestedPercentage = parseFloat(data.deforested_percentage) || 0;

        // Update percentage displays
        const forestValueElement = document.getElementById('forest-percentage');
        const deforestedValueElement = document.getElementById('deforested-percentage');

        if (forestValueElement) {
            animateValue(forestValueElement, 0, forestPercentage, 1500, '%');
        }
        
        if (deforestedValueElement) {
            animateValue(deforestedValueElement, 0, deforestedPercentage, 1500, '%');
        }

        // Update progress rings
        updateProgressRing('forest-progress', forestPercentage);
        updateProgressRing('deforested-progress', deforestedPercentage);
    }

    // Animate value changes
    function animateValue(element, start, end, duration, suffix = '') {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const current = start + (end - start) * easeOutCubic(progress);
            element.textContent = current.toFixed(1) + suffix;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // Easing function for smooth animations
    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    // Update circular progress rings
    function updateProgressRing(ringId, percentage) {
        const ring = document.getElementById(ringId);
        if (ring) {
            const circumference = 2 * Math.PI * 50; // radius = 50
            const offset = circumference - (percentage / 100) * circumference;
            
            // Animate the stroke-dashoffset
            setTimeout(() => {
                ring.style.strokeDashoffset = offset;
            }, 500);
        }
    }

    // Add image click handlers for fullscreen view
    function addImageClickHandlers() {
        const imageCards = document.querySelectorAll('.image-card');
        imageCards.forEach(card => {
            card.addEventListener('click', function() {
                const img = card.querySelector('.result-image');
                if (img && img.src) {
                    const titleElement = card.querySelector('.image-header h4');
                    const title = titleElement ? titleElement.textContent : 'Image';
                    openImageFullscreen(img.src, title);
                }
            });
        });
    }

    // Open image in fullscreen modal
    function openImageFullscreen(src, title) {
        const modal = document.createElement('div');
        modal.className = 'fullscreen-modal';
        modal.innerHTML = `
            <div class="fullscreen-backdrop">
                <div class="fullscreen-content">
                    <div class="fullscreen-header">
                        <h4>${title}</h4>
                        <button class="close-fullscreen">
                            <i class="bi bi-x-lg"></i>
                        </button>
                    </div>
                    <div class="fullscreen-image-container">
                        <img src="${src}" alt="${title}" class="fullscreen-image">
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        
        // Add styles for fullscreen modal
        if (!document.getElementById('fullscreen-styles')) {
            const styles = document.createElement('style');
            styles.id = 'fullscreen-styles';
            styles.textContent = `
                .fullscreen-modal {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    z-index: 10000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .fullscreen-backdrop {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.95);
                    backdrop-filter: blur(10px);
                }
                
                .fullscreen-content {
                    position: relative;
                    max-width: 90vw;
                    max-height: 90vh;
                    background: var(--glass-bg);
                    border: 1px solid var(--glass-border);
                    border-radius: 20px;
                    overflow: hidden;
                }
                
                .fullscreen-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1rem 1.5rem;
                    border-bottom: 1px solid var(--glass-border);
                    background: var(--bg-secondary);
                }
                
                .close-fullscreen {
                    background: none;
                    border: none;
                    color: var(--text-primary);
                    font-size: 1.2rem;
                    cursor: pointer;
                    padding: 0.5rem;
                    border-radius: 8px;
                    transition: all 0.3s ease;
                }
                
                .close-fullscreen:hover {
                    background: rgba(255, 255, 255, 0.1);
                }
                
                .fullscreen-image-container {
                    padding: 1rem;
                }
                
                .fullscreen-image {
                    width: 100%;
                    height: auto;
                    max-height: 70vh;
                    object-fit: contain;
                    border-radius: 12px;
                }
            `;
            document.head.appendChild(styles);
        }

        // Close modal handlers
        const closeBtn = modal.querySelector('.close-fullscreen');
        const backdrop = modal.querySelector('.fullscreen-backdrop');
        
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                document.body.removeChild(modal);
            });
        }
        
        if (backdrop) {
            backdrop.addEventListener('click', (e) => {
                if (e.target === backdrop) {
                    document.body.removeChild(modal);
                }
            });
        }
        
        // ESC key handler
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                document.body.removeChild(modal);
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
    }
}); 