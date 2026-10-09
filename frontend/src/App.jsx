import React, { useState } from 'react';
import { 
  Upload, 
  FileImage, 
  Activity, 
  RefreshCw, 
  AlertCircle, 
  CheckCircle, 
  Info, 
  Eye,
  Cpu
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Validate and handle file selection
  const processFile = (file) => {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (PNG, JPG, JPEG, BMP, TIF).');
      return;
    }

    setError(null);
    setResult(null);
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  // Drag and Drop Handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  // Send request to FastAPI /predict backend
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Prediction request failed.');
      }

      setResult(data);
    } catch (err) {
      console.error('API Error:', err);
      setError(err.message || 'Failed to connect to OncoVision backend server.');
    } finally {
      setIsLoading(false);
    }
  };

  // Reset dashboard state
  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  // Helper for badge CSS classes
  const getBadgeClass = (pred) => {
    const p = (pred || '').toLowerCase();
    if (p === 'normal') return 'badge-result badge-normal';
    if (p === 'benign') return 'badge-result badge-benign';
    if (p === 'malignant') return 'badge-result badge-malignant';
    return 'badge-result';
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-badge">
          <Activity size={14} /> Medical Research Platform
        </div>
        <h1>ONCOVISION</h1>
        <p>Breast Tumor Detection & Explainable AI Dashboard</p>
      </header>

      <main className="main-card">
        {/* Error Alert */}
        {error && (
          <div className="error-alert">
            <AlertCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
            {error}
          </div>
        )}

        {/* State 1: Upload Dropzone */}
        {!selectedFile && !result && (
          <div 
            className={`dropzone ${isDragging ? 'active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Upload className="dropzone-icon" />
            <h3>Upload Ultrasound Scan Image</h3>
            <p>Drag & drop your B-mode breast ultrasound file here, or click to browse</p>
            <label className="btn-select">
              Select Image File
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleFileInput} 
                style={{ display: 'none' }} 
              />
            </label>
          </div>
        )}

        {/* State 2: Preview & Analyze Button */}
        {selectedFile && !result && !isLoading && (
          <div className="preview-container">
            <div className="image-preview-wrapper">
              <img src={previewUrl} alt="Ultrasound Scan Preview" className="image-preview" />
            </div>
            <div className="preview-actions">
              <button className="btn-analyze" onClick={handleAnalyze}>
                <Activity size={18} /> Analyze Ultrasound Scan
              </button>
              <button className="btn-reset" onClick={handleReset}>
                <RefreshCw size={16} /> Choose Different Image
              </button>
            </div>
          </div>
        )}

        {/* State 3: Loading Indicator */}
        {isLoading && (
          <div className="loading-box">
            <div className="spinner"></div>
            <h3>Analyzing Ultrasound Image...</h3>
            <p style={{ color: 'var(--text-secondary)', marginTop: '6px' }}>
              Executing EfficientNet-B0 inference & calculating Grad-CAM attention heatmap
            </p>
          </div>
        )}

        {/* State 4: Prediction & Grad-CAM Results */}
        {result && !isLoading && (
          <div>
            {/* Model Metadata Tag */}
            <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
              <span style={{ 
                display: 'inline-flex', 
                alignItems: 'center', 
                gap: '6px', 
                background: '#e0f2fe', 
                color: '#0369a1', 
                padding: '4px 12px', 
                borderRadius: '12px', 
                fontSize: '0.85rem',
                fontWeight: '600'
              }}>
                <Cpu size={14} /> Primary Model: {result.model_name || "EfficientNet-B0"}
              </span>
            </div>

            {/* Header: Class & Confidence */}
            <div className="results-header">
              <div className="prediction-badge-wrapper">
                <span className="prediction-label">Prediction:</span>
                <span className={getBadgeClass(result.prediction)}>
                  {result.prediction}
                </span>
              </div>
              <div className="confidence-box">
                <div className="confidence-val">{(result.confidence * 100).toFixed(2)}%</div>
                <div className="confidence-lbl">Model Confidence</div>
              </div>
            </div>

            {/* Class Probabilities Breakdown */}
            {result.probabilities && (
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(3, 1fr)', 
                gap: '12px', 
                margin: '1.2rem 0',
                background: '#f8fafc',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid #e2e8f0'
              }}>
                {Object.entries(result.probabilities).map(([cls, prob]) => (
                  <div key={cls} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.8rem', color: '#64748b', fontWeight: '500' }}>{cls}</div>
                    <div style={{ fontSize: '1rem', fontWeight: '700', color: cls === result.prediction ? '#0284c7' : '#334155' }}>
                      {(prob * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Explainability Banner */}
            <div className="explain-box">
              <Info size={18} style={{ float: 'left', marginRight: '10px', marginTop: '2px' }} />
              <strong>Grad-CAM Explainability:</strong> The heatmap highlights the spatial acoustic regions 
              within the ultrasound scan that contributed most significantly to EfficientNet-B0's prediction.
            </div>

            {/* Image Comparison Grid */}
            <div className="image-grid">
              <div className="image-card">
                <div className="image-card-title">
                  <FileImage size={18} /> Original Ultrasound Scan
                </div>
                <img src={previewUrl} alt="Original Scan" className="image-display" />
              </div>

              <div className="image-card">
                <div className="image-card-title">
                  <Eye size={18} /> Grad-CAM Attention Map
                </div>
                <img src={result.gradcam_image} alt="Grad-CAM Explanation" className="image-display" />
              </div>
            </div>

            {/* Actions */}
            <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
              <button className="btn-reset" onClick={handleReset}>
                <RefreshCw size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
                Analyze Another Ultrasound Scan
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Research Disclaimer Footer */}
      <footer className="disclaimer-footer">
        <span>
          This system is developed for academic/research purposes and is not a substitute for professional medical diagnosis.
        </span>
      </footer>
    </div>
  );
}
