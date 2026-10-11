import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import PatientHistoryModal from './PatientHistoryModal';
import { 
  Upload, 
  FileImage, 
  Activity, 
  RefreshCw, 
  AlertCircle, 
  Info, 
  Eye, 
  Cpu, 
  UserCheck, 
  ArrowLeft, 
  History,
  CheckCircle,
  Download
} from 'lucide-react';
import { API_BASE_URL } from '../services/api';

export default function ScanAnalysis({ patient, onReturnToPortal }) {
  const { accessToken } = useAuth();

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisStage, setAnalysisStage] = useState('Initializing scan analysis...');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [pdfError, setPdfError] = useState(null);

  useEffect(() => {
    let timer;
    if (isLoading) {
      setElapsedSeconds(0);
      setAnalysisStage('Uploading scan and preparing deep learning model...');
      timer = setInterval(() => {
        setElapsedSeconds((prev) => {
          const next = prev + 1;
          if (next >= 10) {
            setAnalysisStage('Uploading scan artifacts & saving medical record to database...');
          } else if (next >= 3) {
            setAnalysisStage('Executing EfficientNet-B0 inference & computing Grad-CAM explainability...');
          }
          return next;
        });
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isLoading]);

  const handleDownloadPdf = async (scanId, patientCode) => {
    setIsDownloadingPdf(true);
    setPdfError(null);
    try {
      const headers = {};
      if (accessToken) {
        headers['Authorization'] = `Bearer ${accessToken}`;
      }
      const response = await fetch(`${API_BASE_URL}/scans/${scanId}/report.pdf`, { headers });
      
      if (!response.ok) {
        let errorMsg = 'Failed to download PDF report.';
        try {
          const errorData = await response.json();
          errorMsg = errorData.detail || errorMsg;
        } catch (_) {}
        throw new Error(errorMsg);
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `OncoVision_Report_${patientCode || 'Patient'}_${scanId.substring(0, 8)}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      console.error('PDF Download Error:', err);
      setPdfError(err.message || 'Failed to download PDF report.');
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  if (!patient) {
    return (
      <div style={{ textAlign: 'center', padding: '3rem' }}>
        <div className="error-alert" style={{ marginBottom: '1.5rem' }}>
          <AlertCircle size={20} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
          No patient selected. A valid patient record must be selected before uploading an ultrasound scan.
        </div>
        <button className="btn-analyze" onClick={onReturnToPortal}>
          <ArrowLeft size={18} /> Return to Patient Portal
        </button>
      </div>
    );
  }

  // File Selection
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

  // Drag and Drop
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

  // Analyze & Save Scan
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('patient_id', patient.id);

    try {
      const headers = {};
      if (accessToken) {
        headers['Authorization'] = `Bearer ${accessToken}`;
      }

      let response;
      try {
        response = await fetch(`${API_BASE_URL}/predict`, {
          method: 'POST',
          headers: headers,
          body: formData
        });
      } catch (networkErr) {
        console.error('Scan analysis network connection error:', networkErr);
        throw new Error(
          'Network connection error: Unable to connect to the OncoVision backend server. ' +
          'The server may be waking up from an idle state or experiencing network latency. Please check your connection and try submitting the scan again.'
        );
      }

      let data = null;
      try {
        data = await response.json();
      } catch (_) {
        data = null;
      }

      if (response.status === 401) {
        throw new Error(
          (data && data.detail) ||
          'Your session has expired or the authorization token is invalid. Please log in again.'
        );
      }

      if (response.status === 502 || response.status === 504) {
        throw new Error(
          `The backend service timed out or was temporarily unavailable (HTTP ${response.status}). The server may still be waking up. Please wait a moment and try again.`
        );
      }

      if (!response.ok) {
        const errorDetail = (data && data.detail) || `Prediction request failed with status code ${response.status}.`;
        throw new Error(errorDetail);
      }

      if (!data || !data.prediction) {
        throw new Error('Received an invalid or incomplete response from the prediction service.');
      }

      setResult(data);
    } catch (err) {
      console.error('Prediction API Error:', err);
      setError(err.message || 'Failed to process ultrasound scan.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const getBadgeClass = (pred) => {
    const p = (pred || '').toLowerCase();
    if (p === 'normal') return 'badge-result badge-normal';
    if (p === 'benign') return 'badge-result badge-benign';
    if (p === 'malignant') return 'badge-result badge-malignant';
    return 'badge-result';
  };

  return (
    <div style={{ maxWidth: '880px', margin: '0 auto' }}>
      {/* Linked Patient Header Banner */}
      <div style={{
        display: 'flex',
        justify: 'space-between',
        alignItems: 'center',
        background: 'rgba(2, 132, 199, 0.15)',
        border: '1px solid rgba(56, 189, 248, 0.4)',
        borderRadius: '10px',
        padding: '12px 18px',
        marginBottom: '1.5rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#38bdf8' }}>
          <UserCheck size={20} />
          <span>
            Linked Patient: <strong style={{ color: '#ffffff', fontSize: '1.05rem' }}>{patient.full_name}</strong>{' '}
            <span style={{ background: '#0284c7', color: '#fff', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem', fontWeight: '700' }}>
              {patient.patient_code}
            </span>
          </span>
        </div>

        <button
          onClick={onReturnToPortal}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            background: 'rgba(255, 255, 255, 0.08)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            color: '#cbd5e1',
            padding: '6px 12px',
            borderRadius: '6px',
            fontSize: '0.82rem',
            cursor: 'pointer',
            fontWeight: '600'
          }}
        >
          <ArrowLeft size={14} /> Back to Patient Portal
        </button>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="error-alert" style={{ marginBottom: '1.2rem', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
            <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px', color: '#f87171' }} />
            <div>
              <div style={{ fontWeight: '700', marginBottom: '2px' }}>Scan Analysis Error</div>
              <div style={{ fontSize: '0.88rem', lineHeight: '1.4' }}>{error}</div>
            </div>
          </div>
          <button
            onClick={() => setError(null)}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#f87171',
              cursor: 'pointer',
              fontSize: '0.82rem',
              fontWeight: '600',
              padding: '2px 6px',
              textDecoration: 'underline'
            }}
          >
            Dismiss
          </button>
        </div>
      )}

      {/* State 1: Dropzone */}
      {!selectedFile && !result && (
        <div 
          className={`dropzone ${isDragging ? 'active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <Upload className="dropzone-icon" />
          <h3>Upload Ultrasound Scan Image</h3>
          <p>
            Drag &amp; drop B-mode ultrasound file for <strong>{patient.full_name}</strong> ({patient.patient_code}), or click to browse
          </p>
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

      {/* State 2: Preview & Analyze CTA */}
      {selectedFile && !result && !isLoading && (
        <div className="preview-container">
          <div className="image-preview-wrapper">
            <img src={previewUrl} alt="Ultrasound Scan Preview" className="image-preview" />
          </div>
          <div className="preview-actions">
            <button className="btn-analyze" onClick={handleAnalyze}>
              <Activity size={18} /> Analyze &amp; Save Scan for {patient.patient_code}
            </button>
            <button className="btn-reset" onClick={handleReset}>
              <RefreshCw size={16} /> Choose Different Image
            </button>
          </div>
        </div>
      )}

      {/* State 3: Loading Indicator */}
      {isLoading && (
        <div className="loading-box" style={{ padding: '2.5rem 1.5rem', textAlign: 'center' }}>
          <div className="spinner" style={{ margin: '0 auto 1.2rem auto' }}></div>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '8px' }}>Analyzing Ultrasound Image &amp; Persisting Scan...</h3>
          <p style={{ color: '#38bdf8', fontWeight: '600', fontSize: '0.95rem', marginBottom: '6px' }}>
            {analysisStage}
          </p>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            <span>Time elapsed: <strong>{elapsedSeconds}s</strong></span>
          </div>
          {elapsedSeconds >= 12 && (
            <div style={{
              marginTop: '1.2rem',
              background: 'rgba(2, 132, 199, 0.1)',
              border: '1px solid rgba(56, 189, 248, 0.25)',
              borderRadius: '8px',
              padding: '10px 14px',
              fontSize: '0.82rem',
              color: '#94a3b8',
              maxWidth: '520px',
              margin: '1.2rem auto 0 auto'
            }}>
              <Info size={14} style={{ verticalAlign: 'middle', marginRight: '6px', color: '#38bdf8' }} />
              Notice: If the backend service is waking up from idle, the cold start may take up to 45–60 seconds. Please keep this page open.
            </div>
          )}
        </div>
      )}

      {/* State 4: Prediction & Grad-CAM Results */}
      {result && !isLoading && (
        <div>
          {/* Saved Scan Confirmation Badge */}
          {result.saved_scan ? (
            <div style={{
              background: 'rgba(16, 185, 129, 0.15)',
              border: '1px solid rgba(52, 211, 153, 0.4)',
              color: '#34d399',
              borderRadius: '8px',
              padding: '12px 18px',
              textAlign: 'center',
              marginBottom: '1.2rem',
              fontSize: '0.92rem',
              fontWeight: '600',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}>
              <CheckCircle size={18} />
              <span>
                Scan &amp; Grad-CAM saved to patient record for <strong>{patient.patient_code}</strong> (Scan ID: {result.saved_scan.id.substring(0, 8)}...)
              </span>
            </div>
          ) : (
            <div style={{
              background: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid rgba(239, 68, 68, 0.4)',
              color: '#f87171',
              borderRadius: '8px',
              padding: '12px 18px',
              textAlign: 'center',
              marginBottom: '1.2rem',
              fontSize: '0.92rem',
              fontWeight: '600',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}>
              <AlertCircle size={18} />
              <span>
                Notice: Scan was completed for preview but could not be saved to patient record.
              </span>
            </div>
          )}

          {pdfError && (
            <div className="error-alert" style={{ marginBottom: '1.2rem', fontSize: '0.88rem' }}>
              <AlertCircle size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
              {pdfError}
            </div>
          )}

          {/* Model Tag */}
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
              <Cpu size={14} /> Model: {result.model_name || "EfficientNet-B0"}
            </span>
          </div>

          {/* Results Header: Class & Confidence */}
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

          {/* Side-by-Side Image Grid */}
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

          {/* Bottom Action Bar */}
          <div style={{ textAlign: 'center', marginTop: '1.5rem', display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
            {result.saved_scan && (
              <button
                onClick={() => handleDownloadPdf(result.saved_scan.id, patient.patient_code)}
                disabled={isDownloadingPdf}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: '#0284c7',
                  color: '#ffffff',
                  border: 'none',
                  padding: '8px 18px',
                  borderRadius: '8px',
                  fontSize: '0.88rem',
                  fontWeight: '600',
                  cursor: isDownloadingPdf ? 'not-allowed' : 'pointer',
                  opacity: isDownloadingPdf ? 0.7 : 1
                }}
              >
                {isDownloadingPdf ? (
                  <>
                    <div className="spinner" style={{ width: '14px', height: '14px', borderTopColor: '#ffffff' }}></div>
                    Downloading Report...
                  </>
                ) : (
                  <>
                    <Download size={16} /> Download PDF Report
                  </>
                )}
              </button>
            )}
            <button className="btn-reset" onClick={handleReset}>
              <RefreshCw size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
              Analyze Another Scan for {patient.patient_code}
            </button>
            <button
              className="btn-analyze"
              style={{ width: 'auto', padding: '0 16px', fontSize: '0.88rem' }}
              onClick={() => setShowHistoryModal(true)}
            >
              <History size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
              View Patient Scan History
            </button>
            <button
              onClick={onReturnToPortal}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                background: 'transparent',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                color: '#cbd5e1',
                padding: '0 16px',
                borderRadius: '8px',
                fontSize: '0.88rem',
                cursor: 'pointer'
              }}
            >
              <ArrowLeft size={16} /> Back to Patient Portal
            </button>
          </div>
        </div>
      )}

      {/* SCAN HISTORY MODAL */}
      {showHistoryModal && (
        <PatientHistoryModal
          patient={patient}
          onClose={() => setShowHistoryModal(false)}
        />
      )}
    </div>
  );
}
