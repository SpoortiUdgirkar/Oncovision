import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { 
  History, 
  Calendar, 
  Eye, 
  FileImage, 
  AlertCircle, 
  X, 
  RefreshCw,
  Download
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function PatientHistoryModal({ patient, onClose }) {
  const { accessToken } = useAuth();
  const [scans, setScans] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloadingScanId, setDownloadingScanId] = useState(null);
  const [pdfError, setPdfError] = useState(null);

  const handleDownloadPdf = async (scanId, patientCode) => {
    setDownloadingScanId(scanId);
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
      setDownloadingScanId(null);
    }
  };

  const fetchScans = async () => {
    if (!patient || !patient.id) return;
    setIsLoading(true);
    setError(null);
    try {
      const headers = {};
      if (accessToken) {
        headers['Authorization'] = `Bearer ${accessToken}`;
      }
      const response = await fetch(`${API_BASE_URL}/patients/${patient.id}/scans`, { headers });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to retrieve scan history from database.');
      }

      setScans(data.scans || []);
    } catch (err) {
      console.error('Fetch Scan History Error:', err);
      setError(err.message || 'Could not load patient scan history.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchScans();
  }, [patient?.id]);

  const getBadgeClass = (pred) => {
    const p = (pred || '').toLowerCase();
    if (p === 'normal') return 'badge-result badge-normal';
    if (p === 'benign') return 'badge-result badge-benign';
    if (p === 'malignant') return 'badge-result badge-malignant';
    return 'badge-result';
  };

  if (!patient) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.78)',
      backdropFilter: 'blur(5px)',
      display: 'flex',
      justify: 'center',
      alignItems: 'center',
      zIndex: 1000,
      padding: '20px'
    }}>
      <div style={{
        background: '#0f172a',
        border: '1px solid var(--panel-border)',
        borderRadius: '16px',
        width: '100%',
        maxWidth: '900px',
        maxHeight: '90vh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        boxShadow: '0 25px 60px rgba(0, 0, 0, 0.6)'
      }}>
        {/* Modal Header */}
        <div style={{
          padding: '16px 24px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          display: 'flex',
          justify: 'space-between',
          alignItems: 'center',
          background: 'rgba(15, 23, 42, 0.95)'
        }}>
          <div>
            <h3 style={{ margin: 0, color: '#f8fafc', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <History size={22} style={{ color: '#38bdf8' }} />
              Patient Scan History: <span style={{ color: '#ffffff' }}>{patient.full_name}</span>
              <span style={{ background: '#0284c7', color: '#fff', padding: '2px 10px', borderRadius: '6px', fontSize: '0.85rem', fontWeight: '700' }}>
                {patient.patient_code}
              </span>
            </h3>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              onClick={fetchScans}
              title="Refresh History"
              style={{
                background: 'rgba(255, 255, 255, 0.08)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                color: '#cbd5e1',
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '0.82rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <RefreshCw size={14} /> Refresh
            </button>
            <button
              onClick={onClose}
              style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
            >
              <X size={24} />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '24px', overflowY: 'auto', flex: 1 }}>
          {error && (
            <div className="error-alert" style={{ marginBottom: '1rem' }}>
              <AlertCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
              {error}
            </div>
          )}
          {pdfError && (
            <div className="error-alert" style={{ marginBottom: '1rem' }}>
              <AlertCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
              {pdfError}
            </div>
          )}

          {isLoading ? (
            <div style={{ textAlign: 'center', padding: '4rem 0' }}>
              <div className="spinner" style={{ margin: '0 auto 16px auto' }}></div>
              <p style={{ color: '#94a3b8', fontSize: '0.95rem' }}>
                Loading patient history…
              </p>
            </div>
          ) : scans.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '4rem 0', color: '#94a3b8' }}>
              <FileImage size={48} style={{ margin: '0 auto 16px auto', opacity: 0.4 }} />
              <h4 style={{ color: '#cbd5e1', margin: '0 0 8px 0', fontSize: '1.1rem' }}>No Historical Scans Found</h4>
              <p style={{ fontSize: '0.9rem', margin: 0 }}>
                No ultrasound scans have been persisted for <strong>{patient.full_name}</strong> ({patient.patient_code}) yet.
              </p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: '#38bdf8', fontSize: '0.85rem', fontWeight: '600', paddingBottom: '4px' }}>
                <span>Showing saved scan records ({scans.length})</span>
                <span style={{ color: '#94a3b8', fontWeight: 'normal' }}>Newest to oldest</span>
              </div>
              {scans.map((scan, index) => {
                const scanDateStr = new Date(scan.created_at).toLocaleString();
                return (
                  <div
                    key={scan.id || index}
                    style={{
                      background: 'rgba(30, 41, 59, 0.65)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '12px',
                      padding: '1.25rem',
                      boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)'
                    }}
                  >
                    {/* Scan Header */}
                    <div style={{
                      display: 'flex',
                      justify: 'space-between',
                      alignItems: 'center',
                      paddingBottom: '12px',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
                      marginBottom: '14px',
                      flexWrap: 'wrap',
                      gap: '10px'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <span className={getBadgeClass(scan.predicted_class)} style={{ fontSize: '0.9rem', padding: '4px 14px' }}>
                          {scan.predicted_class}
                        </span>
                        <span style={{ color: '#38bdf8', fontSize: '0.88rem', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <Calendar size={15} /> {scanDateStr}
                        </span>
                        {index === 0 && (
                          <span style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399', border: '1px solid rgba(52, 211, 153, 0.3)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: '700' }}>
                            LATEST SCAN
                          </span>
                        )}
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <div style={{ fontSize: '0.8rem', color: '#64748b', fontFamily: 'monospace' }}>
                          Scan ID: <strong style={{ color: '#94a3b8' }}>{scan.id ? scan.id.substring(0, 13) : 'N/A'}</strong>
                        </div>
                        {scan.id && (
                          <button
                            onClick={() => handleDownloadPdf(scan.id, patient.patient_code)}
                            disabled={downloadingScanId === scan.id}
                            title="Download PDF Report for this scan"
                            style={{
                              background: '#0284c7',
                              color: '#ffffff',
                              border: 'none',
                              padding: '5px 12px',
                              borderRadius: '6px',
                              fontSize: '0.8rem',
                              fontWeight: '600',
                              cursor: downloadingScanId === scan.id ? 'not-allowed' : 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                              opacity: downloadingScanId === scan.id ? 0.7 : 1
                            }}
                          >
                            {downloadingScanId === scan.id ? (
                              <>
                                <div className="spinner" style={{ width: '12px', height: '12px', borderTopColor: '#ffffff' }}></div>
                                Downloading...
                              </>
                            ) : (
                              <>
                                <Download size={14} /> Download PDF
                              </>
                            )}
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Probabilities Breakdown Grid */}
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(3, 1fr)',
                      gap: '10px',
                      background: 'rgba(15, 23, 42, 0.6)',
                      padding: '10px 14px',
                      borderRadius: '8px',
                      border: '1px solid rgba(255, 255, 255, 0.06)',
                      marginBottom: '16px'
                    }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '0.78rem', color: '#64748b' }}>Normal Prob</div>
                        <div style={{ fontSize: '0.95rem', fontWeight: '700', color: scan.predicted_class === 'Normal' ? '#38bdf8' : '#cbd5e1' }}>
                          {(scan.normal_probability * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '0.78rem', color: '#64748b' }}>Benign Prob</div>
                        <div style={{ fontSize: '0.95rem', fontWeight: '700', color: scan.predicted_class === 'Benign' ? '#38bdf8' : '#cbd5e1' }}>
                          {(scan.benign_probability * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '0.78rem', color: '#64748b' }}>Malignant Prob</div>
                        <div style={{ fontSize: '0.95rem', fontWeight: '700', color: scan.predicted_class === 'Malignant' ? '#ef4444' : '#cbd5e1' }}>
                          {(scan.malignant_probability * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    {/* Requirement 3: BOTH Images Displayed Side-by-Side */}
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: '16px'
                    }}>
                      {/* Left: Original Ultrasound Image */}
                      <div>
                        <div style={{ fontSize: '0.85rem', color: '#cbd5e1', fontWeight: '600', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <FileImage size={16} style={{ color: '#0284c7' }} /> Original Ultrasound Scan
                        </div>
                        {scan.original_image_url ? (
                          <img
                            src={scan.original_image_url}
                            alt={`Original Scan ${scanDateStr}`}
                            style={{
                              width: '100%',
                              borderRadius: '8px',
                              border: '1px solid rgba(255, 255, 255, 0.12)',
                              background: '#0f172a',
                              maxHeight: '280px',
                              objectFit: 'contain'
                            }}
                            onError={(e) => {
                              e.target.style.display = 'none';
                              if (e.target.nextSibling) e.target.nextSibling.style.display = 'block';
                            }}
                          />
                        ) : null}
                        <div style={{
                          display: scan.original_image_url ? 'none' : 'block',
                          padding: '30px 15px',
                          background: '#0f172a',
                          borderRadius: '8px',
                          textAlign: 'center',
                          color: '#f87171',
                          fontSize: '0.85rem',
                          border: '1px solid rgba(239, 68, 68, 0.2)'
                        }}>
                          <AlertCircle size={20} style={{ margin: '0 auto 6px auto' }} />
                          Failed to load saved image from storage
                        </div>
                      </div>

                      {/* Right: Corresponding Grad-CAM Explanation Image */}
                      <div>
                        <div style={{ fontSize: '0.85rem', color: '#cbd5e1', fontWeight: '600', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <Eye size={16} style={{ color: '#38bdf8' }} /> Grad-CAM Explanation
                        </div>
                        {scan.gradcam_image_url ? (
                          <img
                            src={scan.gradcam_image_url}
                            alt={`Grad-CAM Overlay ${scanDateStr}`}
                            style={{
                              width: '100%',
                              borderRadius: '8px',
                              border: '1px solid rgba(255, 255, 255, 0.12)',
                              background: '#0f172a',
                              maxHeight: '280px',
                              objectFit: 'contain'
                            }}
                            onError={(e) => {
                              e.target.style.display = 'none';
                              if (e.target.nextSibling) e.target.nextSibling.style.display = 'block';
                            }}
                          />
                        ) : null}
                        <div style={{
                          display: scan.gradcam_image_url ? 'none' : 'block',
                          padding: '30px 15px',
                          background: '#0f172a',
                          borderRadius: '8px',
                          textAlign: 'center',
                          color: '#f87171',
                          fontSize: '0.85rem',
                          border: '1px solid rgba(239, 68, 68, 0.2)'
                        }}>
                          <AlertCircle size={20} style={{ margin: '0 auto 6px auto' }} />
                          Failed to load saved Grad-CAM heatmap from storage
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div style={{
          padding: '14px 24px',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          display: 'flex',
          justify: 'space-between',
          alignItems: 'center',
          background: 'rgba(15, 23, 42, 0.95)'
        }}>
          <span style={{ fontSize: '0.82rem', color: '#94a3b8' }}>
            Review previous ultrasound scans and their AI-generated explanations, arranged from newest to oldest.
          </span>
          <button
            className="btn-reset"
            style={{ padding: '6px 18px', fontSize: '0.88rem' }}
            onClick={onClose}
          >
            Close History
          </button>
        </div>
      </div>
    </div>
  );
}
