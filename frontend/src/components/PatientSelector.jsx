import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Search, UserCheck, History, Activity, Calendar, Eye, FileImage, AlertCircle, RefreshCw, X, ChevronRight } from 'lucide-react';
import { API_BASE_URL } from '../services/api';

export default function PatientSelector({ selectedPatient, onSelectPatient, onNavigateRegister }) {
  const { accessToken } = useAuth();
  const [searchQuery, setSearchQuery] = useState('');
  const [patients, setPatients] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Patient History State
  const [historyPatient, setHistoryPatient] = useState(null);
  const [patientScans, setPatientScans] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [selectedScanDetail, setSelectedScanDetail] = useState(null);

  // Fetch patients list with optional search query
  const fetchPatients = async (query = '') => {
    setIsLoading(true);
    setError(null);
    try {
      const headers = {};
      if (accessToken) {
        headers['Authorization'] = `Bearer ${accessToken}`;
      }
      const url = query.trim() 
        ? `${API_BASE_URL}/patients?search=${encodeURIComponent(query.trim())}`
        : `${API_BASE_URL}/patients`;
        
      const response = await fetch(url, { headers });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch patients.');
      }

      setPatients(data.patients || []);
    } catch (err) {
      console.error('Fetch Patients Error:', err);
      setError(err.message || 'Could not load patient records.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients('');
  }, []);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    fetchPatients(searchQuery);
  };

  // Fetch scan history for a selected patient
  const fetchPatientScans = async (patient) => {
    setHistoryPatient(patient);
    setIsLoadingHistory(true);
    setSelectedScanDetail(null);
    try {
      const headers = {};
      if (accessToken) {
        headers['Authorization'] = `Bearer ${accessToken}`;
      }
      const response = await fetch(`${API_BASE_URL}/patients/${patient.id}/scans`, { headers });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch scan history.');
      }

      setPatientScans(data.scans || []);
    } catch (err) {
      console.error('Fetch Scans Error:', err);
      setError(`Failed to load history for ${patient.patient_code}: ${err.message}`);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const getBadgeClass = (pred) => {
    const p = (pred || '').toLowerCase();
    if (p === 'normal') return 'badge-result badge-normal';
    if (p === 'benign') return 'badge-result badge-benign';
    if (p === 'malignant') return 'badge-result badge-malignant';
    return 'badge-result';
  };

  return (
    <div style={{ maxWidth: '820px', margin: '0 auto' }}>
      <div style={{ marginBottom: '1.5rem', textAlign: 'center' }}>
        <h2 style={{ fontSize: '1.5rem', color: '#f8fafc', marginBottom: '6px' }}>
          Select Patient for Ultrasound Scan
        </h2>
        <p style={{ color: '#94a3b8', fontSize: '0.95rem' }}>
          Search existing patients by Patient ID (e.g. OV-P-XXXXXX) or Name to link ultrasound predictions & Grad-CAM scans.
        </p>
      </div>

      {/* Selected Patient Banner if active */}
      {selectedPatient && (
        <div style={{
          background: 'rgba(16, 185, 129, 0.12)',
          border: '1px solid rgba(52, 211, 153, 0.4)',
          borderRadius: '10px',
          padding: '14px 18px',
          marginBottom: '1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#34d399', fontWeight: '700', fontSize: '0.95rem' }}>
              <UserCheck size={20} />
              <span>ACTIVE SELECTED PATIENT</span>
            </div>
            <div style={{ marginTop: '4px', color: '#f8fafc', fontSize: '1.1rem', fontWeight: '600' }}>
              {selectedPatient.full_name} <span style={{ color: '#38bdf8', fontSize: '0.95rem' }}>({selectedPatient.patient_code})</span>
            </div>
            <div style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '2px' }}>
              Age: {selectedPatient.age} yrs | Gender: {selectedPatient.gender}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              className="btn-analyze"
              style={{ padding: '8px 16px', fontSize: '0.85rem' }}
              onClick={() => onSelectPatient(selectedPatient)}
            >
              <Activity size={16} /> Proceed to Scan
            </button>
            <button
              style={{
                background: 'rgba(239, 68, 68, 0.2)',
                color: '#f87171',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '8px',
                padding: '8px 12px',
                fontSize: '0.85rem',
                cursor: 'pointer',
                fontWeight: '600'
              }}
              onClick={() => onSelectPatient(null)}
            >
              Unselect
            </button>
          </div>
        </div>
      )}

      {/* Search Bar */}
      <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: '10px', marginBottom: '1.5rem' }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <Search size={18} style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)', color: '#64748b' }} />
          <input
            type="text"
            placeholder="Search by Patient ID (e.g. OV-P-A8X9K2) or Name (e.g. Jane)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              width: '100%',
              padding: '12px 14px 12px 42px',
              borderRadius: '8px',
              border: '1px solid var(--panel-border)',
              background: 'rgba(15, 23, 42, 0.6)',
              color: '#f8fafc',
              fontSize: '0.95rem',
              outline: 'none'
            }}
          />
        </div>
        <button
          type="submit"
          className="btn-analyze"
          style={{ width: 'auto', padding: '0 20px', fontSize: '0.9rem' }}
        >
          Search
        </button>
        {onNavigateRegister && (
          <button
            type="button"
            className="btn-reset"
            style={{ width: 'auto', padding: '0 16px', fontSize: '0.88rem' }}
            onClick={onNavigateRegister}
          >
            + Register New
          </button>
        )}
      </form>

      {/* Error Alert */}
      {error && (
        <div className="error-alert" style={{ marginBottom: '1rem' }}>
          <AlertCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
          {error}
        </div>
      )}

      {/* Patients List */}
      {isLoading ? (
        <div style={{ textAlign: 'center', padding: '2rem 0' }}>
          <div className="spinner" style={{ margin: '0 auto 10px auto' }}></div>
          <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Searching patient database...</p>
        </div>
      ) : patients.length === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '2.5rem',
          background: 'rgba(15, 23, 42, 0.4)',
          borderRadius: '12px',
          border: '1px solid var(--panel-border)'
        }}>
          <p style={{ color: '#94a3b8', fontSize: '0.95rem', marginBottom: '1rem' }}>
            {searchQuery ? `No patient records found matching "${searchQuery}".` : 'No registered patients found in database.'}
          </p>
          {onNavigateRegister && (
            <button className="btn-analyze" style={{ margin: '0 auto' }} onClick={onNavigateRegister}>
              Register New Patient Now
            </button>
          )}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {patients.map((pt) => {
            const isSelected = selectedPatient && selectedPatient.id === pt.id;
            return (
              <div
                key={pt.id}
                style={{
                  background: isSelected ? 'rgba(2, 132, 199, 0.15)' : 'rgba(15, 23, 42, 0.5)',
                  border: '1px solid',
                  borderColor: isSelected ? '#38bdf8' : 'rgba(255, 255, 255, 0.1)',
                  borderRadius: '10px',
                  padding: '14px 18px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  transition: 'all 0.2s'
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{
                      background: '#0284c7',
                      color: '#ffffff',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontWeight: '700',
                      fontSize: '0.85rem'
                    }}>
                      {pt.patient_code}
                    </span>
                    <strong style={{ color: '#f8fafc', fontSize: '1.05rem' }}>{pt.full_name}</strong>
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '6px' }}>
                    Age: {pt.age} yrs | Gender: {pt.gender} | Registered: {new Date(pt.created_at).toLocaleDateString()}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={() => fetchPatientScans(pt)}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
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
                    <History size={14} /> Scan History
                  </button>

                  <button
                    onClick={() => onSelectPatient(pt)}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      background: isSelected ? '#10b981' : '#0284c7',
                      border: 'none',
                      color: '#ffffff',
                      padding: '6px 14px',
                      borderRadius: '6px',
                      fontSize: '0.82rem',
                      cursor: 'pointer',
                      fontWeight: '600'
                    }}
                  >
                    {isSelected ? <UserCheck size={14} /> : <ChevronRight size={14} />}
                    {isSelected ? 'Selected' : 'Select Patient'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* PATIENT SCAN HISTORY MODAL / EXPANDED VIEW */}
      {historyPatient && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000,
          padding: '20px'
        }}>
          <div style={{
            background: '#0f172a',
            border: '1px solid var(--panel-border)',
            borderRadius: '14px',
            width: '100%',
            maxWidth: '850px',
            maxHeight: '90vh',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.5)'
          }}>
            {/* Modal Header */}
            <div style={{
              padding: '16px 20px',
              borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              background: 'rgba(15, 23, 42, 0.9)'
            }}>
              <div>
                <h3 style={{ margin: 0, color: '#f8fafc', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <History size={20} style={{ color: '#38bdf8' }} />
                  Scan History: {historyPatient.full_name} ({historyPatient.patient_code})
                </h3>
              </div>
              <button
                onClick={() => setHistoryPatient(null)}
                style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
              >
                <X size={22} />
              </button>
            </div>

            {/* Modal Content */}
            <div style={{ padding: '20px', overflowY: 'auto', flex: 1 }}>
              {isLoadingHistory ? (
                <div style={{ textAlign: 'center', padding: '3rem 0' }}>
                  <div className="spinner" style={{ margin: '0 auto 12px auto' }}></div>
                  <p style={{ color: '#94a3b8' }}>Loading patient history…</p>
                </div>
              ) : patientScans.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '3rem 0', color: '#94a3b8' }}>
                  <FileImage size={40} style={{ margin: '0 auto 12px auto', opacity: 0.5 }} />
                  <p>No previous ultrasound scans found for this patient.</p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  {patientScans.map((scan, idx) => (
                    <div
                      key={scan.id || idx}
                      style={{
                        background: 'rgba(30, 41, 59, 0.6)',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        borderRadius: '10px',
                        padding: '14px 18px'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <span className={getBadgeClass(scan.predicted_class)}>
                            {scan.predicted_class}
                          </span>
                          <span style={{ color: '#94a3b8', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <Calendar size={14} /> {new Date(scan.created_at).toLocaleString()}
                          </span>
                        </div>
                        <span style={{ fontSize: '0.78rem', color: '#64748b', fontFamily: 'monospace' }}>
                          Scan ID: {scan.id ? scan.id.substring(0, 12) : 'N/A'}
                        </span>
                      </div>

                      {/* Probabilities Row */}
                      <div style={{ display: 'flex', gap: '16px', fontSize: '0.85rem', color: '#cbd5e1', marginBottom: '10px' }}>
                        <div>Normal: <strong>{(scan.normal_probability * 100).toFixed(1)}%</strong></div>
                        <div>Benign: <strong>{(scan.benign_probability * 100).toFixed(1)}%</strong></div>
                        <div>Malignant: <strong>{(scan.malignant_probability * 100).toFixed(1)}%</strong></div>
                      </div>

                      {/* Reopen Images Action */}
                      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                        <button
                          onClick={() => setSelectedScanDetail(selectedScanDetail === scan ? null : scan)}
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '6px',
                            background: '#0284c7',
                            border: 'none',
                            color: '#ffffff',
                            padding: '6px 12px',
                            borderRadius: '6px',
                            fontSize: '0.82rem',
                            fontWeight: '600',
                            cursor: 'pointer'
                          }}
                        >
                          <Eye size={14} /> {selectedScanDetail === scan ? 'Close Saved Images' : 'Reopen Saved Images & Grad-CAM'}
                        </button>
                      </div>

                      {/* Expanded Image Viewer Panel */}
                      {selectedScanDetail === scan && (
                        <div style={{
                          marginTop: '14px',
                          paddingTop: '14px',
                          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                          display: 'grid',
                          gridTemplateColumns: '1fr 1fr',
                          gap: '12px'
                        }}>
                          <div>
                            <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '6px', fontWeight: '600' }}>
                              Original Ultrasound Scan
                            </div>
                            {scan.original_image_url ? (
                              <img
                                src={scan.original_image_url}
                                alt="Original Scan"
                                style={{ width: '100%', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)' }}
                              />
                            ) : (
                              <div style={{ padding: '20px', background: '#0f172a', borderRadius: '8px', textAlign: 'center', color: '#64748b' }}>
                                Image preview unavailable
                              </div>
                            )}
                          </div>

                          <div>
                            <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '6px', fontWeight: '600' }}>
                              Grad-CAM Heatmap
                            </div>
                            {scan.gradcam_image_url ? (
                              <img
                                src={scan.gradcam_image_url}
                                alt="Grad-CAM Overlay"
                                style={{ width: '100%', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)' }}
                              />
                            ) : (
                              <div style={{ padding: '20px', background: '#0f172a', borderRadius: '8px', textAlign: 'center', color: '#64748b' }}>
                                Grad-CAM preview unavailable
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div style={{
              padding: '12px 20px',
              borderTop: '1px solid rgba(255, 255, 255, 0.1)',
              display: 'flex',
              justify: 'space-between',
              alignItems: 'center',
              background: 'rgba(15, 23, 42, 0.9)'
            }}>
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>
                Chronological order (Newest scan first)
              </span>
              <button
                className="btn-reset"
                style={{ padding: '6px 14px', fontSize: '0.82rem' }}
                onClick={() => setHistoryPatient(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
