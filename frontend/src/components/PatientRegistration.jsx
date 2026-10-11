import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { UserPlus, CheckCircle, AlertCircle, RefreshCw, Activity, FileText, User } from 'lucide-react';
import { API_BASE_URL } from '../services/api';

export default function PatientRegistration({ onSelectPatientForScan }) {
  const { accessToken } = useAuth();
  const [fullName, setFullName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('Female');
  const [notes, setNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [registeredPatient, setRegisteredPatient] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    const parsedAge = parseInt(age, 10);
    if (isNaN(parsedAge) || parsedAge < 0 || parsedAge > 120) {
      setError('Please enter a valid age between 0 and 120.');
      return;
    }

    if (!fullName.trim()) {
      setError('Patient full name is required.');
      return;
    }

    setIsSubmitting(true);

    try {
      const headers = {
        'Content-Type': 'application/json',
      };
      if (accessToken) {
        headers['Authorization'] = `Bearer ${accessToken}`;
      }

      const response = await fetch(`${API_BASE_URL}/patients`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          full_name: fullName.trim(),
          age: parsedAge,
          gender,
          notes: notes.trim() || null,
        }),
      });

      const data = await response.json();

      if (response.status === 401) {
        throw new Error(data.detail || 'Authentication token expired or missing. Please log in again.');
      }

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to register patient in database.');
      }

      setRegisteredPatient(data.patient);
    } catch (err) {
      console.error('Patient Registration Error:', err);
      setError(err.message || 'An error occurred during patient registration.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleResetForm = () => {
    setFullName('');
    setAge('');
    setGender('Female');
    setNotes('');
    setError(null);
    setRegisteredPatient(null);
  };

  return (
    <div style={{ maxWidth: '680px', margin: '0 auto' }}>
      {/* Registration Success Card */}
      {registeredPatient ? (
        <div style={{
          background: 'rgba(16, 185, 129, 0.12)',
          border: '1px solid rgba(52, 211, 153, 0.4)',
          borderRadius: '12px',
          padding: '2rem',
          textAlign: 'center'
        }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '56px',
            height: '56px',
            borderRadius: '50%',
            background: 'rgba(16, 185, 129, 0.2)',
            color: '#34d399',
            marginBottom: '1rem'
          }}>
            <CheckCircle size={32} />
          </div>

          <h2 style={{ fontSize: '1.5rem', color: '#f8fafc', marginBottom: '0.5rem' }}>
            Patient Registered Successfully
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '0.95rem', marginBottom: '1.5rem' }}>
            A unique server-side Patient Code has been generated and saved to the patient database.
          </p>

          {/* Generated Patient Details Box */}
          <div style={{
            background: 'rgba(15, 23, 42, 0.6)',
            border: '1px solid var(--panel-border)',
            borderRadius: '10px',
            padding: '1.25rem',
            textAlign: 'left',
            marginBottom: '1.75rem'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              paddingBottom: '12px',
              borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
              marginBottom: '12px'
            }}>
              <span style={{ fontSize: '0.85rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '600' }}>
                Generated Patient ID
              </span>
              <span style={{
                background: '#0284c7',
                color: '#ffffff',
                padding: '4px 12px',
                borderRadius: '6px',
                fontWeight: '700',
                fontSize: '1.05rem',
                letterSpacing: '0.05em'
              }}>
                {registeredPatient.patient_code}
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.92rem' }}>
              <div>
                <span style={{ color: '#64748b' }}>Full Name: </span>
                <strong style={{ color: '#f8fafc' }}>{registeredPatient.full_name}</strong>
              </div>
              <div>
                <span style={{ color: '#64748b' }}>Age: </span>
                <strong style={{ color: '#f8fafc' }}>{registeredPatient.age} years</strong>
              </div>
              <div>
                <span style={{ color: '#64748b' }}>Gender: </span>
                <strong style={{ color: '#f8fafc' }}>{registeredPatient.gender}</strong>
              </div>
              <div>
                <span style={{ color: '#64748b' }}>Database UUID: </span>
                <span style={{ color: '#94a3b8', fontSize: '0.8rem', fontFamily: 'monospace' }}>
                  {registeredPatient.id ? `${registeredPatient.id.substring(0, 8)}...` : 'N/A'}
                </span>
              </div>
            </div>

            {registeredPatient.notes && (
              <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid rgba(255, 255, 255, 0.08)', fontSize: '0.9rem' }}>
                <span style={{ color: '#64748b' }}>Clinical Notes: </span>
                <span style={{ color: '#cbd5e1' }}>{registeredPatient.notes}</span>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
            <button className="btn-analyze" onClick={() => onSelectPatientForScan && onSelectPatientForScan(registeredPatient)}>
              <Activity size={18} /> Analyze Scan for {registeredPatient.patient_code}
            </button>
            <button className="btn-reset" onClick={handleResetForm}>
              <RefreshCw size={16} /> Register Another Patient
            </button>
          </div>
        </div>
      ) : (
        /* Patient Registration Form */
        <div>
          <div style={{ marginBottom: '1.5rem', textAlign: 'center' }}>
            <h2 style={{ fontSize: '1.5rem', color: '#f8fafc', marginBottom: '6px' }}>
              New Patient Registration
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.95rem' }}>
              Create a record in Supabase to link future breast ultrasound scans & Grad-CAM heatmaps
            </p>
          </div>

          {error && (
            <div className="error-alert">
              <AlertCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
            {/* Full Name */}
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                Full Name <span style={{ color: '#ef4444' }}>*</span>
              </label>
              <div style={{ position: 'relative' }}>
                <User size={18} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: '#64748b' }} />
                <input
                  type="text"
                  required
                  placeholder="e.g. Jane Doe"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px 10px 38px',
                    borderRadius: '8px',
                    border: '1px solid var(--panel-border)',
                    background: 'rgba(15, 23, 42, 0.6)',
                    color: '#f8fafc',
                    fontSize: '0.95rem',
                    outline: 'none'
                  }}
                />
              </div>
            </div>

            {/* Grid: Age & Gender */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              {/* Age */}
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                  Age (Years 0-120) <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <input
                  type="number"
                  required
                  min="0"
                  max="120"
                  placeholder="e.g. 45"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: '8px',
                    border: '1px solid var(--panel-border)',
                    background: 'rgba(15, 23, 42, 0.6)',
                    color: '#f8fafc',
                    fontSize: '0.95rem',
                    outline: 'none'
                  }}
                />
              </div>

              {/* Gender */}
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                  Gender <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <select
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: '8px',
                    border: '1px solid var(--panel-border)',
                    background: '#0f172a',
                    color: '#f8fafc',
                    fontSize: '0.95rem',
                    outline: 'none'
                  }}
                >
                  <option value="Female">Female</option>
                  <option value="Male">Male</option>
                  <option value="Other">Other</option>
                  <option value="Prefer not to say">Prefer not to say</option>
                </select>
              </div>
            </div>

            {/* Clinical Notes */}
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: '600', color: '#cbd5e1', marginBottom: '6px' }}>
                Clinical Notes / Medical Observations (Optional)
              </label>
              <div style={{ position: 'relative' }}>
                <FileText size={18} style={{ position: 'absolute', left: '12px', top: '14px', color: '#64748b' }} />
                <textarea
                  rows="3"
                  placeholder="e.g. Palpable mass upper outer quadrant, routine screening"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px 10px 38px',
                    borderRadius: '8px',
                    border: '1px solid var(--panel-border)',
                    background: 'rgba(15, 23, 42, 0.6)',
                    color: '#f8fafc',
                    fontSize: '0.95rem',
                    outline: 'none',
                    resize: 'vertical'
                  }}
                />
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isSubmitting}
              className="btn-analyze"
              style={{ justifyContent: 'center', marginTop: '0.5rem', opacity: isSubmitting ? 0.7 : 1 }}
            >
              {isSubmitting ? (
                <>
                  <div className="spinner" style={{ width: '18px', height: '18px', borderTopColor: '#ffffff' }}></div>
                  Registering Patient...
                </>
              ) : (
                <>
                  <UserPlus size={18} /> Register Patient &amp; Generate ID
                </>
              )}
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
