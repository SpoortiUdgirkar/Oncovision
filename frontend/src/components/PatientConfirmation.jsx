import React, { useState } from 'react';
import PatientHistoryModal from './PatientHistoryModal';
import { 
  UserCheck, 
  Activity, 
  History, 
  ArrowLeft 
} from 'lucide-react';

export default function PatientConfirmation({ patient, onContinueToScan, onReturnToPortal }) {
  const [showHistoryModal, setShowHistoryModal] = useState(false);

  if (!patient) return null;

  return (
    <div style={{ maxWidth: '680px', margin: '0 auto' }}>
      <div style={{
        background: 'rgba(16, 185, 129, 0.12)',
        border: '1px solid rgba(52, 211, 153, 0.4)',
        borderRadius: '14px',
        padding: '2rem',
        textAlign: 'center'
      }}>
        {/* Success Icon */}
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          background: 'rgba(16, 185, 129, 0.2)',
          color: '#34d399',
          marginBottom: '1rem'
        }}>
          <UserCheck size={36} />
        </div>

        <h2 style={{ fontSize: '1.6rem', color: '#f8fafc', marginBottom: '0.4rem' }}>
          Patient Selected &amp; Confirmed
        </h2>
        <p style={{ color: '#94a3b8', fontSize: '0.95rem', marginBottom: '1.5rem' }}>
          This patient record is linked to your current clinician session. Ready for ultrasound scan analysis.
        </p>

        {/* Patient Details Card */}
        <div style={{
          background: 'rgba(15, 23, 42, 0.7)',
          border: '1px solid var(--panel-border)',
          borderRadius: '12px',
          padding: '1.4rem',
          textAlign: 'left',
          marginBottom: '1.75rem'
        }}>
          <div style={{
            display: 'flex',
            justify: 'space-between',
            alignItems: 'center',
            paddingBottom: '12px',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
            marginBottom: '14px'
          }}>
            <span style={{ fontSize: '0.85rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '600' }}>
              Patient ID
            </span>
            <span style={{
              background: '#0284c7',
              color: '#ffffff',
              padding: '4px 14px',
              borderRadius: '6px',
              fontWeight: '700',
              fontSize: '1.1rem',
              letterSpacing: '0.05em'
            }}>
              {patient.patient_code}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.95rem' }}>
            <div>
              <span style={{ color: '#64748b' }}>Full Name: </span>
              <strong style={{ color: '#f8fafc' }}>{patient.full_name}</strong>
            </div>
            <div>
              <span style={{ color: '#64748b' }}>Age: </span>
              <strong style={{ color: '#f8fafc' }}>{patient.age} years</strong>
            </div>
            <div>
              <span style={{ color: '#64748b' }}>Gender: </span>
              <strong style={{ color: '#f8fafc' }}>{patient.gender}</strong>
            </div>
            <div>
              <span style={{ color: '#64748b' }}>Registered Date: </span>
              <span style={{ color: '#cbd5e1' }}>
                {patient.created_at ? new Date(patient.created_at).toLocaleDateString() : 'Active Session'}
              </span>
            </div>
          </div>

          {patient.notes && (
            <div style={{ marginTop: '14px', paddingTop: '12px', borderTop: '1px solid rgba(255, 255, 255, 0.08)', fontSize: '0.9rem' }}>
              <span style={{ color: '#64748b' }}>Clinical Notes: </span>
              <span style={{ color: '#cbd5e1' }}>{patient.notes}</span>
            </div>
          )}

          <div style={{ marginTop: '12px', fontSize: '0.8rem', color: '#64748b', fontFamily: 'monospace' }}>
            Database UUID: {patient.id}
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center' }}>
          {/* PRIMARY CTA: Continue to Ultrasound Scan */}
          <button
            className="btn-analyze"
            onClick={onContinueToScan}
            style={{
              width: '100%',
              maxWidth: '420px',
              justify: 'center',
              padding: '12px 24px',
              fontSize: '1.05rem',
              fontWeight: '700'
            }}
          >
            <Activity size={20} /> Continue to Ultrasound Scan
          </button>

          <div style={{ display: 'flex', gap: '12px', marginTop: '4px' }}>
            {/* View Scan History */}
            <button
              onClick={() => setShowHistoryModal(true)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                background: 'rgba(255, 255, 255, 0.08)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                color: '#cbd5e1',
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '0.88rem',
                cursor: 'pointer',
                fontWeight: '600'
              }}
            >
              <History size={16} /> View Patient History
            </button>

            {/* Select Different Patient / Return to Portal */}
            <button
              onClick={onReturnToPortal}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                background: 'transparent',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                color: '#94a3b8',
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '0.88rem',
                cursor: 'pointer'
              }}
            >
              <ArrowLeft size={16} /> Return to Patient Portal
            </button>
          </div>
        </div>
      </div>

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
