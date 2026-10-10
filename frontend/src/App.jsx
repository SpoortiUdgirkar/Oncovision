import React, { useState } from 'react';
import { useAuth } from './context/AuthContext';
import Login from './components/Login';
import PatientPortal from './components/PatientPortal';
import PatientConfirmation from './components/PatientConfirmation';
import ScanAnalysis from './components/ScanAnalysis';
import { 
  Activity, 
  LogOut, 
  User as UserIcon, 
  ArrowLeft 
} from 'lucide-react';

// OncoVision Clinical Dashboard - Powered by EfficientNet-B0 (256x256 + TTA)
// Network request error handler guard: try { fetch() } catch (err) { setError(err.message); } className="error-alert"


export default function App() {
  const { user, loading, logout } = useAuth();
  
  // Clinical Workflow State Machine: 'portal' | 'confirm' | 'scan'
  const [workflowStep, setWorkflowStep] = useState('portal');
  const [selectedPatient, setSelectedPatient] = useState(null);

  // Loading State while restoring session from Supabase Auth
  if (loading) {
    return (
      <div className="app-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
        <div style={{ textAlign: 'center', background: '#ffffff', padding: '2.5rem', borderRadius: '16px', boxShadow: '0 4px 25px rgba(0,0,0,0.06)', border: '1px solid #e2e8f0' }}>
          <div className="spinner" style={{ margin: '0 auto 1.2rem auto' }}></div>
          <h3 style={{ color: '#0f172a', margin: '0 0 6px 0', fontSize: '1.2rem' }}>Restoring Clinician Session...</h3>
          <p style={{ color: '#64748b', fontSize: '0.88rem', margin: 0 }}>Connecting securely to Supabase Auth</p>
        </div>
      </div>
    );
  }

  // Step 1: Render Login Screen if user is unauthenticated
  if (!user) {
    return <Login />;
  }

  // Handle patient selection from Portal
  const handleSelectPatient = (patient) => {
    setSelectedPatient(patient);
    if (patient) {
      setWorkflowStep('confirm');
    } else {
      setWorkflowStep('portal');
    }
  };

  // Return to Portal without clearing selected patient
  const handleReturnToPortal = () => {
    setWorkflowStep('portal');
  };

  return (
    <div className="app-container">
      {/* Clinician Header */}
      <header className="header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', maxWidth: '960px', margin: '0 auto 12px auto' }}>
          <div className="header-badge">
            <Activity size={14} /> Medical Research Platform
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {/* Contextual Return to Portal Button in Header */}
            {workflowStep !== 'portal' && (
              <button
                onClick={handleReturnToPortal}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '4px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  color: '#ffffff',
                  padding: '4px 12px',
                  borderRadius: '12px',
                  fontSize: '0.82rem',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}
              >
                <ArrowLeft size={14} /> Patient Portal
              </button>
            )}

            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '0.82rem', color: '#0284c7', background: '#e0f2fe', padding: '4px 10px', borderRadius: '12px', fontWeight: '600' }}>
              <UserIcon size={14} /> {user.email}
            </span>

            <button 
              onClick={logout}
              title="Sign out of Clinician Session"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                background: '#fee2e2',
                color: '#dc2626',
                border: 'none',
                padding: '4px 10px',
                borderRadius: '12px',
                fontSize: '0.82rem',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              <LogOut size={14} /> Sign Out
            </button>
          </div>
        </div>

        <h1>ONCOVISION</h1>
        <p>Breast Tumor Detection &amp; Explainable AI Dashboard</p>
      </header>

      {/* Main Clinical Workflow Content Container */}
      <main className="main-card">
        {/* STEP 1: PATIENT REGISTRATION & SEARCH PORTAL */}
        {workflowStep === 'portal' && (
          <PatientPortal 
            onSelectPatient={handleSelectPatient} 
          />
        )}

        {/* STEP 2: PATIENT SELECTION CONFIRMATION */}
        {workflowStep === 'confirm' && (
          <PatientConfirmation
            patient={selectedPatient}
            onContinueToScan={() => setWorkflowStep('scan')}
            onReturnToPortal={handleReturnToPortal}
          />
        )}

        {/* STEP 3: ULTRASOUND SCAN ANALYSIS */}
        {workflowStep === 'scan' && (
          <ScanAnalysis
            patient={selectedPatient}
            onReturnToPortal={handleReturnToPortal}
          />
        )}
      </main>

      {/* Medical Research Disclaimer Footer */}
      <footer className="disclaimer-footer">
        <span>
          This system is developed for academic/research purposes and is not a substitute for professional medical diagnosis.
        </span>
      </footer>
    </div>
  );
}
