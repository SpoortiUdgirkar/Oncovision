import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Activity, Lock, Mail, AlertCircle, LogIn, ShieldCheck } from 'lucide-react';

export default function Login() {
  const { login, authError, clearAuthError } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) {
      return;
    }

    setIsSubmitting(true);
    clearAuthError();

    const result = await login(email, password);
    setIsSubmitting(false);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-badge">
          <Activity size={14} /> Medical Research Platform
        </div>
        <h1>ONCOVISION</h1>
        <p>Clinician Portal & Ultrasound Diagnostics Authentication</p>
      </header>

      <main className="main-card" style={{ maxWidth: '480px', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '48px',
            height: '48px',
            borderRadius: '50%',
            background: '#e0f2fe',
            color: '#0284c7',
            marginBottom: '10px'
          }}>
            <ShieldCheck size={26} />
          </div>
          <h2 style={{ fontSize: '1.4rem', color: '#0f172a', margin: '4px 0' }}>
            Clinician Sign In
          </h2>
          <p style={{ fontSize: '0.88rem', color: '#64748b' }}>
            Authenticate with your Supabase credentials to access tumor analysis & patient diagnostics
          </p>
        </div>

        {/* Auth Error Banner */}
        {authError && (
          <div className="error-alert" style={{ marginBottom: '1.2rem' }}>
            <AlertCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
            {authError}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {/* Email Input */}
          <div style={{ marginBottom: '1.2rem' }}>
            <label style={{
              display: 'block',
              fontSize: '0.85rem',
              fontWeight: '600',
              color: '#334155',
              marginBottom: '6px'
            }}>
              Clinician Email Address
            </label>
            <div style={{ position: 'relative' }}>
              <Mail 
                size={18} 
                style={{ 
                  position: 'absolute', 
                  left: '12px', 
                  top: '50%', 
                  transform: 'translateY(-50%)', 
                  color: '#94a3b8' 
                }} 
              />
              <input 
                type="email"
                required
                placeholder="clinician@hospital.org"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px 12px 10px 38px',
                  borderRadius: '8px',
                  border: '1px solid #cbd5e1',
                  fontSize: '0.95rem',
                  outline: 'none',
                  boxSizing: 'border-box'
                }}
              />
            </div>
          </div>

          {/* Password Input */}
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{
              display: 'block',
              fontSize: '0.85rem',
              fontWeight: '600',
              color: '#334155',
              marginBottom: '6px'
            }}>
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <Lock 
                size={18} 
                style={{ 
                  position: 'absolute', 
                  left: '12px', 
                  top: '50%', 
                  transform: 'translateY(-50%)', 
                  color: '#94a3b8' 
                }} 
              />
              <input 
                type="password"
                required
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px 12px 10px 38px',
                  borderRadius: '8px',
                  border: '1px solid #cbd5e1',
                  fontSize: '0.95rem',
                  outline: 'none',
                  boxSizing: 'border-box'
                }}
              />
            </div>
          </div>

          {/* Submit Button */}
          <button 
            type="submit"
            disabled={isSubmitting}
            className="btn-analyze"
            style={{ width: '100%', justifyContent: 'center', opacity: isSubmitting ? 0.7 : 1 }}
          >
            {isSubmitting ? (
              <>
                <div className="spinner" style={{ width: '16px', height: '16px', borderTopColor: '#ffffff' }}></div>
                Authenticating...
              </>
            ) : (
              <>
                <LogIn size={18} /> Sign In to Dashboard
              </>
            )}
          </button>
        </form>

        {/* Demo Helper Box */}
        <div style={{
          marginTop: '1.5rem',
          padding: '12px',
          background: '#f8fafc',
          borderRadius: '8px',
          border: '1px solid #e2e8f0',
          fontSize: '0.82rem',
          color: '#64748b'
        }}>
          <strong>Demo Setup Notice:</strong> Clinician accounts are created in your Supabase Auth dashboard under 
          <em> Authentication &gt; Users &gt; Add User</em>.
        </div>
      </main>

      {/* Footer */}
      <footer className="disclaimer-footer">
        <span>
          Secured with Supabase Auth &amp; Cryptographic Bearer Verification.
        </span>
      </footer>
    </div>
  );
}
