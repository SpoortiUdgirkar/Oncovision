import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import PatientHistoryModal from './PatientHistoryModal';
import { 
  Search, 
  UserPlus, 
  UserCheck, 
  History, 
  AlertTriangle, 
  AlertCircle, 
  User, 
  FileText,
  ChevronLeft,
  ChevronRight,
  X
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function PatientPortal({ onSelectPatient }) {
  const { accessToken } = useAuth();

  // Search & Pagination State
  const [searchQuery, setSearchQuery] = useState('');
  const [patients, setPatients] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalCount, setTotalCount] = useState(0);

  // Registration State
  const [fullName, setFullName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('Female');
  const [notes, setNotes] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [registerError, setRegisterError] = useState(null);

  // Duplicate Warning State
  const [duplicateMatches, setDuplicateMatches] = useState(null);

  // Patient Scan History Modal State
  const [historyPatient, setHistoryPatient] = useState(null);

  // Fetch patient records with query and page parameter (10 results per page)
  const fetchPatients = async (query = '', page = 1) => {
    const trimmed = query.trim();
    if (!trimmed) {
      setPatients([]);
      setHasSearched(false);
      setTotalCount(0);
      setTotalPages(1);
      setCurrentPage(1);
      return;
    }

    setIsSearching(true);
    setSearchError(null);
    try {
      const headers = {};
      if (accessToken) {
        headers['Authorization'] = `Bearer ${accessToken}`;
      }
      const url = `${API_BASE_URL}/patients?search=${encodeURIComponent(trimmed)}&page=${page}&page_size=10`;

      const response = await fetch(url, { headers });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch patients.');
      }

      setPatients(data.patients || []);
      setTotalCount(data.total ?? (data.patients ? data.patients.length : 0));
      setTotalPages(data.total_pages || 1);
      setCurrentPage(data.page || page);
      setHasSearched(true);
    } catch (err) {
      console.error('Fetch Patients Error:', err);
      setSearchError(err.message || 'Failed to connect to patient database.');
    } finally {
      setIsSearching(false);
    }
  };

  const handleSearchSubmit = (e) => {
    if (e) e.preventDefault();
    fetchPatients(searchQuery, 1);
  };

  const handlePageChange = (newPage) => {
    if (newPage < 1 || newPage > totalPages) return;
    fetchPatients(searchQuery, newPage);
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setPatients([]);
    setHasSearched(false);
    setTotalCount(0);
    setTotalPages(1);
    setCurrentPage(1);
    setSearchError(null);
  };

  // Perform Registration with Duplicate Name Checking
  const handleRegisterSubmit = async (e, forceOverride = false) => {
    if (e) e.preventDefault();
    setRegisterError(null);

    const parsedAge = parseInt(age, 10);
    if (isNaN(parsedAge) || parsedAge < 0 || parsedAge > 120) {
      setRegisterError('Please enter a valid age between 0 and 120 years.');
      return;
    }

    const trimmedName = fullName.trim();
    if (!trimmedName) {
      setRegisterError('Patient full name is required.');
      return;
    }

    // Step A: Check for existing matching names unless forceOverride is true
    if (!forceOverride) {
      try {
        const headers = {};
        if (accessToken) {
          headers['Authorization'] = `Bearer ${accessToken}`;
        }
        const checkRes = await fetch(`${API_BASE_URL}/patients?search=${encodeURIComponent(trimmedName)}`, { headers });
        const checkData = await checkRes.json();
        
        if (checkRes.ok && checkData.patients && checkData.patients.length > 0) {
          const nameMatches = checkData.patients.filter(
            (p) => p.full_name.toLowerCase() === trimmedName.toLowerCase() || p.patient_code.toLowerCase() === trimmedName.toLowerCase()
          );

          if (nameMatches.length > 0) {
            setDuplicateMatches(nameMatches);
            return;
          }
        }
      } catch (err) {
        console.warn('Duplicate check warning:', err);
      }
    }

    // Step B: Proceed with Registration
    setIsRegistering(true);
    setDuplicateMatches(null);

    try {
      const headers = { 'Content-Type': 'application/json' };
      if (accessToken) {
        headers['Authorization'] = `Bearer ${accessToken}`;
      }

      const response = await fetch(`${API_BASE_URL}/patients`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          full_name: trimmedName,
          age: parsedAge,
          gender,
          notes: notes.trim() || null
        })
      });

      const data = await response.json();

      if (response.status === 401) {
        throw new Error(data.detail || 'Authentication token expired. Please log in again.');
      }

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to register patient in database.');
      }

      if (data.patient) {
        onSelectPatient(data.patient);
      }
    } catch (err) {
      console.error('Registration Error:', err);
      setRegisterError(err.message || 'An error occurred during patient registration.');
    } finally {
      setIsRegistering(false);
    }
  };

  return (
    <div style={{ maxWidth: '960px', margin: '0 auto' }}>
      {/* Portal Header */}
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.6rem', color: '#f8fafc', marginBottom: '6px' }}>
          Patient Registration &amp; Search Portal
        </h2>
        <p style={{ color: '#94a3b8', fontSize: '0.95rem' }}>
          Search existing records or register a new patient before proceeding to ultrasound scan analysis.
        </p>
      </div>

      {/* Grid Layout: Left = Search & List, Right = Registration Form */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'start' }}>
        
        {/* PANEL 1: SEARCH & SELECT EXISTING PATIENT */}
        <div style={{
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid var(--panel-border)',
          borderRadius: '12px',
          padding: '1.5rem'
        }}>
          <h3 style={{ fontSize: '1.15rem', color: '#f8fafc', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Search size={18} style={{ color: '#38bdf8' }} /> Search Existing Patients
          </h3>

          {/* Search Form */}
          <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: '8px', marginBottom: '1.2rem' }}>
            <div style={{ position: 'relative', flex: 1 }}>
              <Search size={16} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#64748b' }} />
              <input
                type="text"
                placeholder="Search Patient ID (OV-P-...) or Name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  width: '100%',
                  padding: searchQuery ? '8px 30px 8px 34px' : '8px 10px 8px 34px',
                  borderRadius: '6px',
                  border: '1px solid var(--panel-border)',
                  background: '#0f172a',
                  color: '#f8fafc',
                  fontSize: '0.88rem',
                  outline: 'none'
                }}
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={handleClearSearch}
                  title="Clear search query"
                  style={{
                    position: 'absolute',
                    right: '8px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    color: '#94a3b8',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '2px'
                  }}
                >
                  <X size={14} />
                </button>
              )}
            </div>
            <button
              type="submit"
              className="btn-analyze"
              style={{ width: 'auto', padding: '0 14px', fontSize: '0.85rem' }}
            >
              Search
            </button>
          </form>

          {searchError && (
            <div className="error-alert" style={{ marginBottom: '1rem', fontSize: '0.85rem' }}>
              <AlertCircle size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
              {searchError}
            </div>
          )}

          {/* Results Area */}
          {isSearching ? (
            <div style={{ textAlign: 'center', padding: '2.5rem 0' }}>
              <div className="spinner" style={{ margin: '0 auto 8px auto', width: '20px', height: '20px' }}></div>
              <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Searching database...</p>
            </div>
          ) : !hasSearched ? (
            <div style={{
              textAlign: 'center',
              padding: '2.5rem 1rem',
              background: 'rgba(15, 23, 42, 0.4)',
              borderRadius: '10px',
              border: '1px dashed rgba(255, 255, 255, 0.12)'
            }}>
              <Search size={32} style={{ color: '#38bdf8', marginBottom: '10px', opacity: 0.8 }} />
              <h4 style={{ color: '#f8fafc', fontSize: '0.98rem', marginBottom: '6px', fontWeight: '600' }}>
                Find Patient Record
              </h4>
              <p style={{ color: '#94a3b8', fontSize: '0.85rem', maxWidth: '320px', margin: '0 auto', lineHeight: '1.4' }}>
                Search by Patient ID (e.g. <span style={{ color: '#38bdf8', fontWeight: '600' }}>OV-P-XXXXXX</span>) or Name to view matching records.
              </p>
            </div>
          ) : patients.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem 1rem', background: 'rgba(15, 23, 42, 0.4)', borderRadius: '8px', color: '#94a3b8', fontSize: '0.88rem' }}>
              No patient records found matching "{searchQuery}".
            </div>
          ) : (
            <div>
              {/* Pagination / Count Summary Bar */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', fontSize: '0.8rem', color: '#94a3b8' }}>
                <span>Showing <strong>{patients.length}</strong> of <strong>{totalCount}</strong> matching record(s)</span>
                <span>Page {currentPage} of {totalPages}</span>
              </div>

              {/* Patient Cards List */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '380px', overflowY: 'auto', paddingRight: '4px' }}>
                {patients.map((pt) => (
                  <div
                    key={pt.id}
                    style={{
                      background: 'rgba(30, 41, 59, 0.5)',
                      border: '1px solid rgba(255, 255, 255, 0.08)',
                      borderRadius: '8px',
                      padding: '12px 14px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '8px'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{
                        background: '#0284c7',
                        color: '#ffffff',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontWeight: '700',
                        fontSize: '0.82rem'
                      }}>
                        {pt.patient_code}
                      </span>
                      <span style={{ color: '#64748b', fontSize: '0.78rem' }}>
                        {new Date(pt.created_at).toLocaleDateString()}
                      </span>
                    </div>

                    <div>
                      <strong style={{ color: '#f8fafc', fontSize: '0.98rem' }}>{pt.full_name}</strong>
                      <div style={{ color: '#94a3b8', fontSize: '0.82rem', marginTop: '2px' }}>
                        Age: {pt.age} yrs | Gender: {pt.gender}
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
                      <button
                        onClick={() => onSelectPatient(pt)}
                        className="btn-analyze"
                        style={{ padding: '6px 12px', fontSize: '0.82rem', flex: 1, justifyContent: 'center' }}
                      >
                        <UserCheck size={14} /> Select Patient
                      </button>
                      <button
                        onClick={() => setHistoryPatient(pt)}
                        style={{
                          background: 'rgba(255, 255, 255, 0.08)',
                          border: '1px solid rgba(255, 255, 255, 0.15)',
                          color: '#cbd5e1',
                          padding: '6px 10px',
                          borderRadius: '6px',
                          fontSize: '0.8rem',
                          cursor: 'pointer',
                          fontWeight: '600',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px'
                        }}
                      >
                        <History size={14} /> Scan History
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Pagination Controls */}
              {totalPages > 1 && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem', paddingTop: '0.8rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
                  <button
                    type="button"
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage <= 1 || isSearching}
                    style={{
                      background: 'rgba(255, 255, 255, 0.08)',
                      border: '1px solid rgba(255, 255, 255, 0.15)',
                      color: currentPage <= 1 ? '#64748b' : '#f8fafc',
                      padding: '6px 12px',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      fontWeight: '600',
                      cursor: currentPage <= 1 ? 'not-allowed' : 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      opacity: currentPage <= 1 ? 0.5 : 1
                    }}
                  >
                    <ChevronLeft size={16} /> Prev
                  </button>

                  <span style={{ fontSize: '0.8rem', color: '#cbd5e1', fontWeight: '600' }}>
                    Page {currentPage} of {totalPages}
                  </span>

                  <button
                    type="button"
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage >= totalPages || isSearching}
                    style={{
                      background: 'rgba(255, 255, 255, 0.08)',
                      border: '1px solid rgba(255, 255, 255, 0.15)',
                      color: currentPage >= totalPages ? '#64748b' : '#f8fafc',
                      padding: '6px 12px',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      fontWeight: '600',
                      cursor: currentPage >= totalPages ? 'not-allowed' : 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      opacity: currentPage >= totalPages ? 0.5 : 1
                    }}
                  >
                    Next <ChevronRight size={16} />
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* PANEL 2: REGISTER NEW PATIENT FORM */}
        <div style={{
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid var(--panel-border)',
          borderRadius: '12px',
          padding: '1.5rem'
        }}>
          <h3 style={{ fontSize: '1.15rem', color: '#f8fafc', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <UserPlus size={18} style={{ color: '#38bdf8' }} /> Register New Patient
          </h3>

          {registerError && (
            <div className="error-alert" style={{ marginBottom: '1rem', fontSize: '0.85rem' }}>
              <AlertCircle size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
              {registerError}
            </div>
          )}

          {/* DUPLICATE NAME WARNING BANNER / DIALOG */}
          {duplicateMatches ? (
            <div style={{
              background: 'rgba(234, 179, 8, 0.12)',
              border: '1px solid rgba(234, 179, 8, 0.4)',
              borderRadius: '10px',
              padding: '1.2rem',
              marginBottom: '1rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#fde047', fontWeight: '700', fontSize: '0.92rem', marginBottom: '8px' }}>
                <AlertTriangle size={18} />
                <span>Matching Patient Record(s) Found</span>
              </div>
              <p style={{ color: '#cbd5e1', fontSize: '0.85rem', marginBottom: '12px', lineHeight: '1.4' }}>
                Found {duplicateMatches.length} existing patient record(s) matching <strong>"{fullName}"</strong>. 
                Please select an existing patient or confirm registration if this is a different person.
              </p>

              {/* List matching patients */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '14px' }}>
                {duplicateMatches.map((dm) => (
                  <div key={dm.id} style={{
                    background: '#0f172a',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    border: '1px solid rgba(255,255,255,0.1)',
                    display: 'flex',
                    justify: 'space-between',
                    alignItems: 'center'
                  }}>
                    <div>
                      <span style={{ color: '#38bdf8', fontWeight: '700', fontSize: '0.82rem', marginRight: '8px' }}>{dm.patient_code}</span>
                      <strong style={{ color: '#ffffff', fontSize: '0.88rem' }}>{dm.full_name}</strong>
                      <span style={{ color: '#94a3b8', fontSize: '0.78rem', marginLeft: '8px' }}>({dm.age} yrs, {dm.gender})</span>
                    </div>
                    <button
                      onClick={() => onSelectPatient(dm)}
                      style={{
                        background: '#0284c7',
                        color: '#fff',
                        border: 'none',
                        padding: '4px 10px',
                        borderRadius: '4px',
                        fontSize: '0.78rem',
                        fontWeight: '600',
                        cursor: 'pointer'
                      }}
                    >
                      Select Existing
                    </button>
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  type="button"
                  onClick={(e) => handleRegisterSubmit(e, true)}
                  style={{
                    background: '#10b981',
                    color: '#ffffff',
                    border: 'none',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    fontSize: '0.82rem',
                    fontWeight: '700',
                    cursor: 'pointer',
                    flex: 1
                  }}
                >
                  Confirm &amp; Register New Person
                </button>
                <button
                  type="button"
                  onClick={() => setDuplicateMatches(null)}
                  style={{
                    background: 'rgba(255,255,255,0.1)',
                    color: '#94a3b8',
                    border: 'none',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    fontSize: '0.82rem',
                    cursor: 'pointer'
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            /* Standard Registration Form */
            <form onSubmit={(e) => handleRegisterSubmit(e, false)} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {/* Full Name */}
              <div>
                <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: '600', color: '#cbd5e1', marginBottom: '4px' }}>
                  Full Name <span style={{ color: '#ef4444' }}>*</span>
                </label>
                <div style={{ position: 'relative' }}>
                  <User size={16} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#64748b' }} />
                  <input
                    type="text"
                    required
                    placeholder="e.g. Jane Doe"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 10px 8px 34px',
                      borderRadius: '6px',
                      border: '1px solid var(--panel-border)',
                      background: '#0f172a',
                      color: '#f8fafc',
                      fontSize: '0.9rem',
                      outline: 'none'
                    }}
                  />
                </div>
              </div>

              {/* Grid: Age & Gender */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem' }}>
                {/* Age */}
                <div>
                  <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: '600', color: '#cbd5e1', marginBottom: '4px' }}>
                    Age (0-120) <span style={{ color: '#ef4444' }}>*</span>
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
                      padding: '8px 10px',
                      borderRadius: '6px',
                      border: '1px solid var(--panel-border)',
                      background: '#0f172a',
                      color: '#f8fafc',
                      fontSize: '0.9rem',
                      outline: 'none'
                    }}
                  />
                </div>

                {/* Gender */}
                <div>
                  <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: '600', color: '#cbd5e1', marginBottom: '4px' }}>
                    Gender <span style={{ color: '#ef4444' }}>*</span>
                  </label>
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 10px',
                      borderRadius: '6px',
                      border: '1px solid var(--panel-border)',
                      background: '#0f172a',
                      color: '#f8fafc',
                      fontSize: '0.9rem',
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

              {/* Notes */}
              <div>
                <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: '600', color: '#cbd5e1', marginBottom: '4px' }}>
                  Clinical Observations (Optional)
                </label>
                <div style={{ position: 'relative' }}>
                  <FileText size={16} style={{ position: 'absolute', left: '10px', top: '10px', color: '#64748b' }} />
                  <textarea
                    rows="2"
                    placeholder="e.g. Palpable lesion upper outer quadrant"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 10px 8px 34px',
                      borderRadius: '6px',
                      border: '1px solid var(--panel-border)',
                      background: '#0f172a',
                      color: '#f8fafc',
                      fontSize: '0.88rem',
                      outline: 'none',
                      resize: 'vertical'
                    }}
                  />
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isRegistering}
                className="btn-analyze"
                style={{ justifyContent: 'center', marginTop: '0.5rem', opacity: isRegistering ? 0.7 : 1 }}
              >
                {isRegistering ? (
                  <>
                    <div className="spinner" style={{ width: '16px', height: '16px', borderTopColor: '#ffffff' }}></div>
                    Registering Patient...
                  </>
                ) : (
                  <>
                    <UserPlus size={16} /> Register Patient &amp; Select
                  </>
                )}
              </button>
            </form>
          )}
        </div>
      </div>

      {/* PATIENT SCAN HISTORY MODAL */}
      {historyPatient && (
        <PatientHistoryModal
          patient={historyPatient}
          onClose={() => setHistoryPatient(null)}
        />
      )}
    </div>
  );
}
