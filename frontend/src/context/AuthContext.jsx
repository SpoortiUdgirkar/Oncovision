import React, { createContext, useContext, useState, useEffect } from 'react';
import { supabase } from '../services/supabase';

const AuthContext = createContext({
  user: null,
  session: null,
  loading: true,
  authError: null,
  login: async () => {},
  logout: async () => {},
  clearAuthError: () => {},
});

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    // Restore existing session on initial application load
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      if (isMounted) {
        if (error) {
          console.error('[OncoVision Auth] Session restoration error:', error.message);
        }
        setSession(session);
        setUser(session?.user ?? null);
        setLoading(false);
      }
    }).catch((err) => {
      if (isMounted) {
        console.error('[OncoVision Auth] Unexpected session error:', err);
        setLoading(false);
      }
    });

    // Subscribe to live auth state changes (sign in, sign out, token refresh)
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (isMounted) {
        setSession(session);
        setUser(session?.user ?? null);
        setLoading(false);
      }
    });

    return () => {
      isMounted = false;
      subscription?.unsubscribe();
    };
  }, []);

  const login = async (email, password) => {
    setAuthError(null);
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password: password,
      });

      if (error) {
        throw error;
      }

      setSession(data.session);
      setUser(data.user);
      return { success: true, user: data.user };
    } catch (err) {
      const errorMsg = err.message || 'Authentication failed. Please verify credentials.';
      setAuthError(errorMsg);
      return { success: false, error: errorMsg };
    }
  };

  const logout = async () => {
    try {
      await supabase.auth.signOut();
    } catch (err) {
      console.error('[OncoVision Auth] Logout error:', err);
    } finally {
      setUser(null);
      setSession(null);
      setAuthError(null);
    }
  };

  const clearAuthError = () => setAuthError(null);

  const value = {
    user,
    session,
    loading,
    authError,
    login,
    logout,
    clearAuthError,
    accessToken: session?.access_token || null,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
