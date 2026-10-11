/**
 * Shared API Configuration & Base URL Resolution for OncoVision.
 *
 * Ensures all API requests (patient registration, search, duplicate check,
 * scan prediction, history retrieval, and PDF downloads) target the correct backend.
 *
 * Requirements:
 * 1. Single shared API base URL from import.meta.env.VITE_API_BASE_URL.
 * 2. http://localhost:8000 is strictly a fallback for local development, never production.
 * 3. In production builds or hosted deployments (e.g. Vercel), defaults to https://oncovision-api.onrender.com.
 */

export const PRODUCTION_API_URL = 'https://oncovision-api.onrender.com';
export const LOCAL_DEV_API_URL = 'http://localhost:8000';

export function getApiBaseUrl() {
  const envUrl = (import.meta.env.VITE_API_BASE_URL || '').trim().replace(/\/+$/, '');
  const isProd = import.meta.env.PROD === true;
  const isNonLocalhost =
    typeof window !== 'undefined' &&
    window.location &&
    window.location.hostname &&
    !['localhost', '127.0.0.1', '0.0.0.0'].includes(window.location.hostname);

  // Production or deployed environment:
  if (isProd || isNonLocalhost) {
    // If a valid non-localhost remote URL is provided in VITE_API_BASE_URL, use it
    if (envUrl && !envUrl.includes('localhost') && !envUrl.includes('127.0.0.1')) {
      return envUrl;
    }
    // In production, never use localhost:8000; fall back to the deployed Render API
    return PRODUCTION_API_URL;
  }

  // Local development mode fallback:
  if (envUrl) {
    return envUrl;
  }

  return LOCAL_DEV_API_URL;
}

export const API_BASE_URL = getApiBaseUrl();

export default API_BASE_URL;
