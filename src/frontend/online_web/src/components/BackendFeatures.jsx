import { createContext, useContext, useEffect, useState } from 'react'

const Context = createContext(null)
const EMPTY = Object.freeze({ memoryImport: false, memoryExport: false, titleEditing: false })
export const useBackendFeatures = () => useContext(Context)

export function detectBackendFeatures(schema) {
  const paths = schema?.paths || {}
  const has = (path, method) => Object.entries(paths).some(([route, methods]) =>
    route.replace(/\{[^}]+\}/g, '{}') === path && methods?.[method])
  return {
    memoryImport: Boolean(has('/memory-imports', 'post') && has('/memory-imports/{}', 'get')),
    memoryExport: Boolean(has('/session/{}/memories/{}/export', 'get') || has('/memories/{}/export', 'get')),
    titleEditing: Boolean(has('/session/{}/title', 'get') && has('/session/{}/title', 'patch'))
  }
}

// Optional controls follow the connected backend's advertised routes. Existing
// viewing, capture and Q&A remain usable if capability discovery is unavailable.
export function BackendFeaturesProvider({ children }) {
  const [features, setFeatures] = useState(EMPTY)
  useEffect(() => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 5000)
    const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')
    fetch(`${base}/openapi.json`, { signal: controller.signal })
      .then(response => response.ok ? response.json() : null)
      .then(schema => { if (!controller.signal.aborted) setFeatures(detectBackendFeatures(schema)) })
      .catch(() => {})
      .finally(() => clearTimeout(timer))
    return () => { clearTimeout(timer); controller.abort() }
  }, [])
  return <Context.Provider value={features}>{children}</Context.Provider>
}
