import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useState } from 'react'
import { applyTheme, readThemePreference, resolveTheme, THEME_STORAGE_KEY } from '../theme.js'

const ThemeContext = createContext({ theme: 'light', toggleTheme: () => {} })

export function useTheme() {
  return useContext(ThemeContext)
}

export default function ThemeProvider({ children }) {
  const [preference, setPreference] = useState(readThemePreference)
  const [prefersDark, setPrefersDark] = useState(() => Boolean(window.matchMedia?.('(prefers-color-scheme: dark)').matches))
  const theme = resolveTheme(preference, prefersDark)

  useLayoutEffect(() => {
    applyTheme(theme)
  }, [theme])

  useEffect(() => {
    const media = window.matchMedia?.('(prefers-color-scheme: dark)')
    const syncSystemTheme = (event) => setPrefersDark(event.matches)
    const syncPreference = (event) => {
      if (event.key === THEME_STORAGE_KEY || event.key === null) setPreference(readThemePreference())
    }
    media?.addEventListener?.('change', syncSystemTheme)
    window.addEventListener('storage', syncPreference)
    return () => {
      media?.removeEventListener?.('change', syncSystemTheme)
      window.removeEventListener('storage', syncPreference)
    }
  }, [])

  const toggleTheme = useCallback(() => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setPreference(next)
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, next)
    } catch { /* Theme switching remains available without browser storage. */ }
  }, [theme])

  const value = useMemo(() => ({ theme, toggleTheme }), [theme, toggleTheme])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}
