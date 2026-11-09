export const THEME_STORAGE_KEY = 'lightmem-ego.ui-theme.v1'

export function readThemePreference() {
  try {
    const value = window.localStorage.getItem(THEME_STORAGE_KEY)
    return value === 'dark' || value === 'light' ? value : null
  } catch {
    return null
  }
}

export function resolveTheme(preference, prefersDark = false) {
  return preference === 'light' || preference === 'dark'
    ? preference
    : prefersDark ? 'dark' : 'light'
}

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme
  document.documentElement.style.colorScheme = theme
  const meta = document.querySelector('meta[name="theme-color"]')
  if (meta) meta.content = theme === 'dark' ? '#0e1015' : '#f5f5f7'
}
