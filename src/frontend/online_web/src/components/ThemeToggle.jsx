import { Moon, Sun } from 'lucide-react'
import { useLanguage } from '../i18n.jsx'
import { useTheme } from './ThemeProvider.jsx'

export default function ThemeToggle() {
  const { t } = useLanguage()
  const { theme, toggleTheme } = useTheme()
  const label = t(theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme')
  return (
    <button className="theme-toggle" type="button" onClick={toggleTheme} aria-label={label} aria-pressed={theme === 'dark'} title={label}>
      {theme === 'dark' ? <Sun key="sun" size={18} strokeWidth={1.7} aria-hidden="true" /> : <Moon key="moon" size={18} strokeWidth={1.7} aria-hidden="true" />}
    </button>
  )
}
