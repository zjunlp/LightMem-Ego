import { Type } from 'lucide-react'
import { useEffect, useLayoutEffect, useState } from 'react'
import { useLanguage } from '../i18n.jsx'
import {
  applyFontScale,
  FONT_SCALE_STORAGE_KEY,
  FONT_SCALE_VALUES,
  readFontScalePreference
} from '../fontScale.js'

const SCALE_LABELS = { small: 'Small', normal: 'Default', large: 'Large' }

export default function FontSizeToggle() {
  const { t } = useLanguage()
  const [preference, setPreference] = useState(readFontScalePreference)
  // CSS adapts the base type scale to the viewport; the saved choice is relative to it.
  const scale = preference || 'normal'

  useLayoutEffect(() => {
    applyFontScale(scale)
  }, [scale])

  useEffect(() => {
    const syncPreference = (event) => {
      if (event.key === FONT_SCALE_STORAGE_KEY || event.key === null) {
        setPreference(readFontScalePreference())
      }
    }
    window.addEventListener('storage', syncPreference)
    return () => window.removeEventListener('storage', syncPreference)
  }, [])

  const nextScale = FONT_SCALE_VALUES[(FONT_SCALE_VALUES.indexOf(scale) + 1) % FONT_SCALE_VALUES.length]

  const currentLabel = t(SCALE_LABELS[scale])
  const label = t('Adjust font size')
  const title = `${label}: ${currentLabel}`
  const changeScale = () => {
    setPreference(nextScale)
    try { window.localStorage.setItem(FONT_SCALE_STORAGE_KEY, nextScale) } catch { /* Keep working if storage is blocked. */ }
  }

  return (
    <button
      className="font-size-toggle"
      type="button"
      onClick={changeScale}
      aria-label={title}
      title={`${title} · ${t('Click to change')}`}
    >
      <Type size={17} strokeWidth={1.8} aria-hidden="true" />
      <span aria-hidden="true">{scale === 'small' ? 'A−' : scale === 'large' ? 'A+' : 'A'}</span>
    </button>
  )
}
