// @vitest-environment jsdom
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ThemeProvider from './ThemeProvider.jsx'
import ThemeToggle from './ThemeToggle.jsx'
import { LanguageContext, translateText } from '../i18n.jsx'
import { THEME_STORAGE_KEY } from '../theme.js'

let media
beforeEach(() => {
  localStorage.clear()
  media = new EventTarget()
  media.matches = false
  vi.stubGlobal('matchMedia', vi.fn(() => media))
  const meta = document.createElement('meta')
  meta.name = 'theme-color'
  document.head.append(meta)
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  document.querySelector('meta[name="theme-color"]')?.remove()
  delete document.documentElement.dataset.theme
  document.documentElement.style.removeProperty('color-scheme')
})

function renderToggle(language = 'en') {
  return render(
    <ThemeProvider>
      <LanguageContext.Provider value={{ language, t: (text, params) => translateText(language, text, params) }}>
        <ThemeToggle />
      </LanguageContext.Provider>
    </ThemeProvider>
  )
}
function systemTheme(dark) {
  media.matches = dark
  const event = new Event('change')
  event.matches = dark
  act(() => media.dispatchEvent(event))
}

describe('appearance preference', () => {
  it('follows system appearance until a user chooses a theme, then preserves that choice on remount', () => {
    localStorage.setItem('lightmem-ego.ui-language.v1', 'zh')
    const first = renderToggle()
    expect(document.documentElement.dataset.theme).toBe('light')
    systemTheme(true)
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(document.querySelector('meta[name="theme-color"]').content).toBe('#0e1015')
    fireEvent.click(screen.getByRole('button', { name: 'Switch to light theme' }))
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('light')
    systemTheme(false)
    systemTheme(true)
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(localStorage.getItem('lightmem-ego.ui-language.v1')).toBe('zh')
    first.unmount()
    renderToggle()
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(document.documentElement.style.colorScheme).toBe('light')
  })

  it('exposes translated keyboard-operable controls and stores dark mode', () => {
    renderToggle('zh')
    const button = screen.getByRole('button', { name: '切换至深色主题' })
    button.focus()
    expect(document.activeElement).toBe(button)
    fireEvent.click(button)
    const next = screen.getByRole('button', { name: '切换至浅色主题' })
    expect(next.getAttribute('aria-pressed')).toBe('true')
    expect(document.activeElement).toBe(next)
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark')
  })

  it('keeps theme switching available when storage is blocked', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
    renderToggle()
    fireEvent.click(screen.getByRole('button', { name: 'Switch to dark theme' }))
    expect(document.documentElement.dataset.theme).toBe('dark')
    systemTheme(false)
    expect(document.documentElement.dataset.theme).toBe('dark')
  })

  it('syncs a preference changed in another tab and releases its system listener', () => {
    const remove = vi.spyOn(media, 'removeEventListener')
    const view = renderToggle()
    localStorage.setItem(THEME_STORAGE_KEY, 'dark')
    fireEvent(window, new StorageEvent('storage', { key: THEME_STORAGE_KEY, newValue: 'dark' }))
    expect(document.documentElement.dataset.theme).toBe('dark')
    localStorage.removeItem(THEME_STORAGE_KEY)
    fireEvent(window, new StorageEvent('storage', { key: THEME_STORAGE_KEY, newValue: null }))
    expect(document.documentElement.dataset.theme).toBe('light')
    view.unmount()
    expect(remove).toHaveBeenCalledWith('change', expect.any(Function))
  })

  it.each([
    ['dark', false, 'dark'],
    ['light', true, 'light'],
    ['invalid', true, 'dark'],
    [null, false, 'light']
  ])('applies the theme before React starts: stored=%s, systemDark=%s', (stored, systemDark, expected) => {
    const root = { dataset: {}, style: {} }
    const meta = { content: '' }
    runInNewContext(readFileSync('public/theme-init.js', 'utf8'), {
      localStorage: { getItem: () => stored },
      window: { matchMedia: () => ({ matches: systemDark }) },
      document: { documentElement: root, querySelector: () => meta }
    })
    expect(root.dataset.theme).toBe(expected)
    expect(root.style.colorScheme).toBe(expected)
    expect(meta.content).toBe(expected === 'dark' ? '#0e1015' : '#f5f5f7')
  })
})
