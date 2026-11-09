// @vitest-environment jsdom
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import FontSizeToggle from './FontSizeToggle.jsx'
import { LanguageContext, translateText } from '../i18n.jsx'
import { FONT_SCALE_STORAGE_KEY } from '../fontScale.js'

beforeEach(() => localStorage.clear())
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  delete document.documentElement.dataset.fontScale
})

function renderToggle(language = 'en') {
  return render(
    <LanguageContext.Provider value={{ language, t: (text, params) => translateText(language, text, params) }}>
      <FontSizeToggle />
    </LanguageContext.Provider>
  )
}

describe('text size preference', () => {
  it('cycles the three sizes, preserves the saved choice and leaves business/language storage intact', () => {
    localStorage.setItem('lightmem-ego.ui-language.v1', 'zh')
    const first = renderToggle()
    expect(document.documentElement.dataset.fontScale).toBe('normal')
    for (const [name, scale] of [['Default', 'large'], ['Large', 'small'], ['Small', 'normal']]) {
      fireEvent.click(screen.getByRole('button', { name: `Adjust font size: ${name}` }))
      expect(document.documentElement.dataset.fontScale).toBe(scale)
      expect(localStorage.getItem(FONT_SCALE_STORAGE_KEY)).toBe(scale)
    }
    fireEvent.click(screen.getByRole('button', { name: 'Adjust font size: Default' }))
    first.unmount()
    renderToggle()
    expect(document.documentElement.dataset.fontScale).toBe('large')
    expect(localStorage.getItem('lightmem-ego.ui-language.v1')).toBe('zh')
  })

  it('accepts cross-tab updates and falls back from an invalid or cleared preference', () => {
    localStorage.setItem(FONT_SCALE_STORAGE_KEY, 'invalid')
    renderToggle()
    expect(document.documentElement.dataset.fontScale).toBe('normal')
    localStorage.setItem(FONT_SCALE_STORAGE_KEY, 'small')
    fireEvent(window, new StorageEvent('storage', { key: FONT_SCALE_STORAGE_KEY, newValue: 'small' }))
    expect(document.documentElement.dataset.fontScale).toBe('small')
    localStorage.clear()
    fireEvent(window, new StorageEvent('storage', { key: null }))
    expect(document.documentElement.dataset.fontScale).toBe('normal')
  })

  it('still changes text size if browser storage is blocked', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
    renderToggle()
    fireEvent.click(screen.getByRole('button', { name: 'Adjust font size: Default' }))
    expect(document.documentElement.dataset.fontScale).toBe('large')
  })

  it.each(['small', 'normal', 'large', 'invalid', null])('applies the same choice before first paint: %s', (stored) => {
    const root = { dataset: {}, style: {} }
    runInNewContext(readFileSync('public/theme-init.js', 'utf8'), {
      localStorage: { getItem: key => key === FONT_SCALE_STORAGE_KEY ? stored : null },
      window: { innerWidth: 2560, matchMedia: () => ({ matches: false }) },
      document: { documentElement: root, querySelector: () => null }
    })
    expect(root.dataset.fontScale).toBe(['small', 'normal', 'large'].includes(stored) ? stored : 'normal')
  })
})
