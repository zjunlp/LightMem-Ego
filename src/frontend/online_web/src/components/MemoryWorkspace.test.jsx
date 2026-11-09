// @vitest-environment jsdom
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import MemoryWorkspace from './MemoryWorkspace.jsx'

vi.mock('./MemoryPanel.jsx', () => ({ default: ({ standalone, active }) => <div data-testid="memory-catalog">{String(standalone && active)}</div> }))
vi.mock('../hooks/useRealtimeStream.js', () => ({ useRealtimeStream: () => { throw new Error('Standalone memory must not start live capture') } }))
afterEach(() => { cleanup(); localStorage.clear() })

it('opens a dedicated memory workspace with catalog and return links', () => {
  localStorage.setItem('lightmem-ego.ui-language.v1', 'zh')
  render(<MemoryWorkspace />)
  expect(screen.getByTestId('memory-catalog').textContent).toBe('true')
  expect(screen.getByRole('link', { name: /视频提问|视频回看/ }).getAttribute('href')).toBe('/history-demo/')
  expect(screen.getByRole('link', { name: /采集与问答|采集与提问/ }).getAttribute('href')).toBe('/#/collect')
  fireEvent.click(screen.getByRole('button', { name: /切换语言/ }))
  expect(document.documentElement.lang).toBe('en')
  expect(screen.getByRole('heading', { name: 'Memories' })).toBeTruthy()
})
