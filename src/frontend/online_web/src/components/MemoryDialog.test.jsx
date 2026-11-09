// @vitest-environment jsdom

import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import MemoryDialog, { MemoryRecordCard } from './MemoryDialog.jsx'

const memoryApi = vi.hoisted(() => ({ fetchLongTermMemories: vi.fn() }))

vi.mock('../api/lightmem_egoApi.js', async (importOriginal) => ({
  ...await importOriginal(),
  ...memoryApi
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

const exportableMemory = {
  id: 'mlt_ep_abc',
  status: 'archived',
  webVisible: true,
  canView: true,
  canExport: true,
  deletedAt: null,
  archiveCompletedAt: '2026-09-26T06:30:12.000Z',
  content: 'A safe memory'
}

describe('MemoryRecordCard export controls', () => {
  it('always shows the export button and reports clicks for backend validation', () => {
    const onExport = vi.fn()
    const { rerender } = render(
      <MemoryRecordCard memory={exportableMemory} sessionId="session-1" onExport={onExport} />
    )

    const button = screen.getByRole('button', { name: 'Export memory mlt_ep_abc as JSON' })
    expect(button.getAttribute('title')).toBe('Export JSON')
    button.click()
    expect(onExport).toHaveBeenCalledWith(exportableMemory, 'session-1')

    rerender(
      <MemoryRecordCard
        memory={{ ...exportableMemory, canExport: false }}
        sessionId="session-1"
        onExport={onExport}
      />
    )
    expect(screen.getByRole('button', { name: 'Export memory mlt_ep_abc as JSON' })).toBeTruthy()

    rerender(
      <MemoryRecordCard memory={{ content: 'Missing ID' }} sessionId="session-1" onExport={onExport} />
    )
    expect(screen.getByRole('button', { name: 'Export memory unknown as JSON' })).toBeTruthy()
  })
})

it('loads more records using the whole memory dialog as its scroll root', async () => {
  const observers = []
  vi.stubGlobal('IntersectionObserver', class {
    constructor(callback, options) {
      this.callback = callback
      this.root = options.root
      observers.push(this)
    }
    observe() {}
    disconnect() {}
  })
  const memory = {
    sessionId: 'session-1', memoryVersion: 1, qaAligned: true,
    counts: { total: 101, episodic: 0, semantic: 101, visual: 0 },
    episodic: { '30sec': [], '3min': [], '10min': [], '1h': [] },
    semantic: Array.from({ length: 101 }, (_, index) => ({ id: `record-${index}`, content: `Memory content ${index}` })),
    visual: []
  }
  const user = userEvent.setup()
  render(<MemoryDialog session={{ sessionId: 'session-1', memoryVersion: 1 }} cachedMemory={memory} initialTab="records" onClose={vi.fn()} />)
  expect(screen.getByRole('button', { name: /Semantic Memory/ }).getAttribute('aria-expanded')).toBe('false')
  await user.click(screen.getByRole('button', { name: /Semantic Memory/ }))
  expect(observers.length).toBeGreaterThan(0)
  expect(observers.at(-1).root).toBe(screen.getByRole('dialog'))
  expect(screen.queryByText('Memory content 100')).toBeNull()
  act(() => observers.at(-1).callback([{ isIntersecting: true }]))
  expect(screen.getByText('Memory content 100')).toBeTruthy()
})

it('shows the published server version in the detail header after a mutation', async () => {
  vi.stubGlobal('requestAnimationFrame', vi.fn())
  const first = detailMemory(1)
  const updated = detailMemory(2)
  memoryApi.fetchLongTermMemories.mockResolvedValue(first)
  const props = {
    session: { sessionId: 'session-1', memoryVersion: 1, sessionKind: 'video' },
    cachedMemory: first,
    initialTab: 'records',
    onClose: vi.fn()
  }
  const { rerender } = render(<MemoryDialog {...props} />)
  await waitForVersion(1)

  rerender(<MemoryDialog {...props} authoritativeUpdate={{
    sessionId: 'session-1', revision: 1, memory: updated, graph: null, graphStatus: 'refreshing'
  }} />)

  await waitForVersion(2)
})

it('shows a version-matched Session summary above collapsed episodic records', async () => {
  vi.stubGlobal('requestAnimationFrame', vi.fn())
  const memory = detailMemory(1)
  memory.sessionSummary = 'A concise account of this Session.'
  memory.sessionSummaryVersion = 1
  memory.sessionSummaryStatus = 'ready'
  memory.episodic['30sec'] = [{ id: 'event-1', content: 'An episodic record' }]
  memory.counts.total = 1
  memory.counts.episodic = 1
  memory.counts.episodicByGranularity['30sec'] = 1
  memoryApi.fetchLongTermMemories.mockResolvedValue(memory)

  render(<MemoryDialog session={{ sessionId: 'session-1', memoryVersion: 1 }} cachedMemory={memory} initialTab="records" onClose={vi.fn()} />)

  const summary = await screen.findByRole('complementary', { name: 'Session summary' })
  expect(summary.textContent).toContain('A concise account of this Session.')
  const episodicToggle = screen.getByRole('button', { name: /Episodic Memory/ })
  expect(episodicToggle.getAttribute('aria-expanded')).toBe('false')
  await userEvent.setup().click(episodicToggle)
  expect(screen.getByRole('button', { name: /30 seconds/ }).getAttribute('aria-expanded')).toBe('false')
  expect(screen.queryByText('An episodic record')).toBeNull()
  expect(summary.compareDocumentPosition(screen.getByRole('button', { name: /30 seconds/ })) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
})

it.each([
  ['unavailable', 1, 'Old summary'],
  ['ready', 0, 'Stale summary'],
  ['ready', 1, '   ']
])('hides a %s Session summary when status, version, or content is invalid', async (status, summaryVersion, summaryText) => {
  vi.stubGlobal('requestAnimationFrame', vi.fn())
  const memory = detailMemory(1)
  memory.sessionSummary = summaryText
  memory.sessionSummaryVersion = summaryVersion
  memory.sessionSummaryStatus = status
  memoryApi.fetchLongTermMemories.mockResolvedValue(memory)

  render(<MemoryDialog session={{ sessionId: 'session-1', memoryVersion: 1 }} cachedMemory={memory} initialTab="records" onClose={vi.fn()} />)

  await waitFor(() => expect(memoryApi.fetchLongTermMemories).toHaveBeenCalled())
  expect(screen.queryByRole('complementary', { name: 'Session summary' })).toBeNull()
})

it('keeps manually expanded groups open when a new complete snapshot is published', async () => {
  vi.stubGlobal('requestAnimationFrame', vi.fn())
  const first = detailMemory(1)
  const updated = detailMemory(2)
  memoryApi.fetchLongTermMemories.mockResolvedValue(first)
  const props = {
    session: { sessionId: 'session-1', memoryVersion: 1 },
    cachedMemory: first,
    initialTab: 'records',
    onClose: vi.fn()
  }
  const user = userEvent.setup()
  const { rerender, unmount } = render(<MemoryDialog {...props} />)

  const episodicToggle = screen.getByRole('button', { name: /Episodic Memory/ })
  expect(episodicToggle.getAttribute('aria-expanded')).toBe('false')
  await user.click(episodicToggle)
  expect(episodicToggle.getAttribute('aria-expanded')).toBe('true')

  rerender(<MemoryDialog {...props} authoritativeUpdate={{
    sessionId: 'session-1', revision: 1, memory: updated, graph: null
  }} />)
  expect(screen.getByRole('button', { name: /Episodic Memory/ }).getAttribute('aria-expanded')).toBe('true')

  unmount()
  render(<MemoryDialog {...props} />)
  expect(screen.getByRole('button', { name: /Episodic Memory/ }).getAttribute('aria-expanded')).toBe('false')
})

async function waitForVersion(version) {
  await waitFor(() => {
    expect(document.querySelector('.memory-dialog-version-chips')?.firstElementChild?.textContent).toBe(`M_lt v${version}`)
  })
}

function detailMemory(version) {
  return {
    sessionId: 'session-1', memoryVersion: version,
    componentVersions: { episodic: version, semantic: version, visual: version },
    activeRootKind: 'em2mem', qaAligned: true, updatedAt: null,
    editSupported: true, canEdit: false, editDisabledReason: 'no_editable_records',
    sessionSummary: '', sessionSummaryVersion: null, sessionSummaryStatus: 'unavailable',
    canRollback: false, editPolicy: { field: 'episodic.30sec[].content', maxCharsPerRecord: 4000, maxTotalChars: 32000 },
    propagation: null, fullPropagationSupported: false,
    availability: { episodic: 'ready', semantic: 'ready', visual: 'ready' },
    counts: { total: 0, episodic: 0, semantic: 0, visual: 0,
      episodicByGranularity: { '30sec': 0, '3min': 0, '10min': 0, '1h': 0 } },
    episodic: { '30sec': [], '3min': [], '10min': [], '1h': [] }, semantic: [], visual: []
  }
}
