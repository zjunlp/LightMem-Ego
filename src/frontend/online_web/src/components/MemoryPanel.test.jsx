// @vitest-environment jsdom
import { StrictMode } from 'react'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'
import { fetchLongTermMemorySessions } from '../api/lightmem_egoApi.js'
import MemoryPanel from './MemoryPanel.jsx'

const titleApi = vi.hoisted(() => ({
  getSessionTitle: vi.fn(),
  updateSessionTitle: vi.fn(),
  fetchLongTermMemoryGraph: vi.fn()
}))

vi.mock('../api/lightmem_egoApi.js', async (importOriginal) => ({
  ...await importOriginal(),
  fetchLongTermMemorySessions: vi.fn(),
  ...titleApi
}))

afterEach(() => { cleanup(); vi.clearAllMocks(); vi.unstubAllGlobals() })

it('loads the standalone catalog after effect replay and ignores the interrupted response', async () => {
  // jsdom has no native element scrolling; this test covers catalog requests.
  vi.stubGlobal('requestAnimationFrame', vi.fn())
  const requests = []
  fetchLongTermMemorySessions.mockImplementation(({ signal }) => new Promise((resolve) => {
    requests.push({ signal, resolve })
  }))
  const catalog = (title) => ({
    items: [{ sessionId: 'session-1', title, memoryVersion: 1,
      componentVersions: { episodic: 1, semantic: 1, visual: 1 },
      activeRootKind: 'em2mem', qaAligned: true, updatedAt: null }],
    totalReadySessions: 1, hasMore: false, nextCursor: null
  })
  render(<StrictMode><MemoryPanel standalone /></StrictMode>)
  expect(requests[0].signal.aborted).toBe(true)
  const current = requests.find((request) => !request.signal.aborted)
  expect(current).toBeTruthy()
  await act(async () => current.resolve(catalog('Current memory')))
  expect(screen.getByRole('button', { name: /Current memory/ })).toBeTruthy()
  await act(async () => requests[0].resolve(catalog('Interrupted memory')))
  expect(screen.queryByText('Interrupted memory')).toBeNull()
  expect(screen.getByRole('button', { name: /Current memory/ })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Refresh' }).disabled).toBe(false)
})

it('keeps the newer title cache when an older catalog response arrives', async () => {
  vi.stubGlobal('requestAnimationFrame', vi.fn())
  const firstPage = {
    items: [{ sessionId: 'session-1', title: 'Old title', titleRevision: 1, memoryVersion: 1,
      componentVersions: { episodic: 1, semantic: 1, visual: 1 },
      activeRootKind: 'em2mem', qaAligned: true, updatedAt: null }],
    totalReadySessions: 1, hasMore: false, nextCursor: null
  }
  fetchLongTermMemorySessions.mockResolvedValue(firstPage)
  const { rerender } = render(<MemoryPanel standalone sessionTitles={{}} />)
  expect(await screen.findByRole('button', { name: /Old title/ })).toBeTruthy()

  rerender(<MemoryPanel standalone sessionTitles={{
    'session-1': { sessionId: 'session-1', title: 'New title', titleRevision: 2, titleUpdatedAt: null }
  }} />)
  expect(screen.getByRole('button', { name: /New title/ })).toBeTruthy()

  fetchLongTermMemorySessions.mockResolvedValueOnce(firstPage)
  await userEvent.setup().click(screen.getByRole('button', { name: 'Refresh' }))
  await waitFor(() => expect(fetchLongTermMemorySessions).toHaveBeenCalledTimes(2))
  expect(screen.getByRole('button', { name: /New title/ })).toBeTruthy()
  expect(screen.queryByRole('button', { name: /Old title/ })).toBeNull()
})

it('keeps each Session publication version through batched updates and stale catalog responses', async () => {
  vi.stubGlobal('requestAnimationFrame', vi.fn())
  const firstPage = {
    items: [catalogSession('session-1', 'First session', 1), catalogSession('session-2', 'Second session', 5)],
    totalReadySessions: 2, hasMore: false, nextCursor: null
  }
  fetchLongTermMemorySessions.mockResolvedValue(firstPage)
  const { rerender } = render(<MemoryPanel standalone />)
  const firstCard = await screen.findByRole('button', { name: /First session/ })
  const secondCard = screen.getByRole('button', { name: /Second session/ })
  expect(firstCard.textContent).toContain('v1')
  expect(secondCard.textContent).toContain('v5')

  const updates = new Map([
    ['session-1', publication('session-1', 2, 1)],
    ['session-2', publication('session-2', 6, 2)]
  ])
  rerender(<MemoryPanel standalone authoritativeUpdates={updates} />)
  await waitFor(() => {
    expect(screen.getByRole('button', { name: /First session/ }).textContent).toContain('v2')
    expect(screen.getByRole('button', { name: /Second session/ }).textContent).toContain('v6')
  })

  fetchLongTermMemorySessions.mockResolvedValueOnce(firstPage)
  await userEvent.setup().click(screen.getByRole('button', { name: 'Refresh' }))
  await waitFor(() => expect(fetchLongTermMemorySessions).toHaveBeenCalledTimes(2))
  expect(screen.getByRole('button', { name: /First session/ }).textContent).toContain('v2')
  expect(screen.getByRole('button', { name: /Second session/ }).textContent).toContain('v6')
})

it('propagates a successful rename from the detail dialog back to its catalog card', async () => {
  vi.stubGlobal('requestAnimationFrame', vi.fn())
  fetchLongTermMemorySessions.mockResolvedValue({
    items: [{ sessionId: 'session-1', title: 'Old title', titleRevision: 1, memoryVersion: 1,
      componentVersions: { episodic: 1, semantic: 1, visual: 1 },
      activeRootKind: 'em2mem', qaAligned: true, updatedAt: null }],
    totalReadySessions: 1, hasMore: false, nextCursor: null
  })
  titleApi.getSessionTitle.mockResolvedValue(titleSnapshot('session-1', 'Old title', 1))
  titleApi.updateSessionTitle.mockResolvedValue(titleSnapshot('session-1', 'New title', 2))
  titleApi.fetchLongTermMemoryGraph.mockRejectedValue(Object.assign(new Error('Unavailable'), {
    code: 'memory_graph_not_available'
  }))

  const user = userEvent.setup()
  render(<MemoryPanel standalone />)
  await user.click(await screen.findByRole('button', { name: /Old title/ }))
  expect(await screen.findByRole('heading', { name: 'Old title' })).toBeTruthy()
  expect(screen.getAllByText('session-1').length).toBeGreaterThanOrEqual(2)

  await user.click(screen.getByRole('button', { name: 'Rename Session' }))
  const input = await screen.findByRole('textbox', { name: 'Session name' })
  await user.clear(input)
  await user.type(input, 'New title')
  await user.click(screen.getByRole('button', { name: 'Save name' }))

  expect(await screen.findByRole('heading', { name: 'New title' })).toBeTruthy()
  expect(screen.getByRole('button', { name: /New title/ })).toBeTruthy()
  expect(screen.getAllByText('session-1').length).toBeGreaterThanOrEqual(2)
  expect(titleApi.updateSessionTitle).toHaveBeenCalledWith(expect.objectContaining({
    sessionId: 'session-1', title: 'New title', expectedRevision: 1
  }))
})

function titleSnapshot(sessionId, title, titleRevision) {
  return { status: 'ok', sessionId, title, titleRevision, titleUpdatedAt: null }
}

function catalogSession(sessionId, title, version) {
  return {
    sessionId,
    title,
    titleRevision: 1,
    memoryVersion: version,
    componentVersions: { episodic: version, semantic: version, visual: version },
    activeRootKind: 'em2mem',
    qaAligned: true,
    updatedAt: null
  }
}

function publication(sessionId, version, revision) {
  return {
    sessionId,
    revision,
    memory: {
      ...catalogSession(sessionId, sessionId, version),
      counts: { total: 1, episodic: 1, semantic: 0, visual: 0 },
      canEdit: true,
      canRollback: true,
      editDisabledReason: null,
      queryReady: true
    }
  }
}
