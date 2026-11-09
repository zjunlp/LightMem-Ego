// @vitest-environment jsdom
import { StrictMode, useState } from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import MemoryDialog from './MemoryDialog.jsx'
import CurrentSessionMemoryEditor from './CurrentSessionMemoryEditor.jsx'
import { AdminModeProvider } from './AdminMode.jsx'
import { MemoryEditProvider } from '../hooks/useSessionMemoryEditor.jsx'
import { buildMemory, buildMixedImport, buildNoEditable, buildMutation, buildGraph } from '../test/memoryFixtures.js'
import * as api from '../api/lightmem_egoApi.js'
import * as importApi from '../api/memoryImportApi.js'

vi.mock('../api/lightmem_egoApi.js', async (original) => ({
  ...await original(), fetchLongTermMemories: vi.fn(), updateThirtySecondMemories: vi.fn(),
  rollbackThirtySecondMemories: vi.fn(), fetchLongTermMemoryGraph: vi.fn(), exportLongTermMemoryJson: vi.fn()
}))
vi.mock('../api/memoryImportApi.js', async (original) => ({
  ...await original(), getMemorySessionStatus: vi.fn(), getSessionQueryWarmup: vi.fn()
}))
vi.mock('./SessionTitleEditor.jsx', () => ({ default: () => null }))
vi.mock('./memory-graph/MemoryGraphView.jsx', () => ({ default: ({ authoritativeUpdate }) => <div>Graph version {authoritativeUpdate?.memory?.memoryVersion}</div> }))

const normalized = (memory) => api.normalizeLongTermMemoryResponse(memory, memory.sessionId)
const session = { sessionId: 'session-1', title: 'Historical Session', memoryVersion: 1, activeRootKind: 'em2mem' }
beforeEach(() => {
  vi.clearAllMocks()
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  api.fetchLongTermMemoryGraph.mockResolvedValue({ graph: buildGraph(), etag: 'v2' })
  importApi.getMemorySessionStatus.mockResolvedValue({ sessionId: 'session-1', queryAllowed: true, queryReady: false })
  importApi.getSessionQueryWarmup.mockResolvedValue({ sessionId: 'session-1', status: 'warming_query', memoryVersion: 2 })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('historical Session detail editing', () => {
  it('keeps viewing and return navigation available while memory editing is locked', async () => {
    sessionStorage.removeItem('lightmem-history.admin-mode.v1')
    const memory = normalized(buildMemory())
    api.fetchLongTermMemories.mockResolvedValue(memory)
    render(<AdminModeProvider><MemoryDialog session={session} cachedMemory={memory} initialTab="records" onClose={vi.fn()} /></AdminModeProvider>)
    await screen.findByText('Enter admin mode to edit memories.')
    expect(screen.queryByRole('button', {name:'Edit long-term memory'})).toBeNull()
    expect(document.querySelector('.session-memory-actions')).toBeTruthy()
    expect(api.updateThirtySecondMemories).not.toHaveBeenCalled()
  })
  it.each(['ordinary', 'legacy'])('edits a non-active %s Session inside the existing dialog', async (kind) => {
    const user = userEvent.setup()
    const memory = normalized(buildMemory({ kind }))
    api.fetchLongTermMemories.mockResolvedValue(memory)
    api.updateThirtySecondMemories.mockResolvedValue(buildMutation(normalized(buildMemory({ version: 2, content: 'changed', canRollback: true }))))
    const onPublication = vi.fn()
    render(<MemoryDialog session={session} cachedMemory={memory} initialTab="records" onPublication={onPublication} onClose={vi.fn()} />)
    await user.click(screen.getByRole('button', { name: 'Edit long-term memory' }))
    const input = await screen.findByRole('textbox', { name: /record-1/ })
    expect(screen.getAllByRole('dialog')).toHaveLength(1)
    await user.clear(input)
    await user.type(input, 'changed')
    await user.click(screen.getByRole('button', { name: 'Save Changes' }))
    await waitFor(() => expect(api.updateThirtySecondMemories).toHaveBeenCalledTimes(1))
    expect(api.updateThirtySecondMemories.mock.calls[0][0]).toMatchObject({ sessionId: 'session-1', baseVersion: 1, records: [{ id: 'record-1', content: 'changed' }] })
    await screen.findByText('Memory changes saved successfully. The latest version is displayed.')
    expect(screen.queryByRole('textbox', { name: /record-1/ })).toBeNull()
    expect(onPublication).toHaveBeenCalledWith(expect.objectContaining({ sessionId: 'session-1', memory: expect.objectContaining({ memoryVersion: 2 }) }))
  })

  it('confirms an accepted save from the refresh button when the first read is stale', async () => {
    const user = userEvent.setup()
    const memory = normalized(buildMemory())
    const updated = normalized(buildMemory({ version: 2, content: 'updated', canRollback: true }))
    const reads = [memory, memory, memory, updated]
    api.fetchLongTermMemories.mockImplementation(async () => reads.shift() || updated)
    api.updateThirtySecondMemories.mockRejectedValueOnce(Object.assign(new Error('inconsistent response'), {
      code: 'invalid_memory_mutation_response', serverAccepted: true, acceptedVersion: 2
    }))
    const onPublication = vi.fn()
    render(<MemoryDialog session={session} cachedMemory={memory} initialTab="records" onPublication={onPublication} onClose={vi.fn()} />)
    await expandThirtySecondRecords(user)

    await user.click(screen.getByRole('button', { name: 'Edit long-term memory' }))
    const input = await screen.findByRole('textbox', { name: /record-1/ })
    await user.clear(input)
    await user.type(input, 'updated')
    await user.click(screen.getByRole('button', { name: 'Save Changes' }))

    await screen.findByText(/server accepted the change/)
    await user.click(screen.getByRole('button', { name: 'Refresh latest memory' }))
    await waitFor(() => expect(onPublication).toHaveBeenCalledWith(expect.objectContaining({
      sessionId: 'session-1', memory: expect.objectContaining({ memoryVersion: 2 })
    })))
    expect(screen.queryByText(/server accepted the change/)).toBeNull()
    expect(screen.getByText('updated')).toBeTruthy()
  })

  it('keeps the save success message without a graph refresh reminder when graph service is unavailable', async () => {
    const user = userEvent.setup()
    const memory = normalized(buildMemory())
    api.fetchLongTermMemories.mockResolvedValue(memory)
    api.updateThirtySecondMemories.mockResolvedValue(buildMutation(
      normalized(buildMemory({ version: 2, content: 'updated', canRollback: true }))
    ))
    api.fetchLongTermMemoryGraph.mockRejectedValue({ status: 404, code: 'memory_graph_disabled' })
    render(<MemoryDialog session={session} cachedMemory={memory} initialTab="records" onClose={vi.fn()} />)

    await user.click(screen.getByRole('button', { name: 'Edit long-term memory' }))
    const input = await screen.findByRole('textbox', { name: /record-1/ })
    await user.clear(input)
    await user.type(input, 'updated')
    await user.click(screen.getByRole('button', { name: 'Save Changes' }))

    expect(await screen.findByText('Memory changes saved successfully. The latest version is displayed.')).toBeTruthy()
    await waitFor(() => expect(api.fetchLongTermMemoryGraph).toHaveBeenCalled())
    expect(screen.queryByText('Memory updated. Graph is still refreshing.')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Retry Graph' })).toBeNull()
  })

  it('preserves the draft on conflict and reloads only after explicit discard', async () => {
    const user = userEvent.setup()
    const memory = normalized(buildMemory())
    api.fetchLongTermMemories.mockResolvedValue(memory)
    api.updateThirtySecondMemories.mockRejectedValue({ status: 409, code: 'memory_version_conflict' })
    render(<MemoryDialog session={session} cachedMemory={memory} initialTab="records" onClose={vi.fn()} />)
    await user.click(screen.getByRole('button', { name: 'Edit long-term memory' }))
    const input = await screen.findByRole('textbox', { name: /record-1/ })
    await user.clear(input)
    await user.type(input, 'my draft')
    const reads = api.fetchLongTermMemories.mock.calls.length
    await user.click(screen.getByRole('button', { name: 'Save Changes' }))
    await screen.findByText(/Your draft is preserved/)
    expect(input.value).toBe('my draft')
    expect(api.fetchLongTermMemories).toHaveBeenCalledTimes(reads)
    expect(screen.getByRole('button', { name: 'Copy draft' })).toBeTruthy()
    api.fetchLongTermMemories.mockResolvedValue(normalized(buildMemory({ version: 3, content: 'latest' })))
    await user.click(screen.getByRole('button', { name: 'Discard and reload' }))
    await waitFor(() => expect(input.value).toBe('latest'))
  })

  it('only edits real 30sec imported records and disables query after saving', async () => {
    const user = userEvent.setup()
    const memory = normalized(buildMixedImport())
    api.fetchLongTermMemories.mockResolvedValue(memory)
    api.updateThirtySecondMemories.mockResolvedValue(buildMutation(normalized(buildMixedImport({ version: 2, content: 'changed', canRollback: true })), { queryReady: false }))
    render(<MemoryDialog session={{ ...session, sessionKind: 'imported_memory' }} cachedMemory={memory} initialTab="records" onClose={vi.fn()} />)
    await user.click(screen.getByRole('button', { name: 'Edit long-term memory' }))
    expect(await screen.findByText(/Imported summaries and semantic memories may be replaced/)).toBeTruthy()
    expect(screen.getAllByRole('textbox')).toHaveLength(1)
    const input = screen.getByRole('textbox', { name: /record-1/ })
    await user.clear(input)
    await user.type(input, 'changed')
    await user.click(screen.getByRole('button', { name: 'Save Changes' }))
    await waitFor(() => expect(screen.queryByRole('textbox', { name: /record-1/ })).toBeNull())
    await user.click(screen.getByRole('tab', { name: 'Ask Memory' }))
    expect(screen.getByRole('textbox', { name: 'Ask about the imported memory…' }).disabled).toBe(true)
    expect(screen.getAllByText('Memory saved. Preparing query service…').length).toBeGreaterThan(0)
    importApi.getMemorySessionStatus.mockResolvedValue({ queryAllowed: true, queryReady: true })
    importApi.getSessionQueryWarmup.mockResolvedValue({ status: 'ready', memoryVersion: 2 })
    await user.click(screen.getByRole('button', { name: 'Refresh status' }))
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Ask about the imported memory…' }).disabled).toBe(false))
  })

  it('keeps readonly imported data viewable and permits rollback independently', async () => {
    const user = userEvent.setup()
    const memory = normalized(buildNoEditable({ canRollback: true }))
    api.fetchLongTermMemories.mockResolvedValue(memory)
    render(<MemoryDialog session={session} cachedMemory={memory} initialTab="records" onClose={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Edit long-term memory' }).disabled).toBe(true)
    expect(screen.getByRole('button', { name: 'Undo Last Change' }).disabled).toBe(false)
    await user.click(screen.getByRole('button', { name: /Visual Memory/ }))
    expect(screen.getByText('Original visual record')).toBeTruthy()
  })

  it('requires discard confirmation before closing an unsaved detail', async () => {
    const user = userEvent.setup()
    const memory = normalized(buildMemory())
    api.fetchLongTermMemories.mockResolvedValue(memory)
    const onClose = vi.fn()
    render(<MemoryDialog session={session} cachedMemory={memory} initialTab="records" onClose={onClose} />)
    await user.click(screen.getByRole('button', { name: 'Edit long-term memory' }))
    const input = await screen.findByRole('textbox', { name: /record-1/ })
    await user.type(input, ' draft')
    window.confirm.mockReturnValue(false)
    await user.click(screen.getByRole('button', { name: 'Close Memory' }))
    expect(onClose).not.toHaveBeenCalled()
    expect(input.value).toBe('original draft')
    window.confirm.mockReturnValue(true)
    await user.click(screen.getByRole('button', { name: 'Close Memory' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('prevents an old export from downloading after publication', async () => {
    const user = userEvent.setup()
    const memory = normalized(buildMemory())
    api.fetchLongTermMemories.mockResolvedValue(memory)
    let resolveExport
    api.exportLongTermMemoryJson.mockImplementation(() => new Promise((resolve) => { resolveExport = resolve }))
    const { rerender } = render(<MemoryDialog session={session} cachedMemory={memory} initialTab="records" onClose={vi.fn()} />)
    await expandThirtySecondRecords(user)
    await user.click(screen.getByRole('button', { name: 'Export memory record-1 as JSON' }))
    rerender(<MemoryDialog session={session} cachedMemory={memory} initialTab="records" onClose={vi.fn()}
      authoritativeUpdate={{ sessionId: 'session-1', memory: normalized(buildMemory({ version: 2 })), revision: 1 }} />)
    await act(async () => resolveExport({ blob: new Blob(['old']), filename: 'old.json' }))
    expect(screen.getByText('Memory changed during export. Export the latest record again.')).toBeTruthy()
  })

  it('shares the transaction across history and current entry points under StrictMode', async () => {
    const user = userEvent.setup()
    const memory = normalized(buildMemory())
    api.fetchLongTermMemories.mockResolvedValue(memory)
    api.updateThirtySecondMemories.mockRejectedValueOnce(new TypeError('connection dropped'))
      .mockResolvedValueOnce(buildMutation(normalized(buildMemory({ version: 2, content: 'saved' }))))
    const publish = vi.fn()
    function App() {
      const [history, setHistory] = useState(true)
      return <MemoryEditProvider onPublication={publish}>
        <button onClick={() => setHistory(!history)}>Switch entry</button>
        {history ? <MemoryDialog session={session} cachedMemory={memory} initialTab="records" onClose={() => setHistory(false)} />
          : <CurrentSessionMemoryEditor sessionId="session-1" />}
      </MemoryEditProvider>
    }
    render(<StrictMode><App /></StrictMode>)
    await user.click(screen.getByRole('button', { name: 'Edit long-term memory' }))
    const input = await screen.findByRole('textbox', { name: /record-1/ })
    await user.type(input, ' draft')
    await user.click(screen.getByRole('button', { name: 'Save Changes' }))
    await screen.findByText(/network failed/)
    await user.click(screen.getByRole('button', { name: 'Close Memory' }))
    const first = api.updateThirtySecondMemories.mock.calls[0][0]
    // The current entry can check exactly the same operation after history unmounts.
    await user.click(screen.getAllByRole('button', { name: 'Continue checking' }).at(-1))
    await waitFor(() => expect(api.updateThirtySecondMemories).toHaveBeenCalledTimes(2))
    expect(api.updateThirtySecondMemories.mock.calls[1][0].idempotencyKey).toBe(first.idempotencyKey)
    expect(publish).toHaveBeenCalledWith(expect.objectContaining({ sessionId: 'session-1', memory: expect.objectContaining({ memoryVersion: 2 }) }))
  })
})

async function expandThirtySecondRecords(user) {
  await user.click(screen.getByRole('button', { name: /Episodic Memory/ }))
  await user.click(screen.getByRole('button', { name: /30 seconds/ }))
}
