import { afterEach, describe, expect, it, vi } from 'vitest'
import { createMemoryEditStore, codePointLength } from './memoryEditStore.js'
import { buildMemory, buildMixedImport, buildNoEditable, buildMutation, buildGraph } from '../test/memoryFixtures.js'

const stores = []
const flush = async () => { for (let i = 0; i < 8; i++) await Promise.resolve() }
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done }); return { promise, resolve } }
function setup(memory = buildMemory()) {
  const api = {
    fetchLongTermMemories: vi.fn().mockResolvedValue(memory),
    updateThirtySecondMemories: vi.fn().mockResolvedValue(buildMutation()),
    rollbackThirtySecondMemories: vi.fn().mockResolvedValue(buildMutation(buildMemory({ version: 3, canEdit: false }))),
    fetchLongTermMemoryGraph: vi.fn().mockResolvedValue({ graph: buildGraph(), etag: 'v2' })
  }
  const onPublication = vi.fn()
  const store = createMemoryEditStore({ api, onPublication })
  stores.push(store)
  store.seed(memory.sessionId, memory)
  return { store, api, onPublication, state: () => store.getSnapshot(memory.sessionId) }
}
afterEach(() => { stores.splice(0).forEach((store) => store.dispose()); vi.useRealTimers() })

describe('Session memory transactions', () => {
  it('serializes writes to one Session while allowing an independent Session', async () => {
    const { store, api } = setup()
    const first = deferred()
    api.updateThirtySecondMemories.mockReturnValueOnce(first.promise)
    store.setDraft('session-1', 'record-1', 'changed')
    store.submit('session-1')
    store.submit('session-1')
    store.rollback('session-1')
    expect(api.updateThirtySecondMemories).toHaveBeenCalledTimes(1)
    expect(api.rollbackThirtySecondMemories).not.toHaveBeenCalled()
    store.seed('session-2', buildMemory({ sessionId: 'session-2' }))
    store.setDraft('session-2', 'record-1', 'other')
    api.updateThirtySecondMemories.mockResolvedValueOnce(buildMutation(buildMemory({ sessionId: 'session-2', version: 2 })))
    store.submit('session-2')
    expect(api.updateThirtySecondMemories).toHaveBeenCalledTimes(2)
    first.resolve(buildMutation())
    await flush()
    expect(store.getSnapshot('session-1').memory.memoryVersion).toBe(2)
  })

  it('freezes the request and retries the same key after an uncertain network failure', async () => {
    const { store, api, state } = setup()
    api.updateThirtySecondMemories.mockRejectedValueOnce(new TypeError('Connection dropped'))
    store.setDraft('session-1', 'record-1', 'changed')
    store.submit('session-1')
    await flush()
    expect(state().error.outcomeUnknown).toBe(true)
    expect(store.discard('session-1')).toBe(false)
    store.setDraft('session-1', 'record-1', 'must not replace request')
    store.submit('session-1')
    expect(api.updateThirtySecondMemories).toHaveBeenCalledTimes(1)
    await store.retry('session-1')
    const [first, second] = api.updateThirtySecondMemories.mock.calls.map(([options]) => options)
    expect(second.idempotencyKey).toBe(first.idempotencyKey)
    expect(second.records).toBe(first.records)
    expect(second.records).toEqual([{ id: 'record-1', content: 'changed' }])
    expect(state().pendingOperation).toBeNull()
    store.setDraft('session-1', 'record-1', 'another change')
    store.submit('session-1')
    expect(api.updateThirtySecondMemories.mock.calls[2][0].idempotencyKey).not.toBe(first.idempotencyKey)
  })

  it('retains the original key after the write timeout and checks it again', async () => {
    vi.useFakeTimers()
    const { store, api, state } = setup()
    api.updateThirtySecondMemories.mockImplementationOnce(({ signal }) => new Promise((_, reject) => {
      signal.addEventListener('abort', () => reject(Object.assign(new Error('timeout'), { name: 'AbortError' })))
    }))
    store.setDraft('session-1', 'record-1', 'changed')
    store.submit('session-1')
    await vi.advanceTimersByTimeAsync(300000)
    expect(state().error.code).toBe('request_timeout')
    await store.retry('session-1')
    expect(api.updateThirtySecondMemories.mock.calls[1][0].idempotencyKey).toBe(api.updateThirtySecondMemories.mock.calls[0][0].idempotencyKey)
  })

  it('checks an in-progress operation with backoff and does not create another transaction', async () => {
    vi.useFakeTimers()
    const { store, api, state } = setup()
    api.updateThirtySecondMemories.mockRejectedValueOnce({ code: 'memory_update_in_progress' })
    store.setDraft('session-1', 'record-1', 'changed')
    store.submit('session-1')
    await flush()
    expect(state().pendingOperation.attempt).toBe(1)
    await vi.advanceTimersByTimeAsync(1999)
    expect(api.updateThirtySecondMemories).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(api.updateThirtySecondMemories).toHaveBeenCalledTimes(2)
    expect(api.updateThirtySecondMemories.mock.calls[1][0].idempotencyKey).toBe(api.updateThirtySecondMemories.mock.calls[0][0].idempotencyKey)
  })

  it('keeps a conflicted draft and base version until explicit discard', async () => {
    const { store, api, state } = setup()
    api.updateThirtySecondMemories.mockRejectedValueOnce({ status: 409, code: 'memory_version_conflict' })
    store.setDraft('session-1', 'record-1', 'my draft')
    store.submit('session-1')
    await flush()
    expect(state().drafts['record-1']).toBe('my draft')
    expect(state().memory.memoryVersion).toBe(1)
    expect(api.fetchLongTermMemories).not.toHaveBeenCalled()
    store.submit('session-1')
    expect(api.updateThirtySecondMemories).toHaveBeenCalledTimes(1)
    api.fetchLongTermMemories.mockResolvedValueOnce(buildMemory({ version: 3 }))
    expect(store.discard('session-1')).toBe(true)
    await store.load('session-1', { discard: true })
    expect(state().memory.memoryVersion).toBe(3)
  })

  it('uses code points at both limits and submits only changed allowed records', async () => {
    const memory = buildMixedImport()
    memory.editPolicy = { ...memory.editPolicy, maxCharsPerRecord: 2, maxTotalChars: 3 }
    memory.episodic['30sec'].push({ id: 'second', content: 'unchanged', editable: true })
    const { store, api, state } = setup(memory)
    expect(codePointLength('中😀')).toBe(2)
    store.setDraft('session-1', 'record-1', '中😀!')
    store.submit('session-1')
    expect(state().fieldErrors['record-1']).toBeTruthy()
    store.setDraft('session-1', 'record-1', '中😀')
    store.setDraft('session-1', 'second', '😀😀')
    store.submit('session-1')
    expect(state().fieldErrors._global).toBeTruthy()
    expect(api.updateThirtySecondMemories).not.toHaveBeenCalled()
    store.setDraft('session-1', 'second', 'unchanged')
    store.setDraft('session-1', 'summary-1', 'cannot edit derived summary')
    store.submit('session-1')
    expect(api.updateThirtySecondMemories.mock.calls[0][0].records).toEqual([{ id: 'record-1', content: '中😀' }])
    await flush()
  })

  it('accepts an imported publication before query warmup and keeps visual data', async () => {
    const original = buildMixedImport()
    const { store, api, state, onPublication } = setup(original)
    const updated = buildMixedImport({ version: 2, content: 'changed', canRollback: true })
    updated.episodic['3min'][0].content = 'Rebuilt summary'
    api.updateThirtySecondMemories.mockResolvedValueOnce(buildMutation(updated, { queryReady: false }))
    store.setDraft('session-1', 'record-1', 'changed')
    store.submit('session-1')
    await flush()
    expect(state().notice).toBe('Memory saved. Preparing query service…')
    expect(state().memory.visual).toEqual(original.visual)
    expect(state().memory.episodic['3min'][0].content).toBe('Rebuilt summary')
    expect(onPublication.mock.calls[0][0]).toMatchObject({ sessionId: 'session-1', queryReady: false, graph: null })
    store.setQueryState('session-1', false, true)
    expect(state().notice).toBe('Memory saved. Query preparation failed.')
    expect(state().memory.memoryVersion).toBe(2)
  })

  it('reports success when the server accepted an update but its response versions were inconsistent', async () => {
    const updated = buildMemory({ version: 2, content: 'changed', canRollback: true })
    const { store, api, state, onPublication } = setup()
    api.updateThirtySecondMemories.mockRejectedValueOnce(Object.assign(new Error('inconsistent response'), {
      code: 'invalid_memory_mutation_response', serverAccepted: true, acceptedVersion: 3
    }))
    api.fetchLongTermMemories.mockResolvedValueOnce(updated)

    store.setDraft('session-1', 'record-1', 'changed')
    store.submit('session-1')
    await flush()

    expect(state()).toMatchObject({
      memory: { memoryVersion: 2, episodic: { '30sec': [{ content: 'changed' }] } },
      pendingOperation: null,
      checkingResult: false,
      error: null,
      notice: 'Memory changes saved successfully. The latest version is displayed.'
    })
    expect(onPublication).toHaveBeenCalledWith(expect.objectContaining({ sessionId: 'session-1', memory: updated }))
  })

  it('offers a manual latest-memory refresh when the first confirmation read is stale', async () => {
    const updated = buildMemory({ version: 2, content: 'changed', canRollback: true })
    const { store, api, state } = setup()
    api.updateThirtySecondMemories.mockRejectedValueOnce(Object.assign(new Error('inconsistent response'), {
      code: 'invalid_memory_mutation_response', serverAccepted: true, acceptedVersion: 2
    }))
    api.fetchLongTermMemories.mockResolvedValueOnce(buildMemory()).mockResolvedValueOnce(updated)

    store.setDraft('session-1', 'record-1', 'changed')
    store.submit('session-1')
    await flush()
    expect(state().error).toMatchObject({ outcomeUnknown: true })
    expect(state().pendingOperation).not.toBeNull()

    await store.refreshLatest('session-1')

    expect(state()).toMatchObject({
      memory: { memoryVersion: 2, episodic: { '30sec': [{ content: 'changed' }] } },
      pendingOperation: null,
      checkingResult: false,
      error: null,
      notice: 'Memory changes saved successfully. The latest version is displayed.'
    })
  })

  it('confirms an accepted rollback against the refreshed complete snapshot', async () => {
    const restored = buildNoEditable({ version: 3 })
    const { store, api, state } = setup(buildMemory({ version: 2, canRollback: true }))
    api.rollbackThirtySecondMemories.mockRejectedValueOnce(Object.assign(new Error('inconsistent response'), {
      code: 'invalid_memory_mutation_response', serverAccepted: true, acceptedVersion: 4
    }))
    api.fetchLongTermMemories.mockResolvedValueOnce(restored)

    store.rollback('session-1')
    await flush()

    expect(state()).toMatchObject({
      memory: { memoryVersion: 3, canEdit: false, episodic: { '30sec': [] } },
      pendingOperation: null,
      error: null,
      notice: 'Memory rollback succeeded. The latest version is displayed.'
    })
  })

  it('restores a complete readonly legacy snapshot with an independent rollback key', async () => {
    const { store, api, state } = setup(buildMemory({ version: 2, canRollback: true }))
    const restored = buildNoEditable({ version: 3 })
    restored.activeRootKind = 'lightmem_ego'
    restored.qaAligned = false
    api.rollbackThirtySecondMemories.mockResolvedValueOnce(buildMutation(restored, { queryReady: false }))
    store.rollback('session-1')
    await flush()
    expect(api.rollbackThirtySecondMemories.mock.calls[0][0].currentVersion).toBe(2)
    expect(state().memory).toMatchObject({ memoryVersion: 3, canEdit: false, activeRootKind: 'lightmem_ego', visual: restored.visual })
  })

  it('does not install a late read or graph over a publication', async () => {
    const { store, api, state, onPublication } = setup()
    const read = deferred()
    const graph = deferred()
    api.fetchLongTermMemories.mockReturnValueOnce(read.promise)
    api.fetchLongTermMemoryGraph.mockReturnValueOnce(graph.promise)
    const loading = store.load('session-1')
    // A second view publishes a complete newer snapshot while this read is pending.
    store.seed('session-1', buildMemory({ version: 2 }))
    read.resolve(buildMemory())
    await loading
    expect(state().memory.memoryVersion).toBe(2)
    store.setDraft('session-1', 'record-1', 'change')
    api.updateThirtySecondMemories.mockResolvedValueOnce(buildMutation(buildMemory({ version: 3 })))
    store.submit('session-1')
    await flush()
    store.seed('session-1', buildMemory({ version: 4 }))
    graph.resolve({ graph: buildGraph(3), etag: 'old' })
    await flush()
    expect(state().memory.memoryVersion).toBe(4)
    expect(onPublication.mock.calls.some(([update]) => update.graph?.memoryVersion === 3)).toBe(false)
  })

  it('keeps an operation when the editor subscription leaves and returns', async () => {
    const { store, api, state } = setup()
    api.updateThirtySecondMemories.mockRejectedValueOnce(new TypeError('network'))
    const unsubscribe = store.subscribe('session-1', vi.fn())
    store.setDraft('session-1', 'record-1', 'draft')
    store.submit('session-1')
    await flush()
    const operation = state().pendingOperation
    unsubscribe()
    store.subscribe('session-1', vi.fn())
    store.seed('session-1', buildMemory({ version: 2 }))
    expect(state().pendingOperation).toBe(operation)
    await store.retry('session-1')
    expect(api.updateThirtySecondMemories.mock.calls[1][0].idempotencyKey).toBe(operation.key)
  })
})
