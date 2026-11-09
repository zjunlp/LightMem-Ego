// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import CurrentSessionMemoryEditor from './CurrentSessionMemoryEditor.jsx'

const api = vi.hoisted(() => ({
  fetchLongTermMemories: vi.fn(),
  fetchLongTermMemoryGraph: vi.fn(),
  rollbackThirtySecondMemories: vi.fn(),
  updateThirtySecondMemories: vi.fn(),
  getSessionTitle: vi.fn(),
  updateSessionTitle: vi.fn(),
  validateSessionTitle: vi.fn(() => '')
}))

vi.mock('../api/lightmem_egoApi.js', () => api)

beforeEach(() => {
  Object.values(api).forEach((mock) => mock.mockReset())
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  api.fetchLongTermMemories.mockResolvedValue(buildMemory())
})

afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('CurrentSessionMemoryEditor', () => {
  it('loads Memory, edits only permitted records, and cancels without a write', async () => {
    const user = userEvent.setup()
    render(<CurrentSessionMemoryEditor sessionId="session-1" />)

    await user.click(screen.getByRole('button', { name: 'Edit Current Session Long-Term Memory' }))
    const editor = await screen.findByRole('textbox', { name: /record-1/ })
    expect(screen.queryByRole('textbox', { name: /record-2/ })).toBeNull()
    expect(screen.getByText('Read-only: Stable record ID is missing')).toBeTruthy()

    await user.clear(editor)
    await user.type(editor, 'changed')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(api.updateThirtySecondMemories).not.toHaveBeenCalled()
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('submits only changed records and publishes the refreshed graph', async () => {
    const user = userEvent.setup()
    const updatedMemory = buildMemory({ version: 2, content: 'changed', canRollback: true })
    const graph = buildGraph(2)
    api.updateThirtySecondMemories.mockResolvedValue({
      status: 'ok',
      success: true,
      message: '当前记忆已修改，请刷新页面。',
      sessionId: 'session-1',
      memoryVersion: 2,
      memory: updatedMemory,
      graphUpdate: {
        status: 'ready',
        scale: '30sec',
        memoryVersion: 2,
        componentVersions: { episodic: 2, graph: 2, semantic: 2 },
        graphVersions: { episodicGraph: 2, semanticGraph: 2 }
      },
      propagation: buildPropagation(2),
      canRollback: true
    })
    api.fetchLongTermMemoryGraph.mockResolvedValue({ status: 'ok', graph, etag: '"mg-2"' })
    const onPublication = vi.fn()
    render(<CurrentSessionMemoryEditor sessionId="session-1" onPublication={onPublication} />)

    await user.click(screen.getByRole('button', { name: 'Edit Current Session Long-Term Memory' }))
    const editor = await screen.findByRole('textbox', { name: /record-1/ })
    await user.clear(editor)
    await user.type(editor, 'changed')
    await user.click(screen.getByRole('button', { name: 'Save Changes' }))

    await waitFor(() => expect(api.updateThirtySecondMemories).toHaveBeenCalledTimes(1))
    expect(api.updateThirtySecondMemories.mock.calls[0][0]).toMatchObject({
      sessionId: 'session-1',
      baseVersion: 1,
      records: [{ id: 'record-1', content: 'changed' }]
    })
    expect(await screen.findByText('Memory changes saved successfully. The latest version is displayed.')).toBeTruthy()
    await waitFor(() => expect(onPublication).toHaveBeenLastCalledWith(expect.objectContaining({
      sessionId: 'session-1',
      memory: updatedMemory,
      graph,
      graphStatus: 'ready'
    })))
    expect(screen.getByRole('button', { name: 'Undo Changes' })).toBeTruthy()
  })

  it('blocks an empty change before calling the backend', async () => {
    const user = userEvent.setup()
    render(<CurrentSessionMemoryEditor sessionId="session-1" />)
    await user.click(screen.getByRole('button', { name: 'Edit Current Session Long-Term Memory' }))
    const editor = await screen.findByRole('textbox', { name: /record-1/ })
    await user.clear(editor)
    await user.type(editor, '   ')
    await user.click(screen.getByRole('button', { name: 'Save Changes' }))
    expect(await screen.findByText('Content cannot be empty.')).toBeTruthy()
    expect(api.updateThirtySecondMemories).not.toHaveBeenCalled()
  })

  it('rolls back once and publishes the restored Memory and Graph', async () => {
    const user = userEvent.setup()
    const editableMemory = buildMemory({ version: 2, content: 'changed', canRollback: true })
    const restoredMemory = buildMemory({ version: 3, content: 'original', canRollback: false })
    const graph = buildGraph(3)
    api.fetchLongTermMemories.mockResolvedValue(editableMemory)
    api.rollbackThirtySecondMemories.mockResolvedValue({
      status: 'ok',
      success: true,
      message: '已撤回上一次记忆修改，请刷新页面。',
      sessionId: 'session-1',
      memoryVersion: 3,
      memory: restoredMemory,
      graphUpdate: {
        status: 'ready',
        scale: '30sec',
        memoryVersion: 3,
        componentVersions: { episodic: 3, graph: 3, semantic: 3 },
        graphVersions: { episodicGraph: 3, semanticGraph: 3 }
      },
      propagation: buildPropagation(3),
      canRollback: false
    })
    api.fetchLongTermMemoryGraph.mockResolvedValue({ status: 'ok', graph, etag: '"mg-3"' })
    const onPublication = vi.fn()
    render(<CurrentSessionMemoryEditor sessionId="session-1" onPublication={onPublication} />)

    await user.click(screen.getByRole('button', { name: 'Edit Current Session Long-Term Memory' }))
    await screen.findByRole('dialog')
    await user.click(screen.getByRole('button', { name: 'Undo Last Change' }))

    await waitFor(() => expect(api.rollbackThirtySecondMemories).toHaveBeenCalledWith(expect.objectContaining({
      sessionId: 'session-1',
      currentVersion: 2
    })))
    expect(await screen.findByText('Memory rollback succeeded. The latest version is displayed.')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Undo Changes' })).toBeNull()
    await waitFor(() => expect(onPublication).toHaveBeenLastCalledWith(expect.objectContaining({ graph })))
  })

  it('ignores a stale GET after the Session changes', async () => {
    const user = userEvent.setup()
    let resolveFirst
    api.fetchLongTermMemories
      .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve }))
      .mockResolvedValueOnce(buildMemory({ sessionId: 'session-2', content: 'session two' }))
    const { rerender } = render(<CurrentSessionMemoryEditor sessionId="session-1" />)

    await user.click(screen.getByRole('button', { name: 'Edit Current Session Long-Term Memory' }))
    rerender(<CurrentSessionMemoryEditor sessionId="session-2" />)
    resolveFirst(buildMemory({ content: 'stale session one' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())

    await user.click(screen.getByRole('button', { name: 'Edit Current Session Long-Term Memory' }))
    const editor = await screen.findByRole('textbox', { name: /record-1/ })
    expect(editor.value).toBe('session two')
  })

  it('shows an explicit empty state when the Session has no long-term memory', async () => {
    const user = userEvent.setup()
    api.fetchLongTermMemories.mockResolvedValue(buildMemory({ empty: true }))
    render(<CurrentSessionMemoryEditor sessionId="session-1" />)

    await user.click(screen.getByRole('button', { name: 'Edit Current Session Long-Term Memory' }))

    expect(await screen.findByText('No long-term memory available.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Save Changes' }).disabled).toBe(true)
    expect(screen.queryByRole('textbox')).toBeNull()
  })

  it('keeps a backend without full propagation in read-only mode', async () => {
    const user = userEvent.setup()
    api.fetchLongTermMemories.mockResolvedValue(buildMemory({ propagationSupported: false }))
    render(<CurrentSessionMemoryEditor sessionId="session-1" />)

    await user.click(screen.getByRole('button', { name: 'Edit Current Session Long-Term Memory' }))

    expect(await screen.findByText('The editing API is not enabled by the backend')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Save Changes' }).disabled).toBe(true)
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(api.updateThirtySecondMemories).not.toHaveBeenCalled()
  })

  it('retries a derivation failure with the same immutable operation', async () => {
    const user = userEvent.setup()
    const failure = Object.assign(new Error('unsafe backend detail'), {
      code: 'memory_derivation_failed',
      details: { component: 'semantic' },
      requestId: 'request-derive'
    })
    const updatedMemory = buildMemory({ version: 2, content: 'changed', canRollback: true })
    api.updateThirtySecondMemories
      .mockRejectedValueOnce(failure)
      .mockResolvedValueOnce(buildMutationResult(updatedMemory, 2))
    api.fetchLongTermMemoryGraph.mockResolvedValue({ status: 'ok', graph: buildGraph(2), etag: '"mg-2"' })
    render(<CurrentSessionMemoryEditor sessionId="session-1" />)

    await user.click(screen.getByRole('button', { name: 'Edit Current Session Long-Term Memory' }))
    const editor = await screen.findByRole('textbox', { name: /record-1/ })
    await user.clear(editor)
    await user.type(editor, 'changed')
    await user.click(screen.getByRole('button', { name: 'Save Changes' }))

    expect(await screen.findByText(/Component: semantic/)).toBeTruthy()
    await user.click(screen.getByRole('button', { name: 'Retry rebuild' }))
    await screen.findByText('Memory changes saved successfully. The latest version is displayed.')

    expect(api.updateThirtySecondMemories).toHaveBeenCalledTimes(2)
    const first = api.updateThirtySecondMemories.mock.calls[0][0]
    const second = api.updateThirtySecondMemories.mock.calls[1][0]
    expect(second.idempotencyKey).toBe(first.idempotencyKey)
    expect(second.records).toBe(first.records)
  })
})

function buildMemory({ sessionId = 'session-1', version = 1, content = 'original', canRollback = false, empty = false, propagationSupported = true } = {}) {
  const records = empty
    ? []
    : [
        {
          id: 'record-1',
          content,
          editable: propagationSupported,
          editDisabledReason: propagationSupported ? null : 'full_propagation_not_supported'
        },
        { id: 'record-2', content: 'readonly', editable: false, editDisabledReason: 'missing_stable_id' }
      ]
  return {
    status: 'ok',
    schemaVersion: 1,
    sessionId,
    memorySource: 'M_lt',
    canEdit: propagationSupported && !empty,
    editDisabledReason: propagationSupported && !empty ? null : 'no_editable_records',
    editSupported: propagationSupported,
    fullPropagationSupported: propagationSupported,
    memoryVersion: version,
    componentVersions: { episodic: version, semantic: version, visual: version },
    activeRootKind: 'em2mem',
    qaAligned: true,
    updatedAt: null,
    completeRecordSet: true,
    fieldTruncation: false,
    truncatedFieldCount: 0,
    availability: { episodic: 'ready', semantic: 'ready', visual: 'ready' },
    counts: { total: records.length, episodic: records.length, semantic: 0, visual: 0, episodicByGranularity: { '30sec': records.length, '3min': 0, '10min': 0, '1h': 0 } },
    episodic: {
      '30sec': records,
      '3min': [],
      '10min': [],
      '1h': []
    },
    semantic: [],
    visual: [],
    editPolicy: { field: 'episodic.30sec[].content', maxCharsPerRecord: 4000, maxTotalChars: 32000, lengthUnit: 'unicode_code_points' },
    propagation: propagationSupported ? buildPropagation(version) : null,
    canRollback
  }
}

function buildGraph(version) {
  return {
    sessionId: 'session-1',
    memoryVersion: version,
    componentVersions: { episodic: version, graph: version, semantic: version },
    graphVersions: { episodicGraph: version, semanticGraph: version },
    episodic: { nodes: [], edges: [] },
    semantic: { nodes: [], edges: [], timeline: [] },
    documents: {},
    indexes: {},
    warnings: {},
    clientWarnings: { invalidNodeCount: 0, duplicateNodeCount: 0, invalidEdgeCount: 0, missingRelationEventIdCount: 0 }
  }
}

function buildPropagation(version) {
  return {
    status: 'ready',
    sourceScale: '30sec',
    targetMemoryVersion: version,
    derivedScales: {
      '3min': { status: 'ready', recomputedRecords: 0 },
      '10min': { status: 'ready', recomputedRecords: 0 },
      '1h': { status: 'ready', recomputedRecords: 0 }
    },
    semantic: { status: 'ready', version, changedFacts: 0 },
    graphs: {
      episodic: { status: 'ready', version },
      semantic: { status: 'ready', version }
    },
    backends: {
      multiscale: 'llm',
      triplets: 'llm_openie',
      semantic: 'llm_semantic_extraction_consolidation'
    }
  }
}

function buildMutationResult(memory, version) {
  return {
    status: 'ok',
    success: true,
    message: 'updated',
    sessionId: memory.sessionId,
    memoryVersion: version,
    memory,
    graphUpdate: {
      status: 'ready',
      scale: '30sec',
      memoryVersion: version,
      componentVersions: { episodic: version, graph: version, semantic: version },
      graphVersions: { episodicGraph: version, semanticGraph: version }
    },
    propagation: buildPropagation(version),
    canRollback: memory.canRollback
  }
}
