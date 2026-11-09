import { buildMemory, buildPropagation, buildGraphResponse, buildMixedImport, buildNoEditable, buildMutation } from '../test/memoryFixtures.js'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  buildMemoryExportFilename,
  exportLongTermMemoryJson,
  fetchLongTermMemories,
  fetchLongTermMemorySessions,
  getSessionTitle,
  normalizeLongTermMemoryResponse,
  normalizeMemoryMutationResponse,
  normalizeEvidenceFrames,
  sanitizeMemoryExportFilenamePart,
  startImageUploadStream,
  uploadAudioChunk,
  updateSessionTitle,
  updateThirtySecondMemories,
  validateSessionTitle
} from './lightmem_egoApi.js'
import { normalizeMemoryGraphResponse } from '../utils/memoryGraphUtils.js'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('long-term memory edit API', () => {
  it('normalizes Session summary fields and keeps older responses readable', () => {
    const source = buildMemory()
    source.session_summary = 'A concise account of this Session.'
    source.session_summary_version = 1
    source.session_summary_status = 'ready'

    expect(normalizeLongTermMemoryResponse(source, 'session-1')).toMatchObject({
      sessionSummary: 'A concise account of this Session.',
      sessionSummaryVersion: 1,
      sessionSummaryStatus: 'ready'
    })

    const legacy = normalizeLongTermMemoryResponse(buildMemory(), 'session-1')
    expect(legacy).toMatchObject({
      sessionSummary: '',
      sessionSummaryVersion: null,
      sessionSummaryStatus: 'unavailable'
    })
  })

  it('preserves unavailable summary state without exposing stale summary text', () => {
    const source = buildMemory()
    source.sessionSummary = ''
    source.sessionSummaryVersion = 2
    source.sessionSummaryStatus = 'unavailable'

    expect(normalizeLongTermMemoryResponse(source, 'session-1')).toMatchObject({
      sessionSummary: '',
      sessionSummaryVersion: 2,
      sessionSummaryStatus: 'unavailable'
    })
  })

  it('normalizes the edit contract and record permissions', () => {
    const memory = normalizeLongTermMemoryResponse(buildMemory(), 'session-1')
    expect(memory.editPolicy.maxCharsPerRecord).toBe(4000)
    expect(memory.fullPropagationSupported).toBe(true)
    expect(memory.episodic['30sec'][0]).toMatchObject({
      id: 'record-1',
      editable: true,
      editDisabledReason: null
    })
  })

  it('loads the latest Session memory without reusing a cached response', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(buildMemory()), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    }))

    await fetchLongTermMemories({ sessionId: 'session-1' })

    expect(fetchMock.mock.calls[0][1]).toMatchObject({
      cache: 'no-store',
      headers: { 'Cache-Control': 'no-cache' }
    })
  })

  it('keeps legacy backend Memory readable while disabling edits', () => {
    const legacy = buildMemory()
    delete legacy.propagation
    delete legacy.canEdit
    delete legacy.editDisabledReason

    const memory = normalizeLongTermMemoryResponse(legacy, 'session-1')
    expect(memory.editSupported).toBe(false)
    expect(memory.fullPropagationSupported).toBe(false)
    expect(memory.canRollback).toBe(false)
    expect(memory.episodic['30sec'][0]).toMatchObject({
      editable: false,
      editDisabledReason: 'backend_edit_not_supported'
    })
  })

  it('sends only the requested changes with one idempotency key', async () => {
    const responseMemory = buildMemory({ version: 2, content: 'updated', canRollback: true })
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      status: 'ok',
      success: true,
      message: '当前记忆已修改，请刷新页面。',
      sessionId: 'session-1',
      memoryVersion: 2,
      memory: responseMemory,
      graphUpdate: {
        status: 'ready',
        scale: '30sec',
        memoryVersion: 2,
        componentVersions: { episodic: 2, graph: 2, semantic: 2 },
        graphVersions: { episodicGraph: 2, semanticGraph: 2 },
        etag: '"mg-2"'
      },
      propagation: buildPropagation(2),
      canRollback: true
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json', 'X-Request-ID': 'request-1' }
    }))

    const result = await updateThirtySecondMemories({
      sessionId: 'session-1',
      baseVersion: 1,
      records: [{ id: 'record-1', content: 'updated' }],
      idempotencyKey: 'operation-1'
    })

    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toMatch(/\/session\/session-1\/memories\/30sec$/)
    expect(options.method).toBe('PUT')
    expect(options.headers['Idempotency-Key']).toBe('operation-1')
    expect(JSON.parse(options.body)).toEqual({
      baseVersion: 1,
      records: [{ id: 'record-1', content: 'updated' }]
    })
    expect(result.memory.memoryVersion).toBe(2)
    expect(result.propagation.status).toBe('ready')
    expect(result.requestId).toBe('request-1')
  })

  it('keeps server acceptance metadata when a successful response has incomplete propagation', async () => {
    const responseMemory = buildMemory({ version: 2, content: 'updated', canRollback: true })
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      status: 'ok',
      success: true,
      message: 'updated',
      sessionId: 'session-1',
      memoryVersion: 3,
      memory: responseMemory,
      graphUpdate: {
        status: 'ready',
        scale: '30sec',
        memoryVersion: 2,
        componentVersions: { episodic: 2, graph: 2, semantic: 2 },
        graphVersions: { episodicGraph: 2, semanticGraph: 2 }
      },
      canRollback: true
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))

    await expect(updateThirtySecondMemories({
      sessionId: 'session-1',
      baseVersion: 1,
      records: [{ id: 'record-1', content: 'updated' }],
      idempotencyKey: 'operation-1'
    })).rejects.toMatchObject({
      code: 'invalid_memory_mutation_response',
      serverAccepted: true,
      acceptedVersion: 3
    })
  })

  it('preserves structured backend errors and request IDs', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      error: {
        code: 'memory_version_conflict',
        message: 'reload',
        details: { currentVersion: 3 }
      }
    }), {
      status: 409,
      headers: { 'Content-Type': 'application/json', 'X-Request-ID': 'request-conflict' }
    }))

    await expect(updateThirtySecondMemories({
      sessionId: 'session-1',
      baseVersion: 1,
      records: [{ id: 'record-1', content: 'updated' }],
      idempotencyKey: 'operation-1'
    })).rejects.toMatchObject({
      status: 409,
      code: 'memory_version_conflict',
      details: { currentVersion: 3 },
      requestId: 'request-conflict'
    })
  })

  it('normalizes both graph versions while keeping old read-only Graph responses compatible', () => {
    const current = buildGraphResponse(2)
    expect(normalizeMemoryGraphResponse(current, 'session-1').graphVersions).toEqual({
      episodicGraph: 2,
      semanticGraph: 2
    })

    const legacy = buildGraphResponse(1)
    delete legacy.graph_versions
    expect(normalizeMemoryGraphResponse(legacy, 'session-1').graphVersions).toBeNull()
  })
})

describe('long-term memory JSON export API', () => {
  it('uses the Session-scoped route and server filename for a Blob response', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{"exportVersion":"1.0"}', {
      status: 200,
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Disposition': "attachment; filename*=UTF-8''memory-%E7%AC%94%E8%AE%B0.json",
        'X-Request-ID': 'request-export'
      }
    }))

    const result = await exportLongTermMemoryJson({
      sessionId: 'session/1',
      memoryId: 'memory 1',
      memory: { id: 'memory 1', title: null }
    })

    expect(fetchMock.mock.calls[0][0]).toMatch(/\/session\/session%2F1\/memories\/memory%201\/export\?format=json$/)
    expect(fetchMock.mock.calls[0][1]).toMatchObject({
      method: 'GET',
      credentials: 'same-origin',
      headers: { Accept: 'application/json' }
    })
    expect(result.filename).toBe('memory-笔记.json')
    expect(await result.blob.text()).toBe('{"exportVersion":"1.0"}')
    expect(result.requestId).toBe('request-export')
  })

  it('uses the global route only when Session ID is unavailable', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{}', { status: 200 }))

    await exportLongTermMemoryJson({ memoryId: 'mlt_ep_abc' })

    expect(fetchMock.mock.calls[0][0]).toMatch(/\/memories\/mlt_ep_abc\/export\?format=json$/)
  })

  it('preserves backend error code, request ID, and retry metadata', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      status: 'error',
      error: {
        code: 'memory_export_rate_limited',
        message: 'try later',
        details: { windowSeconds: 60 },
        requestId: 'request-limit'
      }
    }), {
      status: 429,
      headers: { 'Content-Type': 'application/json', 'Retry-After': '30' }
    }))

    await expect(exportLongTermMemoryJson({ sessionId: 'session-1', memoryId: 'memory-1' })).rejects.toMatchObject({
      status: 429,
      code: 'memory_export_rate_limited',
      details: { windowSeconds: 60 },
      requestId: 'request-limit',
      retryAfter: '30'
    })
  })

  it('cleans unsafe filename parts and creates a UTC fallback filename', () => {
    expect(sanitizeMemoryExportFilenamePart(' ../笔记\\\n')).toBe('笔记')
    expect(buildMemoryExportFilename(
      { id: 'mlt/ep/1', title: null },
      new Date('2026-09-26T06:30:12.000Z')
    )).toBe('memory-mlt-ep-1-untitled-20260926-063012.json')
  })
})

describe('Session title API', () => {
  it('defaults missing optional catalog titles to the Session ID and revision zero', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      status: 'ok', schema_version: 1, memory_source: 'M_lt', page_size: 10,
      total_ready_sessions: 1, has_more: false, next_cursor: null,
      items: [{
        session_id: 'session-1', memory_version: 1,
        component_versions: { episodic: 1, semantic: 1, visual: 1 },
        active_root_kind: 'em2mem', qa_aligned: true, updated_at: null
      }]
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))

    await expect(fetchLongTermMemorySessions()).resolves.toMatchObject({
      items: [{ sessionId: 'session-1', title: 'session-1', titleRevision: 0, titleUpdatedAt: null }]
    })
  })

  it('reads a title from the same-origin encoded Session route', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      status: 'ok', session_id: 'session/1', title: '办公室 🧠', title_revision: 3,
      title_updated_at: '2026-10-17T08:30:00Z'
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))

    await expect(getSessionTitle({ sessionId: 'session/1' })).resolves.toEqual({
      status: 'ok', sessionId: 'session/1', title: '办公室 🧠', titleRevision: 3,
      titleUpdatedAt: '2026-10-17T08:30:00Z'
    })
    expect(fetchMock.mock.calls[0][0]).toBe('/api/session/session%2F1/title')
    expect(fetchMock.mock.calls[0][1].method).toBe('GET')
  })

  it('patches only the trimmed title and expected revision', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      status: 'ok', session_id: 'session-1', title: '项目 🧠', title_revision: 1, title_updated_at: null
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))

    await updateSessionTitle({ sessionId: 'session-1', title: ' 项目 🧠 ', expectedRevision: 0 })
    expect(fetchMock.mock.calls[0][0]).toBe('/api/session/session-1/title')
    expect(fetchMock.mock.calls[0][1].method).toBe('PATCH')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ title: '项目 🧠', expected_revision: 0 })
  })

  it('normalizes conflict details and preserves request and retry metadata', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      error: {
        code: 'session_title_conflict', message: 'conflict',
        details: { current: { title: 'Server name', title_revision: 4, title_updated_at: null } }
      }
    }), {
      status: 409,
      headers: { 'Content-Type': 'application/json', 'X-Request-ID': 'title-conflict', 'Retry-After': '12' }
    }))

    await expect(updateSessionTitle({ sessionId: 'session-1', title: 'Draft', expectedRevision: 3 })).rejects.toMatchObject({
      code: 'session_title_conflict',
      requestId: 'title-conflict',
      retryAfter: '12',
      details: { current: { sessionId: 'session-1', title: 'Server name', titleRevision: 4 } }
    })
  })

  it('validates trimmed Unicode code points and rejects controls', () => {
    expect(validateSessionTitle('  名称 🧠  ')).toBe('')
    expect(validateSessionTitle('😀'.repeat(100))).toBe('')
    expect(validateSessionTitle('😀'.repeat(101))).toBe('session_title_too_long')
    expect(validateSessionTitle(' \t ')).toBe('session_title_required')
    expect(validateSessionTitle('line 1\nline 2')).toBe('session_title_control_character')
  })
})

describe('Photo Stream API', () => {
  it('formats relative upload time from milliseconds for evidence labels', () => {
    const [frame] = normalizeEvidenceFrames({
      evidence_frames: [{ id: 'frame-2', relative_ts_ms: 2450, caption: 'A scene' }]
    }, 'session-1')

    expect(frame.timestamp).toBe(2.45)
    expect(frame.timestampText).toBe('2.5s')
  })

  it('starts an image upload session through the existing stream endpoint', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      session_id: 'photo-session',
      input_mode: 'frame_audio_stream',
      can_ask: false
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))

    await startImageUploadStream()

    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toMatch(/\/stream\/start$/)
    expect(options.method).toBe('POST')
    expect(JSON.parse(options.body).metadata).toMatchObject({
      source: 'web_image_upload',
      mode: 'image_upload'
    })
  })

  it('uploads a local audio file through the existing audio chunk endpoint', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      session_id: 'media-session',
      audio_index: 0,
      audio_ready: true
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    const audio = new File(['audio'], 'sample.wav', { type: 'audio/wav' })

    await uploadAudioChunk('media-session', audio, {
      audioIndex: 0,
      relativeTsMs: 1250,
      durationMs: 3000,
      source: 'web_audio_upload',
      filename: audio.name
    })

    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toMatch(/\/stream\/media-session\/audio_chunk$/)
    expect(options.method).toBe('POST')
    expect(options.body.get('audio')).toMatchObject({ name: 'sample.wav', type: 'audio/wav', size: 5 })
    expect(options.body.get('audio_index')).toBe('0')
    expect(options.body.get('relative_ts_ms')).toBe('1250')
    expect(options.body.get('duration_ms')).toBe('3000')
    expect(options.body.get('source')).toBe('web_audio_upload')
  })
})


describe('historical Session edit capabilities', () => {
  it.each(['ordinary', 'imported', 'legacy'])('uses backend permissions for %s Sessions without an active Session', (kind) => {
    const memory = normalizeLongTermMemoryResponse(buildMemory({ kind }), 'session-1')
    expect(memory.canEdit).toBe(true)
    expect(memory.editSupported).toBe(true)
    expect(memory.episodic['30sec'][0].editable).toBe(true)
  })

  it('keeps non-30sec imported types readonly and retains real timestamps', () => {
    const source = buildMixedImport()
    const memory = normalizeLongTermMemoryResponse(source, 'session-1')
    expect(memory.episodic['30sec'][0].editable).toBe(true)
    expect(memory.episodic['3min'][0].editable).toBe(false)
    expect(memory.semantic[0].editable).toBe(false)
    expect(memory.visual).toEqual(source.visual)
  })

  it('accepts the new editing capability before any previous derivation publication', () => {
    const source = buildMemory({ kind: 'legacy' })
    source.propagation = null
    const memory = normalizeLongTermMemoryResponse(source, 'session-1')
    expect(memory.canEdit).toBe(true)
    expect(memory.fullPropagationSupported).toBe(false)
  })

  it('keeps a Session without editable records readable and rollback independent', () => {
    const source = buildNoEditable({ canRollback: true })
    const memory = normalizeLongTermMemoryResponse(source, 'session-1')
    expect(memory.canEdit).toBe(false)
    expect(memory.canRollback).toBe(true)
    expect(memory.counts.total).toBe(3)
  })

  it('accepts a saved imported snapshot with queryReady=false', () => {
    const updated = buildMixedImport({ version: 2, canRollback: true })
    updated.sessionSummary = 'Updated Session summary.'
    updated.sessionSummaryVersion = 2
    updated.sessionSummaryStatus = 'ready'
    const result = normalizeMemoryMutationResponse(buildMutation(updated, { queryReady: false }), 'session-1')
    expect(result.queryReady).toBe(false)
    expect(result.memory.memoryVersion).toBe(2)
    expect(result.memory).toMatchObject({
      sessionSummary: 'Updated Session summary.',
      sessionSummaryVersion: 2,
      sessionSummaryStatus: 'ready'
    })
  })

  it('accepts a rollback to a complete readonly snapshot', () => {
    const restored = buildNoEditable({ version: 3 })
    restored.sessionSummary = 'Restored Session summary.'
    restored.sessionSummaryVersion = 3
    restored.sessionSummaryStatus = 'ready'
    const result = normalizeMemoryMutationResponse(buildMutation(restored), 'session-1')
    expect(result.memory.canEdit).toBe(false)
    expect(result.memory.visual).toHaveLength(1)
    expect(result.memory.memoryVersion).toBe(3)
    expect(result.memory.sessionSummary).toBe('Restored Session summary.')
  })

  it('rejects wrong Session IDs and mismatched component versions', () => {
    expect(() => normalizeLongTermMemoryResponse(buildMemory(), 'another-session')).toThrow()
    const result = buildMutation()
    result.graphUpdate.graphVersions.semanticGraph = 1
    expect(() => normalizeMemoryMutationResponse(result, 'session-1')).toThrow()
  })
})
