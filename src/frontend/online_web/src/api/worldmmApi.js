const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://omnispark.zjukg.cn/api'
const DEMO_API_BASE_URL = import.meta.env.VITE_DEMO_API_BASE_URL || API_BASE_URL

const DEFAULT_START_PAYLOAD = {
  input_mode: 'frame_audio_stream',
  chunk_duration: 1,
  metadata: {
    source: 'web_client',
    mode: 'frame_audio_stream'
  }
}

function joinUrl(baseUrl, path) {
  if (/^https?:\/\//i.test(String(path || ''))) {
    return String(path)
  }
  const base = String(baseUrl || API_BASE_URL).replace(/\/+$/, '')
  const suffix = String(path || '').startsWith('/') ? path : `/${path}`
  return `${base}${suffix}`
}

export function buildApiUrl(path, options = {}) {
  return joinUrl(options.baseUrl, path)
}

function snakeToCamel(value) {
  return String(value).replace(/_([a-z])/g, (_, char) => char.toUpperCase())
}

export function snakeToCamelDeep(value) {
  if (Array.isArray(value)) {
    return value.map((item) => snakeToCamelDeep(item))
  }

  if (!value || Object.prototype.toString.call(value) !== '[object Object]') {
    return value
  }

  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [snakeToCamel(key), snakeToCamelDeep(item)])
  )
}

async function parseJsonResponse(response) {
  const text = await response.text()
  let raw = null

  if (text) {
    try {
      raw = JSON.parse(text)
    } catch (error) {
      raw = { message: text }
    }
  }

  if (!response.ok) {
    const message = raw?.message || raw?.error || `${response.status} ${response.statusText}`
    const err = new Error(message)
    err.status = response.status
    err.raw = raw
    throw err
  }

  return raw || {}
}

async function requestJson(path, options = {}) {
  const response = await fetch(joinUrl(options.baseUrl, path), {
    method: options.method || 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal
  })

  return parseJsonResponse(response)
}

async function postForm(path, formData, options = {}) {
  const response = await fetch(joinUrl(options.baseUrl, path), {
    method: 'POST',
    body: formData,
    signal: options.signal
  })

  return parseJsonResponse(response)
}

export function buildDemoApiUrl(path, options = {}) {
  return joinUrl(options.baseUrl || DEMO_API_BASE_URL, path)
}

export async function uploadDemoTestVideos(day1File, day2File, options = {}) {
  const formData = new FormData()
  formData.append('day1_video', day1File)
  formData.append('day2_video', day2File)
  formData.append('day1_start', options.day1Start || '2026-06-30 18:56:02')
  formData.append('day2_start', options.day2Start || '2026-07-01 21:26:32')
  formData.append('sample_fps', String(options.sampleFps ?? 1))
  formData.append('auto_prepare', String(options.autoPrepare !== false))
  formData.append('enqueue_offline', String(!!options.enqueueOffline))
  if (options.ownerId) formData.append('owner_id', options.ownerId)
  if (options.deviceId) formData.append('device_id', options.deviceId)
  if (options.deviceType) formData.append('device_type', options.deviceType)
  if (options.metadata) formData.append('metadata', JSON.stringify(options.metadata))

  const raw = await postForm('/demo-test/upload', formData, {
    baseUrl: options.baseUrl || DEMO_API_BASE_URL,
    signal: options.signal
  })

  return normalizeDemoTestSession(raw, options)
}

export async function startDemoTestPlayback(sessionId, options = {}) {
  const raw = await requestJson(`/demo-test/${encodeURIComponent(sessionId)}/start`, {
    baseUrl: options.baseUrl || DEMO_API_BASE_URL,
    method: 'POST',
    signal: options.signal,
    body: {
      clip_id: options.clipId || 'day1',
      current_time: options.currentTime ?? 0,
      playback_speed: options.playbackSpeed ?? 1
    }
  })

  return snakeToCamelDeep(raw)
}

export async function tickDemoTestPlayback(sessionId, options = {}) {
  const raw = await requestJson(`/demo-test/${encodeURIComponent(sessionId)}/tick`, {
    baseUrl: options.baseUrl || DEMO_API_BASE_URL,
    method: 'POST',
    signal: options.signal,
    body: {
      clip_id: options.clipId || 'day1',
      current_time: options.currentTime ?? 0,
      paused: !!options.paused,
      playback_speed: options.playbackSpeed ?? 1
    }
  })

  return snakeToCamelDeep(raw)
}

export async function getDemoTestStatus(sessionId, options = {}) {
  const raw = await requestJson(`/demo-test/${encodeURIComponent(sessionId)}/status`, {
    baseUrl: options.baseUrl || DEMO_API_BASE_URL,
    signal: options.signal
  })

  return snakeToCamelDeep(raw)
}

export async function enqueueDemoTestOffline(sessionId, options = {}) {
  const raw = await requestJson(`/demo-test/${encodeURIComponent(sessionId)}/enqueue_offline`, {
    baseUrl: options.baseUrl || DEMO_API_BASE_URL,
    method: 'POST',
    signal: options.signal,
    body: {
      force_preprocess: !!options.forcePreprocess,
      enqueue_evidence: !!options.enqueueEvidence,
      force_evidence: !!options.forceEvidence,
      ...(options.body || {})
    }
  })

  return snakeToCamelDeep(raw)
}

export async function buildDemoTestMemory(sessionId, options = {}) {
  const raw = await requestJson(`/demo-test/${encodeURIComponent(sessionId)}/build_memory`, {
    baseUrl: options.baseUrl || DEMO_API_BASE_URL,
    method: 'POST',
    signal: options.signal,
    body: {
      force: options.force !== false,
      allow_manifest_fallback: !!options.allowManifestFallback,
      skip_semantic: !!options.skipSemantic,
      ...(options.body || {})
    }
  })

  return snakeToCamelDeep(raw)
}

export async function uploadDemoVideo(file, options = {}) {
  const formData = new FormData()
  formData.append('video', file)
  formData.append('sample_fps', String(options.sampleFps ?? 1))
  formData.append('auto_prepare', String(options.autoPrepare !== false))
  formData.append('enqueue_preprocess', String(!!options.enqueuePreprocess))
  if (options.ownerId) formData.append('owner_id', options.ownerId)
  if (options.deviceId) formData.append('device_id', options.deviceId)
  if (options.deviceType) formData.append('device_type', options.deviceType)
  if (options.metadata) formData.append('metadata', JSON.stringify(options.metadata))
  if (options.forcePreprocess !== undefined) {
    formData.append('force_preprocess', String(!!options.forcePreprocess))
  }

  const raw = await postForm('/demo/upload', formData, {
    baseUrl: options.baseUrl,
    signal: options.signal
  })

  return normalizeDemoSession(raw, options)
}

export async function startDemoPlayback(sessionId, options = {}) {
  const raw = await requestJson(`/demo/${encodeURIComponent(sessionId)}/start`, {
    baseUrl: options.baseUrl,
    method: 'POST',
    signal: options.signal,
    body: {
      current_time: options.currentTime ?? 0,
      playback_speed: options.playbackSpeed ?? 1
    }
  })

  return snakeToCamelDeep(raw)
}

export async function tickDemoPlayback(sessionId, options = {}) {
  const raw = await requestJson(`/demo/${encodeURIComponent(sessionId)}/tick`, {
    baseUrl: options.baseUrl,
    method: 'POST',
    signal: options.signal,
    body: {
      current_time: options.currentTime ?? 0,
      paused: !!options.paused,
      playback_speed: options.playbackSpeed ?? 1
    }
  })

  return snakeToCamelDeep(raw)
}

export async function pauseDemoPlayback(sessionId, options = {}) {
  const raw = await requestJson(`/demo/${encodeURIComponent(sessionId)}/pause`, {
    baseUrl: options.baseUrl,
    method: 'POST',
    signal: options.signal,
    body: {
      current_time: options.currentTime ?? 0,
      playback_speed: options.playbackSpeed ?? 1
    }
  })

  return snakeToCamelDeep(raw)
}

export async function stopDemoPlayback(sessionId, options = {}) {
  const raw = await requestJson(`/demo/${encodeURIComponent(sessionId)}/stop`, {
    baseUrl: options.baseUrl,
    method: 'POST',
    signal: options.signal,
    body: {
      current_time: options.currentTime ?? 0,
      playback_speed: options.playbackSpeed ?? 1
    }
  })

  return snakeToCamelDeep(raw)
}

export async function startStream(options = {}) {
  const inputMode = options.inputMode || options.payload?.input_mode || DEFAULT_START_PAYLOAD.input_mode
  if (String(inputMode).startsWith('rokid_')) {
    const error = new Error('Rokid input modes are served by /rokid/stream/start. Use the dedicated Rokid API instead of /stream/start.')
    error.raw = { rokid_start_url: '/rokid/stream/start' }
    throw error
  }
  const payload = {
    ...(inputMode === 'frame_audio_stream' ? DEFAULT_START_PAYLOAD : {}),
    ...(options.payload || {}),
    input_mode: inputMode,
    ...(options.ownerId ? { owner_id: options.ownerId } : {}),
    ...(options.deviceId ? { device_id: options.deviceId } : {}),
    ...(options.deviceType ? { device_type: options.deviceType } : {}),
    metadata: {
      source: 'web_client',
      mode: inputMode,
      ...(options.metadata || {}),
      ...(options.payload?.metadata || {})
    }
  }

  const raw = await requestJson('/stream/start', {
    baseUrl: options.baseUrl,
    method: 'POST',
    body: payload,
    signal: options.signal
  })

  return normalizeStartStream(raw, inputMode)
}

export async function getActiveRokidStream(options = {}) {
  const raw = await requestJson('/rokid/stream/active', {
    baseUrl: options.baseUrl,
    signal: options.signal
  })

  if (!raw.active) {
    return {
      ...snakeToCamelDeep(raw),
      active: false,
      sessionId: raw.session_id || raw.sessionId || '',
      raw
    }
  }

  return {
    ...normalizeStartStream(raw, raw.input_mode || raw.inputMode || 'rokid_live_rtmp'),
    active: true,
    source: raw.source || '',
    raw
  }
}

export async function startRokidStream(options = {}) {
  const inputMode = options.inputMode || options.payload?.input_mode || 'rokid_frame_audio'
  const raw = await requestJson('/rokid/stream/start', {
    baseUrl: options.baseUrl,
    method: 'POST',
    body: {
      ...(options.payload || {}),
      input_mode: inputMode,
      owner_id: options.ownerId || options.payload?.owner_id || 'web_user',
      device_id: options.deviceId || options.payload?.device_id || 'rokid_web',
      device_type: options.deviceType || options.payload?.device_type || 'rokid',
      metadata: {
        source: 'web_frontend',
        mode: inputMode,
        ...(options.metadata || {}),
        ...(options.payload?.metadata || {})
      }
    },
    signal: options.signal
  })

  return normalizeStartStream(raw, inputMode)
}

export function startFrameAudioStream(options = {}) {
  return startStream({
    ...options,
    inputMode: 'frame_audio_stream',
    payload: {
      input_mode: 'frame_audio_stream',
      chunk_duration: 1,
      metadata: {
        source: 'web_client',
        mode: 'frame_audio_stream'
      },
      ...(options.payload || {})
    }
  })
}

export function startWebRtcWhipStream(options = {}) {
  return startStream({
    ...options,
    inputMode: 'web_webrtc_whip',
    payload: {
      input_mode: 'web_webrtc_whip',
      metadata: {
        source: 'web_client',
        mode: 'web_webrtc_whip'
      },
      ...(options.payload || {})
    }
  })
}

export async function startLiveIngest(sessionId, options = {}) {
  const raw = await requestJson(`/stream/${encodeURIComponent(sessionId)}/live/ingest/start`, {
    baseUrl: options.baseUrl,
    method: 'POST',
    body: options.body,
    signal: options.signal
  })

  return snakeToCamelDeep(raw)
}

export async function startRokidLiveIngest(sessionId, options = {}) {
  const raw = await requestJson(`/rokid/${encodeURIComponent(sessionId)}/live/ingest/start`, {
    baseUrl: options.baseUrl,
    method: 'POST',
    body: options.body,
    signal: options.signal
  })

  return snakeToCamelDeep(raw)
}

export async function stopLiveIngest(sessionId, options = {}) {
  const raw = await requestJson(`/stream/${encodeURIComponent(sessionId)}/live/ingest/stop`, {
    baseUrl: options.baseUrl,
    method: 'POST',
    body: options.body,
    signal: options.signal
  })

  return snakeToCamelDeep(raw)
}

export async function stopRokidLiveIngest(sessionId, options = {}) {
  const raw = await requestJson(`/rokid/${encodeURIComponent(sessionId)}/live/ingest/stop`, {
    baseUrl: options.baseUrl,
    method: 'POST',
    body: options.body,
    signal: options.signal
  })

  return snakeToCamelDeep(raw)
}

export async function getRokidStatus(sessionId, options = {}) {
  const raw = await requestJson(`/rokid/${encodeURIComponent(sessionId)}/status`, {
    baseUrl: options.baseUrl,
    signal: options.signal
  })

  return {
    ...snakeToCamelDeep(raw),
    canAsk: !!(raw.can_ask || raw.canAsk),
    raw
  }
}

export async function uploadFrame(sessionId, frameBlob, options = {}) {
  const formData = new FormData()
  formData.append('frame', frameBlob, options.filename || `frame-${options.frameIndex ?? 0}.${options.format || 'jpg'}`)
  formData.append('frame_index', String(options.frameIndex ?? 0))
  formData.append('client_ts_ms', String(options.clientTsMs || Date.now()))
  formData.append('relative_ts_ms', String(options.relativeTsMs ?? 0))
  formData.append('width', String(options.width || ''))
  formData.append('height', String(options.height || ''))
  formData.append('format', options.format || 'jpg')
  formData.append('source', options.source || 'web_canvas')

  const raw = await postForm(`/stream/${encodeURIComponent(sessionId)}/frame`, formData, {
    baseUrl: options.baseUrl,
    signal: options.signal
  })

  return normalizeUploadFrame(raw)
}

export async function uploadAudioChunk(sessionId, audioBlob, options = {}) {
  const format = options.format || inferAudioFormat(audioBlob.type)
  const formData = new FormData()
  formData.append('audio', audioBlob, options.filename || `audio-${options.audioIndex ?? 0}.${format}`)
  formData.append('audio_index', String(options.audioIndex ?? 0))
  formData.append('client_ts_ms', String(options.clientTsMs || Date.now()))
  formData.append('relative_ts_ms', String(options.relativeTsMs ?? 0))
  formData.append('duration_ms', String(options.durationMs || 1000))
  formData.append('sample_rate', String(options.sampleRate || ''))
  formData.append('channels', String(options.channels || 1))
  formData.append('format', format)
  formData.append('source', options.source || 'web_media_recorder')

  const raw = await postForm(`/stream/${encodeURIComponent(sessionId)}/audio_chunk`, formData, {
    baseUrl: options.baseUrl,
    signal: options.signal
  })

  return normalizeUploadAudio(raw)
}

export async function uploadRokidFrame(sessionId, frameBlob, options = {}) {
  const format = options.format || 'jpeg'
  const formData = new FormData()
  formData.append('frame', frameBlob, options.filename || `rokid-frame-${options.frameIndex ?? 0}.${format}`)
  formData.append('frame_index', String(options.frameIndex ?? 0))
  formData.append('relative_ts_ms', String(options.relativeTsMs ?? 0))
  formData.append('format', format)
  formData.append('source', options.source || 'rokid_sdk_video')
  if (options.clientTsMs !== undefined) formData.append('client_ts_ms', String(options.clientTsMs))
  if (options.width !== undefined) formData.append('width', String(options.width))
  if (options.height !== undefined) formData.append('height', String(options.height))

  const raw = await postForm(`/rokid/${encodeURIComponent(sessionId)}/frame`, formData, {
    baseUrl: options.baseUrl,
    signal: options.signal
  })

  return normalizeUploadFrame(raw)
}

export async function uploadRokidAudioChunk(sessionId, audioBlob, options = {}) {
  const format = options.format || inferAudioFormat(audioBlob.type)
  const formData = new FormData()
  formData.append('audio', audioBlob, options.filename || `rokid-audio-${options.audioIndex ?? 0}.${format}`)
  formData.append('audio_index', String(options.audioIndex ?? 0))
  formData.append('relative_ts_ms', String(options.relativeTsMs ?? 0))
  formData.append('duration_ms', String(options.durationMs || 1000))
  formData.append('format', format)
  formData.append('source', options.source || 'rokid_sdk_audio')
  if (options.clientTsMs !== undefined) formData.append('client_ts_ms', String(options.clientTsMs))
  if (options.sampleRate !== undefined) formData.append('sample_rate', String(options.sampleRate))
  if (options.channels !== undefined) formData.append('channels', String(options.channels))

  const raw = await postForm(`/rokid/${encodeURIComponent(sessionId)}/audio_chunk`, formData, {
    baseUrl: options.baseUrl,
    signal: options.signal
  })

  return normalizeUploadAudio(raw)
}

export async function askQuestion(sessionId, question, options = {}) {
  const inputMethod = options.inputMethod || 'manual'
  const raw = await requestJson(`/ask/${encodeURIComponent(sessionId)}`, {
    baseUrl: options.baseUrl,
    method: 'POST',
    signal: options.signal,
    body: {
      question,
      mode: options.mode || 'async',
      retrieval_mode: options.retrievalMode || 'auto',
      memory_mode: options.memoryMode || 'auto',
      priority: options.priority || 'high',
      prefer_current: options.preferCurrent !== false,
      top_k: options.topK || 5,
      use_image_evidence: options.useImageEvidence || 'auto',
      max_image_evidence: options.maxImageEvidence || 6,
      debug_router: options.debugRouter !== false,
      use_interaction_cache: options.useInteractionCache !== false,
      response_mode: options.responseMode || 'legacy',
      client_source: options.clientSource || 'frontend',
      input_method: inputMethod,
      ...(options.useCurrent !== undefined ? { use_current: options.useCurrent } : {}),
      ...(options.useShortTerm !== undefined ? { use_short_term: options.useShortTerm } : {}),
      ...(options.useLongTerm !== undefined ? { use_long_term: options.useLongTerm } : {}),
      ...(options.cacheMode ? { cache_mode: options.cacheMode } : {}),
      ...(options.longTermRetrievalScheme ? { long_term_retrieval_scheme: options.longTermRetrievalScheme } : {}),
      ...(options.body || {})
    }
  })

  if (raw.status === 'queued' || raw.queued === true || (raw.task_id && !raw.answer && !raw.result)) {
    return {
      status: 'queued',
      queued: true,
      done: false,
      taskId: raw.task_id || raw.taskId || '',
      taskPath: raw.task_path || raw.taskPath || '',
      message: raw.message || 'Query queued',
      streamContext: snakeToCamelDeep(raw.stream_context || raw.streamContext || {}),
      raw
    }
  }

  return {
    status: raw.status || 'done',
    queued: false,
    done: true,
    result: normalizeAskResult(raw),
    raw
  }
}

export async function streamAskQuestion(sessionId, question, options = {}) {
  const endpoint = options.endpoint || `/ask/${encodeURIComponent(sessionId)}/stream`
  const inputMethod = options.inputMethod || 'manual'
  const requestPayload = {
    question,
    retrieval_mode: options.retrievalMode || 'auto',
    memory_mode: options.memoryMode || 'auto',
    priority: options.priority || 'high',
    prefer_current: options.preferCurrent !== false,
    top_k: options.topK || 5,
    use_image_evidence: options.useImageEvidence || 'auto',
    max_image_evidence: options.maxImageEvidence || 6,
    debug_router: options.debugRouter !== false,
    use_interaction_cache: options.useInteractionCache !== false,
    response_mode: 'stream',
    client_source: options.clientSource || 'frontend',
    input_method: inputMethod,
    ...(options.useCurrent !== undefined ? { use_current: options.useCurrent } : {}),
    ...(options.useShortTerm !== undefined ? { use_short_term: options.useShortTerm } : {}),
    ...(options.useLongTerm !== undefined ? { use_long_term: options.useLongTerm } : {}),
    ...(options.cacheMode ? { cache_mode: options.cacheMode } : {}),
    ...(options.longTermRetrievalScheme ? { long_term_retrieval_scheme: options.longTermRetrievalScheme } : {}),
    ...(options.body || {})
  }
  let response = null
  let lastEvent = null
  let rawBody = ''

  try {
    response = await fetch(joinUrl(options.baseUrl, endpoint), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...(options.headers || {})
      },
      body: JSON.stringify(requestPayload),
      signal: options.signal
    })

    if (!response.ok) {
      await parseJsonResponse(response)
    }

    if (!response.body) {
      throw new Error('Streaming response body is unavailable.')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunkText = decoder.decode(value, { stream: true })
        rawBody = trimDebugText(rawBody + chunkText)
        buffer += chunkText
        const parts = buffer.split(/\r?\n\r?\n/)
        buffer = parts.pop() || ''

        for (const part of parts) {
          const event = parseSseEvent(part)
          if (event) {
            lastEvent = event
            options.onEvent?.(event)
          }
        }
      }

      const trailingText = decoder.decode()
      if (trailingText) {
        rawBody = trimDebugText(rawBody + trailingText)
        buffer += trailingText
      }
      const trailingEvent = parseSseEvent(buffer)
      if (trailingEvent) {
        lastEvent = trailingEvent
        options.onEvent?.(trailingEvent)
      }
    } finally {
      reader.releaseLock()
    }
  } catch (error) {
    error.raw = {
      ...(error.raw || {}),
      httpStatus: response?.status || error.status || null,
      endpoint,
      sessionId,
      requestPayload,
      lastEvent,
      rawBody
    }
    throw error
  }

  return {
    httpStatus: response.status,
    endpoint,
    sessionId,
    requestPayload,
    lastEvent,
    rawBody
  }
}

export async function appendQaHistory(sessionId, payload = {}, options = {}) {
  const raw = await requestJson(`/session/${encodeURIComponent(sessionId)}/qa_history`, {
    baseUrl: options.baseUrl,
    method: 'POST',
    signal: options.signal,
    body: {
      question: payload.question || '',
      answer: payload.answer || '',
      client_source: payload.clientSource || payload.client_source || 'frontend',
      input_method: payload.inputMethod || payload.input_method || 'manual',
      status: payload.status || 'done',
      error: payload.error || '',
      metadata: payload.metadata || {}
    }
  })
  return raw
}

function trimDebugText(text, maxLength = 12000) {
  const value = String(text || '')
  return value.length > maxLength ? value.slice(value.length - maxLength) : value
}

export async function getQueryTask(taskId, options = {}) {
  const raw = await requestJson(`/query_task/${encodeURIComponent(taskId)}`, {
    baseUrl: options.baseUrl,
    signal: options.signal
  })

  return normalizeQueryTask(raw, taskId)
}

export async function pollQueryTask(taskId, options = {}) {
  const intervalMs = options.intervalMs || 900
  const timeoutMs = options.timeoutMs || 45000
  const startedAt = Date.now()
  let attempt = 0

  while (Date.now() - startedAt < timeoutMs) {
    if (options.shouldContinue && !options.shouldContinue()) {
      return { status: 'cancelled', done: false, cancelled: true, taskId }
    }

    attempt += 1
    const task = await getQueryTask(taskId, options)
    options.onPoll?.({ ...task, attempt })

    if (task.done || task.status === 'failed' || task.status === 'cancelled' || task.status === 'canceled') {
      return task
    }

    await sleep(intervalMs)
  }

  return {
    status: 'timeout',
    done: false,
    taskId,
    message: 'Query timed out'
  }
}

export async function getStreamStatus(sessionId, options = {}) {
  const raw = await requestJson(`/stream/${encodeURIComponent(sessionId)}/status`, {
    baseUrl: options.baseUrl,
    signal: options.signal
  })

  return {
    ...snakeToCamelDeep(raw),
    canAsk: !!(raw.can_ask || raw.canAsk),
    raw
  }
}

export async function getStreamPreview(sessionId, options = {}) {
  const raw = await requestJson(`/stream/${encodeURIComponent(sessionId)}/preview`, {
    baseUrl: options.baseUrl,
    signal: options.signal
  })

  return {
    ...snakeToCamelDeep(raw),
    canAsk: !!(raw.can_ask || raw.canAsk),
    raw
  }
}

export function buildEvidenceFrameUrl(sessionId, path, options = {}) {
  if (!sessionId || !path) return ''
  return joinUrl(options.baseUrl, `/session/${encodeURIComponent(sessionId)}/file?path=${encodeURIComponent(String(path).replace(/^\/+/, ''))}`)
}

function resolveEvidenceFrameImageUrl(item, sessionId, rawPath, options = {}) {
  const relativeFileUrl = firstDefined(item.relativeFileUrl, item.relative_file_url)
  if (relativeFileUrl) return resolveMaybeRelativeUrl(relativeFileUrl, options)

  const fileUrlSource = firstDefined(item.fileUrl, item.file_url, item.imageUrl, item.image_url, item.url)
  const ownerSessionId = firstDefined(item.ownerSessionId, item.owner_session_id, extractSessionIdFromFileUrl(fileUrlSource))
  if (ownerSessionId && rawPath) return buildEvidenceFrameUrl(ownerSessionId, rawPath, options)

  if (rawPath) return buildEvidenceFrameUrl(sessionId, rawPath, options)

  const directUrl = firstDefined(
    item.imageUrl,
    item.image_url,
    item.displayUrl,
    item.display_url,
    item.url,
    item.src,
    item.thumbnailUrl,
    item.thumbnail_url
  )
  return directUrl ? resolveMaybeRelativeUrl(directUrl, options) : ''
}

export function normalizeEvidenceFrames(result, sessionId, options = {}) {
  const raw = result || {}
  const candidates = [
    raw.evidenceFrames,
    raw.evidence_frames,
    raw.visualEvidence,
    raw.visual_evidence,
    raw.selectedEvidence,
    raw.selected_evidence,
    raw.raw?.evidenceFrames,
    raw.raw?.evidence_frames,
    raw.raw?.visualEvidence,
    raw.raw?.visual_evidence
  ]

  const frames = candidates.find((item) => Array.isArray(item) && item.length) || []

  return frames.slice(0, options.limit || 8).map((frame, index) => normalizeEvidenceFrame(frame, index, sessionId, options))
}

function normalizeEvidenceFrame(frame, index, sessionId, options = {}) {
  const item = frame || {}
  const fileUrlSource = firstDefined(item.relative_file_url, item.relativeFileUrl, item.file_url, item.fileUrl, item.image_url, item.imageUrl, item.display_url, item.displayUrl, item.url)
  const rawPath = firstDefined(
    item.path,
    item.image_path,
    item.imagePath,
    item.file_path,
    item.filePath,
    extractPathFromUrl(fileUrlSource)
  )
  const timestamp = firstDefined(item.timestamp, item.time, item.ts, item.second, item.seconds)
  const start = firstDefined(item.start, item.startTime, item.start_time, item.startSec, item.start_sec, item.begin)
  const end = firstDefined(item.end, item.endTime, item.end_time, item.endSec, item.end_sec)
  const score = firstDefined(item.finalScore, item.final_score, item.fusedScore, item.fused_score, item.score, item.visualScore, item.visual_score)
  const source = firstDefined(item.source, item.memorySource, item.memory_source, item.memoryMode, item.memory_mode, item.route, 'Memory')
  const sourceType = firstDefined(item.sourceType, item.source_type, item.type, item.modality, '')
  const caption = firstDefined(item.caption, item.text, item.description, item.transcript, item.memory, item.summary, 'Evidence frame')
  const id = String(firstDefined(item.id, item.frameId, item.frame_id, rawPath, item.image_url, item.url, `${source}-${index}`))

  return {
    id,
    raw: item,
    rawPath,
    imageUrl: resolveEvidenceFrameImageUrl(item, sessionId, rawPath, options),
    caption: String(caption),
    source: String(source || 'Memory'),
    sourceType: String(sourceType || ''),
    timestamp,
    timestampText: firstDefined(item.timestampText, item.timestamp_text, formatTimestamp(timestamp), ''),
    timeRangeText: firstDefined(item.timeRangeText, item.time_range_text, item.timeRange, item.time_range, formatTimeRange(start, end), ''),
    score,
    scoreText: formatScore(score)
  }
}

function normalizeStartStream(raw, requestedInputMode = 'frame_audio_stream') {
  const mapped = snakeToCamelDeep(raw)
  const webrtc = mapped.webrtc || {}
  const live = mapped.live || {}
  const metadata = mapped.metadata || {}
  const sessionId = raw.session_id || raw.sessionId || ''
  const inputMode = raw.input_mode || raw.inputMode || requestedInputMode
  const deviceType = firstDefined(raw.device_type, raw.deviceType, mapped.deviceType, metadata.deviceType, '')
  const streamName = firstDefined(
    raw.stream_name,
    raw.streamName,
    raw.webrtc?.stream_name,
    raw.webrtc?.streamName,
    raw.live?.stream_name,
    raw.live?.streamName,
    ''
  )
  const whipUrl = firstDefined(
    raw.webrtc?.whip_url,
    raw.webrtc?.whipUrl,
    raw.webrtc?.whip_url_public,
    raw.webrtc?.whipUrlPublic,
    raw.live?.whip_url_public,
    raw.live?.whipUrlPublic,
    raw.push_url,
    raw.pushUrl,
    ''
  )
  const webrtcPlayUrl = firstDefined(
    raw.live?.webrtc_play_url_public,
    raw.live?.webrtcPlayUrlPublic,
    raw.webrtc_play_url_public,
    raw.webrtcPlayUrlPublic,
    live.webrtcPlayUrlPublic,
    live.webrtcPlayUrl,
    ''
  )
  const liveIngestStartUrl = raw.live_ingest_start_url || raw.liveIngestStartUrl || ''
  const liveIngestStopUrl = raw.live_ingest_stop_url || raw.liveIngestStopUrl || ''
  const frameUploadUrl = firstDefined(raw.frame_upload_url, raw.frameUploadUrl, mapped.frameUploadUrl, '')
  const audioUploadUrl = firstDefined(raw.audio_upload_url, raw.audioUploadUrl, mapped.audioUploadUrl, '')
  const statusUrl = firstDefined(raw.status_url, raw.statusUrl, mapped.statusUrl, '')
  const askStreamUrl = firstDefined(raw.ask_stream_url, raw.askStreamUrl, mapped.askStreamUrl, sessionId ? `/ask/${sessionId}/stream` : '')

  return {
    ...mapped,
    sessionId,
    streamId: raw.stream_id || raw.streamId || '',
    deviceKind: resolveDeviceKind(inputMode, deviceType),
    ownerId: firstDefined(raw.owner_id, raw.ownerId, mapped.ownerId, metadata.ownerId, ''),
    deviceId: firstDefined(raw.device_id, raw.deviceId, mapped.deviceId, metadata.deviceId, ''),
    deviceType,
    frameUploadUrl,
    audioUploadUrl,
    statusUrl,
    askStreamUrl,
    singleActiveSession: firstDefined(raw.single_active_session, raw.singleActiveSession, mapped.singleActiveSession, null),
    streamName,
    status: raw.status || 'started',
    canAsk: !!(raw.can_ask || raw.canAsk),
    inputMode,
    whipUrl,
    whipUrlAvailable: !!whipUrl,
    webrtcPlayUrl,
    webrtcPlayUrlAvailable: !!webrtcPlayUrl,
    liveIngestStartUrl,
    liveIngestStopUrl,
    webrtc,
    live,
    message: raw.message || '',
    raw
  }
}

function resolveDeviceKind(inputMode, deviceType = '') {
  const normalizedDeviceType = String(deviceType || '').toLowerCase()
  if (normalizedDeviceType.includes('rokid')) return 'rokid'
  if (String(inputMode || '').startsWith('rokid_')) return 'rokid'
  if (String(inputMode || '') === 'demo_video') return 'demo'
  if (String(inputMode || '') === 'demo_test') return 'demo'
  return 'web'
}

function normalizeUploadFrame(raw) {
  return {
    sessionId: raw.session_id || raw.sessionId || '',
    status: raw.status || '',
    frameIndex: raw.frame_index ?? raw.frameIndex ?? null,
    frameId: raw.frame_id || raw.frameId || '',
    savedPath: raw.saved_path || raw.savedPath || '',
    currentFramePath: raw.current_frame_path || raw.currentFramePath || '',
    mcurReady: !!raw.mcur_ready,
    mcurVersion: raw.mcur_version ?? raw.mcurVersion ?? null,
    canAsk: !!raw.can_ask,
    message: raw.message || '',
    raw
  }
}

function normalizeUploadAudio(raw) {
  return {
    sessionId: raw.session_id || raw.sessionId || '',
    status: raw.status || '',
    audioIndex: raw.audio_index ?? raw.audioIndex ?? null,
    audioId: raw.audio_id || raw.audioId || '',
    savedPath: raw.saved_path || raw.audio_path || raw.savedPath || '',
    audioReady: !!(raw.audio_ready || raw.ready),
    asrEnqueue: snakeToCamelDeep(raw.asr_enqueue || raw.asrEnqueue || null),
    message: raw.message || '',
    raw
  }
}

function normalizeAskResult(raw) {
  const result = raw?.result?.result && !raw.result.answer && !raw.result.final_answer ? raw.result.result : raw.result || raw

  return {
    ...snakeToCamelDeep(result),
    answer: firstDefined(result.answer, result.finalAnswer, result.final_answer, result.response, result.text, ''),
    latency: firstDefined(result.latency, result.query_latency, result.queryLatency, null),
    usedMemorySources: firstDefined(result.usedMemorySources, result.used_memory_sources, []),
    raw: result
  }
}

function normalizeQueryTask(raw, fallbackTaskId) {
  const resultPayload = raw.result || {}
  const nestedResult = resultPayload.result || {}
  const rawStatus = raw.status || resultPayload.status || 'queued'
  const status = String(rawStatus === 'in_progress' ? 'running' : rawStatus).toLowerCase()
  const queueState = raw.queue_state || raw.queueState || resultPayload.queue_state || resultPayload.queueState || ''
  const taskId = raw.task_id || raw.taskId || fallbackTaskId

  if (status === 'done' && (!queueState || queueState === 'query_done')) {
    const finalPayload = raw.result && raw.result.result && !raw.result.answer && !raw.result.final_answer
      ? raw.result.result
      : (raw.result || raw)

    return {
      status: 'done',
      done: true,
      running: false,
      taskId,
      sessionId: raw.session_id || raw.sessionId || resultPayload.session_id || resultPayload.sessionId || nestedResult.session_id || '',
      result: normalizeAskResult(finalPayload),
      raw
    }
  }

  return {
    status: status === 'done' ? 'running' : status,
    done: false,
    running: !['failed', 'cancelled', 'canceled', 'not_found', 'aborted'].includes(status),
    taskId,
    queueState,
    message: raw.message || resultPayload.message || '',
    error: raw.error || resultPayload.error || null,
    raw
  }
}

function normalizeDemoSession(raw, options = {}) {
  const mapped = snakeToCamelDeep(raw)
  const sessionId = raw.session_id || raw.sessionId || mapped.sessionId || ''
  const videoUrl = raw.video_url || raw.videoUrl || mapped.videoUrl || ''

  return {
    ...mapped,
    sessionId,
    videoUrl: videoUrl ? joinUrl(options.baseUrl, videoUrl) : '',
    startUrl: raw.start_url || raw.startUrl || mapped.startUrl || '',
    tickUrl: raw.tick_url || raw.tickUrl || mapped.tickUrl || '',
    askStreamUrl: raw.ask_stream_url || raw.askStreamUrl || mapped.askStreamUrl || '',
    prepared: !!raw.prepared,
    duration: raw.duration ?? mapped.duration ?? null,
    frameCount: raw.frame_count ?? raw.frameCount ?? mapped.frameCount ?? null,
    preprocessQueued: !!(raw.preprocess_queued || raw.preprocessQueued),
    raw
  }
}

function normalizeDemoTestSession(raw, options = {}) {
  const mapped = snakeToCamelDeep(raw)
  const baseUrl = options.baseUrl || DEMO_API_BASE_URL
  const sessionId = raw.session_id || raw.sessionId || mapped.sessionId || ''
  const clips = Array.isArray(raw.clips || mapped.clips)
    ? (raw.clips || mapped.clips).map((clip) => normalizeDemoTestClip(clip, baseUrl))
    : []
  const activeClipId = options.activeClipId || raw.active_clip_id || raw.activeClipId || clips[0]?.clipId || 'day1'
  const tickUrl = raw.tick_url || raw.tickUrl || mapped.tickUrl || (sessionId ? `/demo-test/${sessionId}/tick` : '')
  const askStreamUrl = raw.ask_stream_url || raw.askStreamUrl || mapped.askStreamUrl || (sessionId ? `/ask/${sessionId}/stream` : '')

  return {
    ...mapped,
    sessionId,
    clips,
    activeClipId,
    demoApiBaseUrl: baseUrl,
    askStreamUrl,
    askStreamUrlResolved: askStreamUrl ? joinUrl(baseUrl, askStreamUrl) : '',
    tickUrl,
    tickUrlResolved: tickUrl ? joinUrl(baseUrl, tickUrl) : '',
    memoryReady: !!(raw.memory_ready || raw.memoryReady || mapped.memoryReady),
    prepared: raw.prepared !== false && mapped.prepared !== false,
    raw
  }
}

function normalizeDemoTestClip(rawClip, baseUrl) {
  const clip = snakeToCamelDeep(rawClip || {})
  const clipId = rawClip?.clip_id || rawClip?.clipId || clip.clipId || ''
  const videoUrl = rawClip?.video_url || rawClip?.videoUrl || clip.videoUrl || ''

  return {
    ...clip,
    clipId,
    displayDate: rawClip?.display_date || rawClip?.displayDate || clip.displayDate || '',
    startTime: rawClip?.start_time || rawClip?.startTime || clip.startTime || '',
    videoUrl,
    videoUrlResolved: videoUrl ? joinUrl(baseUrl, videoUrl) : ''
  }
}

function parseSseEvent(rawPart) {
  const part = String(rawPart || '').trim()
  if (!part) return null

  const lines = part.split(/\r?\n/)
  let eventType = 'message'
  const dataLines = []

  for (const line of lines) {
    if (!line || line.startsWith(':')) continue
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim() || 'message'
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart())
    }
  }

  if (!dataLines.length) {
    return { event: eventType, data: {}, raw: part }
  }

  const dataText = dataLines.join('\n')
  let data

  try {
    data = JSON.parse(dataText)
  } catch (error) {
    const err = new Error(`Invalid SSE JSON for ${eventType}: ${error.message}`)
    err.raw = dataText
    err.event = eventType
    throw err
  }

  const resolvedType = data?.type || eventType

  if (resolvedType === 'done' && data?.result) {
    data = {
      ...data,
      result: normalizeAskResult(data.result)
    }
  }

  return {
    event: eventType,
    type: resolvedType,
    data,
    raw: part
  }
}

function inferAudioFormat(mimeType = '') {
  if (mimeType.includes('mp4')) return 'm4a'
  if (mimeType.includes('mpeg')) return 'mp3'
  if (mimeType.includes('ogg')) return 'ogg'
  if (mimeType.includes('wav')) return 'wav'
  return 'webm'
}

function extractPathFromUrl(url) {
  if (!url) return ''
  try {
    const parsed = new URL(url)
    return parsed.searchParams.get('path') || ''
  } catch (error) {
    const match = String(url).match(/[?&]path=([^&#]*)/)
    return match ? decodeURIComponent(match[1]) : ''
  }
}

function extractSessionIdFromFileUrl(url) {
  if (!url) return ''
  try {
    const parsed = new URL(String(url), 'https://omnispark.local')
    const match = parsed.pathname.match(/\/session\/([^/]+)\/file\b/)
    return match ? decodeURIComponent(match[1]) : ''
  } catch (error) {
    const match = String(url).match(/\/session\/([^/?#]+)\/file\b/)
    return match ? decodeURIComponent(match[1]) : ''
  }
}

function resolveMaybeRelativeUrl(url, options = {}) {
  const value = String(url || '')
  if (/^(https?:|data:|blob:)/i.test(value)) return value
  if (value.startsWith('/api/')) {
    const base = String(options.baseUrl || API_BASE_URL).replace(/\/+$/, '')
    return /\/api$/i.test(base) ? `${base}${value.slice(4)}` : `${base}${value}`
  }
  return joinUrl(options.baseUrl, value)
}

function firstDefined(...values) {
  return values.find((item) => item !== undefined && item !== null && item !== '')
}

function formatTimestamp(value) {
  if (value === undefined || value === null || value === '') return ''
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return String(value)
  return `${numeric.toFixed(numeric < 10 ? 1 : 0)}s`
}

function formatTimeRange(start, end) {
  if (start === undefined || start === null || start === '') return ''
  const left = formatTimestamp(start)
  const right = formatTimestamp(end)
  return right ? `${left} - ${right}` : left
}

function formatScore(value) {
  if (value === undefined || value === null || value === '') return ''
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric.toFixed(2) : String(value)
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

export { API_BASE_URL, DEMO_API_BASE_URL }
