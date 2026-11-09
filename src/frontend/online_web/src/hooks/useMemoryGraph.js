import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchLongTermMemoryGraph } from '../api/lightmem_egoApi.js'

const NORMAL_POLL_MS = 5000
const RETRY_DELAYS_MS = [2000, 5000, 10000, 30000]
const RETRYABLE_CODES = new Set([
  'memory_graph_not_ready',
  'memory_component_lagging',
  'memory_snapshot_changing'
])

export function useMemoryGraph(sessionId, { authoritativeUpdate, onSessionUnavailable } = {}) {
  const [payload, setPayload] = useState(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState(null)
  const [requestId, setRequestId] = useState('')
  const [lastCheckedAt, setLastCheckedAt] = useState(null)
  const requestGenerationRef = useRef(0)
  const minimumVersionRef = useRef(0)
  const etagRef = useRef('')
  const timerRef = useRef(null)
  const controllerRef = useRef(null)
  const inFlightRef = useRef(false)
  const mountedRef = useRef(false)
  const retryIndexRef = useRef(0)
  const runRef = useRef(null)
  const onSessionUnavailableRef = useRef(onSessionUnavailable)
  onSessionUnavailableRef.current = onSessionUnavailable

  const clearTimer = useCallback(() => {
    window.clearTimeout(timerRef.current)
    timerRef.current = null
  }, [])

  const schedule = useCallback((delay) => {
    clearTimer()
    if (!mountedRef.current || document.hidden) return
    timerRef.current = window.setTimeout(() => runRef.current?.(), delay)
  }, [clearTimer])

  const run = useCallback(async () => {
    if (!mountedRef.current || document.hidden || !sessionId) return
    if (inFlightRef.current) {
      schedule(250)
      return
    }
    clearTimer()
    inFlightRef.current = true
    const controller = new AbortController()
    const generation = ++requestGenerationRef.current
    const timeout = window.setTimeout(() => controller.abort(), 15000)
    controllerRef.current = controller
    setStatus((current) => (payload ? current : 'loading'))

    try {
      const result = await fetchLongTermMemoryGraph({
        sessionId,
        scale: '30sec',
        etag: etagRef.current,
        signal: controller.signal
      })
      if (!mountedRef.current || controller.signal.aborted || generation !== requestGenerationRef.current) return
      if (result.status === 'ok' && (result.graph?.sessionId !== sessionId || result.graph.memoryVersion < minimumVersionRef.current)) {
        throw Object.assign(new Error('Graph is behind the current Memory version.'), { status: 503, code: 'memory_component_lagging' })
      }
      if (result.status === 'ok') minimumVersionRef.current = Math.max(minimumVersionRef.current, result.graph.memoryVersion)
      etagRef.current = result.etag || etagRef.current
      setRequestId(result.requestId || '')
      setLastCheckedAt(Date.now())
      setError(null)
      retryIndexRef.current = 0
      if (result.status === 'ok') setPayload(result.graph)
      setStatus('ready')
      schedule(NORMAL_POLL_MS)
    } catch (loadError) {
      if (!mountedRef.current || generation !== requestGenerationRef.current) return
      if (loadError?.name === 'AbortError') { schedule(5000); return }
      setRequestId(loadError?.requestId || '')
      setError(loadError)
      setLastCheckedAt(Date.now())

      if (loadError?.code === 'session_not_found') {
        setStatus('error')
        onSessionUnavailableRef.current?.(sessionId)
        return
      }

      if (loadError?.status === 503 && (RETRYABLE_CODES.has(loadError?.code) || !loadError?.code)) {
        setStatus('waiting')
        const retryIndex = Math.min(retryIndexRef.current, RETRY_DELAYS_MS.length - 1)
        retryIndexRef.current += 1
        schedule(RETRY_DELAYS_MS[retryIndex])
        return
      }

      setStatus('error')
    } finally {
      window.clearTimeout(timeout)
      if (controllerRef.current === controller) {
        controllerRef.current = null
        inFlightRef.current = false
      }
    }
  }, [clearTimer, payload, schedule, sessionId])

  runRef.current = run

  useEffect(() => {
    mountedRef.current = true
    requestGenerationRef.current += 1
    minimumVersionRef.current = 0
    setPayload(null)
    setError(null)
    setRequestId('')
    setLastCheckedAt(null)
    setStatus('idle')
    etagRef.current = ''
    retryIndexRef.current = 0
    queueMicrotask(() => runRef.current?.())

    const handleVisibility = () => {
      if (document.hidden) {
        clearTimer()
        controllerRef.current?.abort()
      } else {
        queueMicrotask(() => runRef.current?.())
      }
    }
    document.addEventListener('visibilitychange', handleVisibility)

    return () => {
      mountedRef.current = false
      requestGenerationRef.current += 1
      document.removeEventListener('visibilitychange', handleVisibility)
      clearTimer()
      controllerRef.current?.abort()
      controllerRef.current = null
      inFlightRef.current = false
    }
  }, [clearTimer, sessionId])

  useEffect(() => {
    if (!authoritativeUpdate || authoritativeUpdate.sessionId !== sessionId) return
    const nextVersion = authoritativeUpdate.memory?.memoryVersion || 0
    if (nextVersion < minimumVersionRef.current) return
    minimumVersionRef.current = nextVersion
    requestGenerationRef.current += 1
    controllerRef.current?.abort()
    controllerRef.current = null
    inFlightRef.current = false
    clearTimer()
    etagRef.current = authoritativeUpdate.graphEtag || ''
    retryIndexRef.current = 0
    setError(authoritativeUpdate.graphError || null)

    if (authoritativeUpdate.graph) {
      setPayload(authoritativeUpdate.graph)
      setStatus('ready')
      setLastCheckedAt(Date.now())
      schedule(NORMAL_POLL_MS)
      return
    }

    if (authoritativeUpdate.graphStatus === 'refreshing') {
      setPayload(null)
      setStatus('loading')
    } else if (authoritativeUpdate.graphStatus === 'error') {
      setPayload(null)
      setStatus('error')
    }
  }, [authoritativeUpdate?.revision, clearTimer, schedule, sessionId])

  const retry = useCallback(() => {
    retryIndexRef.current = 0
    setError(null)
    setStatus(payload ? 'ready' : 'loading')
    controllerRef.current?.abort()
    queueMicrotask(() => runRef.current?.())
  }, [payload])

  return {
    payload,
    status,
    error,
    requestId,
    lastCheckedAt,
    retry
  }
}
