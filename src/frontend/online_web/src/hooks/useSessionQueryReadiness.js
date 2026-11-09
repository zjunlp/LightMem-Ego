import { useCallback, useEffect, useRef, useState } from 'react'
import { getMemorySessionStatus, getSessionQueryWarmup, retryAfterMs } from '../api/memoryImportApi.js'

const failed = (value) => ['failed', 'error', 'cancelled'].includes(String(value?.state || value?.status || '').toLowerCase())
const readyWarmup = (value) => !!value && !failed(value) && value.queryReady !== false && value.query_ready !== false && (
  value.queryReady === true || value.query_ready === true
  || ['ready', 'done', 'completed'].includes(String(value.state || value.status || '').toLowerCase())
)

export function useSessionQueryReadiness(session, { publication, active = true } = {}) {
  const sessionId = session.sessionId
  const version = publication?.memory?.memoryVersion || session.memoryVersion
  // Graph refreshes reuse the publication ID and must not reset query readiness.
  const publicationId = publication?.publicationId || publication?.memory?.memoryVersion || ''
  const identity = `${sessionId}:${version}:${publicationId}`
  const [result, setResult] = useState({ identity: '', ready: false, loading: false, failed: false, error: null })
  const runRef = useRef(null)
  const generationRef = useRef(0)

  useEffect(() => {
    const generation = ++generationRef.current
    if (!active) return
    let controller = null
    let timer = null
    let timeout = null
    let requestGeneration = 0
    let failures = 0
    let finished = false
    let stopped = false
    const schedule = (delay) => {
      clearTimeout(timer)
      if (!stopped && !document.hidden) timer = setTimeout(() => run(), delay)
    }
    const run = async ({ manual = false } = {}) => {
      if (!active || stopped || document.hidden || controller || (finished && !manual)) return
      clearTimeout(timer)
      const request = new AbortController()
      const requestId = ++requestGeneration
      controller = request
      const requestTimeout = setTimeout(() => request.abort(), 15000)
      timeout = requestTimeout
      setResult((current) => ({ ...current, identity, loading: true, ready: current.identity === identity && current.ready }))
      try {
        const [status, warmup] = await Promise.all([
          getMemorySessionStatus({ sessionId, signal: request.signal }),
          getSessionQueryWarmup({ sessionId, signal: request.signal })
        ])
        if (stopped || generation !== generationRef.current || requestId !== requestGeneration || request.signal.aborted) return
        const versionsMatch = [status.memoryVersion, warmup.memoryVersion, warmup.targetMemoryVersion]
          .every((actual) => actual === undefined || actual === version)
        const hasFailed = versionsMatch && (failed(warmup) || failed(status))
        const ready = versionsMatch && status.queryAllowed === true && status.queryReady === true && readyWarmup(warmup)
        failures = 0
        finished = ready || hasFailed
        setResult({ identity, ready, loading: false, failed: hasFailed, error: null, status, warmup })
        if (!finished) schedule(5000)
      } catch (error) {
        if (stopped || generation !== generationRef.current || requestId !== requestGeneration || document.hidden) return
        failures += 1
        setResult({ identity, ready: false, loading: false, failed: false, error })
        schedule(retryAfterMs(error.retryAfter, failures === 1 ? 10000 : 30000))
      } finally {
        clearTimeout(requestTimeout)
        if (controller === request) controller = null
      }
    }
    runRef.current = () => run({ manual: true })
    const visibility = () => {
      clearTimeout(timer)
      requestGeneration += 1
      controller?.abort()
      controller = null
      clearTimeout(timeout)
      if (!document.hidden) { finished = false; void run() }
    }
    setResult({ identity, ready: false, loading: active, failed: false, error: null })
    if (active) void run()
    document.addEventListener('visibilitychange', visibility)
    return () => {
      stopped = true
      generationRef.current += 1
      controller?.abort()
      clearTimeout(timer)
      clearTimeout(timeout)
      document.removeEventListener('visibilitychange', visibility)
    }
  }, [sessionId, version, publicationId, identity, active])

  const refresh = useCallback(() => runRef.current?.(), [])
  return { ...(result.identity === identity ? result : { ready: false, loading: active, failed: false, error: null }), refresh }
}
