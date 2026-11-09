import { useCallback, useEffect, useRef, useState } from 'react'
import { createMemoryImport, formatMemoryImportError, getMemoryImportTask, retryAfterMs, validateImportFile } from '../api/memoryImportApi.js'

export const IMPORT_STORAGE_KEY = 'lightmem-ego.memory-import.v1'
export const IMPORT_POLL_WINDOW_MS = 30 * 60 * 1000

function restoreTask() {
  try {
    const value = JSON.parse(sessionStorage.getItem(IMPORT_STORAGE_KEY))
    if (typeof value?.taskId === 'string' && value.taskId && typeof value.sessionId === 'string' && value.sessionId && Number.isFinite(value.deadline)) {
      return { taskId: value.taskId, sessionId: value.sessionId, fileName: value.fileName, deadline: value.deadline, state: 'queued' }
    }
  } catch { /* Storage may be disabled. The current page can still track the task. */ }
  return null
}

function persistTask(task) {
  try {
    sessionStorage.setItem(IMPORT_STORAGE_KEY, JSON.stringify({
      taskId: task.taskId, sessionId: task.sessionId, fileName: task.fileName, deadline: task.deadline
    }))
  } catch { /* Do not block imports when browser storage is unavailable. */ }
}

export function useMemoryImport({ enabled = true } = {}) {
  const enabledRef = useRef(enabled)
  enabledRef.current = enabled
  const [task, setTask] = useState(restoreTask)
  const [uploading, setUploading] = useState(false)
  const [checking, setChecking] = useState(false)
  const [paused, setPaused] = useState(false)
  const [unavailable, setUnavailable] = useState(false)
  const [error, setError] = useState('')
  const [cooldownUntil, setCooldownUntil] = useState(0)
  const taskRef = useRef(task)
  const mountedRef = useRef(false)
  const generationRef = useRef(0)
  const controllerRef = useRef(null)
  const timerRef = useRef(null)
  const timeoutRef = useRef(null)
  const failuresRef = useRef(0)
  const pausedRef = useRef(false)
  const unavailableRef = useRef(false)
  const retryAtRef = useRef(0)
  const runRef = useRef(null)

  const clearTimer = useCallback(() => {
    window.clearTimeout(timerRef.current)
    timerRef.current = null
  }, [])

  const installTask = useCallback((next) => {
    taskRef.current = next
    setTask(next)
    persistTask(next)
  }, [])

  const pause = useCallback((message) => {
    pausedRef.current = true
    setPaused(true)
    setError(message)
    clearTimer()
  }, [clearTimer])

  const schedule = useCallback((delay) => {
    clearTimer()
    if (!mountedRef.current || document.hidden || pausedRef.current) return
    const remaining = taskRef.current?.deadline - Date.now()
    timerRef.current = window.setTimeout(() => runRef.current?.(), Math.max(0, Math.min(delay, remaining)))
  }, [clearTimer])

  const run = useCallback(async ({ manual = false } = {}) => {
    const current = taskRef.current
    if (!enabledRef.current || !mountedRef.current || !current || controllerRef.current || document.hidden || unavailableRef.current) return
    if (!manual && (pausedRef.current || current.state === 'failed' || (current.state === 'ready' && current.queryReady === true))) return
    if (!manual && Date.now() >= current.deadline) {
      pause('Automatic status checks paused after 30 minutes. The task may still be running. Refresh or continue checking.')
      return
    }
    if (Date.now() < retryAtRef.current) {
      if (!pausedRef.current) schedule(retryAtRef.current - Date.now())
      return
    }
    clearTimer()
    const controller = new AbortController()
    controllerRef.current = controller
    const generation = generationRef.current
    setChecking(true)
    timeoutRef.current = window.setTimeout(() => controller.abort(), 30000)
    try {
      const result = await getMemoryImportTask({ ...current, signal: controller.signal })
      if (!mountedRef.current || generation !== generationRef.current) return
      const next = { ...current, ...result }
      installTask(next)
      failuresRef.current = 0
      retryAtRef.current = 0
      setError('')
      if (result.state === 'failed') {
        setError(formatMemoryImportError({ code: result.error?.code || result.errorCode || 'memory_import_failed', requestId: result.requestId }))
      } else if (result.state === 'ready') {
        if (result.queryReady !== true) pause('The task reports ready, but the query service is not ready. Refresh the task status.')
        else { pausedRef.current = false; setPaused(false) }
      } else if (!pausedRef.current) schedule(1000)
    } catch (failure) {
      if (!mountedRef.current || generation !== generationRef.current) return
      if (document.hidden && failure.name === 'AbortError') return
      setError(formatMemoryImportError(failure))
      if (failure.status === 404 || failure.status === 410) {
        unavailableRef.current = true
        setUnavailable(true)
        clearTimer()
      } else if (failure.status === 429) {
        const delay = retryAfterMs(failure.retryAfter)
        retryAtRef.current = Date.now() + delay
        if (!pausedRef.current) schedule(delay)
      } else if (!failure.status || failure.status >= 500) {
        failuresRef.current += 1
        if (failure.code === 'invalid_memory_import_response' || failuresRef.current >= 5) pause(`${formatMemoryImportError(failure)} Automatic checks paused.`)
        else if (!pausedRef.current) schedule(Math.min(10000, 1000 * 2 ** failuresRef.current))
      } else pause(formatMemoryImportError(failure))
    } finally {
      if (controllerRef.current === controller) {
        window.clearTimeout(timeoutRef.current)
        controllerRef.current = null
        if (mountedRef.current) setChecking(false)
      }
    }
  }, [clearTimer, installTask, pause, schedule])
  runRef.current = run

  useEffect(() => {
    if (!enabled) return undefined
    mountedRef.current = true
    queueMicrotask(() => runRef.current?.())
    const visibility = () => {
      clearTimer()
      if (document.hidden) controllerRef.current?.abort()
      else runRef.current?.()
    }
    document.addEventListener('visibilitychange', visibility)
    return () => {
      mountedRef.current = false
      generationRef.current += 1
      clearTimer()
      window.clearTimeout(timeoutRef.current)
      controllerRef.current?.abort()
      controllerRef.current = null
      document.removeEventListener('visibilitychange', visibility)
    }
  }, [clearTimer, enabled])

  useEffect(() => {
    if (!cooldownUntil) return undefined
    const timer = window.setTimeout(() => setCooldownUntil(0), Math.max(0, cooldownUntil - Date.now()))
    return () => window.clearTimeout(timer)
  }, [cooldownUntil])

  const start = useCallback(async (file, title) => {
    if (!enabledRef.current) return
    const current = taskRef.current
    if (controllerRef.current || cooldownUntil > Date.now() || (current && !unavailableRef.current && current.state !== 'failed' && !(current.state === 'ready' && current.queryReady === true))) return
    const validation = validateImportFile(file)
    if (validation) { setError(validation); return }
    clearTimer()
    const generation = ++generationRef.current
    const controller = new AbortController()
    controllerRef.current = controller
    setUploading(true)
    setError('')
    timeoutRef.current = window.setTimeout(() => controller.abort(), 120000)
    try {
      const result = await createMemoryImport({ file, title, signal: controller.signal })
      if (!mountedRef.current || generation !== generationRef.current) return
      pausedRef.current = false
      unavailableRef.current = false
      retryAtRef.current = 0
      failuresRef.current = 0
      setPaused(false)
      setUnavailable(false)
      installTask({ ...result, fileName: result.fileName || file.name, deadline: Date.now() + IMPORT_POLL_WINDOW_MS })
      schedule(0)
    } catch (failure) {
      if (!mountedRef.current || generation !== generationRef.current) return
      if (failure.status === 429) setCooldownUntil(Date.now() + retryAfterMs(failure.retryAfter))
      const uncertain = !failure.status || failure.status >= 500
      setError(uncertain
        ? 'Unable to confirm whether the import task was created. Check the Memories catalog. Uploading again may create another Session.'
        : formatMemoryImportError(failure))
    } finally {
      if (controllerRef.current === controller) {
        window.clearTimeout(timeoutRef.current)
        controllerRef.current = null
        if (mountedRef.current) setUploading(false)
      }
    }
  }, [clearTimer, cooldownUntil, installTask, schedule])

  const resume = useCallback(() => {
    if (!taskRef.current || controllerRef.current || unavailableRef.current) return
    pausedRef.current = false
    failuresRef.current = 0
    setPaused(false)
    installTask({ ...taskRef.current, deadline: Date.now() + IMPORT_POLL_WINDOW_MS })
    runRef.current?.()
  }, [installTask])

  const ready = task?.state === 'ready' && task?.queryReady === true
  const locked = uploading || (task && !unavailable && task.state !== 'failed' && !ready)
  return { task, uploading, checking, paused, unavailable, error, ready, locked: !!locked, cooldownUntil, start, refresh: () => run({ manual: true }), resume }
}
