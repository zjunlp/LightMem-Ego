import { Check, LoaderCircle, Pencil, RefreshCw, RotateCcw, X } from 'lucide-react'
import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { getSessionTitle, updateSessionTitle, validateSessionTitle } from '../api/lightmem_egoApi.js'
import { useLanguage } from '../i18n.jsx'
import { useBackendFeatures } from './BackendFeatures.jsx'

const REQUEST_TIMEOUT_MS = 15000

export default function SessionTitleEditor({
  sessionId,
  title,
  onUpdated,
  className = ''
}) {
  const { t } = useLanguage()
  const features = useBackendFeatures()
  const editorId = useId()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [baseline, setBaseline] = useState(null)
  const [busy, setBusy] = useState(false)
  const [loadingTitle, setLoadingTitle] = useState(false)
  const [error, setError] = useState('')
  const [status, setStatus] = useState('')
  const [retryAt, setRetryAt] = useState(0)
  const [clock, setClock] = useState(Date.now())
  const generationRef = useRef(0)
  const sessionIdRef = useRef(sessionId)
  const controllerRef = useRef(null)
  const timeoutRef = useRef(null)
  const mountedRef = useRef(true)

  const clearRequest = useCallback(() => {
    window.clearTimeout(timeoutRef.current)
    timeoutRef.current = null
    controllerRef.current?.abort()
    controllerRef.current = null
  }, [])

  const request = useCallback(async (callback) => {
    clearRequest()
    const controller = new AbortController()
    let timedOut = false
    controllerRef.current = controller
    timeoutRef.current = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, REQUEST_TIMEOUT_MS)
    try {
      return await callback(controller.signal)
    } catch (requestError) {
      if (timedOut) requestError.sessionTitleTimedOut = true
      throw requestError
    } finally {
      if (controllerRef.current === controller) {
        window.clearTimeout(timeoutRef.current)
        timeoutRef.current = null
        controllerRef.current = null
      }
    }
  }, [clearRequest])

  const isCurrent = useCallback((generation, targetSessionId) => (
    mountedRef.current
    && generationRef.current === generation
    && sessionIdRef.current === targetSessionId
  ), [])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      generationRef.current += 1
      clearRequest()
    }
  }, [clearRequest])

  useEffect(() => {
    if (sessionIdRef.current === sessionId) return
    sessionIdRef.current = sessionId
    generationRef.current += 1
    clearRequest()
    setEditing(false)
    setDraft('')
    setBaseline(null)
    setBusy(false)
    setLoadingTitle(false)
    setError('')
    setStatus('')
    setRetryAt(0)
  }, [clearRequest, sessionId])

  useEffect(() => {
    if (!retryAt) return undefined
    const timer = window.setTimeout(() => {
      setRetryAt(0)
      setClock(Date.now())
    }, Math.max(0, retryAt - Date.now()))
    return () => window.clearTimeout(timer)
  }, [retryAt])

  const applySnapshot = useCallback((snapshot) => {
    setBaseline(snapshot)
    onUpdated?.(snapshot)
  }, [onUpdated])

  const loadLatest = useCallback(async ({ preserveDraft = false, generation = generationRef.current, targetSessionId = sessionId } = {}) => {
    setBusy(true)
    setLoadingTitle(true)
    setError('')
    try {
      const snapshot = await request((signal) => getSessionTitle({ sessionId: targetSessionId, signal }))
      if (!isCurrent(generation, targetSessionId)) return null
      applySnapshot(snapshot)
      if (!preserveDraft) setDraft(snapshot.title)
      return snapshot
    } catch (requestError) {
      if (!isCurrent(generation, targetSessionId) || requestError?.name === 'AbortError') return null
      setError(formatTitleError(requestError, t))
      return null
    } finally {
      if (isCurrent(generation, targetSessionId)) {
        setBusy(false)
        setLoadingTitle(false)
      }
    }
  }, [applySnapshot, isCurrent, request, sessionId, t])

  const openEditor = async () => {
    if (!sessionId || busy) return
    const generation = ++generationRef.current
    setEditing(true)
    setStatus('')
    setError('')
    setDraft(title || sessionId)
    await loadLatest({ generation, targetSessionId: sessionId })
  }

  const closeEditor = () => {
    generationRef.current += 1
    clearRequest()
    setEditing(false)
    setBusy(false)
    setLoadingTitle(false)
    setError('')
    setStatus('')
  }

  const completeSave = (snapshot) => {
    applySnapshot(snapshot)
    setDraft(snapshot.title)
    setError('')
    setStatus('Session name updated.')
    setEditing(false)
  }

  const saveTitle = async (value = draft) => {
    if (busy || !baseline) return
    const nextTitle = String(value ?? '').trim()
    const validation = validateSessionTitle(nextTitle)
    if (validation) {
      setError(formatTitleError({ code: validation }, t))
      return
    }
    if (baseline.title === nextTitle) {
      setError('')
      setStatus('')
      setEditing(false)
      return
    }

    const targetSessionId = sessionId
    const generation = ++generationRef.current
    setBusy(true)
    setLoadingTitle(false)
    setError('')
    setStatus('')
    setDraft(nextTitle)
    try {
      const snapshot = await request((signal) => updateSessionTitle({
        sessionId: targetSessionId,
        title: nextTitle,
        expectedRevision: baseline.titleRevision,
        signal
      }))
      if (!isCurrent(generation, targetSessionId)) return
      completeSave(snapshot)
    } catch (requestError) {
      if (!isCurrent(generation, targetSessionId) || requestError?.name === 'AbortError' && !requestError.sessionTitleTimedOut) return

      if (requestError?.code === 'session_title_conflict') {
        let current = requestError.details?.current
        if (!isTitleSnapshot(current, targetSessionId)) {
          current = await loadLatest({ preserveDraft: true, generation, targetSessionId })
        } else {
          applySnapshot(current)
        }
        if (isCurrent(generation, targetSessionId)) {
          setError(current
            ? t('The Session name changed elsewhere. Current name: {value1}. Your draft is kept; review it and save again.', { value1: current.title })
            : t('The Session name changed elsewhere. Your draft is kept. Reload the latest name before saving again.'))
        }
      } else if (requestError?.sessionTitleTimedOut || !requestError?.status) {
        const current = await loadLatest({ preserveDraft: true, generation, targetSessionId })
        if (!isCurrent(generation, targetSessionId)) return
        if (current?.title === nextTitle) completeSave(current)
        else if (current) setError(t('The save result could not be confirmed. The latest name is shown and your draft is kept.'))
        else setError(t('The save result could not be confirmed. Your draft is kept; reload the latest name before retrying.'))
      } else {
        if (requestError?.status === 429) setRetryAt(Date.now() + parseRetryAfter(requestError.retryAfter))
        setError(formatTitleError(requestError, t))
      }
    } finally {
      if (isCurrent(generation, targetSessionId)) setBusy(false)
    }
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    saveTitle()
  }

  const disabled = busy || clock < retryAt

  if (features && !features.titleEditing) return null
  return (
    <div className={`session-title-editor ${className}`.trim()}>
      {!editing && (
        <>
          <button className="icon-button secondary session-title-edit-trigger" type="button" disabled={!sessionId || busy} onClick={openEditor}>
            <Pencil size={15} />
            <span>{t('Rename Session')}</span>
          </button>
          {status && <span className="session-title-status" role="status">{t(status)}</span>}
        </>
      )}

      {editing && (
        <form className="session-title-form" onSubmit={handleSubmit} onKeyDown={(event) => {
          if (event.key === 'Escape') {
            event.preventDefault()
            closeEditor()
          }
        }} aria-busy={busy}>
          <label htmlFor={`session-title-${editorId}`}>{t('Session name')}</label>
          <input
            id={`session-title-${editorId}`}
            type="text"
            value={draft}
            aria-describedby={`session-title-help-${editorId}`}
            disabled={loadingTitle}
            onChange={(event) => { setDraft(event.target.value); setError('') }}
            autoFocus
          />
          <small id={`session-title-help-${editorId}`} className="session-title-counter">
            {Array.from(String(draft ?? '').trim()).length} / 100 · {t('Session ID')}: {sessionId}
          </small>

          {error && (
            <div className="inline-error session-title-error" role="alert">
              <span>{t(error)}</span>
              {loadingTitle && <LoaderCircle size={14} className="memory-spin" />}
            </div>
          )}

          {loadingTitle && !error && (
            <div className="session-title-loading" role="status"><LoaderCircle size={14} className="memory-spin" />{t('Loading Session name…')}</div>
          )}

          {retryAt > clock && <small className="session-title-retry">{t('Please wait before trying again.')}</small>}

          <div className="session-title-actions">
            <button className="icon-button primary" type="submit" disabled={disabled || !baseline}>
              {busy && !loadingTitle ? <LoaderCircle size={15} className="memory-spin" /> : <Check size={15} />}
              <span>{t('Save name')}</span>
            </button>
            <button className="icon-button secondary" type="button" disabled={busy} onClick={closeEditor}>
              <X size={15} /><span>{t('Cancel')}</span>
            </button>
            <button
              className="icon-button secondary"
              type="button"
              disabled={disabled || !baseline || baseline.title === sessionId}
              onClick={() => saveTitle(sessionId)}
            >
              <RotateCcw size={15} /><span>{t('Restore default name')}</span>
            </button>
            {error && !loadingTitle && (
              <button className="icon-button secondary" type="button" disabled={busy} onClick={() => loadLatest({ preserveDraft: true })}>
                <RefreshCw size={15} /><span>{t('Reload name')}</span>
              </button>
            )}
          </div>
        </form>
      )}
    </div>
  )
}

function isTitleSnapshot(value, sessionId) {
  return value
    && value.sessionId === sessionId
    && typeof value.title === 'string'
    && Number.isInteger(value.titleRevision)
}

function parseRetryAfter(value) {
  if (!value) return 10000
  const seconds = Number(value)
  const delay = Number.isFinite(seconds) ? seconds * 1000 : Date.parse(value) - Date.now()
  return Number.isFinite(delay) ? Math.max(1000, delay) : 10000
}

function formatTitleError(error, t) {
  const messages = {
    invalid_session_id: 'The Session ID is invalid.',
    invalid_title_revision: 'The Session name version is invalid. Reload the latest name.',
    invalid_session_title_response: 'The server returned an invalid Session name response.',
    session_title_required: 'Enter a Session name.',
    invalid_session_title: 'Enter a valid Session name.',
    session_title_control_character: 'Session names cannot contain line breaks or control characters.',
    session_title_too_long: 'Session names can contain up to 100 Unicode characters.',
    session_not_found: 'This Session no longer exists. Refresh the memory catalog.',
    session_title_rate_limited: 'Too many rename requests. Please wait before trying again.',
    session_title_unavailable: 'Session names are temporarily unavailable. Your draft is kept.',
    session_title_conflict: 'The Session name changed elsewhere. Your draft is kept.'
  }
  const message = messages[error?.code]
    || (error?.status === 429 ? messages.session_title_rate_limited : '')
    || (error?.status === 404 ? messages.session_not_found : '')
    || (error?.status === 503 ? messages.session_title_unavailable : '')
    || (error?.status >= 500 ? 'The Session name service is temporarily unavailable. Your draft is kept.' : '')
    || (error?.name === 'AbortError' ? 'The Session name request was cancelled.' : 'Unable to update the Session name. Your draft is kept.')
  return [t(message), error?.requestId && `${t('Request ID')}: ${error.requestId}`].filter(Boolean).join(' · ')
}
