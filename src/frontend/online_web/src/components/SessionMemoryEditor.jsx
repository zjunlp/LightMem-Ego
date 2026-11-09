import { AlertTriangle, Pencil, RefreshCw, RotateCcw, X } from 'lucide-react'
import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { MemoryRecordCard, ReadonlyValue } from './MemoryRecordCard.jsx'
import SessionTitleEditor from './SessionTitleEditor.jsx'
import { useLanguage } from '../i18n.jsx'
import { useSessionMemoryEditor } from '../hooks/useSessionMemoryEditor.jsx'
import { changedMemoryRecords, codePointLength } from '../hooks/memoryEditStore.js'
import { useAdminMode } from './AdminMode.jsx'

export default function SessionMemoryEditor({ sessionId, sessionTitle, initialMemory, onSessionTitleUpdated,
  onPublication, queryState, active = true, embedded = false, current = false, closeRequestRef, onEditingChange }) {
  const { t } = useLanguage()
  const admin = useAdminMode()
  const { state, store } = useSessionMemoryEditor(sessionId, { initialMemory, onPublication })
  const { memory, drafts, loading, submitting, pendingOperation, checkingResult, error, fieldErrors, notice } = state
  const [open, setOpen] = useState(false)
  const [copyNotice, setCopyNotice] = useState('')
  const openerRef = useRef(null)
  const openedVersionRef = useRef(0)
  const changedRecords = changedMemoryRecords(state)
  const dirty = changedRecords.length > 0
  const blocking = submitting || (!!pendingOperation && !error)

  const requestClose = useCallback(() => {
    if (blocking) return false
    if (pendingOperation) {
      if (!window.confirm(t('The save result is not confirmed. Processing may continue. Close this view?'))) return false
    } else if (dirty) {
      if (!window.confirm(t('Discard unsaved memory changes?'))) return false
      store.discard(sessionId)
    }
    setOpen(false)
    requestAnimationFrame(() => openerRef.current?.focus())
    return true
  }, [blocking, pendingOperation, dirty, store, sessionId, t])
  const previousActiveRef = useRef(active)
  useEffect(() => {
    const wasActive = previousActiveRef.current
    previousActiveRef.current = active
    if (wasActive && !active) requestClose()
  }, [active, requestClose])
  useEffect(() => { setOpen(false) }, [sessionId])
  useEffect(() => {
    if (queryState?.ready || queryState?.failed) store.setQueryState(sessionId, queryState.ready, queryState.failed)
  }, [store, sessionId, queryState?.ready, queryState?.failed])
  useEffect(() => { onEditingChange?.({ open, blocking, dirty }) }, [open, blocking, dirty, onEditingChange])
  useEffect(() => {
    if (open && notice && !pendingOperation && memory?.memoryVersion > openedVersionRef.current) {
      setOpen(false)
      requestAnimationFrame(() => openerRef.current?.focus())
    }
  }, [open, notice, pendingOperation, memory?.memoryVersion])
  useEffect(() => {
    if (!dirty && !pendingOperation) return
    const beforeUnload = (event) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', beforeUnload)
    return () => window.removeEventListener('beforeunload', beforeUnload)
  }, [dirty, pendingOperation])
  useEffect(() => {
    if (!closeRequestRef) return
    closeRequestRef.current = requestClose
    return () => { closeRequestRef.current = null }
  }, [closeRequestRef, requestClose])

  const openEditor = (event) => {
    openerRef.current = event.currentTarget
    openedVersionRef.current = memory?.memoryVersion || 0
    setOpen(true)
    if (!pendingOperation) void store.load(sessionId)
  }
  const discardAndReload = () => {
    if (dirty && !window.confirm(t('Discard unsaved memory changes?'))) return
    if (store.discard(sessionId)) void store.load(sessionId, { discard: true })
  }
  const rollback = () => {
    if (dirty) {
      if (!window.confirm(t('Discard unsaved memory changes and undo the last saved change?'))) return
      if (!store.discard(sessionId)) return
    }
    store.rollback(sessionId)
  }
  const copyDraft = async () => {
    try {
      await navigator.clipboard.writeText(changedRecords.map((record) => `${record.id}\n${record.content}`).join('\n\n'))
      setCopyNotice(t('Draft copied.'))
    } catch { setCopyNotice(t('Unable to copy automatically. Select and copy your draft text.')) }
  }
  const canRetryPending = !!pendingOperation && !submitting
  if (admin && !admin.unlocked) return <section className="current-memory-action">
    <div className="session-memory-actions" />
    <p className="current-memory-warning" role="status">{t('Enter admin mode to edit memories.')}</p>
  </section>
  return <>
    <section className="current-memory-action" aria-label={t('Session long-term memory actions')}>
      {!open && <div className="session-memory-actions">
        <button ref={openerRef} className="icon-button secondary current-memory-edit-button" type="button"
          disabled={!sessionId || submitting || (!!memory && !memory.canEdit && !pendingOperation)} onClick={openEditor}>
          <Pencil size={16} /><span>{t(pendingOperation ? 'Continue checking' : current ? 'Edit Current Session Long-Term Memory' : 'Edit long-term memory')}</span>
        </button>
        {memory?.canRollback && <button className="icon-button secondary" type="button" disabled={submitting || !!pendingOperation} onClick={rollback}>
          <RotateCcw size={15} />{t(notice ? 'Undo Changes' : 'Undo Last Change')}
        </button>}
      </div>}
      {!open && memory && !memory.canEdit && <div className="current-memory-warning" role="status">{t(formatDisabledReason(memory.editDisabledReason))}</div>}
      {!open && loading && <div role="status">{t('Loading long-term memory…')}</div>}
      {!open && checkingResult && <div className="current-memory-notice" role="status"><span className="memory-spinner" />{t('Checking whether the memory change was saved…')}</div>}
      {!open && submitting && !checkingResult && <div className="current-memory-notice" role="status"><span className="memory-spinner" />{t('Rebuilding complete long-term memory and knowledge graphs…')}</div>}
      {notice && <div className="current-memory-notice" role="status">{t(notice)}</div>}
      {!open && error && !checkingResult && <div className="inline-error current-memory-error" role="alert">
        <span>{formatMemoryEditError(error, t)}</span>
        {canRetryPending && <button className="icon-button secondary" type="button" onClick={() => store.retry(sessionId)}>{t('Continue checking')}</button>}
        <button className="icon-button secondary" type="button" disabled={loading || submitting} onClick={() => store.refreshLatest(sessionId)}>
          <RefreshCw size={15} />{t('Refresh latest memory')}
        </button>
      </div>}
    </section>
    {copyNotice && <div role="status">{copyNotice}</div>}
    {open && <MemoryEditDialog embedded={embedded} sessionId={sessionId} sessionTitle={sessionTitle}
      onSessionTitleUpdated={onSessionTitleUpdated} memory={memory} drafts={drafts} loading={loading}
      submitting={submitting} pendingOperation={pendingOperation} checkingResult={checkingResult} error={error} fieldErrors={fieldErrors}
      changedRecords={changedRecords} onCopyDraft={copyDraft} canRetryPending={canRetryPending}
      canDiscardPending={!!error?.retryableRebuild} retryPendingLabel={t(error?.retryableRebuild ? 'Retry rebuild' : 'Continue checking')}
      operationBlocking={blocking} onDraftChange={(id, content) => store.setDraft(sessionId, id, content)}
      onRetryLoad={() => store.load(sessionId)} onRetryPending={() => store.retry(sessionId)} onRefreshLatest={() => store.refreshLatest(sessionId)}
      onDiscardPending={discardAndReload} onRollback={rollback} onSubmit={() => store.submit(sessionId)} onClose={requestClose} />}
  </>
}

function MemoryEditDialog({
  embedded,
  sessionId,
  sessionTitle,
  onSessionTitleUpdated,
  memory,
  drafts,
  loading,
  submitting,
  pendingOperation,
  checkingResult,
  error,
  fieldErrors,
  canRetryPending,
  changedRecords,
  onCopyDraft,
  canDiscardPending,
  retryPendingLabel,
  operationBlocking,
  onDraftChange,
  onRetryLoad,
  onRetryPending,
  onRefreshLatest,
  onDiscardPending,
  onRollback,
  onSubmit,
  onClose
}) {
  const { t } = useLanguage()
  const titleId = useId()
  const dialogRef = useRef(null)
  const closeRef = useRef(null)
  const hasLongTermMemory = Number(memory?.counts?.total || 0) > 0
  const hasEditableRecord = !!memory?.episodic?.['30sec']?.some((record) => record.editable)

  useEffect(() => {
    if (embedded) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const frame = requestAnimationFrame(() => closeRef.current?.focus())
    return () => {
      cancelAnimationFrame(frame)
      document.body.style.overflow = previousOverflow
    }
  }, [embedded])
  useEffect(() => {
    if (!embedded || !memory || loading) return undefined
    const frame = requestAnimationFrame(() => {
      const target = dialogRef.current?.querySelector('textarea:not([disabled])') || dialogRef.current
      target?.focus()
    })
    return () => cancelAnimationFrame(frame)
  }, [embedded, memory?.memoryVersion, loading])

  const handleKeyDown = (event) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      if (!operationBlocking) onClose()
      return
    }
    trapDialogFocus(event, dialogRef.current)
  }

  const content = (
      <section ref={dialogRef} className={embedded ? 'session-memory-edit-content' : 'current-memory-dialog'}
        role={embedded ? undefined : 'dialog'} aria-modal={embedded ? undefined : true}
        aria-labelledby={embedded ? undefined : titleId} tabIndex={-1} onKeyDown={embedded ? undefined : handleKeyDown}>
        {!embedded && <header className="current-memory-dialog-header">
          <div><span>{t('Session Long-Term Memory')}</span><h2 id={titleId}>{sessionTitle || sessionId}</h2>
            <small>{sessionId}</small>
            <SessionTitleEditor sessionId={sessionId} title={sessionTitle || sessionId} onUpdated={onSessionTitleUpdated} />
            {memory && <small>{t('M_lt v')}{memory.memoryVersion}</small>}
          </div>
          <button ref={closeRef} className="icon-button secondary" type="button" disabled={operationBlocking} aria-label={t('Close')} onClick={onClose}><X size={17} /></button>
        </header>}
        <div className="current-memory-dialog-body">
          {loading && !memory && <div className="memory-loading"><span className="memory-spinner" />{t('Loading long-term memory…')}</div>}
          {checkingResult && <div className="memory-loading" role="status">
            <span className="memory-spinner" />{t('Checking whether the memory change was saved…')}
          </div>}

          {error && !checkingResult && (
            <div className="inline-error current-memory-error" role="alert">
              <span>{formatMemoryEditError(error, t)}</span>
              {error.requestId && <small>{t("Request")} {error.requestId}</small>}
              {!memory && !pendingOperation && <button className="icon-button secondary" type="button" onClick={onRetryLoad}>{t("Reload")}</button>}
              {canRetryPending && <button className="icon-button secondary" type="button" onClick={onRetryPending}>{t(retryPendingLabel)}</button>}
              {memory && <button className="icon-button secondary" type="button" disabled={loading || submitting} onClick={onRefreshLatest}>
                <RefreshCw size={15} />{t('Refresh latest memory')}
              </button>}
              {error?.code === 'memory_version_conflict' && <>
                <button className="icon-button secondary" type="button" onClick={onCopyDraft}>{t('Copy draft')}</button>
                <button className="icon-button secondary" type="button" onClick={onDiscardPending}>{t('Discard and reload')}</button>
              </>}
              {canDiscardPending && <button className="icon-button secondary" type="button" onClick={onDiscardPending}>{t("Discard and reload")}</button>}
            </div>
          )}

          {submitting && !checkingResult && (
            <div className="memory-loading" role="status">
              <span className="memory-spinner" />
              {t('Rebuilding complete long-term memory and knowledge graphs…')}
            </div>
          )}

          {memory && (
            <>
              {!hasLongTermMemory ? (
                <div className="memory-empty current-memory-empty" role="status">{t('No long-term memory available.')}</div>
              ) : !memory.editSupported ? (
                <div className="current-memory-warning" role="status">
                  <AlertTriangle size={16} />
                  <span>{t('The editing API is not enabled by the backend')}</span>
                </div>
              ) : !memory.canEdit && (
                <div className="current-memory-warning" role="status">
                  <AlertTriangle size={16} />
                  <span>{t(formatDisabledReason(memory.editDisabledReason))}</span>
                </div>
              )}
              {hasLongTermMemory && (
                <>
                  <div className="current-memory-warning" role="status">
                    {t('Saving regenerates summaries at other granularities and semantic memories, and updates graphs and retrieval indexes.')}
                    {memory.sessionKind === 'imported_memory' && <span>{t('Imported summaries and semantic memories may be replaced. Original visual records, media, and real timestamps are preserved.')}</span>}
                  </div>
                  <section className="current-memory-edit-section">
                    <div className="current-memory-section-heading">
                      <div>
                        <span>{t("Editable scope")}</span>
                      <h3>{t('30-Second Memory')}</h3>
                      </div>
                      <small>{memory.editPolicy.maxCharsPerRecord} {t("per record ·")} {memory.editPolicy.maxTotalChars} {t("characters per update")}</small>
                    </div>
                    {!memory.episodic['30sec'].length && <div className="memory-empty">{t("No 30-second memory available.")}</div>}
                    <div className="current-memory-record-list">
                      {memory.episodic['30sec'].map((record) => (
                        <ThirtySecondEditor
                          key={record.id}
                          record={record}
                          value={drafts[record.id] ?? record.content}
                          maxLength={memory.editPolicy.maxCharsPerRecord}
                          disabled={loading || submitting || !!pendingOperation}
                          error={fieldErrors[record.id]}
                          onChange={(content) => onDraftChange(record.id, content)}
                        />
                      ))}
                    </div>
                    {fieldErrors._global && <div className="inline-error" role="alert">{t(fieldErrors._global)}</div>}
                  </section>

                  <ReadonlyMemory memory={memory} />
                </>
              )}
            </>
          )}
        </div>

        <footer className="current-memory-dialog-footer">
          {memory?.canRollback && (
            <button className="icon-button secondary" type="button" disabled={loading || submitting || !!pendingOperation} onClick={onRollback}>
              <RotateCcw size={15} />{t("Undo Last Change")}</button>
          )}
          <span className="memory-edit-total">{t('Changed records: {count} · {length} / {limit} code points', {
            count: changedRecords.length, length: changedRecords.reduce((sum, record) => sum + codePointLength(record.content), 0), limit: memory?.editPolicy?.maxTotalChars || 32000
          })}</span>
          <button className="icon-button secondary" type="button" disabled={operationBlocking} onClick={onClose}>{t('Cancel')}</button>
          <button className="current-memory-confirm" type="button" disabled={!memory?.canEdit || !hasEditableRecord || !changedRecords.length || loading || submitting || !!pendingOperation || error?.code === 'memory_version_conflict'} onClick={onSubmit}>
            {submitting && <span className="button-spinner" />}
            {submitting ? t('Rebuilding…') : t('Save Changes')}
          </button>
        </footer>
      </section>
  )
  if (embedded) return content
  return createPortal(<div className="memory-dialog-overlay current-memory-dialog-overlay fade-in" onMouseDown={(event) => {
    if (event.target === event.currentTarget && !operationBlocking) onClose()
  }}>{content}</div>, document.body)
}

function ThirtySecondEditor({ record, value, maxLength, disabled, error, onChange }) {
  const { t } = useLanguage()
  const length = codePointLength(value)
  const errorId = useId()
  const reason = !record.editable ? formatDisabledReason(record.editDisabledReason) : ''
  return (
    <article className={`current-memory-record ${record.editable ? '' : 'is-readonly'}`}>
      <div className="current-memory-record-meta">
        <code>{record.id}</code>
        <span>{length} / {maxLength}</span>
      </div>
      {record.editable ? (
        <textarea
          value={value}
          disabled={disabled}
          aria-label={t("Edit 30-second memory {value1}", { value1: record.id })}
          aria-invalid={!!error}
          aria-describedby={error ? errorId : undefined}
          onChange={(event) => onChange(event.target.value)}
        />
      ) : (
        <p>{record.content || '—'}</p>
      )}
      {reason && <div className="current-memory-readonly-reason">{t("Read-only:")} {t(reason)}</div>}
      {error && <div id={errorId} className="current-memory-field-error" role="alert">{t(error)}</div>}
      {record.data && <ReadonlyValue value={record.data} root />}
    </article>
  )
}

function ReadonlyMemory({ memory }) {
  const { t } = useLanguage()
  const groups = [
    ['3-Minute Memory', memory.episodic['3min']],
    ['10-Minute Memory', memory.episodic['10min']],
    ['1-Hour Memory', memory.episodic['1h']],
    ['Semantic Memory', memory.semantic],
    ['Visual Memory', memory.visual]
  ]
  return (
    <section className="current-memory-readonly-section">
      <h3>{t("Other Long-Term Memory (Read-Only)")}</h3>
      {groups.map(([label, records]) => (
        <details key={label}>
          <summary>{t(label)}<span>{records.length} {t("records")}</span></summary>
          {records.length ? (
            <div className="memory-list">
              {records.map((record) => <MemoryRecordCard memory={record} key={record.id} />)}
            </div>
          ) : <div className="memory-empty is-compact">{t("No records.")}</div>}
        </details>
      ))}
    </section>
  )
}

export function formatDisabledReason(code) {
  const labels = {
    no_editable_30sec: 'No editable 30-second episodic memories are available.',
    no_editable_records: 'No editable 30-second episodic memories are available.',
    missing_stable_id: 'Stable record ID is missing',
    duplicate_stable_id: 'Stable record ID is duplicated',
    unsupported_source_record: 'Source record does not support editing',
    missing_editable_text_field: 'No editable text field is available',
    backend_edit_not_supported: 'The editing API is not enabled by the backend',
    full_propagation_not_supported: 'Full memory propagation is not available'
  }
  return labels[code] || code || 'This record cannot be edited'
}

function formatMemoryEditError(error, t = (text, params = {}) => text.replace(/\{(\w+)\}/g, (match, key) => params[key] ?? match)) {
  const labels = {
    session_not_found: 'This Session does not exist.',
    memory_not_ready: 'No long-term memory available.',
    memory_version_conflict: 'This Session was updated elsewhere. Your draft is preserved. Copy it or reload the latest version.',
    rollback_unavailable: 'The one-time rollback has already been used.',
    idempotency_key_conflict: 'The request key conflicts with another operation. Preserve your draft and reload.',
    memory_update_in_progress: 'The server is still processing this request.',
    memory_update_failed: 'The server could not update Memory or Graph. The previous version is unchanged.',
    semantic_rebuild_unavailable: 'Semantic rebuilding is unavailable. Retry the rebuild with the same request.',
    memory_derivation_failed: 'A derived memory component failed to rebuild. Retry with the same request.',
    memory_too_long: 'The update exceeds the server length limit.',
    invalid_thirty_second_memory: 'The 30-second memory content is invalid.',
    request_timeout: 'The request timed out; the update result is not confirmed.',
    network_error: 'The network failed; the update result is not confirmed.',
    invalid_memory_mutation_response: error?.serverAccepted
      ? 'The server accepted the change, but the latest version is not loaded yet. Refresh latest memory.'
      : 'The save response versions did not match. Refresh the latest memory to confirm whether the change was saved.'
  }
  const component = safeDerivationComponent(error?.details?.component)
  const message = labels[error?.code] || 'Unable to process the current long-term memory.'
  return component ? t('{message} Component: {component}.', { message: t(message), component: t(component) }) : t(message)
}

function safeDerivationComponent(value) {
  const components = new Set([
    '30sec',
    '3min',
    '10min',
    '1h',
    'multiscale',
    'sidecar',
    'semantic',
    'openie',
    'episodic_graph',
    'semantic_graph',
    'query_runtime',
    'snapshot'
  ])
  return components.has(value) ? value : (value ? 'derived component' : '')
}

function trapDialogFocus(event, root) {
  if (event.key !== 'Tab') return
  const focusable = Array.from(root?.querySelectorAll([
    'button:not([disabled])',
    'textarea:not([disabled])',
    'summary',
    '[tabindex]:not([tabindex="-1"])'
  ].join(',')) || [])
  if (!focusable.length) return
  const first = focusable[0]
  const last = focusable[focusable.length - 1]
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault()
    last.focus()
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault()
    first.focus()
  }
}
