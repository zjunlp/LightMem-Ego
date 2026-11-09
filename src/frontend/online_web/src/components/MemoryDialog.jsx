import { ChevronDown, ChevronUp, LoaderCircle, Sparkles, X } from 'lucide-react'
import { lazy, Suspense, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  buildMemoryExportFilename,
  exportLongTermMemoryJson,
  fetchLongTermMemories
} from '../api/lightmem_egoApi.js'
import SessionTitleEditor from './SessionTitleEditor.jsx'
import ImportedMemoryAsk from './ImportedMemoryAsk.jsx'
import SessionMemoryEditor from './SessionMemoryEditor.jsx'
import { useSessionQueryReadiness } from '../hooks/useSessionQueryReadiness.js'
import { MemoryRecordCard, ReadonlyValue, humanizeKey } from './MemoryRecordCard.jsx'
export { MemoryRecordCard, ReadonlyValue } from './MemoryRecordCard.jsx'
import { useLanguage } from '../i18n.jsx'

const MemoryGraphView = lazy(() => import('./memory-graph/MemoryGraphView.jsx'))

const PAGE_SIZE = 100
const EPISODIC_GROUPS = [
  ['30sec', '30 seconds'],
  ['3min', '3 minutes'],
  ['10min', '10 minutes'],
  ['1h', '1 hour']
]
const GROUP_IDS = [
  ...EPISODIC_GROUPS.map(([key]) => `episodic-${key}`),
  'semantic',
  'visual'
]

function createVisibleCounts() {
  return Object.fromEntries(GROUP_IDS.map((key) => [key, PAGE_SIZE]))
}

function createExpandedGroups() {
  return new Set()
}

export default function MemoryDialog({
  session,
  active = true,
  cachedMemory,
  authoritativeUpdate: externalUpdate,
  onPublication,
  openerElement,
  fallbackFocusElement,
  onMemoryLoaded,
  onSessionTitleUpdated,
  onSessionUnavailable,
  onClose,
  initialTab = 'graph'
}) {
  const { t } = useLanguage()
  const isImported = session.sessionKind === 'imported_memory'
  const [localUpdate, setLocalUpdate] = useState(null)
  const lastPublicationRef = useRef(null)
  const matchingExternalUpdate = externalUpdate?.sessionId === session.sessionId ? externalUpdate : null
  const matchingLocalUpdate = localUpdate?.sessionId === session.sessionId ? localUpdate : null
  const candidate = !matchingExternalUpdate ? matchingLocalUpdate
    : !matchingLocalUpdate ? matchingExternalUpdate
      : comparePublicationUpdates(matchingExternalUpdate, matchingLocalUpdate) >= 0
        ? matchingExternalUpdate : matchingLocalUpdate
  if (candidate?.sessionId === session.sessionId && (!lastPublicationRef.current
    || candidate.memory.memoryVersion > lastPublicationRef.current.memory.memoryVersion
    || (candidate.memory.memoryVersion === lastPublicationRef.current.memory.memoryVersion
      && candidate.revision > lastPublicationRef.current.revision))) lastPublicationRef.current = candidate
  const authoritativeUpdate = lastPublicationRef.current
  const [editing, setEditing] = useState({ open: false, blocking: false, dirty: false })
  const previousActiveRef = useRef(active)
  const handleEditingChange = useCallback((next) => setEditing((current) =>
    current.open === next.open && current.blocking === next.blocking && current.dirty === next.dirty ? current : next), [])
  const editorCloseRef = useRef(null)
  const exportGenerationRef = useRef(0)
  const latestMemoryRef = useRef(cachedMemory)
  const handlePublication = useCallback((update) => {
    setLocalUpdate(update)
    onPublication?.(update)
    onMemoryLoaded?.({ ...update.memory, ...(update.queryReady !== undefined ? { queryReady: update.queryReady } : {}) })
  }, [onPublication, onMemoryLoaded])
  const handleClose = () => {
    if (editorCloseRef.current && !editorCloseRef.current()) return
    onClose()
  }
  useEffect(() => {
    const wasActive = previousActiveRef.current
    previousActiveRef.current = active
    if (wasActive && !active) handleClose()
  }, [active])
  const [activeTab, setActiveTab] = useState(initialTab)
  const [memory, setMemory] = useState(cachedMemory)
  const editorMemory = useMemo(() => memory ? { ...memory, sessionKind: memory.sessionKind || session.sessionKind } : null, [memory, session.sessionKind])
  const queryState = useSessionQueryReadiness({ ...session, memoryVersion: memory?.memoryVersion || session.memoryVersion }, { publication: authoritativeUpdate, active: isImported })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedGroups, setExpandedGroups] = useState(createExpandedGroups)
  const [visibleCounts, setVisibleCounts] = useState(createVisibleCounts)
  const [exportingMemoryId, setExportingMemoryId] = useState('')
  const [exportNotice, setExportNotice] = useState(null)
  const titleId = useId()
  const dialogRef = useRef(null)
  const closeButtonRef = useRef(null)
  const scrollRootRef = dialogRef
  const requestControllerRef = useRef(null)
  const requestGenerationRef = useRef(0)
  const mountedRef = useRef(true)
  const onMemoryLoadedRef = useRef(onMemoryLoaded)
  const onSessionUnavailableRef = useRef(onSessionUnavailable)
  const exportNoticeTimerRef = useRef(null)

  onMemoryLoadedRef.current = onMemoryLoaded
  onSessionUnavailableRef.current = onSessionUnavailable

  const reportExportEvent = useCallback((event, payload = {}) => {
    const detail = { event, ...payload }
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('memory_export', { detail }))
      window.analytics?.track?.(event, detail)
    }
  }, [])

  const showExportNotice = useCallback((notice) => {
    window.clearTimeout(exportNoticeTimerRef.current)
    setExportNotice(notice)
    if (notice?.tone !== 'error') {
      exportNoticeTimerRef.current = window.setTimeout(() => setExportNotice(null), 5000)
    }
  }, [])

  const handleExport = useCallback(async (record) => {
    if (exportingMemoryId) return

    const exportGeneration = exportGenerationRef.current
    const startedAt = performance.now()
    const memoryId = record?.id || ''
    reportExportEvent('export_click', { memoryId, sessionId: session.sessionId })
    setExportingMemoryId(memoryId || '__missing_memory_id__')
    setExportNotice(null)

    try {
      const result = await exportLongTermMemoryJson({
        sessionId: session.sessionId,
        memoryId,
        memory: record
      })
      if (!mountedRef.current) return
      if (exportGeneration !== exportGenerationRef.current) {
        showExportNotice({ tone: 'error', message: 'Memory changed during export. Export the latest record again.' })
        return
      }
      const objectUrl = URL.createObjectURL(result.blob)
      const link = document.createElement('a')
      link.href = objectUrl
      link.download = normalizeDownloadFilename(result.filename, record)
      link.style.display = 'none'
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(objectUrl)
      reportExportEvent('export_success', {
        memoryId,
        sessionId: session.sessionId,
        durationMs: Math.round(performance.now() - startedAt),
        bytes: result.blob.size,
        requestId: result.requestId
      })
      showExportNotice({ tone: 'success', message: 'JSON downloaded.' })
    } catch (exportError) {
      if (exportError?.name === 'AbortError') return
      reportExportEvent('export_fail', {
        memoryId,
        sessionId: session.sessionId,
        durationMs: Math.round(performance.now() - startedAt),
        statusCode: exportError?.status,
        errorCode: exportError?.code || '',
        requestId: exportError?.requestId || ''
      })
      showExportNotice({ tone: 'error', message: formatExportError(exportError) })
    } finally {
      if (mountedRef.current) setExportingMemoryId('')
    }
  }, [exportingMemoryId, reportExportEvent, session.sessionId, showExportNotice])

  const loadMemory = useCallback(async ({ force = false } = {}) => {
    if (requestControllerRef.current && !force) return
    if (force) requestControllerRef.current?.abort()

    const controller = new AbortController()
    const generation = ++requestGenerationRef.current
    requestControllerRef.current = controller
    setLoading(true)
    setError(null)

    try {
      const result = await fetchLongTermMemories({
        sessionId: session.sessionId,
        signal: controller.signal
      })
      if (
        !mountedRef.current
        || generation !== requestGenerationRef.current
        || result.sessionId !== session.sessionId
      ) return

      if (result.memoryVersion < (latestMemoryRef.current?.memoryVersion || 0)) return
      latestMemoryRef.current = result
      setMemory(result)
      setVisibleCounts(createVisibleCounts())
      onMemoryLoadedRef.current?.(result)
    } catch (loadError) {
      if (loadError?.name === 'AbortError') return
      if (!mountedRef.current || generation !== requestGenerationRef.current) return
      if (loadError?.code === 'session_not_found' || loadError?.status === 404) {
        onSessionUnavailableRef.current?.(session.sessionId)
        return
      }
      setError(loadError)
    } finally {
      if (requestControllerRef.current === controller) requestControllerRef.current = null
      if (mountedRef.current && generation === requestGenerationRef.current) setLoading(false)
    }
  }, [session.sessionId])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      requestControllerRef.current?.abort()
      requestControllerRef.current = null
      window.clearTimeout(exportNoticeTimerRef.current)
      requestGenerationRef.current += 1
    }
  }, [loadMemory])

  useEffect(() => { void loadMemory() }, [loadMemory])

  const showRecords = () => {
    setActiveTab('records')
    if (!memory && !requestControllerRef.current) loadMemory()
  }

  useEffect(() => {
    setActiveTab(initialTab)
  }, [initialTab, session.sessionId])

  useEffect(() => {
    if (activeTab === 'records' && !memory && !requestControllerRef.current) loadMemory()
  }, [activeTab, loadMemory, memory])

  useEffect(() => {
    const authoritativeMemory = authoritativeUpdate?.memory
    if (!authoritativeMemory || authoritativeMemory.sessionId !== session.sessionId) return
    if (authoritativeMemory.memoryVersion < (latestMemoryRef.current?.memoryVersion || 0)) return
    requestControllerRef.current?.abort()
    requestGenerationRef.current += 1
    exportGenerationRef.current += 1
    latestMemoryRef.current = authoritativeMemory
    setLoading(false)
    setError(null)
    setMemory(authoritativeMemory)
    setVisibleCounts(createVisibleCounts())
  }, [authoritativeUpdate, session.sessionId])

  useEffect(() => {
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const focusFrame = requestAnimationFrame(() => closeButtonRef.current?.focus())
    return () => {
      cancelAnimationFrame(focusFrame)
      document.body.style.overflow = previousOverflow
      const focusTarget = openerElement?.isConnected ? openerElement : fallbackFocusElement
      requestAnimationFrame(() => focusTarget?.focus())
    }
  }, [fallbackFocusElement, openerElement])

  const toggleGroup = (groupId) => {
    setExpandedGroups((current) => {
      const next = new Set(current)
      if (next.has(groupId)) next.delete(groupId)
      else next.add(groupId)
      return next
    })
  }

  const showMore = (groupId, total) => {
    setVisibleCounts((current) => ({
      ...current,
      [groupId]: Math.min(total, (current[groupId] || PAGE_SIZE) + PAGE_SIZE)
    }))
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      if (exportNotice?.tone === 'error') {
        setExportNotice(null)
        return
      }
      if (editing.open) editorCloseRef.current?.()
      else handleClose()
      return
    }
    if (event.key !== 'Tab') return

    const focusable = Array.from(dialogRef.current?.querySelectorAll([
      'button:not([disabled])',
      'a[href]',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      'summary',
      '[tabindex]:not([tabindex="-1"])'
    ].join(',')) || []).filter((element) => !element.closest('[hidden]'))

    if (!focusable.length) {
      event.preventDefault()
      dialogRef.current?.focus()
      return
    }

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

  const formattedError = error ? formatMemoryError(error) : ''
  const retryAllowed = error && error?.status !== 413 && error?.code !== 'memory_response_too_large'

  return createPortal(
    <div
      className="memory-dialog-overlay fade-in"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) handleClose()
      }}
    >
      <section
        ref={dialogRef}
        className={`memory-dialog${editing.open ? ' is-editing' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onKeyDown={handleKeyDown}
      >
        <header className="memory-dialog-header">
          <div className="memory-dialog-title-block">
            <div>
              <span>{t('Complete business-level M_lt')}</span>
              <h2 id={titleId}>{session.title || session.sessionId}</h2>
              <small className="memory-dialog-session-id">{session.sessionId}</small>
              <SessionTitleEditor
                sessionId={session.sessionId}
                title={session.title || session.sessionId}
                onUpdated={onSessionTitleUpdated}
                className="memory-dialog-title-editor"
              />
            </div>
            <div className="memory-dialog-version-chips" aria-label={t("Memory component versions")}>
              <span>{t("M_lt v")}{memory?.memoryVersion || session.memoryVersion}</span>
              <span>{t('Episodic')} {t(formatVersion(memory?.componentVersions?.episodic ?? session.componentVersions?.episodic))}</span>
              <span>{t('Semantic')} {t(formatVersion(memory?.componentVersions?.semantic ?? session.componentVersions?.semantic))}</span>
            </div>
          </div>
          <button ref={closeButtonRef} className="icon-button secondary" type="button" aria-label={t('Close Memory')} disabled={editing.blocking} onClick={handleClose}>
            <X size={17} />
            <span>{t('Close')}</span>
          </button>
        </header>

        <div className="memory-history-editor">
          <SessionMemoryEditor embedded sessionId={session.sessionId} sessionTitle={session.title || session.sessionId}
            initialMemory={editorMemory}
            onSessionTitleUpdated={onSessionTitleUpdated} onPublication={handlePublication} queryState={queryState}
            closeRequestRef={editorCloseRef} onEditingChange={handleEditingChange} />
        </div>
        <div className="memory-dialog-body-wrap" hidden={editing.open}>
        <div className="memory-dialog-tabs" role="tablist" aria-label={t('Long-term memory detail view')}>
          <button
            id={`${titleId}-graph-tab`}
            type="button"
            role="tab"
            aria-selected={activeTab === 'graph'}
            aria-controls={`${titleId}-graph-panel`}
            className={activeTab === 'graph' ? 'active' : ''}
            onClick={() => setActiveTab('graph')}
          >
            {t('Graph')}
          </button>
          <button
            id={`${titleId}-records-tab`}
            type="button"
            role="tab"
            aria-selected={activeTab === 'records'}
            aria-controls={`${titleId}-records-panel`}
            className={activeTab === 'records' ? 'active' : ''}
            onClick={showRecords}
          >
            {t('Records')}
          </button>
          {isImported && (
            <button
              id={`${titleId}-ask-tab`}
              type="button"
              role="tab"
              aria-selected={activeTab === 'ask'}
              aria-controls={`${titleId}-ask-panel`}
              className={activeTab === 'ask' ? 'active' : ''}
              onClick={() => { setActiveTab('ask'); queryState.refresh() }}
            >
              {t('Ask Memory')}
            </button>
          )}
        </div>

        {exportNotice?.tone === 'success' && (
          <div className="memory-export-notice is-success" role="status">
            {t(exportNotice.message)}
          </div>
        )}

        {exportNotice?.tone === 'error' && (
          <div className="memory-export-error-overlay" onMouseDown={(event) => {
            if (event.target === event.currentTarget) setExportNotice(null)
          }}>
            <section
              className="memory-export-error-dialog"
              role="alertdialog"
              aria-modal="true"
              aria-labelledby={`${titleId}-export-error-title`}
              aria-describedby={`${titleId}-export-error-message`}
            >
              <div>
                <strong id={`${titleId}-export-error-title`}>{t("Unable to export JSON")}</strong>
                <p id={`${titleId}-export-error-message`}>{t(exportNotice.message)}</p>
              </div>
              <button className="icon-button secondary" type="button" autoFocus onClick={() => setExportNotice(null)}>{t("OK")}</button>
            </section>
          </div>
        )}

        <div className={`memory-dialog-body is-${activeTab}`}>
          {activeTab === 'graph' && (
            <div
              id={`${titleId}-graph-panel`}
              role="tabpanel"
              aria-labelledby={`${titleId}-graph-tab`}
            >
              <Suspense fallback={(
                <div className="memory-graph-request-state" aria-live="polite">
                  <span className="memory-spinner" />
                  <strong>{t('Loading graph viewer…')}</strong>
                </div>
              )}>
                <MemoryGraphView
                  session={session}
                  authoritativeUpdate={authoritativeUpdate}
                  onSessionUnavailable={onSessionUnavailableRef.current}
                />
              </Suspense>
            </div>
          )}

          <div
            id={`${titleId}-records-panel`}
            role="tabpanel"
            aria-labelledby={`${titleId}-records-tab`}
            hidden={activeTab !== 'records'}
          >
            {loading && !memory && (
              <div className="memory-loading" aria-live="polite">
                <span className="memory-spinner" />
                {t('Loading complete long-term memory…')}
              </div>
            )}

            {formattedError && (
              <div className="inline-error memory-error" role="alert">
                <span>{t(formattedError)}</span>
                {retryAllowed && (
                  <button className="icon-button secondary" type="button" onClick={() => loadMemory({ force: true })}>{t("Retry")}</button>
                )}
              </div>
            )}

            {memory && (
              <MemoryContent
                memory={memory}
                expandedGroups={expandedGroups}
                visibleCounts={visibleCounts}
                scrollRootRef={scrollRootRef}
                onToggleGroup={toggleGroup}
                onShowMore={showMore}
                sessionId={session.sessionId}
                onExport={handleExport}
                exportingMemoryId={exportingMemoryId}
              />
            )}

            {!loading && !memory && !formattedError && (
              <div className="memory-empty">{t('Open Records to load the complete long-term memory.')}</div>
            )}
          </div>

          {isImported && (
            <div
              id={`${titleId}-ask-panel`}
              role="tabpanel"
              aria-labelledby={`${titleId}-ask-tab`}
              hidden={activeTab !== 'ask'}
            >
              <ImportedMemoryAsk session={{ ...session, memoryVersion: memory?.memoryVersion || session.memoryVersion }}
                queryState={queryState} publication={authoritativeUpdate}
                active={activeTab === 'ask' && !editing.open} />
            </div>
          )}
        </div>
        </div>
      </section>
    </div>,
    document.body
  )
}

function MemoryContent({
  memory,
  expandedGroups,
  visibleCounts,
  scrollRootRef,
  onToggleGroup,
  onShowMore,
  sessionId,
  onExport,
  exportingMemoryId
}) {
  const { t } = useLanguage()
  const overviewItems = [
    ['Total', memory.counts.total],
    ['Episodic', memory.counts.episodic],
    ['Semantic', memory.counts.semantic],
    ['Visual', memory.counts.visual]
  ]

  return (
    <>
      <div className="memory-panel-heading">
        <div>
          <span>{t("Complete business-level memory")}</span>
          <strong>{t("M_lt · v")}{memory.memoryVersion}</strong>
          <small>{formatUpdatedAt(memory.updatedAt, t)}</small>
        </div>
      </div>

      <div className="memory-overview" aria-label={t("Long-term memory overview")}>
        {overviewItems.map(([label, value]) => (
          <div className="memory-overview-item" key={label}>
            <span>{t(label)}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>

      <div className="memory-status-row">
        <span className={`memory-status-badge ${memory.qaAligned ? 'is-ready' : 'is-warning'}`}>
          {memory.qaAligned ? t("QA aligned") : t("Legacy M_lt")}
        </span>
        {Object.entries(memory.availability || {}).map(([key, value]) => (
          <span className={`memory-status-badge ${value === 'ready' ? 'is-ready' : 'is-muted'}`} key={key}>
            {t(humanizeKey(key))}: {t(humanizeKey(value))}
          </span>
        ))}
      </div>

      {!memory.qaAligned && (
        <div className="memory-notice is-warning">{t("Legacy M_lt is displayed and may not match the memory currently used by QA.")}</div>
      )}
      {memory.fieldTruncation && (
        <div className="memory-notice">{t("Some repeated metadata arrays were truncated by the server. All Memory records are included.")}</div>
      )}

      <MemorySection
        groupId="episodic"
        title={t("Episodic Memory")}
        count={memory.counts.episodic}
        availability={memory.availability?.episodic}
        expanded={expandedGroups.has('episodic')}
        onToggle={onToggleGroup}
        sessionSummary={getVisibleSessionSummary(memory)}
      >
        {EPISODIC_GROUPS.map(([key, label]) => {
          const groupId = `episodic-${key}`
          const records = memory.episodic[key]
          return (
            <MemorySection
              compact
              groupId={groupId}
              title={t(label)}
              count={memory.counts.episodicByGranularity?.[key] ?? records.length}
              availability={memory.availability?.episodic}
              expanded={expandedGroups.has(groupId)}
              onToggle={onToggleGroup}
              key={key}
            >
              <MemoryList
                groupId={groupId}
                items={records}
                visibleCount={visibleCounts[groupId] || PAGE_SIZE}
                emptyLabel={t("No {value1} episodic Memory in this Session.", { value1: t(label) })}
                scrollRootRef={scrollRootRef}
                onShowMore={onShowMore}
                sessionId={sessionId}
                onExport={onExport}
                exportingMemoryId={exportingMemoryId}
              />
            </MemorySection>
          )
        })}
      </MemorySection>

      <MemorySection
        groupId="semantic"
        title={t("Semantic Memory")}
        count={memory.counts.semantic}
        availability={memory.availability?.semantic}
        expanded={expandedGroups.has('semantic')}
        onToggle={onToggleGroup}
      >
        <MemoryList
          groupId="semantic"
          items={memory.semantic}
          visibleCount={visibleCounts.semantic || PAGE_SIZE}
          emptyLabel={t("No Semantic Memory in this Session.")}
          scrollRootRef={scrollRootRef}
          onShowMore={onShowMore}
          sessionId={sessionId}
          onExport={onExport}
          exportingMemoryId={exportingMemoryId}
        />
      </MemorySection>

      <MemorySection
        groupId="visual"
        title={t("Visual Memory")}
        count={memory.counts.visual}
        availability={memory.availability?.visual}
        expanded={expandedGroups.has('visual')}
        onToggle={onToggleGroup}
      >
        <MemoryList
          groupId="visual"
          items={memory.visual}
          visibleCount={visibleCounts.visual || PAGE_SIZE}
          emptyLabel={t("No Visual Memory in this Session.")}
          scrollRootRef={scrollRootRef}
          onShowMore={onShowMore}
          sessionId={sessionId}
          onExport={onExport}
          exportingMemoryId={exportingMemoryId}
        />
      </MemorySection>
    </>
  )
}

function MemorySection({ groupId, title, count, availability, expanded, onToggle, compact = false, sessionSummary, children }) {
  const { t } = useLanguage()
  return (
    <section className={`memory-section ${compact ? 'is-compact' : ''}`}>
      <button
        className="memory-section-toggle"
        type="button"
        aria-expanded={expanded}
        onClick={() => onToggle(groupId)}
      >
        <span>
          <strong>{t(title)}</strong>
          <small>{availability === 'not_available' ? t("Not available") : t("{value1} records", { value1: count ?? 0 })}</small>
        </span>
        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>
      {sessionSummary && (
        <aside className="memory-session-summary" aria-label={t('Session summary')}>
          <div className="memory-session-summary-label">
            <Sparkles size={16} aria-hidden="true" />
            <strong>{t('Session summary')}</strong>
          </div>
          <p>{sessionSummary}</p>
        </aside>
      )}
      {expanded && (
        <div className="memory-section-content fade-in">
          {availability === 'not_available'
            ? <div className="memory-empty is-compact">{t("This component is not available for this Session.")}</div>
            : children}
        </div>
      )}
    </section>
  )
}

function getVisibleSessionSummary(memory) {
  const content = typeof memory?.sessionSummary === 'string' ? memory.sessionSummary.trim() : ''
  if (!content || memory?.sessionSummaryStatus !== 'ready') return ''
  if (memory.sessionSummaryVersion !== memory.memoryVersion) return ''
  return content
}

function MemoryList({
  groupId,
  items,
  visibleCount,
  emptyLabel,
  scrollRootRef,
  onShowMore,
  sessionId,
  onExport,
  exportingMemoryId
}) {
  const { t } = useLanguage()
  const sentinelRef = useRef(null)
  const shownItems = items.slice(0, visibleCount)
  const hasMore = shownItems.length < items.length

  useEffect(() => {
    if (!hasMore || !sentinelRef.current || typeof IntersectionObserver === 'undefined') return undefined
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) onShowMore(groupId, items.length)
    }, { root: scrollRootRef.current, rootMargin: '0px' })
    observer.observe(sentinelRef.current)
    return () => observer.disconnect()
  }, [groupId, hasMore, items.length, onShowMore, scrollRootRef])

  if (!items.length) return <div className="memory-empty is-compact">{t(emptyLabel)}</div>
  return (
    <>
      <div className="memory-list">
        {shownItems.map((item) => (
          <MemoryRecordCard
            memory={item}
            sessionId={sessionId}
            onExport={onExport}
            exporting={exportingMemoryId === (item.id || '__missing_memory_id__')}
            key={item.id || `${groupId}-${shownItems.indexOf(item)}`}
          />
        ))}
      </div>
      <div className="memory-local-count">{t("Showing")} {shownItems.length} {t("of")} {items.length}</div>
      {hasMore && (
        <div ref={sentinelRef} className="memory-local-sentinel">
          <button className="icon-button secondary" type="button" onClick={() => onShowMore(groupId, items.length)}>{t("Show next")} {Math.min(PAGE_SIZE, items.length - shownItems.length)}
          </button>
        </div>
      )}
    </>
  )
}

function formatUpdatedAt(value, t) {
  if (!value) return t('Updated time unavailable')
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? String(value) : t('Updated {date}', { date: parsed.toLocaleString() })
}

function formatVersion(value) {
  return value === null || value === undefined ? 'unknown' : `v${value}`
}

function normalizeDownloadFilename(filename, memory) {
  const candidate = String(filename || '')
    .normalize('NFKC')
    .replace(/[\u0000-\u001f\u007f]/g, '')
    .replace(/[\\/:*?"<>|]/g, '-')
    .replace(/[^\p{L}\p{N}._-]+/gu, '-')
    .replace(/-+/g, '-')
    .replace(/^[-.]+|[-.]+$/g, '')
    .slice(0, 120)
  if (/\.json$/i.test(candidate) && !candidate.includes('..')) {
    return candidate
  }
  return buildMemoryExportFilename(memory)
}

function comparePublicationUpdates(left, right) {
  const versionDifference = (left.memory?.memoryVersion || 0) - (right.memory?.memoryVersion || 0)
  if (versionDifference) return versionDifference
  return (left.revision || 0) - (right.revision || 0)
}

function formatExportError(error) {
  const code = error?.code || error?.raw?.error?.code
  if (code === 'invalid_memory_id' || error?.status === 400) return 'The memory ID or export format is invalid.'
  if (code === 'memory_not_found' || error?.status === 404) return 'This memory is no longer available.'
  if (code === 'memory_id_ambiguous') return 'This memory ID is ambiguous. Please open the memory from its Session and retry.'
  if (code === 'memory_not_exportable') return 'This memory is not ready for export.'
  if (code === 'memory_snapshot_changing') return 'This memory is being updated. Please try again.'
  if (code === 'memory_export_too_large' || error?.status === 413) return 'This memory is too large to export.'
  if (code === 'memory_export_rate_limited' || error?.status === 429) return 'Too many export requests. Please try again later.'
  if (code === 'memory_export_failed' || error?.status === 500) return 'JSON export failed. Please try again.'
  if (error?.status === 401) return 'Your session has expired. Please sign in again.'
  if (error?.status === 403) return 'You do not have permission to export this memory.'
  return 'Unable to download the JSON file. Check your connection and retry.'
}

function formatMemoryError(error) {
  const code = error?.code || error?.raw?.error?.code
  if (code === 'invalid_session_id' || error?.status === 400) return 'The Session ID is invalid.'
  if (code === 'session_not_found' || error?.status === 404) return 'The Session does not exist.'
  if (code === 'memory_response_too_large' || error?.status === 413) {
    return "This Session's long-term memory is too large to display safely."
  }
  if (code === 'memory_not_ready') return 'Long-term memory is no longer ready for this Session.'
  if (code === 'memory_snapshot_changing') return 'Long-term memory is being updated. Please retry shortly.'
  if (code === 'memory_component_lagging') return 'Long-term memory components are still being published.'
  if (error?.status === 503) return 'Long-term memory is updating or not ready. Please retry shortly.'
  if (code === 'invalid_memory_view_response') return 'The long-term memory response was incomplete or invalid.'
  return error?.message || 'Unable to load long-term memory.'
}
