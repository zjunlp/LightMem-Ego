import { ChevronDown, ChevronUp, Database, RefreshCw, Upload, LoaderCircle } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchLongTermMemorySessions } from '../api/lightmem_egoApi.js'
import MemoryDialog from './MemoryDialog.jsx'
import { useMemoryImport } from '../hooks/useMemoryImport.js'
import { useLanguage } from '../i18n.jsx'
import { useBackendFeatures } from './BackendFeatures.jsx'

const FIRST_PAGE_KEY = '__first_page__'
const CATALOG_PAGE_SIZE = 10
const COMPONENT_KEYS = ['episodic', 'semantic', 'visual']
const EMPTY_AUTHORITATIVE_UPDATES = new Map()

export default function MemoryPanel({
  authoritativeUpdate = null,
  authoritativeUpdates = EMPTY_AUTHORITATIVE_UPDATES,
  onPublication,
  standalone = false,
  active = true,
  sessionTitles = {},
  onSessionTitleUpdated
}) {
  const { t } = useLanguage()
  const features = useBackendFeatures()
  const canImport = features?.memoryImport ?? true
  const [open, setOpen] = useState(standalone)
  const [items, setItems] = useState([])
  const [totalReadySessions, setTotalReadySessions] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [initialLoading, setInitialLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [catalogError, setCatalogError] = useState('')
  const [catalogBlocked, setCatalogBlocked] = useState(false)
  const [catalogNotice, setCatalogNotice] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const [pageInput, setPageInput] = useState('1')
  const [pageError, setPageError] = useState('')
  const [pageLoading, setPageLoading] = useState(false)
  const [targetPage, setTargetPage] = useState(null)
  const [selectedSession, setSelectedSession] = useState(null)
  const [selectedOpener, setSelectedOpener] = useState(null)
  const [importOpen, setImportOpen] = useState(false)
  const [importFile, setImportFile] = useState(null)
  const [importNotice, setImportNotice] = useState('')
  const [importInitialTab, setImportInitialTab] = useState('graph')
  const importInputRef = useRef(null)
  const memoryImport = useMemoryImport({ enabled: canImport })

  const toggleButtonRef = useRef(null)
  const panelRef = useRef(null)
  const catalogControllerRef = useRef(null)
  const catalogGenerationRef = useRef(0)
  const navigationGenerationRef = useRef(0)
  const catalogRequestSequenceRef = useRef(0)
  const inFlightCursorRef = useRef(null)
  const seenCursorsRef = useRef(new Set())
  const cursorRecoveryUsedRef = useRef(false)
  const catalogLoadedRef = useRef(false)
  const pageOperationRef = useRef(false)
  const lastTargetPageRef = useRef(1)
  const currentPageRef = useRef(1)
  const itemsRef = useRef([])
  const nextCursorRef = useRef(null)
  const hasMoreRef = useRef(false)
  const detailCacheRef = useRef(new Map())
  const mountedRef = useRef(true)

  const replaceItems = useCallback((nextItems) => {
    itemsRef.current = nextItems
    setItems(nextItems)
  }, [])

  const mergeCatalogItems = useCallback((incomingItems, requestSequence) => {
    const merged = [...itemsRef.current]
    const positions = new Map(merged.map((item, index) => [item.sessionId, index]))

    for (const incoming of incomingItems) {
      const position = positions.get(incoming.sessionId)
      const existing = position === undefined ? null : merged[position]
      const cacheEntry = detailCacheRef.current.get(incoming.sessionId)
      const publication = authoritativeUpdates.get(incoming.sessionId)
      let nextItem = mergeSessionTitle(existing, incoming)
      nextItem = applyTitleSnapshot(nextItem, sessionTitles[incoming.sessionId])

      if (publication?.memory?.memoryVersion >= incoming.memoryVersion) {
        nextItem = applyAuthoritativeMemory(nextItem, publication.memory)
        detailCacheRef.current.set(incoming.sessionId, {
          versionKey: buildVersionKey(publication.memory),
          activeRootKind: publication.memory.activeRootKind,
          memory: publication.memory,
          confirmedRequestSequence: requestSequence
        })
      } else if (cacheEntry?.confirmedRequestSequence >= requestSequence) {
        nextItem = applyAuthoritativeMemory(nextItem, cacheEntry.memory)
      } else if (cacheEntry && cacheEntry.memory.memoryVersion > incoming.memoryVersion) {
        nextItem = applyAuthoritativeMemory(nextItem, cacheEntry.memory)
      } else if (cacheEntry) {
        if (isCacheCompatible(incoming, cacheEntry)) {
          nextItem = preserveUnknownComponentVersions(nextItem, cacheEntry.memory)
        } else {
          detailCacheRef.current.delete(incoming.sessionId)
        }
      }

      if (position === undefined) {
        positions.set(nextItem.sessionId, merged.length)
        merged.push(nextItem)
      } else {
        merged[position] = nextItem
      }
    }

    replaceItems(merged)
    setSelectedSession((current) => {
      if (!current) return current
      const replacement = merged.find((item) => item.sessionId === current.sessionId)
      return replacement ? { ...current, ...replacement } : current
    })
    return merged
  }, [authoritativeUpdates, replaceItems, sessionTitles])

  const fetchCatalogBatch = useCallback(async ({ cursor = null, mode = 'page' } = {}) => {
    const requestKey = cursor || FIRST_PAGE_KEY
    if (catalogControllerRef.current || inFlightCursorRef.current === requestKey) return { status: 'busy' }

    if (cursor && seenCursorsRef.current.has(cursor)) {
      nextCursorRef.current = null
      hasMoreRef.current = false
      setHasMore(false)
      setCatalogBlocked(true)
      setCatalogError('The server returned a repeated pagination cursor. Refresh the Catalog to continue.')
      return { status: 'loop' }
    }

    const controller = new AbortController()
    const generation = catalogGenerationRef.current
    const requestSequence = ++catalogRequestSequenceRef.current
    catalogControllerRef.current = controller
    inFlightCursorRef.current = requestKey
    setCatalogError('')
    setCatalogBlocked(false)
    if (mode === 'initial') setInitialLoading(true)
    if (mode === 'refresh') setRefreshing(true)

    try {
      const result = await fetchLongTermMemorySessions({ cursor, signal: controller.signal })
      if (!mountedRef.current || generation !== catalogGenerationRef.current) return { status: 'stale' }
      result.items.forEach((item) => onSessionTitleUpdated?.(toTitleSnapshot(item)))

      const cursorLoop = !!result.nextCursor && seenCursorsRef.current.has(result.nextCursor)
      if (cursor) seenCursorsRef.current.add(cursor)
      const merged = mergeCatalogItems(result.items, requestSequence)
      catalogLoadedRef.current = true
      setTotalReadySessions(result.totalReadySessions)

      if (cursorLoop) {
        nextCursorRef.current = null
        hasMoreRef.current = false
        setHasMore(false)
        setCatalogBlocked(true)
        setCatalogError('The server returned a cursor that was already loaded. Refresh the Catalog to resync.')
        return { status: 'loop', items: merged }
      }

      nextCursorRef.current = result.nextCursor
      hasMoreRef.current = result.hasMore
      setHasMore(result.hasMore)
      return { status: 'ok', items: merged, result }
    } catch (error) {
      if (error?.name === 'AbortError') return { status: 'aborted' }
      if (!mountedRef.current || generation !== catalogGenerationRef.current) return { status: 'stale' }

      if (error?.code === 'invalid_cursor' || error?.status === 400) {
        if (!cursorRecoveryUsedRef.current) return { status: 'invalid_cursor' }
      }

      if (error?.code === 'invalid_memory_catalog_response') {
        nextCursorRef.current = null
        hasMoreRef.current = false
        setHasMore(false)
        setCatalogBlocked(true)
      }
      setCatalogError(formatCatalogError(error))
      if (error?.code !== 'invalid_memory_catalog_response') setCatalogBlocked(false)
      return { status: 'error', error }
    } finally {
      if (catalogControllerRef.current === controller) {
        catalogControllerRef.current = null
        if (inFlightCursorRef.current === requestKey) inFlightCursorRef.current = null
      }
      if (mountedRef.current && generation === catalogGenerationRef.current) {
        setInitialLoading(false)
        setRefreshing(false)
      }
    }
  }, [mergeCatalogItems, onSessionTitleUpdated])

  useEffect(() => {
    if (!sessionTitles) return
    const currentItems = itemsRef.current
    const nextItems = currentItems.map((item) => applyTitleSnapshot(item, sessionTitles[item.sessionId]))
    if (nextItems.some((item, index) => item !== currentItems[index])) replaceItems(nextItems)
    setSelectedSession((current) => {
      if (!current) return current
      return applyTitleSnapshot(current, sessionTitles[current.sessionId])
    })
  }, [replaceItems, sessionTitles])

  const resetAfterInvalidCursor = useCallback(() => {
    cursorRecoveryUsedRef.current = true
    catalogGenerationRef.current += 1
    catalogControllerRef.current = null
    inFlightCursorRef.current = null
    seenCursorsRef.current = new Set()
    catalogLoadedRef.current = false
    nextCursorRef.current = null
    hasMoreRef.current = false
    detailCacheRef.current.clear()
    replaceItems([])
    setSelectedSession(null)
    setSelectedOpener(null)
    setTotalReadySessions(0)
    setHasMore(false)
    currentPageRef.current = 1
    setCurrentPage(1)
    setCatalogError('')
    setCatalogBlocked(false)
    setCatalogNotice('The pagination cursor expired. The Catalog was restarted from the first page.')
  }, [replaceItems])

  const goToPage = useCallback(async (requestedPage, { mode = 'page' } = {}) => {
    if (pageOperationRef.current) return false
    const page = Number(requestedPage)
    if (!Number.isInteger(page) || page < 1) return false

    const navigationGeneration = navigationGenerationRef.current
    pageOperationRef.current = true
    lastTargetPageRef.current = page
    setPageError('')
    setCatalogError('')
    if (mode === 'page') {
      setPageLoading(true)
      setTargetPage(page)
    }

    let shouldRestart = false
    try {
      do {
        shouldRestart = false
        if (!catalogLoadedRef.current) {
          const firstOutcome = await fetchCatalogBatch({ cursor: null, mode })
          if (firstOutcome.status === 'invalid_cursor' && !cursorRecoveryUsedRef.current) {
            resetAfterInvalidCursor()
            shouldRestart = true
            continue
          }
          if (firstOutcome.status !== 'ok') return false
        }

        while (itemsRef.current.length < page * CATALOG_PAGE_SIZE && hasMoreRef.current) {
          const cursor = nextCursorRef.current
          if (!cursor) break
          const outcome = await fetchCatalogBatch({ cursor, mode: 'page' })
          if (outcome.status === 'invalid_cursor' && !cursorRecoveryUsedRef.current) {
            resetAfterInvalidCursor()
            shouldRestart = true
            break
          }
          if (outcome.status !== 'ok') return false
        }
      } while (shouldRestart)

      if (!mountedRef.current || navigationGeneration !== navigationGenerationRef.current) return false
      const availablePages = Math.max(1, Math.ceil(itemsRef.current.length / CATALOG_PAGE_SIZE))
      const pageHasRecords = page === 1 || itemsRef.current.length > (page - 1) * CATALOG_PAGE_SIZE
      if (page <= availablePages && pageHasRecords) {
        currentPageRef.current = page
        setCurrentPage(page)
        setPageInput(String(page))
        setPageError('')
        requestAnimationFrame(() => panelRef.current?.scrollTo({ top: 0, behavior: 'smooth' }))
        return true
      }

      setPageInput(String(currentPageRef.current))
      setPageError(`Only ${availablePages} ${availablePages === 1 ? 'page is' : 'pages are'} currently available.`)
      return false
    } finally {
      if (navigationGeneration === navigationGenerationRef.current) {
        pageOperationRef.current = false
        setPageLoading(false)
        setTargetPage(null)
      }
    }
  }, [fetchCatalogBatch, resetAfterInvalidCursor])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      navigationGenerationRef.current += 1
      catalogGenerationRef.current += 1
      catalogControllerRef.current?.abort()
      catalogControllerRef.current = null
      inFlightCursorRef.current = null
      pageOperationRef.current = false
    }
  }, [])

  const toggleOpen = () => {
    const nextOpen = !open
    setOpen(nextOpen)
    if (nextOpen && !catalogLoadedRef.current && !pageOperationRef.current) {
      goToPage(1, { mode: 'initial' })
    }
  }

  useEffect(() => {
    if (!standalone) return
    setOpen(true)
    if (!catalogLoadedRef.current && !pageOperationRef.current) goToPage(1, { mode: 'initial' })
  }, [goToPage, standalone])

  const refreshCatalog = ({ preserveSelection = false } = {}) => {
    catalogControllerRef.current?.abort()
    catalogControllerRef.current = null
    inFlightCursorRef.current = null
    catalogGenerationRef.current += 1
    navigationGenerationRef.current += 1
    pageOperationRef.current = false
    seenCursorsRef.current = new Set()
    cursorRecoveryUsedRef.current = false
    catalogLoadedRef.current = false
    nextCursorRef.current = null
    hasMoreRef.current = false
    detailCacheRef.current.clear()
    replaceItems([])
    if (!preserveSelection) {
      setSelectedSession(null)
      setSelectedOpener(null)
    }
    setTotalReadySessions(0)
    setHasMore(false)
    currentPageRef.current = 1
    setCurrentPage(1)
    setPageInput('1')
    setPageError('')
    setPageLoading(false)
    setTargetPage(null)
    setCatalogError('')
    setCatalogBlocked(false)
    setCatalogNotice('')
    setInitialLoading(false)
    setRefreshing(true)
    queueMicrotask(() => goToPage(1, { mode: 'refresh' }))
  }

  const submitPage = (event) => {
    event.preventDefault()
    const normalizedInput = pageInput.trim()
    const page = Number(normalizedInput)
    const loadedPages = Math.max(1, Math.ceil(itemsRef.current.length / CATALOG_PAGE_SIZE))
    const estimatedPages = hasMoreRef.current
      ? Math.max(loadedPages, Math.ceil(totalReadySessions / CATALOG_PAGE_SIZE), 1)
      : loadedPages
    if (!/^[1-9]\d*$/.test(normalizedInput) || !Number.isSafeInteger(page) || page > estimatedPages) {
      setPageError(`Enter a page from 1 to ${estimatedPages}.`)
      return
    }
    goToPage(page)
  }

  const openSession = (session, opener) => {
    setCatalogNotice('')
    setSelectedSession(session)
    setSelectedOpener(opener)
  }

  const openImportedSession = useCallback((tab = 'records') => {
    const imported = memoryImport.task
    if (!imported?.sessionId || !memoryImport.ready) return
    const session = applyTitleSnapshot(itemsRef.current.find((item) => item.sessionId === imported.sessionId) || {
      sessionId: imported.sessionId,
      title: imported.sessionId,
      titleRevision: 0,
      sessionKind: 'imported_memory',
      interactionMode: 'long_term_only',
      queryAllowed: true,
      queryReady: true,
      memoryVersion: 1,
      componentVersions: { episodic: null, semantic: null, visual: null },
      activeRootKind: 'em2mem',
      qaAligned: true,
      updatedAt: imported.updatedAt || null
    }, sessionTitles[imported.sessionId])
    setImportInitialTab(tab)
    setSelectedSession(session)
    setSelectedOpener(null)
  }, [memoryImport.ready, memoryImport.task, sessionTitles])

  useEffect(() => {
    if (!memoryImport.ready || !memoryImport.task?.sessionId) return
    onSessionTitleUpdated?.(toTitleSnapshot(memoryImport.task))
    setImportNotice('Import complete. The new memory Session is ready to view and query.')
    const hadSelection = Boolean(selectedSession)
    refreshCatalog({ preserveSelection: hadSelection })
    if (!hadSelection) queueMicrotask(() => openImportedSession('records'))
  }, [memoryImport.ready, memoryImport.task?.sessionId, onSessionTitleUpdated])

  const handleMemoryLoaded = (memory) => {
    onSessionTitleUpdated?.(toTitleSnapshot(memory))
    if ((detailCacheRef.current.get(memory.sessionId)?.memory?.memoryVersion || 0) > memory.memoryVersion) return
    const cacheEntry = {
      versionKey: buildVersionKey(memory),
      activeRootKind: memory.activeRootKind,
      memory,
      confirmedRequestSequence: catalogRequestSequenceRef.current
    }
    detailCacheRef.current.set(memory.sessionId, cacheEntry)

    const nextItems = itemsRef.current.map((item) => item.sessionId === memory.sessionId
      ? applyAuthoritativeMemory(mergeSessionTitle(item, memory), memory)
      : item)
    replaceItems(nextItems)
    setSelectedSession((current) => (
      current?.sessionId === memory.sessionId
        ? applyAuthoritativeMemory(mergeSessionTitle(current, memory), memory)
        : current
    ))
  }

  const handleSessionTitleUpdated = (snapshot) => {
    onSessionTitleUpdated?.(snapshot)
    const nextItems = itemsRef.current.map((item) => item.sessionId === snapshot.sessionId
      ? applyTitleSnapshot(item, snapshot)
      : item)
    replaceItems(nextItems)
    setSelectedSession((current) => current?.sessionId === snapshot.sessionId
      ? applyTitleSnapshot(current, snapshot)
      : current)
  }

  useEffect(() => {
    const updates = new Map(authoritativeUpdates)
    if (authoritativeUpdate?.sessionId) {
      const current = updates.get(authoritativeUpdate.sessionId)
      if (!current || (authoritativeUpdate.memory?.memoryVersion || 0) >= (current.memory?.memoryVersion || 0)) {
        updates.set(authoritativeUpdate.sessionId, authoritativeUpdate)
      }
    }
    if (!updates.size) return

    let nextItems = itemsRef.current
    for (const update of updates.values()) {
      const memory = update?.memory
      if (!memory?.sessionId) continue
      if ((detailCacheRef.current.get(memory.sessionId)?.memory?.memoryVersion || 0) > memory.memoryVersion) continue

      detailCacheRef.current.set(memory.sessionId, {
        versionKey: buildVersionKey(memory),
        activeRootKind: memory.activeRootKind,
        memory,
        confirmedRequestSequence: catalogRequestSequenceRef.current
      })
      nextItems = nextItems.map((item) => (
        item.sessionId === memory.sessionId ? applyAuthoritativeMemory(item, memory) : item
      ))
    }
    replaceItems(nextItems)
    setSelectedSession((current) => {
      let next = current
      for (const update of updates.values()) {
        const memory = update?.memory
        if (next?.sessionId === memory?.sessionId
          && memory.memoryVersion >= (next.memoryVersion || 0)) next = applyAuthoritativeMemory(next, memory)
      }
      return next
    })
  }, [authoritativeUpdate?.revision, authoritativeUpdates, replaceItems])

  const handleSessionUnavailable = (sessionId) => {
    detailCacheRef.current.delete(sessionId)
    const nextItems = itemsRef.current.filter((item) => item.sessionId !== sessionId)
    replaceItems(nextItems)
    setTotalReadySessions((current) => Math.max(0, current - 1))
    setSelectedSession(null)
    setCatalogNotice('The selected Session is no longer available and was removed from the Catalog.')

    const availablePages = Math.max(1, Math.ceil(nextItems.length / CATALOG_PAGE_SIZE))
    if (currentPage > availablePages) {
      currentPageRef.current = availablePages
      setCurrentPage(availablePages)
      setPageInput(String(availablePages))
    } else if (nextItems.length < currentPage * CATALOG_PAGE_SIZE && hasMoreRef.current) {
      queueMicrotask(() => goToPage(currentPage))
    }
  }

  const loadedPages = Math.max(1, Math.ceil(items.length / CATALOG_PAGE_SIZE))
  const catalogComplete = catalogLoadedRef.current && !hasMore && !catalogBlocked
  const estimatedPages = catalogComplete
    ? loadedPages
    : Math.max(1, loadedPages, Math.ceil(totalReadySessions / CATALOG_PAGE_SIZE))
  const pageStartIndex = (currentPage - 1) * CATALOG_PAGE_SIZE
  const visibleSessions = items.slice(pageStartIndex, pageStartIndex + CATALOG_PAGE_SIZE)
  const visibleStart = visibleSessions.length ? pageStartIndex + 1 : 0
  const visibleEnd = pageStartIndex + visibleSessions.length
  const summaryTotal = catalogComplete ? items.length : Math.max(items.length, totalReadySessions)
  const canGoNext = hasMore || currentPage < loadedPages
  const cachedMemory = selectedSession
    ? getCompatibleCache(selectedSession, detailCacheRef.current.get(selectedSession.sessionId))?.memory || null
    : null

  return (
    <section className={`advanced-tools memory-tools ${standalone ? 'is-standalone' : ''}`}>
      {!standalone && <button
        ref={toggleButtonRef}
        className="advanced-toggle"
        type="button"
        aria-expanded={open}
        aria-controls="long-term-memory-panel"
        onClick={toggleOpen}
      >
        <span>
          <Database size={16} />
          {t('Memories')}
        </span>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>}

      {canImport && <button
        className="icon-button secondary memory-import-toggle"
        type="button"
        aria-expanded={importOpen}
        aria-controls="memory-import-panel"
        onClick={() => setImportOpen((current) => !current)}
      >
        <Upload size={15} />
        <span>{t('Import Memory')}</span>
        {importOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>}

      {importNotice && selectedSession && selectedSession.sessionId !== memoryImport.task?.sessionId && (
        <div className="memory-notice memory-import-complete-notice" role="status">
          <span>{t('Import complete')}</span>
          <strong>{sessionTitles[memoryImport.task?.sessionId]?.title || memoryImport.task?.sessionId}</strong>
          <button className="icon-button secondary" type="button" onClick={() => openImportedSession('records')}>{t('View memory')}</button>
        </div>
      )}

      {canImport && importOpen && (
        <div id="memory-import-panel" className="surface advanced-panel memory-import-panel fade-in">
          <div className="memory-catalog-heading">
            <div>
              <span>{t('Import a long-term memory file')}</span>
              <strong>{t('Create a new memory Session')}</strong>
            </div>
            {memoryImport.checking && <LoaderCircle size={16} className="memory-spin" aria-label={t("Checking import status")} />}
          </div>

          <p className="memory-import-help">{t('Upload one JSON or TXT file. JSON accepts the unified import format, legacy exports, and compatible backend structures. TXT is split into paragraphs by blank lines; no event times are invented.')}</p>
          <p className="memory-import-help">{t('The new Session uses its ID as the default name. Rename it later from memory management.')}</p>

          <div className="memory-import-file-row">
            <input
              ref={importInputRef}
              className="memory-import-file-input"
              type="file"
              accept=".json,.txt,application/json,text/plain"
              onChange={(event) => {
                setImportFile(event.target.files?.[0] || null)
                setImportNotice('')
              }}
              disabled={memoryImport.locked || memoryImport.uploading}
            />
            <button
              className="icon-button secondary"
              type="button"
              disabled={memoryImport.locked || memoryImport.uploading}
              onClick={() => importInputRef.current?.click()}
            >
              <Upload size={15} />
              <span>{t('Choose file')}</span>
            </button>
            {importFile && (
              <span className="memory-import-file-meta">
                {importFile.name} · {formatBytes(importFile.size)}
              </span>
            )}
          </div>

          {memoryImport.task && (
            <div className="memory-import-progress" aria-live="polite">
              <div className="memory-import-progress-heading">
                <strong>{t(formatImportStage(memoryImport.task))}</strong>
                <span>{memoryImport.task.fileName || t("Imported file")}</span>
              </div>
              <div className={`memory-import-progress-track ${isValidProgress(memoryImport.task.progress) ? '' : 'is-indeterminate'}`}>
                <span style={isValidProgress(memoryImport.task.progress) ? { width: `${memoryImport.task.progress}%` } : undefined} />
              </div>
              <div className="memory-import-meta">
                <span>{isValidProgress(memoryImport.task.progress) ? `${memoryImport.task.progress}%` : t('Progress unavailable')}</span>
                <span>{memoryImport.task.recordCount ?? t('Record count unavailable')} {t('records parsed')}</span>
                <span>{t("Session")} {memoryImport.task.sessionId}</span>
              </div>
            </div>
          )}

          {memoryImport.error && <div className="inline-error memory-error" role="alert">{t(memoryImport.error)}</div>}
          {importNotice && <div className="memory-notice" role="status">
            <span>{t(importNotice)}</span>
            {memoryImport.task?.sessionId && <strong>{sessionTitles[memoryImport.task.sessionId]?.title || memoryImport.task.sessionId}</strong>}
          </div>}
          {memoryImport.paused && !memoryImport.unavailable && (
            <div className="memory-import-actions">
              <button className="icon-button secondary" type="button" onClick={memoryImport.resume}>{t('Continue checking')}</button>
              <button className="icon-button secondary" type="button" onClick={memoryImport.refresh} disabled={memoryImport.checking}>{t("Refresh")}</button>
            </div>
          )}
          {memoryImport.unavailable && (
            <div className="memory-import-actions">
              <button className="icon-button secondary" type="button" onClick={() => memoryImport.refresh()} disabled={memoryImport.checking}>{t('Refresh task')}</button>
            </div>
          )}

          <div className="memory-import-actions">
            <button
              className="icon-button primary"
              type="button"
              disabled={!importFile || memoryImport.locked || memoryImport.uploading || Boolean(memoryImport.cooldownUntil && memoryImport.cooldownUntil > Date.now())}
              onClick={() => memoryImport.start(importFile)}
            >
              {memoryImport.uploading ? <LoaderCircle size={15} className="memory-spin" /> : <Upload size={15} />}
              <span>{memoryImport.uploading ? t('Uploading…') : t('Import Memory')}</span>
            </button>
            {memoryImport.ready && (
              <>
                <button className="icon-button secondary" type="button" onClick={() => openImportedSession('records')}>{t('Open Records')}</button>
                <button className="icon-button secondary" type="button" onClick={() => openImportedSession('ask')}>{t('Ask Memory')}</button>
              </>
            )}
          </div>
        </div>
      )}

      {open && (
        <div ref={panelRef} id="long-term-memory-panel" className={`surface advanced-panel memory-panel memory-catalog fade-in ${standalone ? 'standalone-catalog' : ''}`}>
          <div className="memory-catalog-heading">
            <div>
              <span>{t('Ready long-term memory Sessions')}</span>
              <strong>{t('Global M_lt Catalog')}</strong>
              <small>{t('Loaded')} {items.length} / {catalogComplete ? '' : `${t('about')} `}{summaryTotal}</small>
            </div>
            <button
              className="icon-button secondary memory-refresh"
              type="button"
              disabled={initialLoading || pageLoading || refreshing}
              onClick={refreshCatalog}
            >
              <RefreshCw size={15} className={refreshing ? 'memory-spin' : ''} />
              <span>{refreshing ? t('Refreshing') : t('Refresh')}</span>
            </button>
          </div>

          {catalogNotice && <div className="memory-notice" role="status">{t(catalogNotice)}</div>}
          {importNotice && selectedSession?.sessionId !== memoryImport.task?.sessionId && (
            <div className="memory-notice" role="status">
              <span>{t(importNotice)}</span>
              {memoryImport.task?.sessionId && <strong>{sessionTitles[memoryImport.task.sessionId]?.title || memoryImport.task.sessionId}</strong>}
              <button className="icon-button secondary" type="button" onClick={() => openImportedSession('records')}>{t("View imported memory")}</button>
            </div>
          )}

          {initialLoading && items.length === 0 && (
            <div className="memory-loading" aria-live="polite">
              <span className="memory-spinner" />
              {t('Loading ready long-term memory Sessions…')}
            </div>
          )}

          {!initialLoading && catalogError && items.length === 0 && (
            <div className="inline-error memory-error" role="alert">
              <span>{t(catalogError)}</span>
              <button className="icon-button secondary" type="button" onClick={refreshCatalog}>{t('Retry')}</button>
            </div>
          )}

          {!initialLoading && !catalogError && catalogLoadedRef.current && items.length === 0 && (
            <div className="memory-empty">{t('No stable long-term memory Sessions are currently available.')}</div>
          )}

          {visibleSessions.length > 0 && (
            <div className="memory-session-list">
              {visibleSessions.map((session) => (
                <SessionCard session={session} onOpen={openSession} key={session.sessionId} />
              ))}
            </div>
          )}

          {items.length > 0 && (
            <nav className="memory-catalog-pagination" aria-label={t("Long-term memory Session pages")}>
              <div className="memory-catalog-page-summary">
                {t('Showing')} {visibleStart}–{visibleEnd} {t('of')} {catalogComplete ? '' : `${t('about')} `}{summaryTotal} {t('Sessions')}
              </div>

              {catalogError && <div className="memory-catalog-page-error" role="alert">{t(catalogError)}</div>}
              {pageError && <div className="memory-catalog-page-error" role="alert">{t(pageError)}</div>}
              {pageLoading && (
                <div className="memory-catalog-page-loading" aria-live="polite">
                  <span className="memory-spinner is-small" />
                  {t('Loading page')} {targetPage}…
                </div>
              )}

              <div className="memory-catalog-page-actions">
                <button
                  className="icon-button secondary"
                  type="button"
                  disabled={currentPage === 1 || pageLoading || refreshing}
                  onClick={() => goToPage(currentPage - 1)}
                >
                  {t('Previous')}
                </button>
                <span className="memory-catalog-page-status" aria-current="page">
                  {t('Page')} {currentPage} {t('of')} {catalogComplete ? '' : `${t('about')} `}{estimatedPages}
                </span>
                <button
                  className="icon-button secondary"
                  type="button"
                  disabled={!canGoNext || pageLoading || refreshing}
                  onClick={() => goToPage(currentPage + 1)}
                >
                  {t('Next')}
                </button>
              </div>

              <form className="memory-catalog-page-jump" onSubmit={submitPage}>
                <input
                  className="memory-catalog-page-input"
                  type="number"
                  min="1"
                  max={estimatedPages}
                  step="1"
                  inputMode="numeric"
                  aria-label={t("Catalog page number")}
                  value={pageInput}
                  disabled={pageLoading || refreshing}
                  onChange={(event) => {
                    setPageInput(event.target.value)
                    setPageError('')
                  }}
                />
                <button
                  className="icon-button secondary"
                  type="submit"
                  disabled={estimatedPages === 1 || pageLoading || refreshing}
                >
                  {t('Jump')}
                </button>
              </form>

              {catalogBlocked ? (
                <button className="icon-button secondary" type="button" onClick={refreshCatalog}>{t('Refresh Catalog')}</button>
              ) : catalogError ? (
                <button
                  className="icon-button secondary"
                  type="button"
                  disabled={pageLoading}
                  onClick={() => goToPage(lastTargetPageRef.current)}
                >{t("Retry")}</button>
              ) : null}
            </nav>
          )}
        </div>
      )}

      {selectedSession && (
        <MemoryDialog
          key={selectedSession.sessionId}
          session={selectedSession}
          active={active}
          cachedMemory={cachedMemory}
          authoritativeUpdate={getSessionPublication(authoritativeUpdates, selectedSession.sessionId, authoritativeUpdate)}
          openerElement={selectedOpener}
          fallbackFocusElement={toggleButtonRef.current}
          onMemoryLoaded={handleMemoryLoaded}
          onPublication={onPublication}
          onSessionTitleUpdated={handleSessionTitleUpdated}
          onSessionUnavailable={handleSessionUnavailable}
          initialTab={selectedSession.sessionId === memoryImport.task?.sessionId ? importInitialTab : 'graph'}
          onClose={() => setSelectedSession(null)}
        />
      )}
    </section>
  )
}

function SessionCard({ session, onOpen }) {
  const { t } = useLanguage()
  const imported = session.sessionKind === 'imported_memory'
  return (
    <button
      className="memory-session-card"
      type="button"
      onClick={(event) => onOpen(session, event.currentTarget)}
    >
      <span className="memory-session-card-heading">
        <strong>{session.title || session.sessionId}</strong>
        <small>{formatUpdatedAt(session.updatedAt, t)}</small>
      </span>
      <span className="memory-session-id">{session.sessionId}</span>
      <span className="memory-session-versions">
        <VersionItem label={t("M_lt")} value={session.memoryVersion} />
        <VersionItem label={t("Episodic")} value={session.componentVersions?.episodic} />
        <VersionItem label={t("Semantic")} value={session.componentVersions?.semantic} />
        <VersionItem label={t("Visual")} value={session.componentVersions?.visual} />
      </span>
      <span className="memory-status-row">
        {imported && <span className="memory-status-badge is-imported">{t("Imported memory")}</span>}
        <span className={`memory-status-badge ${session.qaAligned ? 'is-ready' : 'is-warning'}`}>
          {session.qaAligned ? t("QA aligned") : t("Legacy M_lt")}
        </span>
        <span className="memory-status-badge is-muted">{session.activeRootKind}</span>
      </span>
    </button>
  )
}

function isValidProgress(value) {
  return Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= 100
}

function formatImportStage(task) {
  const stages = {
    queued: 'Waiting for processing',
    processing: 'Processing import',
    parsing: 'Parsing file',
    normalizing: 'Normalizing memory records',
    building_memory: 'Building long-term memory',
    building_retrieval: 'Building retrieval index',
    warming_query: 'Preparing query service',
    ready: 'Ready to ask',
    failed: 'Import failed'
  }
  return stages[task?.state === 'processing' ? task.stage : task?.state] || 'Checking import status'
}

function formatBytes(value) {
  const bytes = Number(value) || 0
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function VersionItem({ label, value }) {
  const { t } = useLanguage()
  return (
    <span>
      <small>{t(label)}</small>
      <strong>{value === null || value === undefined ? t("Unknown") : t("v{value1}", { value1: value })}</strong>
    </span>
  )
}

function buildVersionKey(memory) {
  const versions = memory.componentVersions || {}
  return [
    memory.sessionId,
    memory.memoryVersion,
    versions.episodic ?? 'na',
    versions.semantic ?? 'na',
    versions.visual ?? 'na'
  ].join('|')
}

function isCacheCompatible(session, cacheEntry) {
  if (!cacheEntry?.memory) return false
  const memory = cacheEntry.memory
  if (memory.sessionId !== session.sessionId) return false
  if (memory.memoryVersion !== session.memoryVersion) return false
  if (memory.activeRootKind !== session.activeRootKind) return false
  return COMPONENT_KEYS.every((key) => (
    session.componentVersions?.[key] === null
    || session.componentVersions?.[key] === undefined
    || session.componentVersions[key] === memory.componentVersions?.[key]
  ))
}

function getCompatibleCache(session, cacheEntry) {
  return isCacheCompatible(session, cacheEntry) ? cacheEntry : null
}

function getSessionPublication(updates, sessionId, fallback) {
  const fromSessionMap = updates?.get(sessionId) || null
  const matchingFallback = fallback?.sessionId === sessionId ? fallback : null
  if (!fromSessionMap) return matchingFallback
  if (!matchingFallback) return fromSessionMap
  if ((matchingFallback.memory?.memoryVersion || 0) !== (fromSessionMap.memory?.memoryVersion || 0)) {
    return (matchingFallback.memory?.memoryVersion || 0) > (fromSessionMap.memory?.memoryVersion || 0)
      ? matchingFallback : fromSessionMap
  }
  return (matchingFallback.revision || 0) > (fromSessionMap.revision || 0)
    ? matchingFallback : fromSessionMap
}

function preserveUnknownComponentVersions(session, memory) {
  return {
    ...session,
    componentVersions: Object.fromEntries(COMPONENT_KEYS.map((key) => [
      key,
      session.componentVersions?.[key] ?? memory.componentVersions?.[key] ?? null
    ]))
  }
}

function applyAuthoritativeMemory(session, memory) {
  return {
    ...session,
    memoryVersion: memory.memoryVersion,
    componentVersions: memory.componentVersions,
    updatedAt: memory.updatedAt,
    activeRootKind: memory.activeRootKind,
    qaAligned: memory.qaAligned,
    counts: memory.counts,
    canEdit: memory.canEdit,
    editDisabledReason: memory.editDisabledReason,
    canRollback: memory.canRollback,
    ...(memory.queryReady !== undefined ? { queryReady: memory.queryReady } : {})
  }
}

function toTitleSnapshot(value) {
  const sessionId = value?.sessionId || ''
  if (!sessionId) return null
  return {
    sessionId,
    title: typeof value.title === 'string' && value.title.trim() ? value.title : sessionId,
    titleRevision: Number.isInteger(value.titleRevision) && value.titleRevision >= 0 ? value.titleRevision : 0,
    titleUpdatedAt: typeof value.titleUpdatedAt === 'string' ? value.titleUpdatedAt : null
  }
}

function mergeSessionTitle(existing, incoming) {
  const merged = { ...(existing || {}), ...(incoming || {}) }
  const incomingTitle = toTitleSnapshot(incoming)
  if (!incomingTitle) return merged
  const existingTitle = toTitleSnapshot(existing)
  const preferredTitle = existingTitle && existingTitle.titleRevision >= incomingTitle.titleRevision
    ? existingTitle
    : incomingTitle
  return { ...merged, ...preferredTitle }
}

function applyTitleSnapshot(session, snapshot) {
  if (!session || !snapshot) return session
  const current = toTitleSnapshot(session)
  const next = toTitleSnapshot(snapshot)
  if (!current || !next || current.titleRevision >= next.titleRevision) return session
  return { ...session, ...next }
}

function formatUpdatedAt(value, t) {
  if (!value) return t('Updated time unavailable')
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? t('Updated time unavailable') : t('Updated {date}', { date: parsed.toLocaleString() })
}

function formatCatalogError(error) {
  if (error?.code === 'invalid_cursor' || error?.status === 400) {
    return 'The Catalog cursor is no longer valid. Refresh the Catalog to restart.'
  }
  if (error?.code === 'memory_catalog_failed' || error?.status === 500) {
    return 'The long-term memory Catalog is temporarily unavailable.'
  }
  if (error?.code === 'invalid_memory_catalog_response') {
    return 'The long-term memory Catalog response was invalid.'
  }
  return error?.message || 'Unable to load the long-term memory Catalog.'
}
