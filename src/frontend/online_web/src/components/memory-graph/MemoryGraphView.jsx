import { useLanguage } from '../../i18n.jsx'
import { AlertTriangle, Network, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMemoryGraph } from '../../hooks/useMemoryGraph.js'
import {
  buildGraphModel,
  DEFAULT_GRAPH_FILTERS,
  deriveGraphView
} from '../../utils/memoryGraphUtils.js'
import EpisodicGraph from './EpisodicGraph.jsx'
import GraphDetailsPanel from './GraphDetailsPanel.jsx'
import GraphToolbar from './GraphToolbar.jsx'
import SemanticGraph from './SemanticGraph.jsx'

export default function MemoryGraphView({ session, authoritativeUpdate, onSessionUnavailable }) {
  const { t } = useLanguage()
  const graphRequest = useMemoryGraph(session.sessionId, {
    authoritativeUpdate,
    onSessionUnavailable
  })
  const [activeMobileGraph, setActiveMobileGraph] = useState('episodic')
  const [filters, setFilters] = useState(() => ({ ...DEFAULT_GRAPH_FILTERS }))
  const [selection, setSelection] = useState(null)
  const [expandedEventId, setExpandedEventId] = useState(null)
  const debouncedQuery = useDebouncedValue(filters.query, 200)
  const model = useMemo(() => graphRequest.payload ? buildGraphModel(graphRequest.payload) : null, [graphRequest.payload])
  const effectiveFilters = useMemo(() => ({ ...filters, query: debouncedQuery }), [debouncedQuery, filters])
  const view = useMemo(
    () => model ? deriveGraphView(model, effectiveFilters, expandedEventId) : null,
    [effectiveFilters, expandedEventId, model]
  )

  useEffect(() => {
    if (!model || !selection) return
    const stillExists = selection.kind === 'event'
      ? model.eventById.has(selection.id)
      : selection.kind === 'fact'
        ? model.factById.has(selection.id)
        : model.episodicEntityById.has(selection.id) || model.semanticEntityById.has(selection.id)
    if (!stillExists) {
      setSelection(null)
      setExpandedEventId(null)
    }
  }, [model, selection])

  useEffect(() => {
    if (expandedEventId && model && !model.eventById.has(expandedEventId)) setExpandedEventId(null)
  }, [expandedEventId, model])

  const updateFilters = useCallback((patch) => setFilters((current) => ({ ...current, ...patch })), [])
  const resetFilters = useCallback(() => setFilters({ ...DEFAULT_GRAPH_FILTERS }), [])
  const clearSelection = useCallback(() => {
    setSelection(null)
    setExpandedEventId(null)
  }, [])

  const selectEpisodic = useCallback((kind, id) => {
    setSelection({ kind, id, origin: 'episodic' })
    if (kind === 'event') setExpandedEventId((current) => current === id ? null : id)
  }, [])

  const selectSemantic = useCallback((kind, id) => {
    setSelection({ kind, id, origin: 'semantic' })
    if (kind === 'entity') {
      const mentionedBy = model?.graph.episodic.edges.find((edge) => (
        edge.kind === 'mentions' && edge.target === id && view?.visibleEventIds.has(edge.source)
      ))
      if (mentionedBy) setExpandedEventId(mentionedBy.source)
    }
  }, [model, view])

  const selectSupportingEvent = useCallback((eventId, outsideFilters) => {
    const event = model?.eventById.get(eventId)
    if (!event) return
    if (outsideFilters) {
      const eventStart = event.eventTime.timelineStartSeconds
      const eventEnd = event.eventTime.timelineEndSeconds
      setFilters((current) => ({
        ...current,
        query: '',
        relations: [],
        episodicMinConfidence: 0,
        startSeconds: Number.isFinite(eventStart)
          ? Math.min(Number.isFinite(current.startSeconds) ? current.startSeconds : model.timeDomain.start, eventStart)
          : current.startSeconds,
        endSeconds: Number.isFinite(eventEnd)
          ? Math.max(Number.isFinite(current.endSeconds) ? current.endSeconds : model.timeDomain.end, eventEnd)
          : current.endSeconds
      }))
    }
    setSelection({ kind: 'event', id: eventId, origin: 'details' })
    setExpandedEventId(eventId)
    setActiveMobileGraph('episodic')
  }, [model])

  if (!graphRequest.payload) {
    return (
      <GraphRequestState
        status={graphRequest.status}
        error={graphRequest.error}
        requestId={graphRequest.requestId}
        onRetry={graphRequest.retry}
      />
    )
  }

  const linkedEventIds = selection?.kind === 'fact' ? model.factIdToEventIds.get(selection.id) || [] : []
  const linkedFactIds = selection?.kind === 'event' ? model.eventIdToFactIds.get(selection.id) || [] : []
  const linkedEntityId = selection?.kind === 'entity' ? selection.id : null
  const visibleLinkedFactIds = linkedFactIds.filter((id) => view.visibleFactIds.has(id))
  const warnings = collectWarnings(graphRequest.payload, t)
  const versions = graphRequest.payload.componentVersions

  return (
    <div className="memory-graph-view">
      <div className="memory-graph-meta">
        <div>
          <span>{t("30-second linked memory graph")}</span>
          <strong>{t("M_lt v")}{graphRequest.payload.memoryVersion ?? session.memoryVersion} {t("· Graph v")}{versions.graph} {t("· Semantic v")}{versions.semantic}</strong>
          <small>{formatUpdatedAt(graphRequest.payload.updatedAt || session.updatedAt, t)}</small>
        </div>
        <GraphStatus status={graphRequest.status} lastCheckedAt={graphRequest.lastCheckedAt} />
      </div>

      {graphRequest.error && (
        <div className="memory-graph-banner is-warning" role="status">
          <AlertTriangle size={15} />
          <span>{t(formatGraphError(graphRequest.error))}</span>
          {graphRequest.status === 'error' && (
            <button className="icon-button secondary" type="button" onClick={graphRequest.retry}>
              <RefreshCw size={14} />{t("Retry")}</button>
          )}
        </div>
      )}

      {warnings.length > 0 && (
        <div className="memory-graph-banner" role="status">
          <AlertTriangle size={15} />
          <span>{warnings.join(' · ')}</span>
        </div>
      )}

      <GraphToolbar
        filters={filters}
        relationOptions={view.relationOptions}
        timeDomain={view.timeDomain}
        onChange={updateFilters}
        onReset={resetFilters}
      />

      <div className="memory-graph-summary">
        <span>{view.visibleEvents.length} {t("events")}</span>
        <span>{t("Showing")} {view.visibleFacts.length} {t("of")} {view.semanticCandidateCount} {t("matching facts")}</span>
        {selection?.kind === 'event' && <span>{linkedFactIds.length} {t("related facts ·")} {visibleLinkedFactIds.length} {t("currently visible")}</span>}
      </div>

      <div className="memory-graph-mobile-tabs" role="tablist" aria-label={t("Memory graph type")}>
        <button type="button" role="tab" aria-selected={activeMobileGraph === 'episodic'} className={activeMobileGraph === 'episodic' ? 'active' : ''} onClick={() => setActiveMobileGraph('episodic')}>{t("Episodic")}</button>
        <button type="button" role="tab" aria-selected={activeMobileGraph === 'semantic'} className={activeMobileGraph === 'semantic' ? 'active' : ''} onClick={() => setActiveMobileGraph('semantic')}>{t("Semantic")}</button>
      </div>

      <div className="memory-graph-board">
        <section className={`memory-graph-pane episodic ${activeMobileGraph === 'episodic' ? 'is-mobile-active' : ''}`}>
          <div className="memory-graph-pane-heading">
            <div>
              <strong>{t("Episodic Graph")}</strong>
              <span>{t("Event timeline · select an Event to expand entities")}</span>
            </div>
            <small>{view.visibleEvents.length} {t("events")}</small>
          </div>
          <EpisodicGraph
            elements={localizeGraphLabels(view.episodic.elements, t)}
            selection={selection}
            linkedEventIds={linkedEventIds}
            linkedEntityId={linkedEntityId}
            focusEventIds={selection?.kind === 'fact' ? linkedEventIds : selection?.id ? [selection.id] : []}
            onSelect={selectEpisodic}
            onClear={clearSelection}
          />
        </section>

        <section className={`memory-graph-pane semantic ${activeMobileGraph === 'semantic' ? 'is-mobile-active' : ''}`}>
          <div className="memory-graph-pane-heading">
            <div>
              <strong>{t("Semantic Graph")}</strong>
              <span>{t("Stable knowledge · edge width represents support")}</span>
            </div>
            <small>{view.visibleFacts.length} / {view.semanticCandidateCount} {t("facts")}</small>
          </div>
          <SemanticGraph
            elements={view.semantic.elements}
            selection={selection}
            linkedFactIds={visibleLinkedFactIds}
            linkedEntityId={linkedEntityId}
            focusFactIds={selection?.kind === 'event' ? visibleLinkedFactIds : selection?.id ? [selection.id] : []}
            onSelect={selectSemantic}
            onClear={clearSelection}
          />
        </section>
      </div>

      <GraphDetailsPanel
        selection={selection}
        model={model}
        view={view}
        onSelectEvent={selectSupportingEvent}
        onClear={clearSelection}
      />
    </div>
  )
}

function GraphRequestState({ status, error, requestId, onRetry }) {
  const { t } = useLanguage()
  const waiting = status === 'waiting'
  const loading = status === 'idle' || status === 'loading'
  if (loading || waiting) {
    return (
      <div className="memory-graph-request-state" aria-live="polite">
        <span className="memory-spinner" />
        <strong>{waiting ? t("Memory graph is being built") : t("Loading linked memory graph…")}</strong>
        <span>{waiting ? t("The page will retry automatically.") : t("Loading Episodic and Semantic data.")}</span>
        {requestId && <small>{t("Request")} {requestId}</small>}
      </div>
    )
  }

  return (
    <div className="memory-graph-request-state is-error" role="alert">
      <Network size={28} />
      <strong>{t(formatGraphError(error))}</strong>
      <span>{error?.code === 'graph_not_supported' ? t("The Records tab remains available for this legacy Session.") : t("The graph could not be displayed.")}</span>
      {requestId && <small>{t("Request")} {requestId}</small>}
      {error?.status !== 409 && error?.status !== 413 && (
        <button className="icon-button secondary" type="button" onClick={onRetry}>
          <RefreshCw size={14} />{t("Retry")}</button>
      )}
    </div>
  )
}

function GraphStatus({ status, lastCheckedAt }) {
  const { t } = useLanguage()
  const label = status === 'waiting' ? 'Building' : status === 'error' ? 'Refresh failed' : 'Synced'
  return (
    <div className={`memory-graph-sync-status is-${status}`} aria-live={status === 'waiting' ? 'polite' : 'off'}>
      <span />
      <strong>{t(label)}</strong>
      {lastCheckedAt && <small>{new Date(lastCheckedAt).toLocaleTimeString()}</small>}
    </div>
  )
}

function collectWarnings(payload, t) {
  const warnings = []
  const serverWarningTotal = Object.values(payload.warnings || {}).reduce((total, value) => total + (Number(value) || 0), 0)
  if (serverWarningTotal > 0) warnings.push(t('{count} server normalization warnings', { count: serverWarningTotal }))
  if (payload.clientWarnings.invalidNodeCount) warnings.push(t('{count} invalid nodes skipped', { count: payload.clientWarnings.invalidNodeCount }))
  if (payload.clientWarnings.duplicateNodeCount) warnings.push(t('{count} duplicate nodes skipped', { count: payload.clientWarnings.duplicateNodeCount }))
  if (payload.clientWarnings.invalidEdgeCount) warnings.push(t('{count} invalid edges skipped', { count: payload.clientWarnings.invalidEdgeCount }))
  if (payload.clientWarnings.missingRelationEventIdCount) warnings.push(t('{count} entity relations cannot be assigned to an Event', { count: payload.clientWarnings.missingRelationEventIdCount }))
  return warnings
}

function formatGraphError(error) {
  const code = error?.code
  if (code === 'invalid_session_id' || code === 'unsupported_graph_scale' || error?.status === 400) return 'The memory graph request is invalid.'
  if (code === 'session_not_found') return 'The Session is no longer available.'
  if (error?.status === 404) return 'The memory graph endpoint is not available in this deployment.'
  if (code === 'graph_not_supported' || error?.status === 409) return 'This legacy Session does not support linked memory graphs.'
  if (code === 'memory_graph_response_too_large' || error?.status === 413) return 'This memory graph is too large to display safely.'
  if (code === 'memory_graph_not_ready') return 'The memory graph is not ready yet.'
  if (code === 'memory_component_lagging') return 'Graph components are still being published.'
  if (code === 'memory_snapshot_changing') return 'The memory graph is being updated.'
  if (code === 'invalid_memory_graph_response') return 'The memory graph response is incomplete or incompatible.'
  if (error?.status === 500) return 'The memory graph service is temporarily unavailable.'
  return error?.message || 'Unable to load the memory graph.'
}

function formatUpdatedAt(value, t) {
  if (!value) return t('30-second scale')
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? t('30-second scale') : t('Updated {date}', { date: parsed.toLocaleString() })
}

function useDebouncedValue(value, delay) {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay)
    return () => window.clearTimeout(timer)
  }, [delay, value])
  return debounced
}

// Translate generated graph annotations without changing titles or relation data.
function localizeGraphLabels(elements, t) {
  return elements.map((element) => {
    const data = element.data
    let label = data.label
    if (data.graphKind === 'event') {
      const lines = String(label).split('\n')
      const confidenceLine = lines.at(-1)?.match(/^confidence ([\d.]+)$/)
      if (confidenceLine) lines[lines.length - 1] = t('confidence {confidence}', { confidence: confidenceLine[1] })
      if (lines[0] === 'TIME —') lines[0] = t('TIME —')
      if (lines[1] === 'Untitled event') lines[1] = t('Untitled event')
      label = lines.join('\n')
    } else if (data.graphKind === 'mentions' && data.occurrenceCount > 1) {
      label = t('mentions ×{count}', { count: data.occurrenceCount })
    }
    return label === data.label ? element : { ...element, data: { ...data, label } }
  })
}
