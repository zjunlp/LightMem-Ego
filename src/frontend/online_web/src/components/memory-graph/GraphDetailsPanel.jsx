import { useLanguage } from '../../i18n.jsx'
import { ArrowRight, CircleX } from 'lucide-react'
import { formatEventRange, safeDisplay } from '../../utils/memoryGraphUtils.js'
import KeyframeGrid from './KeyframeGrid.jsx'

export default function GraphDetailsPanel({ selection, model, view, onSelectEvent, onClear }) {
  const { t } = useLanguage()
  if (!selection) {
    return (
      <section className="memory-graph-details" aria-label={t("Memory graph details")}>
        <div className="memory-graph-details-empty">
          <strong>{t("Select an Event, Fact, or Entity")}</strong>
          <span>{t("Details and cross-graph evidence will appear here.")}</span>
        </div>
      </section>
    )
  }

  let content = null
  if (selection.kind === 'event') content = <EventDetails event={model.eventById.get(selection.id)} model={model} />
  if (selection.kind === 'fact') content = <FactDetails fact={model.factById.get(selection.id)} model={model} view={view} onSelectEvent={onSelectEvent} />
  if (selection.kind === 'entity') content = <EntityDetails entityId={selection.id} model={model} view={view} />

  return (
    <section className="memory-graph-details" aria-label={t("Selected memory graph item details")}>
      <div className="memory-graph-detail-heading">
        <span>{selection.kind === 'fact' ? t("Semantic Fact") : selection.kind === 'event' ? t("Episodic Event") : t("Canonical Entity")}</span>
        <button className="icon-button secondary" type="button" onClick={onClear}>
          <CircleX size={14} />
          <span>{t("Clear")}</span>
        </button>
      </div>
      {content || <div className="memory-graph-details-empty">{t("The selected item is no longer available.")}</div>}
    </section>
  )
}

function EventDetails({ event, model }) {
  const { t } = useLanguage()
  if (!event) return null
  const document = event.docId ? model.documentByDocId.get(event.docId) : null
  const relations = (model.episodicEdgesByEventId.get(event.id) || []).filter((edge) => edge.kind === 'entity_relation')

  return (
    <div className="memory-graph-detail-content">
      <div className="memory-graph-detail-title">
        <strong>{event.title || event.label || t("Untitled event")}</strong>
        <span>{t(formatEventRange(event))}</span>
      </div>
      <DetailGrid items={[
        ['Confidence', formatNumber(event.confidence)],
        ['Status', event.status],
        ['Actions', event.counts?.actions],
        ['Objects', event.counts?.objects ?? event.counts?.visualObjects],
        ['State changes', event.counts?.stateChanges]
      ]} />
      <DetailSection title={t("Description")} value={document?.text || document?.caption || event.title} />
      <DetailSection title={t("Visual summary")} value={document?.visualSummary} />
      <DetailSection title={t("Transcript")} value={document?.transcript} />
      <DetailSection title={t("Scene")} value={document?.scene} />
      <DetailSection title={t("Visual objects")} value={document?.visualObjects} />
      <DetailSection title={t("Actions")} value={document?.actions} />
      <DetailSection title={t("State changes")} value={document?.stateChanges} />
      <DetailSection title={t("Topic threads")} value={document?.topicThreads} />
      <DetailSection title={t("Critical speech")} value={document?.criticalSpeechLines} />
      {relations.length > 0 && (
        <section className="memory-graph-detail-section">
          <h4>{t("Event triples")}</h4>
          <ul className="memory-graph-triples">
            {relations.map((edge) => (
              <li key={edge.id}>
                <span>{model.episodicEntityById.get(edge.source)?.label || edge.source}</span>
                <strong>{edge.relation || edge.label || t("related to")}</strong>
                <span>{model.episodicEntityById.get(edge.target)?.label || edge.target}</span>
                {edge.occurrenceCount > 1 && <small>×{edge.occurrenceCount}</small>}
              </li>
            ))}
          </ul>
        </section>
      )}
      <KeyframeGrid keyframes={document?.keyframes || []} />
    </div>
  )
}

function FactDetails({ fact, model, view, onSelectEvent }) {
  const { t } = useLanguage()
  if (!fact) return null
  const source = model.semanticEntityById.get(fact.source)?.label || fact.source
  const target = model.semanticEntityById.get(fact.target)?.label || fact.target
  const eventIds = model.factIdToEventIds.get(fact.id) || fact.supportEventIds || []

  return (
    <div className="memory-graph-detail-content">
      <div className="memory-graph-fact-title">
        <strong>{source}</strong>
        <span>{fact.relation || t("related to")}</span>
        <ArrowRight size={17} />
        <strong>{target}</strong>
      </div>
      <DetailGrid items={[
        ['Confidence', formatNumber(fact.confidence)],
        ['Support count', fact.supportCount],
        ['Raw support', fact.rawSupportCount],
        ['Habit strength', fact.habitStrength],
        ['Semantic version', fact.semanticVersion],
        ['Unresolved support', fact.unresolvedSupportCount]
      ]} />
      <DetailSection title={t("First seen")} value={fact.firstSeen} />
      <DetailSection title={t("Last seen")} value={fact.lastSeen} />
      <DetailSection title={t("Support days")} value={fact.supportDays} />
      <DetailSection title={t("Support scales")} value={fact.supportScales} />

      <section className="memory-graph-detail-section">
        <h4>{t("Supporting events")}</h4>
        {eventIds.length ? (
          <div className="memory-graph-supporting-events">
            {eventIds.map((eventId) => {
              const event = model.eventById.get(eventId)
              const document = event?.docId ? model.documentByDocId.get(event.docId) : null
              const outsideFilters = !view.visibleEventIds.has(eventId)
              return (
                <button type="button" onClick={() => onSelectEvent(eventId, outsideFilters)} key={eventId}>
                  <span>
                    <strong>{event?.title || event?.label || eventId}</strong>
                    <small>{event ? t(formatEventRange(event)) : t("Event details unavailable")}</small>
                    {(document?.caption || document?.text) && <em>{String(document.caption || document.text).slice(0, 180)}</em>}
                  </span>
                  <span>{outsideFilters ? t("Show event") : t("Open")}</span>
                </button>
              )
            })}
          </div>
        ) : <div className="memory-graph-detail-empty">{t("No resolved supporting events.")}</div>}
      </section>
    </div>
  )
}

function EntityDetails({ entityId, model, view }) {
  const { t } = useLanguage()
  const episodicEntity = model.episodicEntityById.get(entityId)
  const semanticEntity = model.semanticEntityById.get(entityId)
  const entity = semanticEntity || episodicEntity
  if (!entity) return null
  const mentions = model.graph.episodic.edges.filter((edge) => edge.kind === 'mentions' && edge.target === entityId)
  const relatedFacts = [...model.factById.values()].filter((fact) => fact.source === entityId || fact.target === entityId)
  const relatedEvents = new Set(mentions.map((edge) => edge.source))
  const visibleFacts = relatedFacts.filter((fact) => view.visibleFactIds.has(fact.id)).length
  const visibleEvents = [...relatedEvents].filter((eventId) => view.visibleEventIds.has(eventId)).length

  return (
    <div className="memory-graph-detail-content">
      <div className="memory-graph-detail-title">
        <strong>{entity.label}</strong>
        <span>{entity.id}</span>
      </div>
      <DetailGrid items={[
        ['Mentions', mentions.reduce((total, edge) => total + edge.occurrenceCount, 0)],
        ['Events', relatedEvents.size],
        ['Visible events', visibleEvents],
        ['Facts', semanticEntity?.factCount || relatedFacts.length],
        ['Visible facts', visibleFacts],
        ['Total support', semanticEntity?.totalSupportCount || 0]
      ]} />
      <DetailSection title={t("First seen")} value={semanticEntity?.firstSeen || entity.firstSeen} />
      <DetailSection title={t("Last seen")} value={semanticEntity?.lastSeen || entity.lastSeen} />
    </div>
  )
}

function DetailGrid({ items }) {
  const { t } = useLanguage()
  const visibleItems = items.filter(([, value]) => value !== null && value !== undefined && value !== '')
  if (!visibleItems.length) return null
  return (
    <div className="memory-graph-detail-grid">
      {visibleItems.map(([label, value]) => (
        <div key={label}>
          <span>{t(label)}</span>
          <strong>{label === 'Status' ? t(safeDisplay(value)) : safeDisplay(value)}</strong>
        </div>
      ))}
    </div>
  )
}

function DetailSection({ title, value }) {
  const { t } = useLanguage()
  if (value === null || value === undefined || value === '' || (Array.isArray(value) && !value.length)) return null
  const content = Array.isArray(value)
    ? <ul>{value.map((item, index) => <li key={`${title}-${index}`}>{safeDisplay(item)}</li>)}</ul>
    : <p>{safeDisplay(value)}</p>
  return (
    <section className="memory-graph-detail-section">
      <h4>{t(title)}</h4>
      {content}
    </section>
  )
}

function formatNumber(value) {
  return Number.isFinite(value) ? value.toFixed(2) : value
}
