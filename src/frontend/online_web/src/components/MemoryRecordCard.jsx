import { Download, LoaderCircle } from 'lucide-react'
import { memo } from 'react'
import { useLanguage } from '../i18n.jsx'
import { useBackendFeatures } from './BackendFeatures.jsx'
export function humanizeKey(value) {
  return String(value || '').replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase())
}

export const MemoryRecordCard = memo(function MemoryRecordCard({ memory, sessionId, onExport, exporting = false }) {
  const { t } = useLanguage()
  const features = useBackendFeatures()
  const timeLabel = formatEventTime(memory.eventTime)
  return (
    <article className="memory-card">
      <div className="memory-card-header">
        <div className="memory-card-badges">
          <span className="memory-badge">{t(humanizeKey(memory.memoryType || 'Memory'))}</span>
          {memory.granularity && <span className="memory-badge is-muted">{t(memory.granularity)}</span>}
          {memory.status && <span className="memory-badge is-muted">{t(humanizeKey(memory.status))}</span>}
          {memory.sourceId && <span className="memory-badge is-muted">{t("Source")} {memory.sourceId}</span>}
        </div>
        {memory.confidence !== null && memory.confidence !== undefined && (
          <span className="memory-confidence">{t("Confidence")} {formatConfidence(memory.confidence)}</span>
        )}
        {(features?.memoryExport ?? true) && <button
          className="icon-button secondary memory-export-button"
          type="button"
          aria-label={t("Export memory {value1} as JSON", { value1: memory.id || t('unknown') })}
          title={t("Export JSON")}
          aria-busy={exporting}
          disabled={exporting || !onExport}
          onClick={() => onExport?.(memory, sessionId)}
        >
          {exporting ? <LoaderCircle className="memory-export-spinner" size={14} /> : <Download size={14} />}
          <span>{exporting ? t("Exporting…") : t("Export JSON")}</span>
        </button>}
      </div>
      <div className="memory-time">{timeLabel || t('Time unknown')}</div>
      <p className="memory-content">{memory.content || '—'}</p>
      {memory.data && <ReadonlyValue value={memory.data} root />}
    </article>
  )
})

export function ReadonlyValue({ value, root = false }) {
  const { t } = useLanguage()
  if (value === null || value === undefined || value === '') return <span className="memory-param-value">—</span>

  if (Array.isArray(value)) {
    if (!value.length) return <span className="memory-param-value">—</span>
    const content = <MemoryArray items={value} />
    if (value.length <= 4) return content
    return (
      <details className="memory-value-details">
        <summary>{value.length} {t("items")}</summary>
        {content}
      </details>
    )
  }

  if (typeof value === 'object') {
    if (Array.isArray(value.items) && Object.prototype.hasOwnProperty.call(value, 'truncated')) {
      return (
        <div className="memory-truncated-value">
          <div className="memory-truncation-note">{t("Showing")} {value.returnedCount ?? value.items.length} {t("of")} {value.originalCount ?? value.items.length} {t("items.")}</div>
          <ReadonlyValue value={value.items} />
        </div>
      )
    }
    const entries = Object.entries(value)
    if (!entries.length) return <span className="memory-param-value">—</span>
    return (
      <div className={`memory-param-list ${root ? 'is-root' : 'is-nested'}`}>
        {entries.map(([key, item]) => (
          <div className="memory-param-row" key={key}>
            <span className="memory-param-key">{t(humanizeKey(key))}</span>
            <div className="memory-param-value"><ReadonlyValue value={item} /></div>
          </div>
        ))}
      </div>
    )
  }

  return <span className="memory-param-value">{String(value)}</span>
}

function MemoryArray({ items }) {
  return (
    <ul className="memory-array">
      {items.map((item, index) => (
        <li key={`${index}-${primitiveKey(item)}`}><ReadonlyValue value={item} /></li>
      ))}
    </ul>
  )
}

function primitiveKey(value) {
  return typeof value === 'object' ? '' : String(value).slice(0, 32)
}

function formatEventTime(eventTime) {
  if (!eventTime || typeof eventTime !== 'object') return ''
  const date = formatLooseValue(eventTime.date)
  const start = firstValue(eventTime.startSeconds, eventTime.startTime, eventTime.timestampSeconds, eventTime.timestamp)
  const end = firstValue(eventTime.endSeconds, eventTime.endTime)
  const firstSeen = formatLooseValue(eventTime.firstSeen)
  const lastSeen = formatLooseValue(eventTime.lastSeen)
  if (firstSeen || lastSeen) return [firstSeen, lastSeen].filter(Boolean).join(' – ')
  const range = start !== undefined ? `${formatSeconds(start)}${end !== undefined ? ` – ${formatSeconds(end)}` : ''}` : ''
  return [date, range].filter(Boolean).join(' · ')
}

function firstValue(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== '')
}

function formatSeconds(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? `${numeric.toFixed(numeric % 1 ? 1 : 0)}s` : String(value)
}

function formatLooseValue(value) {
  if (value === undefined || value === null || value === '') return ''
  if (typeof value === 'object') return Object.values(value).filter(Boolean).join(' ')
  return String(value)
}

function formatConfidence(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric.toFixed(2) : String(value)
}
