import { useLanguage } from '../../i18n.jsx'
import { RotateCcw, Search } from 'lucide-react'
import { formatTimelineSeconds } from '../../utils/memoryGraphUtils.js'

export default function GraphToolbar({ filters, relationOptions, timeDomain, onChange, onReset }) {
  const { t } = useLanguage()
  const start = Number.isFinite(filters.startSeconds) ? filters.startSeconds : timeDomain.start
  const end = Number.isFinite(filters.endSeconds) ? filters.endSeconds : timeDomain.end
  const rangeDisabled = timeDomain.end <= timeDomain.start

  return (
    <div className="memory-graph-toolbar" aria-label={t("Memory graph filters")}>
      <label className="memory-graph-search">
        <span>{t("Search")}</span>
        <span className="memory-graph-input-shell">
          <Search size={14} />
          <input
            type="search"
            value={filters.query}
            placeholder={t("Events, entities, relations…")}
            onChange={(event) => onChange({ query: event.target.value })}
          />
        </span>
      </label>

      <div className="memory-graph-range-control">
        <span>{t("Time range")}</span>
        <div className="memory-graph-range-labels">
          <output>{formatTimelineSeconds(start)}</output>
          <output>{formatTimelineSeconds(end)}</output>
        </div>
        <div className="memory-graph-range-inputs">
          <input
            type="range"
            min={timeDomain.start}
            max={timeDomain.end}
            step="1"
            value={start}
            disabled={rangeDisabled}
            aria-label={t("Memory graph start time")}
            onChange={(event) => onChange({ startSeconds: Math.min(Number(event.target.value), end) })}
          />
          <input
            type="range"
            min={timeDomain.start}
            max={timeDomain.end}
            step="1"
            value={end}
            disabled={rangeDisabled}
            aria-label={t("Memory graph end time")}
            onChange={(event) => onChange({ endSeconds: Math.max(Number(event.target.value), start) })}
          />
        </div>
      </div>

      <details className="memory-graph-relations">
        <summary>{t("Relations")}{filters.relations.length ? ` (${filters.relations.length})` : ''}</summary>
        <div className="memory-graph-relation-menu">
          {relationOptions.length ? relationOptions.map((relation) => (
            <label key={relation}>
              <input
                type="checkbox"
                checked={filters.relations.includes(relation)}
                onChange={(event) => onChange({
                  relations: event.target.checked
                    ? [...filters.relations, relation]
                    : filters.relations.filter((item) => item !== relation)
                })}
              />
              <span>{relation}</span>
            </label>
          )) : <span className="memory-graph-menu-empty">{t("No relations")}</span>}
        </div>
      </details>

      <label className="memory-graph-number-filter">
        <span>{t("Episodic confidence")}</span>
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          value={filters.episodicMinConfidence}
          onChange={(event) => onChange({ episodicMinConfidence: clamp(Number(event.target.value), 0, 1) })}
        />
      </label>

      <label className="memory-graph-number-filter">
        <span>{t("Semantic confidence")}</span>
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          value={filters.semanticMinConfidence}
          onChange={(event) => onChange({ semanticMinConfidence: clamp(Number(event.target.value), 0, 1) })}
        />
      </label>

      <label className="memory-graph-number-filter">
        <span>{t("Min support")}</span>
        <input
          type="number"
          min="1"
          step="1"
          value={filters.semanticMinSupportCount}
          onChange={(event) => onChange({ semanticMinSupportCount: Math.max(1, Math.floor(Number(event.target.value) || 1)) })}
        />
      </label>

      <button className="icon-button secondary memory-graph-reset" type="button" onClick={onReset}>
        <RotateCcw size={14} />
        <span>{t("Reset")}</span>
      </button>
    </div>
  )
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}
