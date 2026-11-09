import { useLanguage } from '../../i18n.jsx'
import { useEffect, useState } from 'react'
import { NODE_COLOR_PALETTE } from './nodeColorPalette.js'

const RELATION_ITEMS = [
  ['spatial', 'Spatial'],
  ['action', 'Action'],
  ['state', 'State'],
  ['dialogue', 'Dialogue'],
  ['other', 'Other']
]

export default function GraphLegend({ mode }) {
  const { t } = useLanguage()
  const [open, setOpen] = useState(() => !isMobileGraph())

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined
    const media = window.matchMedia('(max-width: 859px)')
    const syncDefault = (event) => setOpen(!event.matches)
    media.addEventListener?.('change', syncDefault)
    return () => media.removeEventListener?.('change', syncDefault)
  }, [])

  return (
    <details
      className="memory-graph-legend"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>{t("Legend")}</summary>
      <div className="memory-graph-legend-content">
        <div className="memory-graph-legend-group" aria-label={t("Node types")}>
          {mode === 'episodic' && <LegendItem swatch="node-event" label={t("Event")} />}
          <LegendItem swatch="node-entity" label={t("Entity")} />
        </div>

        <span className="memory-graph-legend-caption">{t("Node color · stable by ID")}</span>
        <div className="memory-graph-node-palette" aria-label={t("Twelve node colors")}>
          {NODE_COLOR_PALETTE.map((color, index) => (
            <i
              key={color.fill}
              className="memory-graph-node-color"
              style={{ '--node-fill': color.fill, '--node-border': color.border }}
              title={t("Color {value1}", { value1: index + 1 })}
            />
          ))}
        </div>

        <span className="memory-graph-legend-caption">{t("Relation lines")}</span>
        <div className="memory-graph-legend-group is-relations" aria-label={t("Relation groups")}>
          {RELATION_ITEMS.map(([kind, label]) => (
            <LegendItem key={kind} swatch={`relation-${kind}`} label={t(label)} />
          ))}
        </div>

        <div className="memory-graph-legend-notes">
          {mode === 'episodic' ? (
            <>
              <span><i className="memory-graph-line-sample is-before" />{t("Time order")}</span>
              <span><i className="memory-graph-line-sample is-mention" />{t("Mention")}</span>
            </>
          ) : (
            <>
              <span><i className="memory-graph-line-sample is-support" />{t("Width = support")}</span>
              <span><i className="memory-graph-line-sample is-habit" />{t("Dash = habit strength")}</span>
            </>
          )}
        </div>
      </div>
    </details>
  )
}

function LegendItem({ swatch, label }) {
  return (
    <span className="memory-graph-legend-item">
      <i className={`memory-graph-legend-swatch ${swatch}`} />
      {label}
    </span>
  )
}

function isMobileGraph() {
  return typeof window !== 'undefined' && window.matchMedia?.('(max-width: 859px)').matches
}
