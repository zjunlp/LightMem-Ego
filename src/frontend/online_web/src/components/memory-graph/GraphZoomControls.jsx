import { useLanguage } from '../../i18n.jsx'
import { Maximize2, ZoomIn, ZoomOut } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'

const MIN_ZOOM = 0.12
const MAX_ZOOM = 2.5
const ZOOM_FACTOR = 1.2

export default function GraphZoomControls({ cy, disabled = false, label }) {
  const { t } = useLanguage()
  const [zoom, setZoom] = useState(1)
  const animationFrameRef = useRef(null)

  useEffect(() => {
    if (!cy) return undefined
    const syncZoom = () => {
      if (animationFrameRef.current !== null) return
      animationFrameRef.current = window.requestAnimationFrame(() => {
        animationFrameRef.current = null
        setZoom(clampZoom(cy.zoom()))
      })
    }

    syncZoom()
    cy.on('zoom', syncZoom)
    return () => {
      cy.off('zoom', syncZoom)
      if (animationFrameRef.current !== null) window.cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
  }, [cy])

  const setCenteredZoom = useCallback((nextZoom) => {
    if (!cy || disabled) return
    cy.zoom({
      level: clampZoom(nextZoom),
      renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 }
    })
  }, [cy, disabled])

  const fitGraph = useCallback(() => {
    if (!cy || disabled || !cy.elements().length) return
    cy.fit(cy.elements(), 42)
  }, [cy, disabled])

  const controlsDisabled = disabled || !cy
  return (
    <div className="memory-graph-zoom-controls" role="group" aria-label={t("{value1} zoom controls", { value1: label })}>
      <button
        type="button"
        aria-label={t("Zoom out {value1}", { value1: label })}
        title={t("Zoom out")}
        disabled={controlsDisabled || zoom <= MIN_ZOOM}
        onClick={() => setCenteredZoom(zoom / ZOOM_FACTOR)}
      >
        <ZoomOut size={15} />
      </button>
      <input
        type="range"
        min={MIN_ZOOM}
        max={MAX_ZOOM}
        step="0.01"
        value={zoom}
        disabled={controlsDisabled}
        aria-label={t("{value1} zoom level", { value1: label })}
        aria-valuetext={t("{value1} percent", { value1: Math.round(zoom * 100) })}
        onChange={(event) => setCenteredZoom(Number(event.target.value))}
      />
      <output aria-live="off">{Math.round(zoom * 100)}%</output>
      <button
        type="button"
        aria-label={t("Zoom in {value1}", { value1: label })}
        title={t("Zoom in")}
        disabled={controlsDisabled || zoom >= MAX_ZOOM}
        onClick={() => setCenteredZoom(zoom * ZOOM_FACTOR)}
      >
        <ZoomIn size={15} />
      </button>
      <button
        className="memory-graph-zoom-fit"
        type="button"
        aria-label={t("Fit {value1} to canvas", { value1: label })}
        title={t("Fit graph to canvas")}
        disabled={controlsDisabled}
        onClick={fitGraph}
      >
        <Maximize2 size={14} />
        <span>{t("Fit")}</span>
      </button>
    </div>
  )
}

function clampZoom(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return 1
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, numeric))
}
