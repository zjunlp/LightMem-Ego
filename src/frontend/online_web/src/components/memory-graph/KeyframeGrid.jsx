import { useLanguage } from '../../i18n.jsx'
import { ImageOff, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { buildApiUrl } from '../../api/lightmem_egoApi.js'
import { formatTimelineSeconds } from '../../utils/memoryGraphUtils.js'

export default function KeyframeGrid({ keyframes = [] }) {
  const { t } = useLanguage()
  const [failed, setFailed] = useState(() => new Set())
  const [preview, setPreview] = useState(null)
  useEffect(() => {
    if (!preview) return undefined
    const closeOnEscape = (event) => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      event.stopPropagation()
      setPreview(null)
    }
    window.addEventListener('keydown', closeOnEscape, true)
    return () => window.removeEventListener('keydown', closeOnEscape, true)
  }, [preview])
  if (!keyframes.length) return null

  return (
    <section className="memory-graph-detail-section">
      <h4>{t("Keyframes")}</h4>
      <div className="memory-graph-keyframes">
        {keyframes.map((frame, index) => {
          const imageUrl = buildApiUrl(frame.thumbnailUrl)
          const key = `${frame.thumbnailUrl}-${index}`
          const isFailed = failed.has(key)
          return (
            <button
              className="memory-graph-keyframe"
              type="button"
              disabled={isFailed}
              onClick={() => setPreview({ imageUrl, timestampSeconds: frame.timestampSeconds })}
              key={key}
            >
              {isFailed ? (
                <span className="memory-graph-keyframe-placeholder">
                  <ImageOff size={22} />{t("Image unavailable")}</span>
              ) : (
                <img
                  src={imageUrl}
                  alt={Number.isFinite(frame.timestampSeconds) ? t("Event keyframe at {value1}", { value1: formatTimelineSeconds(frame.timestampSeconds) }) : t("Event keyframe {value1}", { value1: index + 1 })}
                  loading="lazy"
                  onError={() => setFailed((current) => new Set(current).add(key))}
                />
              )}
              {Number.isFinite(frame.timestampSeconds) && <small>{formatTimelineSeconds(frame.timestampSeconds)}</small>}
            </button>
          )
        })}
      </div>

      {preview && (
        <div className="memory-graph-image-overlay" role="dialog" aria-modal="true" aria-label={t("Keyframe preview")} onMouseDown={(event) => {
          if (event.target === event.currentTarget) setPreview(null)
        }}>
          <div className="memory-graph-image-preview">
            <button autoFocus className="icon-button secondary" type="button" aria-label={t("Close keyframe preview")} onClick={() => setPreview(null)}>
              <X size={16} />
              <span>{t("Close")}</span>
            </button>
            <img src={preview.imageUrl} alt={t("Selected event keyframe")} />
          </div>
        </div>
      )}
    </section>
  )
}
