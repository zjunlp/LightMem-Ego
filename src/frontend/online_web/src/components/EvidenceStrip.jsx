import { ChevronDown, ChevronUp, Image as ImageIcon } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

export default function EvidenceStrip({ evidenceFrames }) {
  const [expanded, setExpanded] = useState(false)
  const [imageErrors, setImageErrors] = useState({})

  useEffect(() => {
    setImageErrors({})
    setExpanded(false)
  }, [evidenceFrames])

  const visibleFrames = useMemo(() => {
    if (expanded) return evidenceFrames
    return evidenceFrames.slice(0, 3)
  }, [evidenceFrames, expanded])

  return (
    <section className="evidence-section">
      <div className="card-header">
        <div>
          <p className="eyebrow">Evidence Frames</p>
          <h2>{evidenceFrames.length ? 'Memory evidence from the answer' : 'Evidence will appear here after asking.'}</h2>
        </div>

        {evidenceFrames.length > 3 && (
          <button className="icon-button text-button" type="button" onClick={() => setExpanded((value) => !value)}>
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            <span>{expanded ? 'Collapse' : 'View more'}</span>
          </button>
        )}
      </div>

      {visibleFrames.length ? (
        <div className="evidence-strip" aria-live="polite">
          {visibleFrames.map((item) => {
            const hasImage = item.imageUrl && !imageErrors[item.id]
            return (
              <article className="evidence-card" key={item.id}>
                <div className="evidence-thumb-wrap">
                  {hasImage ? (
                    <img
                      className="evidence-thumb"
                      src={item.imageUrl}
                      alt={item.caption}
                      loading="lazy"
                      onError={() => setImageErrors((current) => ({ ...current, [item.id]: true }))}
                    />
                  ) : (
                    <div className="evidence-placeholder">
                      <ImageIcon size={20} />
                      <span>Evidence frame unavailable</span>
                    </div>
                  )}
                </div>
                <div className="evidence-copy">
                  <p>{item.caption}</p>
                  <div className="evidence-meta">
                    <span>{item.timeRangeText || item.timestampText || '-'}</span>
                    {item.scoreText && <span>score {item.scoreText}</span>}
                  </div>
                </div>
              </article>
            )
          })}
        </div>
      ) : (
        <div className="empty-evidence">No visual evidence yet.</div>
      )}
    </section>
  )
}
