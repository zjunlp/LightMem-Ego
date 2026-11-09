import { ChevronDown, ChevronUp, Image as ImageIcon } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useLanguage } from '../i18n.jsx'

export default function EvidenceStrip({ evidenceFrames }) {
  const { t } = useLanguage()
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
          <h2>{t('Evidence Frames')}</h2>
          <p className="section-description">{t(evidenceFrames.length ? 'Memory evidence from the answer' : 'Evidence will appear here after asking.')}</p>
        </div>

        {evidenceFrames.length > 3 && (
          <button className="icon-button text-button" type="button" onClick={() => setExpanded((value) => !value)}>
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            <span>{expanded ? t('Collapse') : t('View more')}</span>
          </button>
        )}
      </div>

      {visibleFrames.length ? (
        <div className="evidence-strip" aria-live="polite">
          {visibleFrames.map((item, index) => {
            const hasImage = item.imageUrl && !imageErrors[item.id]
            return (
              <article className="evidence-card" key={item.id} style={{ '--entry-delay': `${Math.min(index, 5) * 30}ms` }}>
                <div className="evidence-thumb-wrap">
                  {hasImage ? (
                    <img
                      className="evidence-thumb"
                      src={item.imageUrl}
                      alt={item.caption === 'Evidence frame' ? t('Evidence frame') : item.caption}
                      loading="lazy"
                      onError={() => setImageErrors((current) => ({ ...current, [item.id]: true }))}
                    />
                  ) : (
                    <div className="evidence-placeholder">
                      <ImageIcon size={20} />
                      <span>{t('Evidence frame unavailable')}</span>
                    </div>
                  )}
                </div>
                <div className="evidence-copy">
                  <p>{item.caption === 'Evidence frame' ? t('Evidence frame') : item.caption}</p>
                  <div className="evidence-meta">
                    <span>{item.timeRangeText || item.timestampText || '-'}</span>
                    {item.scoreText && <span>{t("score")} {item.scoreText}</span>}
                  </div>
                </div>
              </article>
            )
          })}
        </div>
      ) : (
        <div className="empty-evidence"><ImageIcon size={20} strokeWidth={1.6} aria-hidden="true" /><span>{t('No visual evidence yet.')}</span></div>
      )}
    </section>
  )
}
