import { Activity, MessageSquareText } from 'lucide-react'
import { useLanguage } from '../i18n.jsx'

export default function AnswerCard({ answer, loading, latency, queryStatus, answerMode, answerPhase, error }) {
  const { t } = useLanguage()
  const isProvisional = ['draft', 'streaming', 'final'].includes(answerPhase)

  return (
    <section className={`surface answer-card fade-in ${loading ? 'is-loading' : ''}`} aria-busy={loading}>
      <div className="card-header">
        <div>
          <h2>{t('AI Answer')}</h2>
          <p className="section-description">{t('LightMem-Ego response')}</p>
        </div>
        <div className={`query-pill ${loading ? 'active' : ''}`}>
          <span className="live-dot" />
          <span>{t(formatQueryStatus(queryStatus, loading))}</span>
        </div>
      </div>

      <div className="answer-body">
        {answer ? (
          <>
            {isProvisional && (
              <div className="provisional-label">
                {answerPhase === 'draft' ? t('Provisional draft') : t('Streaming')}
                {loading && (
                  <span className="typing-dots" aria-hidden="true">
                    <i />
                    <i />
                    <i />
                  </span>
                )}
              </div>
            )}
            <p key={answerPhase} className="answer-copy">{answer}</p>
          </>
        ) : loading ? (
          <div className="answer-loading" role="status">
            <div className="answer-skeleton" aria-hidden="true"><i /><i /><i /></div>
            <div className="answer-empty">
              <span>{t('Retrieving live memory and composing an answer')}</span>
              <span className="typing-dots" aria-hidden="true"><i /><i /><i /></span>
            </div>
          </div>
        ) : (
          <div className="answer-idle">
            <span className="empty-state-icon"><MessageSquareText size={21} strokeWidth={1.6} aria-hidden="true" /></span>
            <p className="answer-empty-text">{t('Ask a question while LightMem-Ego is watching.')}</p>
          </div>
        )}
      </div>

      {(answer || error || loading) && (
        <div className="answer-meta">
          <span>
            <Activity size={14} />
            {t('Latency')}: {formatLatency(latency)}
          </span>
        </div>
      )}

      {error && <div className="inline-error">{t(error)}</div>}
    </section>
  )
}

function formatQueryStatus(status, loading) {
  if (loading) {
    if (status === 'queued') return 'Queued'
    if (status === 'streaming') return 'Streaming'
    return 'Thinking'
  }
  if (status === 'done') return 'Done'
  if (status === 'failed') return 'Failed'
  return 'Idle'
}

function formatLatency(value) {
  if (value === undefined || value === null || value === '') return '-'
  if (typeof value === 'number') return value > 100 ? `${Math.round(value)} ms` : `${value.toFixed(2)} s`
  if (typeof value === 'object') {
    const candidates = [value.totalMs, value.total_ms, value.queryMs, value.query_ms, value.total]
    const found = candidates.find((item) => item !== undefined && item !== null)
    return found === undefined ? '-' : formatLatency(found)
  }
  return String(value)
}
