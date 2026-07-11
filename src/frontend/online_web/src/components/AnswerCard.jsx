import { Activity } from 'lucide-react'

export default function AnswerCard({ answer, loading, latency, queryStatus, answerMode, answerPhase, error }) {
  const isProvisional = ['draft', 'streaming', 'final'].includes(answerPhase)

  return (
    <section className="surface answer-card fade-in">
      <div className="card-header">
        <div>
          <p className="eyebrow">AI Answer</p>
          <h2>LightMem-Ego response</h2>
        </div>
        <div className={`query-pill ${loading ? 'active' : ''}`}>
          <span className="live-dot" />
          <span>{formatQueryStatus(queryStatus, loading)}</span>
        </div>
      </div>

      <div className="answer-body">
        {answer ? (
          <>
            {isProvisional && (
              <div className="provisional-label">
                {answerPhase === 'draft' ? 'Provisional draft' : 'Streaming'}
                {loading && (
                  <span className="typing-dots" aria-hidden="true">
                    <i />
                    <i />
                    <i />
                  </span>
                )}
              </div>
            )}
            <p>{answer}</p>
          </>
        ) : loading ? (
          <div className="answer-empty">
            <span>Retrieving live memory and composing an answer</span>
            <span className="typing-dots" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
          </div>
        ) : (
          <p className="answer-empty-text">Ask a question while LightMem-Ego is watching.</p>
        )}
      </div>

      {(answer || error || loading) && (
        <div className="answer-meta">
          <span>
            <Activity size={14} />
            Latency: {formatLatency(latency)}
          </span>
        </div>
      )}

      {error && <div className="inline-error">{error}</div>}
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
