import { RefreshCw, Send } from 'lucide-react'
import { useLayoutEffect } from 'react'
import { formatMemoryImportError } from '../api/memoryImportApi.js'
import { useAskLightMemEgo } from '../hooks/useAskLightMemEgo.js'
import AnswerCard from './AnswerCard.jsx'
import EvidenceStrip from './EvidenceStrip.jsx'
import { useLanguage } from '../i18n.jsx'

const QUICK_QUESTIONS = [
  'Summarize the imported memory.',
  'What important events are recorded?',
  'What projects or topics are mentioned?'
]

export default function ImportedMemoryAsk({ session, queryState, publication }) {
  const { t } = useLanguage()
  const ask = useAskLightMemEgo(session.sessionId, { memoryScope: 'long_term' })
  const { ready: queryAllowed, loading, failed, error, refresh: refreshStatus } = queryState
  const publicationId = publication?.publicationId || publication?.memory?.memoryVersion || ''
  useLayoutEffect(() => { ask.reset() }, [ask.reset, publicationId])
  const disabled = !queryAllowed || ask.loading || loading
  const statusMessage = queryAllowed ? 'Query service ready.'
    : failed ? 'Memory saved. Query preparation failed.'
      : publication ? 'Memory saved. Preparing query service…' : 'Query service is not ready yet.'
  return (
    <div className="memory-ask-panel">
      <div className="memory-ask-status" role="status">
        <span>{t(statusMessage)}</span>
        <button className="icon-button secondary" type="button" onClick={refreshStatus} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'memory-spin' : ''} />
          <span>{t('Refresh status')}</span>
        </button>
      </div>
      {error && <div className="inline-error memory-error" role="alert">{t(formatMemoryImportError(error))}</div>}
      {failed && <p className="inline-error" role="alert">{t('Query preparation failed. Refresh status to check the background retry.')}
        {(queryState.warmup?.error?.code || queryState.warmup?.errorCode) && <small> {queryState.warmup.error?.code || queryState.warmup.errorCode}</small>}
      </p>}
      {!queryAllowed && !failed && <p className="memory-ask-help">{t('The imported Session is still warming up. You can view Records while query service preparation continues.')}</p>}

      <form className="ask-composer" onSubmit={(event) => { event.preventDefault(); if (!disabled) ask.ask() }}>
        <input
          aria-label={t('Ask about the imported memory…')}
          value={ask.question}
          onChange={(event) => ask.setQuestion(event.target.value)}
          placeholder={t('Ask about the imported memory…')}
          disabled={disabled}
        />
        <button className="send-button" type="submit" disabled={disabled || !ask.question.trim()}>
          <Send size={16} />
          <span>{t('Send')}</span>
        </button>
      </form>
      <div className="quick-row">
        {QUICK_QUESTIONS.map((question) => (
          <button className="quick-chip" type="button" key={question} disabled={disabled} onClick={() => { ask.setQuestion(question); ask.ask(question, { inputMethod: 'preset' }) }}>
            {t(question)}
          </button>
        ))}
      </div>

      <AnswerCard
        answer={ask.answer}
        loading={ask.loading}
        latency={ask.latency}
        queryStatus={ask.queryStatus}
        answerMode={ask.answerMode}
        answerPhase={ask.answerPhase}
        error={ask.error}
      />
      {ask.evidenceFrames?.length > 0 && <EvidenceStrip evidenceFrames={ask.evidenceFrames} />}
    </div>
  )
}
