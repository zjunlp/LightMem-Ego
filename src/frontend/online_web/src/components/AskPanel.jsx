import { ChevronLeft, Send, Wrench } from 'lucide-react'
import { useId } from 'react'
import { useLanguage } from '../i18n.jsx'
import AdvancedTools from './AdvancedTools.jsx'
import AnswerCard from './AnswerCard.jsx'
import EvidenceStrip from './EvidenceStrip.jsx'

const QUICK_QUESTIONS = [
  'What is in the current scene?',
  'What just happened?',
  'What did they just say?',
  'Summarize everything so far.'
]

export default function AskPanel({ stream, ask, onOpenLive, onReset }) {
  const { t } = useLanguage()
  const disabled = !stream.sessionId || ask.loading || !stream.canAsk
  const questionId = useId()

  return (
    <section className="ask-panel">
      <header className="ask-header">
        <button className="icon-button secondary back-button" type="button" onClick={onOpenLive}>
          <ChevronLeft size={17} />
          <span>{t('Live')}</span>
        </button>
        <div>
          <h2>{t('Ask LightMem-Ego')}</h2>
          <p>{t('Ask about the live video')}</p>
        </div>
        <a
          className="tools-anchor"
          href="#advanced-tools"
          aria-label={t('Advanced tools')}
          onClick={(event) => {
            event.preventDefault()
            const tools = event.currentTarget.closest('.ask-panel')?.querySelector('#advanced-tools')
            tools?.scrollIntoView({
              block: 'start',
              behavior: window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'
            })
            tools?.querySelector('.advanced-toggle')?.focus({ preventScroll: true })
          }}
        >
          <Wrench size={17} />
        </a>
      </header>

      <section className="surface composer-card">
        <label className="composer-label" htmlFor={questionId}>{t('Ask about the live video')}</label>
        <form
          className="ask-composer"
          onSubmit={(event) => {
            event.preventDefault()
            ask.ask()
          }}
        >
          <input
            id={questionId}
            value={ask.question}
            onChange={(event) => ask.setQuestion(event.target.value)}
            aria-label={t('Ask about the live video...')}
            placeholder={t('Type a question about the live video...')}
            disabled={!stream.sessionId}
          />
          <button className="send-button" type="submit" disabled={disabled || !ask.question.trim()}>
            <Send size={16} />
            <span>{t('Send')}</span>
          </button>
        </form>

        <div className="quick-row">
          {QUICK_QUESTIONS.map((question) => (
            <button
              className="quick-chip"
              type="button"
              key={question}
              disabled={disabled || !stream.canAsk}
              onClick={() => {
                ask.setQuestion(question)
                ask.ask(question, { inputMethod: 'preset' })
              }}
            >
              {t(question)}
            </button>
          ))}
        </div>

        {stream.isWebRtcMode && stream.sessionId && !stream.memoryReady && (
          <p className="first-frame-hint">{t('Live stream is starting. You can ask now; memory may need a moment to become ready.')}</p>
        )}
        {!stream.isWebRtcMode && stream.sessionId && !stream.canAsk && (
          <p className="first-frame-hint">{t('Building first-frame memory. You can ask once the first frame is uploaded.')}</p>
        )}
        {!stream.sessionId && (
          <p className="first-frame-hint">{t('Please start Live View first.')}</p>
        )}
      </section>

      <AnswerCard
        answer={ask.answer}
        loading={ask.loading}
        latency={ask.latency}
        queryStatus={ask.queryStatus}
        answerMode={ask.answerMode}
        answerPhase={ask.answerPhase}
        error={ask.error}
      />

      <EvidenceStrip evidenceFrames={ask.evidenceFrames} />

      <div id="advanced-tools">
        <AdvancedTools stream={stream} ask={ask} onReset={onReset} />
      </div>
    </section>
  )
}
