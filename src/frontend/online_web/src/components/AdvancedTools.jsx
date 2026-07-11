import {
  Bug,
  ChevronDown,
  ChevronUp,
  Film,
  Radio,
  RefreshCw,
  RotateCcw,
  Upload,
  Wrench
} from 'lucide-react'
import { useState } from 'react'
import { ANSWER_MODES, LONG_TERM_RETRIEVAL_SCHEMES } from '../hooks/useAskWorldMM.js'
import { INPUT_MODES } from '../hooks/useRealtimeStream.js'

export default function AdvancedTools({ stream, ask, onReset }) {
  const [open, setOpen] = useState(false)
  const [showRaw, setShowRaw] = useState(false)
  const [demoTestFiles, setDemoTestFiles] = useState({ day1: null, day2: null })
  const activeDemoTestClip = (stream.demoTestSession.clips || []).find((clip) => (
    clip.clipId === stream.demoTestSession.activeClipId
  )) || (stream.demoTestSession.clips || [])[0] || null

  return (
    <section className="advanced-tools">
      <button className="advanced-toggle" type="button" onClick={() => setOpen((value) => !value)}>
        <span>
          <Wrench size={16} />
          Advanced / Tools
        </span>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {open && (
        <div className="surface advanced-panel fade-in">
          <section className="advanced-section">
            <div className="advanced-section-heading">
              <div>
                <span>Input Modes</span>
                <strong>{stream.modeLabel}</strong>
              </div>
            </div>
            <div className="input-mode-switcher" role="radiogroup" aria-label="Realtime input mode">
              <button
                className={stream.inputMode === INPUT_MODES.FRAME_AUDIO ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.FRAME_AUDIO}
                disabled={!stream.canStart}
                onClick={() => stream.setInputMode(INPUT_MODES.FRAME_AUDIO)}
              >
                <Upload size={16} />
                <span>HTTP Frames</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.WEBRTC_WHIP ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.WEBRTC_WHIP}
                disabled={!stream.canStart}
                onClick={() => stream.setInputMode(INPUT_MODES.WEBRTC_WHIP)}
              >
                <Radio size={16} />
                <span>WebRTC</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.ROKID ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.ROKID}
                disabled={!stream.canStart}
                onClick={() => stream.setInputMode(INPUT_MODES.ROKID)}
              >
                <Radio size={16} />
                <span>Rokid</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.ROKID_LIVE_RTMP ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.ROKID_LIVE_RTMP}
                disabled={!stream.canStart}
                onClick={() => stream.setInputMode(INPUT_MODES.ROKID_LIVE_RTMP)}
              >
                <Radio size={16} />
                <span>Rokid RTMP</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.DEMO_VIDEO ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.DEMO_VIDEO}
                disabled={!stream.canStart}
                onClick={() => stream.setInputMode(INPUT_MODES.DEMO_VIDEO)}
              >
                <Film size={16} />
                <span>Demo</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.DEMO_TEST ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.DEMO_TEST}
                disabled={!stream.canStart}
                onClick={() => stream.setInputMode(INPUT_MODES.DEMO_TEST)}
              >
                <Film size={16} />
                <span>Demo Test</span>
              </button>
            </div>
            <p className="mode-description">
              {stream.isDemoTestMode
                ? 'Upload day1 and day2 videos, then replay either day as a simulated cross-day live memory stream.'
                : stream.isLegacyDemoMode
                ? 'Upload a local video, then play it on the main page as a simulated live memory stream.'
                : stream.isRokidMode
                ? 'Rokid Glass uses /rokid/* APIs and keeps the session_id returned by this tab.'
                : stream.isWebRtcMode
                  ? 'Camera and microphone publish through WHIP; the backend live ingest worker extracts frames and audio.'
                  : 'Stable baseline: canvas frames and MediaRecorder chunks upload through HTTP.'}
            </p>
          </section>

          {stream.inputMode === INPUT_MODES.DEMO_VIDEO && (
            <section className="advanced-section">
              <div className="advanced-section-heading">
                <div>
                  <span>Demo Video</span>
                  <strong>{stream.demoSession.sessionId || 'No upload'}</strong>
                </div>
              </div>
              <label className="demo-upload-control">
                <Upload size={16} />
                <span>{stream.demoUploading ? 'Uploading...' : 'Upload video'}</span>
                <input
                  type="file"
                  accept="video/*"
                  disabled={stream.demoUploading || !stream.canStart}
                  onChange={(event) => {
                    const file = event.target.files?.[0]
                    if (file) stream.uploadDemo(file)
                    event.target.value = ''
                  }}
                />
              </label>
              {stream.demoSession.sessionId && (
                <div className="demo-session-summary">
                  <span>{stream.demoSession.prepared ? 'prepared' : 'uploaded'}</span>
                  <span>{formatDuration(stream.demoSession.duration)}</span>
                  <span>{stream.demoSession.frameCount ?? 0} frames</span>
                </div>
              )}
              {stream.demoError && <div className="inline-error">{stream.demoError}</div>}
              <p className="mode-description">After upload, return to Live View and press Start to play the video from the beginning.</p>
            </section>
          )}

          {stream.inputMode === INPUT_MODES.DEMO_TEST && (
            <section className="advanced-section">
              <div className="advanced-section-heading">
                <div>
                  <span>Demo Test</span>
                  <strong>{stream.demoTestSession.sessionId || 'No upload'}</strong>
                </div>
              </div>
              <div className="demo-test-upload-grid">
                <label className="demo-upload-control">
                  <Upload size={16} />
                  <span>{demoTestFiles.day1 ? demoTestFiles.day1.name : 'Choose day1 video'}</span>
                  <input
                    type="file"
                    accept="video/*"
                    disabled={stream.demoTestUploading || !stream.canStart}
                    onChange={(event) => {
                      const file = event.target.files?.[0] || null
                      setDemoTestFiles((current) => ({ ...current, day1: file }))
                    }}
                  />
                </label>
                <label className="demo-upload-control">
                  <Upload size={16} />
                  <span>{demoTestFiles.day2 ? demoTestFiles.day2.name : 'Choose day2 video'}</span>
                  <input
                    type="file"
                    accept="video/*"
                    disabled={stream.demoTestUploading || !stream.canStart}
                    onChange={(event) => {
                      const file = event.target.files?.[0] || null
                      setDemoTestFiles((current) => ({ ...current, day2: file }))
                    }}
                  />
                </label>
              </div>
              <button
                className="icon-button secondary demo-test-upload-button"
                type="button"
                disabled={!demoTestFiles.day1 || !demoTestFiles.day2 || stream.demoTestUploading || !stream.canStart}
                onClick={async () => {
                  const info = await stream.uploadDemoTest(demoTestFiles.day1, demoTestFiles.day2)
                  if (info) setDemoTestFiles({ day1: null, day2: null })
                }}
              >
                <Upload size={16} />
                <span>{stream.demoTestUploading ? 'Uploading...' : 'Upload demo-test videos'}</span>
              </button>

              {stream.demoTestSession.sessionId && (
                <>
                  <div className="demo-clip-switcher" role="radiogroup" aria-label="Demo-test clip">
                    {stream.demoTestSession.clips.map((clip) => (
                      <button
                        key={clip.clipId}
                        className={stream.demoTestSession.activeClipId === clip.clipId ? 'active' : ''}
                        type="button"
                        role="radio"
                        aria-checked={stream.demoTestSession.activeClipId === clip.clipId}
                        disabled={!stream.canStart && !stream.isPaused}
                        onClick={() => stream.setDemoTestActiveClip(clip.clipId)}
                      >
                        <span>{clip.clipId === 'day2' ? 'Day 2' : 'Day 1'}</span>
                        <small>{clip.displayDate} {clip.startTime}</small>
                      </button>
                    ))}
                  </div>
                  <div className="demo-session-summary">
                    <span>{stream.demoTestSession.prepared ? 'prepared' : 'uploaded'}</span>
                    <span>{stream.demoTestSession.memoryReady ? 'memory ready' : 'memory pending'}</span>
                    <span>{activeDemoTestClip ? `${activeDemoTestClip.displayDate} ${activeDemoTestClip.startTime}` : 'clip selected'}</span>
                  </div>
                  <div className="demo-test-actions">
                    <button className="icon-button secondary" type="button" onClick={stream.refreshDemoTestStatus} disabled={stream.demoTestBusy}>
                      <RefreshCw size={16} />
                      <span>Status</span>
                    </button>
                    <button className="icon-button secondary" type="button" onClick={() => stream.enqueueDemoTestMemory({ enqueueEvidence: false })} disabled={stream.demoTestBusy}>
                      <span>Queue preprocess</span>
                    </button>
                    <button className="icon-button secondary" type="button" onClick={() => stream.enqueueDemoTestMemory({ enqueueEvidence: true })} disabled={stream.demoTestBusy}>
                      <span>Queue evidence</span>
                    </button>
                    <button className="icon-button secondary" type="button" onClick={() => stream.buildDemoTestParentMemory({ force: true })} disabled={stream.demoTestBusy}>
                      <span>Build memory</span>
                    </button>
                  </div>
                </>
              )}
              {stream.demoError && <div className="inline-error">{stream.demoError}</div>}
              <p className="mode-description">After upload, pick Day 1 or Day 2, return to Live View, and press Start.</p>
            </section>
          )}

          <section className="advanced-section">
            <div className="advanced-section-heading">
              <div>
                <span>Answer Mode</span>
                <strong>{ask.answerMode === ANSWER_MODES.STREAM ? 'Progressive stream' : 'Legacy polling'}</strong>
              </div>
            </div>
            <div className="input-mode-switcher answer-mode-switcher" role="radiogroup" aria-label="Answer mode">
              <button
                className={ask.answerMode === ANSWER_MODES.LEGACY ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={ask.answerMode === ANSWER_MODES.LEGACY}
                disabled={ask.loading}
                onClick={() => ask.setAnswerMode(ANSWER_MODES.LEGACY)}
              >
                <span>Legacy</span>
              </button>
              <button
                className={ask.answerMode === ANSWER_MODES.STREAM ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={ask.answerMode === ANSWER_MODES.STREAM}
                disabled={ask.loading}
                onClick={() => ask.setAnswerMode(ANSWER_MODES.STREAM)}
              >
                <span>Stream</span>
              </button>
            </div>
            <p className="mode-description">
              {ask.answerMode === ANSWER_MODES.STREAM
                ? 'Shows a draft and token deltas, then reconciles with the final result JSON.'
                : 'Uses the stable queued task flow and waits for the complete result.'}
            </p>
          </section>


          <section className="advanced-section">
            <div className="advanced-section-heading">
              <div>
                <span>Retrieval Scheme</span>
                <strong>{ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.WORLDMM_LEGACY ? 'WorldMM legacy' : 'EM2Memory'}</strong>
              </div>
            </div>
            <div className="input-mode-switcher answer-mode-switcher" role="radiogroup" aria-label="Long-term retrieval scheme">
              <button
                className={ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.EM2MEMORY ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.EM2MEMORY}
                disabled={ask.loading}
                onClick={() => ask.setLongTermRetrievalScheme(LONG_TERM_RETRIEVAL_SCHEMES.EM2MEMORY)}
              >
                <span>EM2Memory</span>
              </button>
              <button
                className={ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.WORLDMM_LEGACY ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.WORLDMM_LEGACY}
                disabled={ask.loading}
                onClick={() => ask.setLongTermRetrievalScheme(LONG_TERM_RETRIEVAL_SCHEMES.WORLDMM_LEGACY)}
              >
                <span>Legacy</span>
              </button>
            </div>
            <p className="mode-description">Selects the long-term memory retrieval path used by ask requests.</p>
          </section>

          <div className="tools-row">
            <button className="icon-button secondary" type="button" onClick={stream.refreshStatus} disabled={!stream.sessionId || stream.diagnosticsLoading}>
              <RefreshCw size={16} />
              <span>{stream.diagnosticsLoading ? 'Refreshing' : 'Refresh status'}</span>
            </button>
            <button className="icon-button secondary danger-text" type="button" onClick={onReset}>
              <RotateCcw size={16} />
              <span>Reset session</span>
            </button>
          </div>

          <div className="diagnostic-id-list">
            <div>
              <span>session_id</span>
              <strong>{stream.sessionId || '-'}</strong>
            </div>
            <div>
              <span>owner_id</span>
              <strong>{stream.activeSession.ownerId || '-'}</strong>
            </div>
            <div>
              <span>device_id</span>
              <strong>{stream.activeSession.deviceId || '-'}</strong>
            </div>
            <div>
              <span>device_type</span>
              <strong>{stream.activeSession.deviceType || stream.activeSession.deviceKind || '-'}</strong>
            </div>
            <div>
              <span>query_task</span>
              <strong>{ask.queryTaskId || '-'}</strong>
            </div>
            <div>
              <span>answer_mode</span>
              <strong>{ask.answerMode || '-'}</strong>
            </div>
            <div>
              <span>answer_phase</span>
              <strong>{ask.answerPhase || '-'}</strong>
            </div>
            <div>
              <span>mode</span>
              <strong>{stream.activeInputMode || 'not started'}</strong>
            </div>
            {stream.isLegacyDemoMode && (
              <div>
                <span>demo_session</span>
                <strong>{stream.demoSession.sessionId || '-'}</strong>
              </div>
            )}
            {stream.isDemoTestMode && (
              <>
                <div>
                  <span>demo_test_session</span>
                  <strong>{stream.demoTestSession.sessionId || '-'}</strong>
                </div>
                <div>
                  <span>demo_test_clip</span>
                  <strong>{stream.demoTestSession.activeClipId || '-'}</strong>
                </div>
              </>
            )}
            <div>
              <span>stream</span>
              <strong>{stream.status}</strong>
            </div>
            {stream.isWebRtcMode && (
              <>
                <div>
                  <span>webrtc_state</span>
                  <strong>{stream.webrtcState}</strong>
                </div>
                <div>
                  <span>stream_name</span>
                  <strong>{stream.streamName || '-'}</strong>
                </div>
                <div>
                  <span>WHIP URL</span>
                  <strong>{stream.whipUrlAvailable ? 'available' : 'missing'}</strong>
                </div>
                <div>
                  <span>live_ingest</span>
                  <strong>{stream.liveIngestStatus}</strong>
                </div>
              </>
            )}
          </div>

          {stream.isWebRtcMode ? (
            <div className="advanced-stat-grid">
              <Stat label="WebRTC" value={stream.webrtcState} tone={stream.webrtcState === 'live' ? 'ready' : 'warn'} />
              <Stat label="Live ingest" value={stream.liveIngestStatus} tone={stream.liveIngestStatus === 'running' ? 'ready' : 'warn'} />
              <Stat label="Frames ingested" value={stream.liveIngestFrames} tone="ready" />
              <Stat label="Audio ingested" value={stream.liveIngestAudioChunks} tone="ready" />
            </div>
          ) : (
            <div className="advanced-stat-grid">
              <Stat label="Frame index" value={stream.stats.frameIndex} />
              <Stat label="Frame uploaded" value={stream.stats.frameUploadedCount} tone="ready" />
              <Stat label="Frame dropped" value={stream.stats.frameDroppedCount} tone="warn" />
              <Stat label="Frame failed" value={stream.stats.frameFailedCount} tone="bad" />
              <Stat label="Audio index" value={stream.stats.audioIndex} />
              <Stat label="Audio uploaded" value={stream.stats.audioUploadedCount} tone="ready" />
              <Stat label="Audio failed" value={stream.stats.audioFailedCount} tone="bad" />
              <Stat label="Can ask" value={stream.canAsk ? 'yes' : 'not yet'} tone={stream.canAsk ? 'ready' : 'warn'} />
            </div>
          )}

          {stream.liveIngestLastError && <div className="inline-error">{stream.liveIngestLastError}</div>}

          {(stream.statusSnapshot || stream.statusError || ask.rawDebug) && (
            <div className="raw-tools">
              <button className="icon-button secondary" type="button" onClick={() => setShowRaw((value) => !value)}>
                <Bug size={16} />
                <span>{showRaw ? 'Hide raw' : 'Show raw'}</span>
              </button>
              {stream.statusError && <div className="inline-error">{stream.statusError}</div>}
              {showRaw && (
                <pre className="raw-box">
                  {JSON.stringify({
                    streamStart: stream.streamInfo,
                    streamStatus: stream.statusSnapshot,
                    demoTest: stream.demoTestSession.status,
                    ask: ask.rawDebug
                  }, null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

function Stat({ label, value, tone = '' }) {
  return (
    <div className={`stat-chip ${tone}`}>
      <span>{label}</span>
      <strong>{value ?? '-'}</strong>
    </div>
  )
}

function formatDuration(value) {
  const seconds = Number(value)
  if (!Number.isFinite(seconds) || seconds <= 0) return 'duration -'
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.round(seconds % 60)
  return `${minutes}:${String(remainder).padStart(2, '0')}`
}
