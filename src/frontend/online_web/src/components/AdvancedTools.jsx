import {
  Bug,
  ChevronDown,
  ChevronUp,
  Film,
  Images,
  Music,
  Radio,
  RefreshCw,
  RotateCcw,
  Upload,
  Wrench
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { useLanguage } from '../i18n.jsx'
import { ANSWER_MODES, LONG_TERM_RETRIEVAL_SCHEMES } from '../hooks/useAskLightMemEgo.js'
import { INPUT_MODES } from '../hooks/useRealtimeStream.js'

export default function AdvancedTools({ stream, ask, onReset }) {
  const { t } = useLanguage()
  const [open, setOpen] = useState(false)
  const [showRaw, setShowRaw] = useState(false)
  const [demoTestFiles, setDemoTestFiles] = useState({ day1: null, day2: null })
  const [clockNow, setClockNow] = useState(() => Date.now())
  useEffect(() => {
    if (!stream.imageUploadNextAt) return undefined
    const timer = window.setInterval(() => setClockNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [stream.imageUploadNextAt])
  const activeDemoTestClip = (stream.demoTestSession.clips || []).find((clip) => (
    clip.clipId === stream.demoTestSession.activeClipId
  )) || (stream.demoTestSession.clips || [])[0] || null

  return (
    <section className="advanced-tools">
      <button className="advanced-toggle" type="button" aria-expanded={open} aria-controls="capture-tools-panel" onClick={() => setOpen((value) => !value)}>
        <span>
          <Wrench size={16} />
          {t('Advanced / Tools')}
        </span>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {open && (
        <div id="capture-tools-panel" className="surface advanced-panel fade-in">
          <section className="advanced-section">
            <div className="advanced-section-heading">
              <div>
                <span>{t('Input Modes')}</span>
                <strong>{t(stream.modeLabel)}</strong>
              </div>
            </div>
            <div className="input-mode-switcher" role="radiogroup" aria-label={t("Realtime input mode")}>
              <button
                className={stream.inputMode === INPUT_MODES.FRAME_AUDIO ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.FRAME_AUDIO}
                disabled={!stream.canChangeInputMode}
                onClick={() => stream.setInputMode(INPUT_MODES.FRAME_AUDIO)}
              >
                <Upload size={16} />
                <span>{t('HTTP Frames')}</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.WEBRTC_WHIP ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.WEBRTC_WHIP}
                disabled={!stream.canChangeInputMode}
                onClick={() => stream.setInputMode(INPUT_MODES.WEBRTC_WHIP)}
              >
                <Radio size={16} />
                <span>{t("WebRTC")}</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.ROKID ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.ROKID}
                disabled={!stream.canChangeInputMode}
                onClick={() => stream.setInputMode(INPUT_MODES.ROKID)}
              >
                <Radio size={16} />
                <span>{t("Rokid")}</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.ROKID_LIVE_RTMP ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.ROKID_LIVE_RTMP}
                disabled={!stream.canChangeInputMode}
                onClick={() => stream.setInputMode(INPUT_MODES.ROKID_LIVE_RTMP)}
              >
                <Radio size={16} />
                <span>{t("Rokid RTMP")}</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.DEMO_VIDEO ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.DEMO_VIDEO}
                disabled={!stream.canChangeInputMode}
                onClick={() => stream.setInputMode(INPUT_MODES.DEMO_VIDEO)}
              >
                <Film size={16} />
                <span>{t("Demo")}</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.DEMO_TEST ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.DEMO_TEST}
                disabled={!stream.canChangeInputMode}
                onClick={() => stream.setInputMode(INPUT_MODES.DEMO_TEST)}
              >
                <Film size={16} />
                <span>{t("Demo Test")}</span>
              </button>
              <button
                className={stream.inputMode === INPUT_MODES.IMAGE_UPLOAD ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={stream.inputMode === INPUT_MODES.IMAGE_UPLOAD}
                disabled={!stream.canChangeInputMode}
                onClick={() => stream.setInputMode(INPUT_MODES.IMAGE_UPLOAD)}
              >
                <Images size={16} />
                <span>{t('Photo / Audio Stream')}</span>
              </button>
            </div>
            <p className="mode-description">
              {stream.isDemoTestMode
                ? t("Upload day1 and day2 videos, then replay either day as a simulated cross-day live memory stream.")
                : stream.isLegacyDemoMode
                ? t("Upload a local video, then play it on the main page as a simulated live memory stream.")
                : stream.isRokidLiveMode
                ? t('Connect to the active Rokid Glass RTMP session. The backend extracts frames and audio; web preview stays black.')
                : stream.isRokidMode
                ? t("Rokid Glass uses /rokid/* APIs and keeps the session_id returned by this tab.")
                : stream.isWebRtcMode
                  ? t("Camera and microphone publish through WHIP; the backend live ingest worker extracts frames and audio.")
                  : stream.isImageUploadMode
                    ? t("Upload local images and audio. By default, the next item starts as soon as the previous analysis is acknowledged.")
                    : t("Stable baseline: canvas frames and MediaRecorder chunks upload through HTTP.")}
            </p>
          </section>

          {stream.inputMode === INPUT_MODES.IMAGE_UPLOAD && (
            <section className="advanced-section photo-stream-section">
              <div className="advanced-section-heading">
                <div>
                  <span>{t("Photo / Audio Stream")}</span>
                  <strong>{stream.imageUploadQueue.length ? t("{value1} media files selected", { value1: stream.imageUploadQueue.length }) : t("No media selected")}</strong>
                </div>
              </div>
              <label className="demo-upload-control photo-stream-picker">
                <Images size={16} />
                <span>{t("Select images or audio")}</span>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp,image/bmp,audio/mpeg,audio/wav,audio/ogg,audio/mp4,audio/x-m4a,audio/aac,audio/webm,audio/flac"
                  multiple
                  disabled={!stream.canChangeInputMode || ['starting', 'running', 'uploading', 'paused'].includes(stream.imageUploadStatus)}
                  onChange={(event) => {
                    stream.setImageUploadFiles(event.target.files)
                    event.target.value = ''
                  }}
                />
              </label>
              {stream.imageUploadQueue.length > 0 && (
                <div className="photo-stream-queue">
                  {stream.imageUploadQueue.map((item, index) => (
                    <div className={`photo-stream-item ${index < stream.imageUploadIndex ? 'uploaded' : ''}`} key={`${item.name}-${index}`}>
                      {item.kind === 'image' ? <img src={item.previewUrl} alt="" /> : <div className="photo-stream-audio-thumb"><Music size={20} /></div>}
                      <div className="photo-stream-item-meta">
                        <strong>{index + 1}. {item.name}</strong>
                        <small>{item.kind === 'image' ? `${item.width} × ${item.height} · ` : t("Audio · ")}{formatBytes(item.size)}</small>
                      </div>
                      <button
                        className="icon-button secondary"
                        type="button"
                        aria-label={t("Remove {value1}", { value1: item.name })}
                        disabled={stream.imageUploadStatus === 'uploading' || stream.imageUploadStatus === 'running'}
                        onClick={() => stream.removeImageUploadFile(index)}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <div className="photo-stream-summary">
                <span>{t("Uploaded")} {Math.min(stream.imageUploadIndex, stream.imageUploadQueue.length)} / {stream.imageUploadQueue.length}</span>
                <span>{stream.imageUploadStatus === 'completed' ? t("Completed") : t(stream.imageUploadStatus)}</span>
                {stream.imageUploadNextAt > 0 && <span>{t("Next upload in")} {formatCountdown(stream.imageUploadNextAt, clockNow)}</span>}
              </div>
              <label className="photo-stream-no-wait-toggle">
                <input
                  type="checkbox"
                  checked={stream.imageUploadNoWait}
                  disabled={['starting', 'running', 'uploading', 'paused'].includes(stream.imageUploadStatus)}
                  onChange={(event) => stream.setImageUploadNoWait(event.target.checked)}
                />
                <span>{t("No enforced wait between media items")}</span>
                <small>{stream.imageUploadNoWait ? t("Upload next after the previous item is analyzed.") : t("Wait 60 seconds between uploads.")}</small>
              </label>
              {stream.imageUploadError && <div className="inline-error">{t(stream.imageUploadError)}</div>}
              <div className="photo-stream-actions">
                <button className="icon-button secondary" type="button" onClick={stream.startImageUpload} disabled={!stream.canStartSelectedMode}>
                  <span>{t("Start Photo / Audio Stream")}</span>
                </button>
                <button className="icon-button secondary" type="button" onClick={stream.pause} disabled={!stream.canPause || !stream.isImageUploadMode}>
                  <span>{t("Pause")}</span>
                </button>
                <button className="icon-button secondary" type="button" onClick={stream.resume} disabled={!stream.isPaused || !stream.isImageUploadMode}>
                  <span>{t("Resume")}</span>
                </button>
                <button className="icon-button secondary" type="button" onClick={stream.retryImageUpload} disabled={stream.imageUploadStatus !== 'error'}>
                  <RefreshCw size={16} />
                  <span>{t("Retry")}</span>
                </button>
                <button className="icon-button secondary" type="button" onClick={stream.clearImageUploadFiles} disabled={!stream.imageUploadQueue.length || stream.imageUploadStatus === 'uploading'}>
                  <RotateCcw size={16} />
                  <span>{t("Clear Queue")}</span>
                </button>
              </div>
              {!stream.imageUploadQueue.length && <p className="mode-description">{t("Select at least one image or audio file to start.")}</p>}
            </section>
          )}

          {stream.inputMode === INPUT_MODES.DEMO_VIDEO && (
            <section className="advanced-section">
              <div className="advanced-section-heading">
                <div>
                  <span>{t("Demo Video")}</span>
                  <strong>{stream.demoSession.sessionId || t("No upload")}</strong>
                </div>
              </div>
              <label className="demo-upload-control">
                <Upload size={16} />
                <span>{stream.demoUploading ? t("Uploading...") : t("Upload video")}</span>
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
                  <span>{stream.demoSession.prepared ? t("prepared") : t("uploaded")}</span>
                  <span>{t(formatDuration(stream.demoSession.duration))}</span>
                  <span>{stream.demoSession.frameCount ?? 0} {t("frames")}</span>
                </div>
              )}
              {stream.demoError && <div className="inline-error">{t(stream.demoError)}</div>}
              <p className="mode-description">{t("After upload, return to Live View and press Start to play the video from the beginning.")}</p>
            </section>
          )}

          {stream.inputMode === INPUT_MODES.DEMO_TEST && (
            <section className="advanced-section">
              <div className="advanced-section-heading">
                <div>
                  <span>{t("Demo Test")}</span>
                  <strong>{stream.demoTestSession.sessionId || t("No upload")}</strong>
                </div>
              </div>
              <div className="demo-test-upload-grid">
                <label className="demo-upload-control">
                  <Upload size={16} />
                  <span>{demoTestFiles.day1 ? demoTestFiles.day1.name : t("Choose day1 video")}</span>
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
                  <span>{demoTestFiles.day2 ? demoTestFiles.day2.name : t("Choose day2 video")}</span>
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
                <span>{stream.demoTestUploading ? t("Uploading...") : t("Upload demo-test videos")}</span>
              </button>

              {stream.demoTestSession.sessionId && (
                <>
                  <div className="demo-clip-switcher" role="radiogroup" aria-label={t("Demo-test clip")}>
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
                        <span>{clip.clipId === 'day2' ? t("Day 2") : t("Day 1")}</span>
                        <small>{clip.displayDate} {clip.startTime}</small>
                      </button>
                    ))}
                  </div>
                  <div className="demo-session-summary">
                    <span>{stream.demoTestSession.prepared ? t("prepared") : t("uploaded")}</span>
                    <span>{stream.demoTestSession.memoryReady ? t("memory ready") : t("memory pending")}</span>
                    <span>{activeDemoTestClip ? `${activeDemoTestClip.displayDate} ${activeDemoTestClip.startTime}` : t("clip selected")}</span>
                  </div>
                  <div className="demo-test-actions">
                    <button className="icon-button secondary" type="button" onClick={stream.refreshDemoTestStatus} disabled={stream.demoTestBusy}>
                      <RefreshCw size={16} />
                      <span>{t("Status")}</span>
                    </button>
                    <button className="icon-button secondary" type="button" onClick={() => stream.enqueueDemoTestMemory({ enqueueEvidence: false })} disabled={stream.demoTestBusy}>
                      <span>{t("Queue preprocess")}</span>
                    </button>
                    <button className="icon-button secondary" type="button" onClick={() => stream.enqueueDemoTestMemory({ enqueueEvidence: true })} disabled={stream.demoTestBusy}>
                      <span>{t("Queue evidence")}</span>
                    </button>
                    <button className="icon-button secondary" type="button" onClick={() => stream.buildDemoTestParentMemory({ force: true })} disabled={stream.demoTestBusy}>
                      <span>{t("Build memory")}</span>
                    </button>
                  </div>
                </>
              )}
              {stream.demoError && <div className="inline-error">{t(stream.demoError)}</div>}
              <p className="mode-description">{t("After upload, pick Day 1 or Day 2, return to Live View, and press Start.")}</p>
            </section>
          )}

          <section className="advanced-section">
            <div className="advanced-section-heading">
              <div>
                <span>{t('Answer Mode')}</span>
                <strong>{ask.answerMode === ANSWER_MODES.STREAM ? t("Progressive stream") : t("Legacy polling")}</strong>
              </div>
            </div>
            <div className="input-mode-switcher answer-mode-switcher" role="radiogroup" aria-label={t("Answer mode")}>
              <button
                className={ask.answerMode === ANSWER_MODES.LEGACY ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={ask.answerMode === ANSWER_MODES.LEGACY}
                disabled={ask.loading}
                onClick={() => ask.setAnswerMode(ANSWER_MODES.LEGACY)}
              >
                <span>{t('Legacy')}</span>
              </button>
              <button
                className={ask.answerMode === ANSWER_MODES.STREAM ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={ask.answerMode === ANSWER_MODES.STREAM}
                disabled={ask.loading}
                onClick={() => ask.setAnswerMode(ANSWER_MODES.STREAM)}
              >
                <span>{t('Stream')}</span>
              </button>
            </div>
            <p className="mode-description">
              {ask.answerMode === ANSWER_MODES.STREAM
                ? t("Shows a draft and token deltas, then reconciles with the final result JSON.")
                : t("Uses the stable queued task flow and waits for the complete result.")}
            </p>
          </section>


          <section className="advanced-section">
            <div className="advanced-section-heading">
              <div>
                <span>{t('Retrieval Scheme')}</span>
                <strong>{ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.LIGHTMEM_EGO_LEGACY ? t("LightMemEgo legacy") : t("EM2Memory")}</strong>
              </div>
            </div>
            <div className="input-mode-switcher answer-mode-switcher" role="radiogroup" aria-label={t("Long-term retrieval scheme")}>
              <button
                className={ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.EM2MEMORY ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.EM2MEMORY}
                disabled={ask.loading}
                onClick={() => ask.setLongTermRetrievalScheme(LONG_TERM_RETRIEVAL_SCHEMES.EM2MEMORY)}
              >
                <span>{t("EM2Memory")}</span>
              </button>
              <button
                className={ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.LIGHTMEM_EGO_LEGACY ? 'active' : ''}
                type="button"
                role="radio"
                aria-checked={ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.LIGHTMEM_EGO_LEGACY}
                disabled={ask.loading}
                onClick={() => ask.setLongTermRetrievalScheme(LONG_TERM_RETRIEVAL_SCHEMES.LIGHTMEM_EGO_LEGACY)}
              >
                <span>{t('Legacy')}</span>
              </button>
            </div>
            <p className="mode-description">{t(ask.longTermRetrievalScheme === LONG_TERM_RETRIEVAL_SCHEMES.LIGHTMEM_EGO_LEGACY
              ? 'LightMemEgo legacy retrieves long-term memory using the original LightMemEgo scheme.'
              : 'EM2Memory retrieves long-term memory using the EM2Memory scheme.')}</p>
          </section>

          <div className="tools-row">
            <button className="icon-button secondary" type="button" onClick={stream.refreshStatus} disabled={!stream.sessionId || stream.diagnosticsLoading}>
              <RefreshCw size={16} />
              <span>{stream.diagnosticsLoading ? t('Refreshing') : t('Refresh status')}</span>
            </button>
            <button className="icon-button secondary danger-text" type="button" onClick={onReset}>
              <RotateCcw size={16} />
              <span>{t('Reset session')}</span>
            </button>
          </div>

          <div className="diagnostic-id-list">
            <div>
              <span>{t("session_id")}</span>
              <strong>{stream.sessionId || '-'}</strong>
            </div>
            <div>
              <span>{t("owner_id")}</span>
              <strong>{stream.activeSession.ownerId || '-'}</strong>
            </div>
            <div>
              <span>{t("device_id")}</span>
              <strong>{stream.activeSession.deviceId || '-'}</strong>
            </div>
            <div>
              <span>{t("device_type")}</span>
              <strong>{stream.activeSession.deviceType || stream.activeSession.deviceKind || '-'}</strong>
            </div>
            <div>
              <span>{t("query_task")}</span>
              <strong>{ask.queryTaskId || '-'}</strong>
            </div>
            <div>
              <span>{t("answer_mode")}</span>
              <strong>{t(ask.answerMode) || '-'}</strong>
            </div>
            <div>
              <span>{t("answer_phase")}</span>
              <strong>{t(ask.answerPhase) || '-'}</strong>
            </div>
            <div>
              <span>{t("mode")}</span>
              <strong>{t(stream.activeInputMode) || t('not started')}</strong>
            </div>
            {stream.isLegacyDemoMode && (
              <div>
                <span>{t("demo_session")}</span>
                <strong>{stream.demoSession.sessionId || '-'}</strong>
              </div>
            )}
            {stream.isDemoTestMode && (
              <>
                <div>
                  <span>{t("demo_test_session")}</span>
                  <strong>{stream.demoTestSession.sessionId || '-'}</strong>
                </div>
                <div>
                  <span>{t("demo_test_clip")}</span>
                  <strong>{stream.demoTestSession.activeClipId || '-'}</strong>
                </div>
              </>
            )}
            <div>
              <span>{t("stream")}</span>
              <strong>{t(stream.status)}</strong>
            </div>
            {stream.isWebRtcMode && (
              <>
                <div>
                  <span>{t("webrtc_state")}</span>
                  <strong>{t(stream.webrtcState)}</strong>
                </div>
                <div>
                  <span>{t("stream_name")}</span>
                  <strong>{stream.streamName || '-'}</strong>
                </div>
                <div>
                  <span>{t("WHIP URL")}</span>
                  <strong>{stream.whipUrlAvailable ? t("available") : t("missing")}</strong>
                </div>
                <div>
                  <span>{t("live_ingest")}</span>
                  <strong>{t(stream.liveIngestStatus)}</strong>
                </div>
              </>
            )}
          </div>

          {stream.isWebRtcMode ? (
            <div className="advanced-stat-grid">
              <Stat label={t("WebRTC")} value={t(stream.webrtcState)} tone={stream.webrtcState === 'live' ? 'ready' : 'warn'} />
              <Stat label={t("Live ingest")} value={t(stream.liveIngestStatus)} tone={stream.liveIngestStatus === 'running' ? 'ready' : 'warn'} />
              <Stat label={t("Frames ingested")} value={stream.liveIngestFrames} tone="ready" />
              <Stat label={t("Audio ingested")} value={stream.liveIngestAudioChunks} tone="ready" />
            </div>
          ) : (
            <div className="advanced-stat-grid">
              <Stat label={t("Frame index")} value={stream.stats.frameIndex} />
              <Stat label={t("Frame uploaded")} value={stream.stats.frameUploadedCount} tone="ready" />
              <Stat label={t("Frame dropped")} value={stream.stats.frameDroppedCount} tone="warn" />
              <Stat label={t("Frame failed")} value={stream.stats.frameFailedCount} tone="bad" />
              <Stat label={t("Audio index")} value={stream.stats.audioIndex} />
              <Stat label={t("Audio uploaded")} value={stream.stats.audioUploadedCount} tone="ready" />
              <Stat label={t("Audio failed")} value={stream.stats.audioFailedCount} tone="bad" />
              <Stat label={t("Can ask")} value={stream.canAsk ? 'yes' : 'not yet'} tone={stream.canAsk ? 'ready' : 'warn'} />
            </div>
          )}

          {stream.liveIngestLastError && <div className="inline-error">{t(stream.liveIngestLastError)}</div>}

          {(stream.statusSnapshot || stream.statusError || ask.rawDebug) && (
            <div className="raw-tools">
              <button className="icon-button secondary" type="button" onClick={() => setShowRaw((value) => !value)}>
                <Bug size={16} />
                <span>{showRaw ? t('Hide raw') : t('Show raw')}</span>
              </button>
              {stream.statusError && <div className="inline-error">{t(stream.statusError)}</div>}
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
  const { t } = useLanguage()
  return (
    <div className={`stat-chip ${tone}`}>
      <span>{t(label)}</span>
      <strong>{typeof value === 'string' ? t(value) : value ?? '-'}</strong>
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

function formatBytes(bytes) {
  const value = Number(bytes) || 0
  if (value < 1024 * 1024) return `${Math.max(1, Math.round(value / 1024))} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

function formatCountdown(timestamp, now = Date.now()) {
  const remaining = Math.max(0, Number(timestamp) - now)
  const seconds = Math.ceil(remaining / 1000)
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}
