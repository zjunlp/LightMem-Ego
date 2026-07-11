import {
  Camera,
  Mic,
  Pause,
  Play,
  Radio,
  RotateCcw,
  Square,
  SwitchCamera,
  Video
} from 'lucide-react'

export default function LiveView({ stream, onOpenAsk, onReset }) {
  const statusLabel = getStatusLabel(stream.status)
  const hideHeroOverlay = stream.isDemoTestMode && !stream.isLive && !stream.isPaused && stream.status !== 'starting'
  const hideDemoVideoBeforeStart = stream.isDemoTestMode && !stream.isLive && !stream.isPaused && stream.status !== 'starting'
  // Rokid RTMP frontend preview is intentionally disabled; backend ingest still receives the glasses video.

  return (
    <section className="live-view">
      <header className="top-header">
        <div>
          <div className="brand">LightMem-Ego</div>
          <div className="subtitle">Online Video Memory</div>
        </div>
        <div className={`status-pill status-${stream.status}`}>
          <span className={`live-dot ${stream.isLive ? 'pulse' : ''}`} />
          <span>{statusLabel}</span>
        </div>
      </header>

      <div className={`video-hero state-${stream.status} ${hideDemoVideoBeforeStart ? 'demo-test-idle' : ''}`}>

        {stream.isDemoMode ? (
          <video
            ref={stream.videoRef}
            className={`live-video demo-video ${hideDemoVideoBeforeStart ? 'demo-video-hidden' : ''}`}
            playsInline
            muted
            preload="metadata"
          />
        ) : stream.isRokidMode && stream.isRokidLiveMode ? (
          <div className="live-video rokid-preview-pending rokid-rtmp-blackout" aria-hidden="true" />
        ) : stream.isRokidMode ? (
          stream.rokidPreviewUrl ? (
            <img
              className="live-video rokid-preview-image"
              src={stream.rokidPreviewUrl}
              alt="Rokid Glass live preview"
            />
          ) : (
            <div className="live-video rokid-preview-pending" aria-hidden="true" />
          )
        ) : (
          <video
            ref={stream.videoRef}
            className="live-video"
            autoPlay
            playsInline
            muted
          />
        )}
        <canvas ref={stream.canvasRef} className="capture-canvas" aria-hidden="true" />

        {!stream.isLive && !stream.isPaused && stream.status !== 'starting' && !(stream.isLegacyDemoMode && stream.demoSession.sessionId) && !(stream.isDemoTestMode && stream.demoTestSession.sessionId) && (
          <div className="hero-empty">
            <div className="hero-empty-mark">
              {stream.isWebRtcMode || stream.isRokidLiveMode ? <Radio size={36} /> : <Video size={36} />}
            </div>
            <h1>{stream.status === 'stopped' ? 'Realtime understanding stopped' : 'Start realtime understanding'}</h1>
            <p>{stream.isDemoMode ? ((stream.isDemoTestMode ? stream.demoTestSession.sessionId : stream.demoSession.sessionId) ? 'Demo video is ready. Start to play it as a live memory stream.' : 'Upload demo video assets from Advanced / Tools.') : (stream.isRokidLiveMode ? 'Rokid RTMP is ingesting on the backend. Web preview is intentionally black.' : (stream.isWebRtcMode ? 'Publish camera and microphone through WHIP.' : 'Live video memory is idle.'))}</p>
          </div>
        )}

        {!hideHeroOverlay && (
          <div className="hero-overlay">
            <div className="hero-overlay-top">
              <div className="glass-pill">
                <span className={`live-dot ${stream.isLive ? 'pulse' : ''}`} />
                <span>{statusLabel}</span>
              </div>
              <div className="glass-pill">{stream.isLegacyDemoMode ? 'Demo Video' : (stream.isRokidMode ? (stream.isRokidLiveMode ? 'Rokid RTMP' : 'Rokid Glass') : (stream.isWebRtcMode ? 'WebRTC' : 'Frame / Audio'))}</div>
            </div>

            {!stream.isDemoTestMode && (
              <div className="hero-overlay-bottom">
                {stream.isLegacyDemoMode ? (
                  <>
                    <div className="signal-chip">
                      <Video size={15} />
                      <span>{stream.demoSession.prepared ? 'Prepared' : 'Uploaded'}</span>
                    </div>
                    <div className="signal-chip">
                      <Camera size={15} />
                      <span>{stream.demoSession.frameCount ? `${stream.demoSession.frameCount} frames` : 'Tick ready'}</span>
                    </div>
                  </>
                ) : (stream.isWebRtcMode || stream.isRokidLiveMode) ? (
                  <>
                    <div className="signal-chip">
                      <Radio size={15} />
                      <span>{stream.webrtcState}</span>
                    </div>
                    <div className="signal-chip">
                      <Video size={15} />
                      <span>Ingest {stream.liveIngestStatus}</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="signal-chip">
                      <Camera size={15} />
                      <span>{stream.isRokidMode ? `HTTP preview ${formatFps(stream.rokidPreviewFps)}` : (stream.stats.firstFrameUploaded ? `${stream.stats.frameUploadedCount} frames` : 'Frame ready')}</span>
                    </div>
                    <div className="signal-chip">
                      <Mic size={15} />
                      <span>{stream.isRokidMode ? `Memory sampling ${formatFps(stream.rokidMemoryFps || stream.rokidMemoryTargetFps)}` : (stream.stats.audioUploadedCount ? `${stream.stats.audioUploadedCount} audio` : 'Audio ready')}</span>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {stream.isDemoTestMode && stream.demoClockText && (
          <div className="demo-clock-overlay">{stream.demoClockText}</div>
        )}
      </div>

      {(stream.streamError || stream.audioError) && (
        <div className="live-error">
          {stream.streamError || stream.audioError}
        </div>
      )}

      <div className="primary-control">
        {stream.canStart && (
          <button className="main-action" type="button" onClick={stream.start}>
            <Play size={20} fill="currentColor" />
            <span>{stream.status === 'stopped' ? 'Restart' : 'Start Live Understanding'}</span>
          </button>
        )}

        {stream.status === 'starting' && (
          <button className="main-action shimmer" type="button" disabled>
            <span className="button-spinner" />
            <span>Starting</span>
          </button>
        )}

        {stream.isLive && (
          <button className="main-action stop-action" type="button" onClick={stream.stop}>
            <Square size={18} fill="currentColor" />
            <span>Stop Live Understanding</span>
          </button>
        )}

        {stream.isPaused && (
          <button className="main-action" type="button" onClick={stream.resume}>
            <Play size={20} fill="currentColor" />
            <span>Resume Live Understanding</span>
          </button>
        )}

        {stream.status === 'stopping' && (
          <button className="main-action shimmer" type="button" disabled>
            <span className="button-spinner" />
            <span>Stopping</span>
          </button>
        )}
      </div>

      <div className="secondary-controls">
        <button className="icon-button secondary" type="button" onClick={stream.pause} disabled={!stream.canPause}>
          <Pause size={16} />
          <span>Pause</span>
        </button>
        <button className="icon-button secondary" type="button" onClick={stream.toggleCamera} disabled={!stream.canFlipCamera}>
          <SwitchCamera size={16} />
          <span>{stream.cameraSwitching ? 'Switching' : stream.cameraFacingLabel}</span>
        </button>
        <button className="icon-button secondary" type="button" onClick={onReset}>
          <RotateCcw size={16} />
          <span>Reset</span>
        </button>
        <button className="icon-button secondary ask-switch" type="button" onClick={onOpenAsk}>
          <span>Ask</span>
        </button>
      </div>
    </section>
  )
}

function formatFps(value) {
  const fps = Number(value)
  if (!Number.isFinite(fps) || fps <= 0) return 'waiting'
  return `${fps.toFixed(fps >= 10 ? 0 : 1)} fps`
}

function getStatusLabel(status) {
  if (status === 'idle') return 'Ready'
  if (status === 'running') return 'Live'
  if (status === 'starting') return 'Starting'
  if (status === 'publishing') return 'Publishing'
  if (status === 'live') return 'Live'
  if (status === 'preview_fallback') return 'Preview'
  if (status === 'paused') return 'Paused'
  if (status === 'stopping') return 'Stopping'
  if (status === 'stopped') return 'Stopped'
  if (status === 'error' || status === 'failed') return 'Failed'
  return status
}
