import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  buildEvidenceFrameUrl,
  buildDemoTestMemory,
  DEMO_API_BASE_URL,
  enqueueDemoTestOffline,
  getActiveRokidStream,
  getDemoTestStatus,
  getStreamStatus,
  getRokidStatus,
  pauseDemoPlayback,
  startFrameAudioStream,
  startDemoPlayback,
  startDemoTestPlayback,
  startLiveIngest,
  startRokidLiveIngest,
  startRokidStream,
  startWebRtcWhipStream,
  stopDemoPlayback,
  stopLiveIngest,
  stopRokidLiveIngest,
  tickDemoPlayback,
  tickDemoTestPlayback,
  uploadDemoVideo,
  uploadDemoTestVideos,
  uploadAudioChunk,
  uploadFrame
} from '../api/lightmem_egoApi.js'
import {
  assertWebRtcSecureContext,
  publishWhip,
  waitForPeerConnectionConnected
} from '../webrtc/whipClient.js'

export const INPUT_MODES = {
  FRAME_AUDIO: 'frame_audio_stream',
  WEBRTC_WHIP: 'web_webrtc_whip',
  ROKID: 'rokid_frame_audio',
  ROKID_LIVE_RTMP: 'rokid_live_rtmp',
  DEMO_VIDEO: 'demo_video',
  DEMO_TEST: 'demo_test'
}

const CAMERA_FACING = {
  BACK: 'environment',
  FRONT: 'user'
}

const FRAME_INTERVAL_MS = 1000
const AUDIO_CHUNK_MS = 1500
const STATUS_POLL_INTERVAL_MS = 2000
const ROKID_PREVIEW_POLL_INTERVAL_MS = 125
const DEMO_TICK_INTERVAL_MS = 750
const DEFAULT_INPUT_MODE = INPUT_MODES.FRAME_AUDIO
const DEFAULT_CAMERA_FACING = CAMERA_FACING.BACK
const LIVE_INGEST_FAILED_STATES = new Set(['failed', 'aborted', 'cancelled', 'canceled'])

const EMPTY_ACTIVE_SESSION = {
  sessionId: '',
  deviceKind: '',
  ownerId: '',
  deviceId: '',
  deviceType: '',
  inputMode: '',
  streamName: '',
  frameUploadUrl: '',
  audioUploadUrl: '',
  statusUrl: '',
  askStreamUrl: '',
  whipUrl: '',
  webrtcPlayUrl: '',
  liveIngestStartUrl: '',
  liveIngestStopUrl: '',
  streamInfo: null
}

const EMPTY_DEMO_SESSION = {
  sessionId: '',
  videoUrl: '',
  prepared: false,
  duration: null,
  frameCount: null,
  preprocessQueued: false,
  raw: null
}

const EMPTY_DEMO_TEST_SESSION = {
  sessionId: '',
  clips: [],
  activeClipId: 'day1',
  demoApiBaseUrl: DEMO_API_BASE_URL,
  askStreamUrl: '',
  askStreamUrlResolved: '',
  tickUrl: '',
  tickUrlResolved: '',
  prepared: false,
  memoryReady: false,
  displayDate: '',
  displayTime: '',
  displayDatetime: '',
  localCurrentTime: 0,
  currentTime: 0,
  status: null,
  raw: null
}

const initialStats = {
  frameIndex: 0,
  audioIndex: 0,
  frameUploadedCount: 0,
  frameFailedCount: 0,
  frameDroppedCount: 0,
  audioUploadedCount: 0,
  audioFailedCount: 0,
  audioDroppedCount: 0,
  firstFrameUploaded: false,
  canAsk: false,
  mcurReady: false,
  mcurVersion: null,
  latestFramePath: '',
  previewFps: null,
  memoryFps: null,
  memoryTargetFps: 1,
  latestAudioPath: '',
  lastFrameAt: null,
  lastAudioAt: null,
  lastFrameUploadMs: null,
  lastAudioUploadMs: null
}

export function useRealtimeStream() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const audioRecorderRef = useRef(null)
  const peerConnectionRef = useRef(null)
  const playbackRef = useRef(null)
  const playbackAbortRef = useRef(null)
  const playbackStartingRef = useRef(false)
  const frameTimerRef = useRef(null)
  const statusTimerRef = useRef(null)
  const rokidPreviewTimerRef = useRef(null)
  const demoTickTimerRef = useRef(null)
  const demoClockTimerRef = useRef(null)
  const lastDemoClockTextRef = useRef('')
  const sessionIdRef = useRef('')
  const activeSessionRef = useRef(EMPTY_ACTIVE_SESSION)
  const demoSessionRef = useRef(EMPTY_DEMO_SESSION)
  const demoTestSessionRef = useRef(EMPTY_DEMO_TEST_SESSION)
  const inputModeRef = useRef(DEFAULT_INPUT_MODE)
  const liveIngestStartedRef = useRef(false)
  const streamStartTimeRef = useRef(0)
  const epochRef = useRef(0)
  const activeRef = useRef(false)
  const pausedRef = useRef(false)
  const frameUploadInFlightRef = useRef(false)
  const frameIndexRef = useRef(0)
  const audioIndexRef = useRef(0)
  const audioChunkStartedAtRef = useRef(0)
  const cameraFacingModeRef = useRef(DEFAULT_CAMERA_FACING)
  const cameraSwitchingRef = useRef(false)
  const clientIdentityRef = useRef(getClientIdentity())

  const [inputMode, setInputModeState] = useState(DEFAULT_INPUT_MODE)
  const [status, setStatus] = useState('idle')
  const [webrtcState, setWebrtcState] = useState('idle')
  const [cameraFacingMode, setCameraFacingMode] = useState(DEFAULT_CAMERA_FACING)
  const [cameraSwitching, setCameraSwitching] = useState(false)
  const [activeSession, setActiveSession] = useState(EMPTY_ACTIVE_SESSION)
  const [streamError, setStreamError] = useState('')
  const [audioError, setAudioError] = useState('')
  const [stats, setStats] = useState(initialStats)
  const [statusSnapshot, setStatusSnapshot] = useState(null)
  const [statusError, setStatusError] = useState('')
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false)
  const [demoSession, setDemoSession] = useState(EMPTY_DEMO_SESSION)
  const [demoTestSession, setDemoTestSession] = useState(EMPTY_DEMO_TEST_SESSION)
  const [demoUploading, setDemoUploading] = useState(false)
  const [demoTestUploading, setDemoTestUploading] = useState(false)
  const [demoTestBusy, setDemoTestBusy] = useState(false)
  const [demoError, setDemoError] = useState('')
  const [demoClockText, setDemoClockText] = useState('')

  const getDemoPlaybackState = useCallback(() => {
    const video = videoRef.current
    return {
      currentTime: video?.currentTime || 0,
      paused: video ? video.paused : true,
      playbackSpeed: video?.playbackRate || 1
    }
  }, [])

  const isCurrentEpoch = useCallback((epoch) => {
    return activeRef.current && epochRef.current === epoch
  }, [])

  const clearDemoTickLoop = useCallback(() => {
    window.clearInterval(demoTickTimerRef.current)
    demoTickTimerRef.current = null
  }, [])

  const clearDemoClockLoop = useCallback((options = {}) => {
    window.clearInterval(demoClockTimerRef.current)
    demoClockTimerRef.current = null
    if (!options.keepText) {
      lastDemoClockTextRef.current = ''
      setDemoClockText('')
    }
  }, [])

  const updateDemoClock = useCallback((options = {}) => {
    const session = demoTestSessionRef.current
    const clip = getDemoTestActiveClip(session)
    const video = videoRef.current
    const currentTime = options.currentTime ?? video?.currentTime ?? session.currentTime ?? 0
    const nextText = formatDemoTestClockAt(clip, currentTime)
    if (nextText && nextText !== lastDemoClockTextRef.current) {
      lastDemoClockTextRef.current = nextText
      setDemoClockText(nextText)
    }
    return nextText
  }, [])

  const startDemoClockLoop = useCallback((epoch) => {
    window.clearInterval(demoClockTimerRef.current)
    updateDemoClock()
    demoClockTimerRef.current = window.setInterval(() => {
      if (!isCurrentEpoch(epoch) || pausedRef.current || inputModeRef.current !== INPUT_MODES.DEMO_TEST) return
      updateDemoClock()
    }, 250)
  }, [isCurrentEpoch, updateDemoClock])

  const updateDemoTestSession = useCallback((patch) => {
    const nextSession = {
      ...demoTestSessionRef.current,
      ...(typeof patch === 'function' ? patch(demoTestSessionRef.current) : patch)
    }
    demoTestSessionRef.current = nextSession
    setDemoTestSession(nextSession)
    return nextSession
  }, [])

  const activateSession = useCallback((info) => {
    const nextSession = {
      sessionId: info.sessionId || '',
      deviceKind: info.deviceKind || '',
      ownerId: info.ownerId || '',
      deviceId: info.deviceId || '',
      deviceType: info.deviceType || '',
      inputMode: info.inputMode || '',
      streamName: info.streamName || '',
      frameUploadUrl: info.frameUploadUrl || '',
      audioUploadUrl: info.audioUploadUrl || '',
      statusUrl: info.statusUrl || '',
      askStreamUrl: info.askStreamUrl || '',
      whipUrl: info.whipUrl || '',
      webrtcPlayUrl: info.webrtcPlayUrl || '',
      liveIngestStartUrl: info.liveIngestStartUrl || '',
      liveIngestStopUrl: info.liveIngestStopUrl || '',
      streamInfo: info
    }

    activeSessionRef.current = nextSession
    sessionIdRef.current = nextSession.sessionId
    setActiveSession(nextSession)
    return nextSession
  }, [])

  const clearActiveSession = useCallback(() => {
    activeSessionRef.current = EMPTY_ACTIVE_SESSION
    sessionIdRef.current = ''
    liveIngestStartedRef.current = false
    setActiveSession(EMPTY_ACTIVE_SESSION)
  }, [])

  const clearFrameTransport = useCallback(() => {
    window.clearInterval(frameTimerRef.current)
    frameTimerRef.current = null
    frameUploadInFlightRef.current = false

    const recorder = audioRecorderRef.current
    audioRecorderRef.current = null
    if (recorder && recorder.state !== 'inactive') {
      try {
        recorder.stop()
      } catch (error) {
        // Tracks are stopped below even if MediaRecorder is already closing.
      }
    }
  }, [])

  const closePeerConnection = useCallback(() => {
    const peerConnection = peerConnectionRef.current
    peerConnectionRef.current = null
    if (!peerConnection) return

    peerConnection.onconnectionstatechange = null
    peerConnection.oniceconnectionstatechange = null
    peerConnection.close()
  }, [])

  const closePlayback = useCallback((options = {}) => {
    const keepDemoTestVideo = !!options.keepDemoTestVideo && inputModeRef.current === INPUT_MODES.DEMO_TEST
    playbackStartingRef.current = false
    playbackAbortRef.current?.abort?.()
    playbackAbortRef.current = null
    playbackRef.current?.close?.()
    playbackRef.current = null
    if (videoRef.current) {
      videoRef.current.pause?.()
      videoRef.current.srcObject = null
      if (inputModeRef.current === INPUT_MODES.DEMO_TEST && !activeRef.current && !keepDemoTestVideo) {
        videoRef.current.removeAttribute('src')
      }
    }
  }, [])

  const stopLocalMedia = useCallback((options = {}) => {
    clearFrameTransport()
    closePeerConnection()
    closePlayback(options)
    clearDemoTickLoop()
    clearDemoClockLoop()
    window.clearInterval(statusTimerRef.current)
    statusTimerRef.current = null
    window.clearInterval(rokidPreviewTimerRef.current)
    rokidPreviewTimerRef.current = null

    const stream = mediaStreamRef.current
    mediaStreamRef.current = null
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
      if (inputModeRef.current !== INPUT_MODES.DEMO_VIDEO && inputModeRef.current !== INPUT_MODES.DEMO_TEST) {
        videoRef.current.removeAttribute('src')
      }
    }
  }, [clearDemoClockLoop, clearDemoTickLoop, clearFrameTransport, closePeerConnection, closePlayback])

  const captureFrame = useCallback(async (epoch) => {
    if (!isCurrentEpoch(epoch) || pausedRef.current || inputModeRef.current !== INPUT_MODES.FRAME_AUDIO) return

    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
      setStats((current) => ({ ...current, frameDroppedCount: current.frameDroppedCount + 1 }))
      return
    }

    if (frameUploadInFlightRef.current) {
      setStats((current) => ({ ...current, frameDroppedCount: current.frameDroppedCount + 1 }))
      return
    }

    const currentSessionId = sessionIdRef.current
    if (!currentSessionId) return

    frameUploadInFlightRef.current = true
    const frameIndex = frameIndexRef.current
    frameIndexRef.current += 1
    setStats((current) => ({ ...current, frameIndex: frameIndexRef.current }))

    try {
      const { blob, format, width, height } = await captureVideoBlob(video, canvas)
      if (!isCurrentEpoch(epoch) || pausedRef.current) return

      const relativeTsMs = Math.max(0, Date.now() - streamStartTimeRef.current)
      const startedAt = performance.now()
      const result = await uploadFrame(currentSessionId, blob, {
        frameIndex,
        relativeTsMs,
        width,
        height,
        format,
        source: 'web_canvas'
      })

      if (!isCurrentEpoch(epoch)) return

      setStats((current) => ({
        ...current,
        frameUploadedCount: current.frameUploadedCount + 1,
        firstFrameUploaded: true,
        canAsk: result.canAsk || result.mcurReady || current.canAsk,
        mcurReady: result.mcurReady || current.mcurReady,
        mcurVersion: result.mcurVersion ?? current.mcurVersion,
        latestFramePath: result.currentFramePath || result.savedPath || current.latestFramePath,
        lastFrameAt: Date.now(),
        lastFrameUploadMs: Math.round(performance.now() - startedAt)
      }))
    } catch (error) {
      if (!isCurrentEpoch(epoch)) return
      setStreamError(error.message || 'Frame upload failed')
      setStats((current) => ({
        ...current,
        frameFailedCount: current.frameFailedCount + 1
      }))
    } finally {
      frameUploadInFlightRef.current = false
    }
  }, [isCurrentEpoch])

  const startFrameLoop = useCallback((epoch) => {
    window.clearInterval(frameTimerRef.current)
    frameTimerRef.current = window.setInterval(() => {
      captureFrame(epoch)
    }, FRAME_INTERVAL_MS)
  }, [captureFrame])

  const startAudioRecorder = useCallback((stream, currentSessionId, epoch) => {
    const audioTracks = stream.getAudioTracks()
    if (!audioTracks.length) {
      setAudioError('Microphone is unavailable; video frames will continue.')
      return
    }

    if (!window.MediaRecorder) {
      setAudioError('MediaRecorder is not supported by this browser.')
      return
    }

    try {
      const audioStream = new MediaStream(audioTracks)
      const mimeType = pickAudioMimeType()
      const recorder = mimeType ? new MediaRecorder(audioStream, { mimeType }) : new MediaRecorder(audioStream)

      recorder.ondataavailable = async (event) => {
        if (!event.data || !event.data.size || !isCurrentEpoch(epoch) || pausedRef.current) return

        const audioIndex = audioIndexRef.current
        audioIndexRef.current += 1
        const now = Date.now()
        const durationMs = audioChunkStartedAtRef.current ? Math.max(250, now - audioChunkStartedAtRef.current) : AUDIO_CHUNK_MS
        audioChunkStartedAtRef.current = now

        setStats((current) => ({ ...current, audioIndex: audioIndexRef.current }))

        try {
          const startedAt = performance.now()
          const result = await uploadAudioChunk(currentSessionId, event.data, {
            audioIndex,
            relativeTsMs: Math.max(0, now - streamStartTimeRef.current),
            durationMs,
            format: inferAudioFormat(event.data.type),
            source: 'web_media_recorder'
          })

          if (!isCurrentEpoch(epoch)) return

          setStats((current) => ({
            ...current,
            audioUploadedCount: current.audioUploadedCount + 1,
            latestAudioPath: result.savedPath || current.latestAudioPath,
            lastAudioAt: Date.now(),
            lastAudioUploadMs: Math.round(performance.now() - startedAt)
          }))
        } catch (error) {
          if (!isCurrentEpoch(epoch)) return
          setAudioError(error.message || 'Audio upload failed')
          setStats((current) => ({
            ...current,
            audioFailedCount: current.audioFailedCount + 1
          }))
        }
      }

      recorder.onerror = (event) => {
        if (!isCurrentEpoch(epoch)) return
        const message = event.error?.message || 'Audio recorder failed'
        setAudioError(message)
        setStats((current) => ({ ...current, audioFailedCount: current.audioFailedCount + 1 }))
      }

      audioRecorderRef.current = recorder
      audioChunkStartedAtRef.current = Date.now()
      recorder.start(AUDIO_CHUNK_MS)
    } catch (error) {
      setAudioError(error.message || 'Audio recorder failed; video frames will continue.')
    }
  }, [isCurrentEpoch])

  const applyPreviewSnapshot = useCallback((result) => {
    const frameStream = result.frameStream || {}
    const latestFramePath = frameStream.latestFramePath || frameStream.latestCurrentFramePath || ''

    setStats((current) => ({
      ...current,
      canAsk: result.canAsk || current.canAsk,
      firstFrameUploaded: current.firstFrameUploaded || !!latestFramePath || !!frameStream.receivedCount || !!frameStream.previewReceivedCount,
      frameUploadedCount: frameStream.previewReceivedCount ?? frameStream.receivedCount ?? current.frameUploadedCount,
      latestFramePath: latestFramePath || current.latestFramePath,
      previewFps: frameStream.previewFps ?? current.previewFps,
      memoryFps: frameStream.memoryFps ?? current.memoryFps,
      memoryTargetFps: frameStream.memoryTargetFps ?? frameStream.targetFps ?? current.memoryTargetFps,
      mcurReady: frameStream.mcurReady || current.mcurReady,
      mcurVersion: frameStream.mcurVersion ?? current.mcurVersion,
      lastFrameAt: frameStream.latestFrameAt ? Date.parse(frameStream.latestFrameAt) || current.lastFrameAt : current.lastFrameAt
    }))
  }, [])

  const applyStatusSnapshot = useCallback((result, options = {}) => {
    applyPreviewSnapshot(result)
    setStatusSnapshot(result)

    if (!options.updateLiveState) return

    const activeMode = inputModeRef.current
    const isLiveMode = activeMode === INPUT_MODES.WEBRTC_WHIP || activeMode === INPUT_MODES.ROKID_LIVE_RTMP
    if (!isLiveMode) return

    const liveIngest = result.liveIngest || {}
    const live = result.live || {}
    const ingestStatus = liveIngest.status || live.ingestStatus || ''
    const framesIngested = toFiniteNumber(liveIngest.framesIngested ?? live.framesIngested ?? 0, 0)
    const failedMessage = liveIngest.lastError || live.lastError || 'Live ingest failed'

    if (LIVE_INGEST_FAILED_STATES.has(String(ingestStatus).toLowerCase())) {
      if (activeMode === INPUT_MODES.ROKID_LIVE_RTMP && framesIngested > 0) {
        setStatus('preview_fallback')
        setWebrtcState('preview_fallback')
        setStreamError(`Rokid RTMP live ingest stopped. Showing latest frame preview: ${failedMessage}`)
      } else {
        setStatus('failed')
        setWebrtcState('failed')
        setStreamError(failedMessage)
      }
      return
    }

    if (ingestStatus === 'running') {
      if (activeMode === INPUT_MODES.ROKID_LIVE_RTMP) {
        setStatus('live')
        setWebrtcState('live')
        return
      }

      setStatus('live')
      setWebrtcState('live')
    } else if (['queued', 'starting', 'waiting_stream', 'waiting_rtmp_output', 'waiting_keyframe'].includes(ingestStatus)) {
      setStatus('publishing')
      setWebrtcState(activeMode === INPUT_MODES.ROKID_LIVE_RTMP ? 'ingest_starting' : 'publishing')
    }
  }, [applyPreviewSnapshot, webrtcState])

  const pollStatus = useCallback(async (epoch) => {
    const currentSessionId = sessionIdRef.current
    if (!currentSessionId || !isCurrentEpoch(epoch)) return

    try {
      const useRokidStatus = [INPUT_MODES.ROKID, INPUT_MODES.ROKID_LIVE_RTMP].includes(activeSessionRef.current.inputMode)
      const result = useRokidStatus ? await getRokidStatus(currentSessionId) : await getStreamStatus(currentSessionId)
      if (!isCurrentEpoch(epoch)) return
      applyStatusSnapshot(result, { updateLiveState: true })
      setStatusError('')
    } catch (error) {
      if (!isCurrentEpoch(epoch)) return
      setStatusError(error.message || 'Status refresh failed')
    }
  }, [applyStatusSnapshot, isCurrentEpoch])

  const pollRokidPreview = useCallback(async (epoch) => {
    const currentSessionId = sessionIdRef.current
    if (!currentSessionId || !isCurrentEpoch(epoch)) return
    if (activeSessionRef.current.inputMode !== INPUT_MODES.ROKID) return

    try {
      const result = await getRokidStatus(currentSessionId)
      if (!isCurrentEpoch(epoch)) return
      applyPreviewSnapshot(result)
    } catch (error) {
      if (!isCurrentEpoch(epoch)) return
      setStatusError(error.message || 'Preview refresh failed')
    }
  }, [applyPreviewSnapshot, isCurrentEpoch])

  const startStatusPolling = useCallback((epoch) => {
    window.clearInterval(statusTimerRef.current)
    statusTimerRef.current = window.setInterval(() => {
      pollStatus(epoch)
    }, STATUS_POLL_INTERVAL_MS)
    pollStatus(epoch)
  }, [pollStatus])

  const startRokidPreviewPolling = useCallback((epoch) => {
    window.clearInterval(rokidPreviewTimerRef.current)
    rokidPreviewTimerRef.current = window.setInterval(() => {
      pollRokidPreview(epoch)
    }, ROKID_PREVIEW_POLL_INTERVAL_MS)
    pollRokidPreview(epoch)
  }, [pollRokidPreview])

  const bindPeerConnectionState = useCallback((peerConnection, epoch) => {
    const handleFailure = (message) => {
      if (!isCurrentEpoch(epoch)) return
      console.warn('[LightMem-Ego WHIP] peer connection reported failure', {
        message,
        iceConnectionState: peerConnection.iceConnectionState,
        connectionState: peerConnection.connectionState,
        iceGatheringState: peerConnection.iceGatheringState
      })
      setWebrtcState('failed')
      setStreamError(message)
    }

    peerConnection.onconnectionstatechange = () => {
      if (peerConnection.connectionState === 'failed') {
        handleFailure('WHIP publish failed: peer connection failed')
      }
    }
    peerConnection.oniceconnectionstatechange = () => {
      if (peerConnection.iceConnectionState === 'failed') {
        handleFailure('ICE failed')
      }
    }
  }, [isCurrentEpoch])

  const prepareStart = useCallback((selectedInputMode) => {
    const epoch = epochRef.current + 1
    epochRef.current = epoch
    activeRef.current = false
    pausedRef.current = false
    stopLocalMedia({ keepDemoTestVideo: selectedInputMode === INPUT_MODES.DEMO_TEST })
    playbackStartingRef.current = false
    setStatus('starting')
    setWebrtcState(selectedInputMode === INPUT_MODES.WEBRTC_WHIP ? 'starting' : 'idle')
    setStreamError('')
    setAudioError('')
    setStatusError('')
    setStats(initialStats)
    setStatusSnapshot(null)
    clearActiveSession()
    frameIndexRef.current = 0
    audioIndexRef.current = 0
    return epoch
  }, [clearActiveSession, stopLocalMedia])

  const requestLocalMedia = useCallback(async (epoch, options = {}) => {
    const stream = await requestCameraAndMic({
      ...options,
      facingMode: cameraFacingModeRef.current
    })

    if (epochRef.current !== epoch) {
      stream.getTracks().forEach((track) => track.stop())
      return stream
    }

    mediaStreamRef.current = stream
    attachPreviewStream(videoRef.current, stream)
    return stream
  }, [])

  const toggleCamera = useCallback(async () => {
    if (cameraSwitchingRef.current || status === 'starting' || status === 'stopping') return

    const previousFacingMode = cameraFacingModeRef.current
    const nextFacingMode = getNextCameraFacingMode(previousFacingMode)
    cameraFacingModeRef.current = nextFacingMode
    setCameraFacingMode(nextFacingMode)
    setStreamError('')

    const currentStream = mediaStreamRef.current
    const shouldHotSwap = currentStream && ['running', 'live', 'paused'].includes(status)
    if (!shouldHotSwap) return

    cameraSwitchingRef.current = true
    setCameraSwitching(true)

    let replacementStream = null
    try {
      replacementStream = await requestCameraOnly(nextFacingMode)
      const nextTrack = replacementStream.getVideoTracks()[0]
      if (!nextTrack) {
        throw new Error('No camera video track returned')
      }

      const activeStream = mediaStreamRef.current
      if (!activeStream) {
        nextTrack.stop()
        return
      }

      const peerConnection = peerConnectionRef.current
      const videoSender = peerConnection?.getSenders?.().find((sender) => sender.track?.kind === 'video')
      if (videoSender) {
        await videoSender.replaceTrack(nextTrack)
      }

      const oldVideoTracks = activeStream.getVideoTracks()
      oldVideoTracks.forEach((track) => activeStream.removeTrack(track))
      activeStream.addTrack(nextTrack)
      oldVideoTracks.forEach((track) => track.stop())
      replacementStream.getTracks().forEach((track) => {
        if (track !== nextTrack) track.stop()
      })

      attachPreviewStream(videoRef.current, activeStream)
      if (videoRef.current) {
        await waitForVideoReady(videoRef.current).catch(() => {})
      }
      if (inputModeRef.current === INPUT_MODES.FRAME_AUDIO && status === 'running') {
        captureFrame(epochRef.current)
      }
    } catch (error) {
      replacementStream?.getTracks().forEach((track) => track.stop())
      cameraFacingModeRef.current = previousFacingMode
      setCameraFacingMode(previousFacingMode)
      setStreamError(formatCameraSwitchError(error))
    } finally {
      cameraSwitchingRef.current = false
      setCameraSwitching(false)
    }
  }, [captureFrame, status])

  const startFrameAudio = useCallback(async () => {
    if (!['idle', 'stopped', 'error', 'failed'].includes(status)) return
    const epoch = prepareStart(INPUT_MODES.FRAME_AUDIO)

    try {
      const mediaPromise = requestLocalMedia(epoch, {
        allowVideoOnly: true
      })
      const identity = clientIdentityRef.current
      const startPromise = startFrameAudioStream({
        ownerId: identity.ownerId,
        deviceId: identity.webDeviceId,
        deviceType: 'web',
        metadata: {
          client: 'web_frontend'
        }
      })
      const [stream, info] = await Promise.all([mediaPromise, startPromise])
      if (epochRef.current !== epoch) return
      if (!info.sessionId) throw new Error('Stream start failed: session_id missing')

      streamStartTimeRef.current = Date.now()
      activeRef.current = true
      activateSession(info)
      setStats((current) => ({ ...current, canAsk: info.canAsk }))
      setStatus('running')
      if (videoRef.current) {
        await waitForVideoReady(videoRef.current)
      }
      if (!isCurrentEpoch(epoch)) return
      startFrameLoop(epoch)
      await captureFrame(epoch)
      startAudioRecorder(stream, info.sessionId, epoch)
    } catch (error) {
      epochRef.current += 1
      activeRef.current = false
      stopLocalMedia()
      setStatus('failed')
      setWebrtcState('idle')
      setStreamError(formatStartError(error))
    }
  }, [
    activateSession,
    captureFrame,
    prepareStart,
    requestLocalMedia,
    startAudioRecorder,
    startFrameLoop,
    status,
    stopLocalMedia
  ])

  const bindDemoVideo = useCallback((session = demoSessionRef.current) => {
    const video = videoRef.current
    return bindVideoElementSource(video, session.videoUrl)
  }, [])

  const bindDemoTestVideo = useCallback((session = demoTestSessionRef.current) => {
    const video = videoRef.current
    const clip = getDemoTestActiveClip(session)
    return bindVideoElementSource(video, clip?.videoUrlResolved)
  }, [])

  const sendDemoTick = useCallback(async (options = {}) => {
    const activeInputMode = options.inputMode || activeSessionRef.current.inputMode || inputModeRef.current
    const currentSessionId = options.sessionId ||
      activeSessionRef.current.sessionId ||
      (activeInputMode === INPUT_MODES.DEMO_TEST ? demoTestSessionRef.current.sessionId : demoSessionRef.current.sessionId)
    if (!currentSessionId) return null
    const playback = getDemoPlaybackState()
    const tickOptions = {
      currentTime: options.currentTime ?? playback.currentTime,
      paused: options.paused ?? playback.paused,
      playbackSpeed: options.playbackSpeed ?? playback.playbackSpeed
    }
    const result = activeInputMode === INPUT_MODES.DEMO_TEST
      ? await tickDemoTestPlayback(currentSessionId, {
          ...tickOptions,
          clipId: options.clipId || demoTestSessionRef.current.activeClipId || 'day1',
          baseUrl: demoTestSessionRef.current.demoApiBaseUrl || DEMO_API_BASE_URL
        })
      : await tickDemoPlayback(currentSessionId, tickOptions)
    if (activeInputMode === INPUT_MODES.DEMO_TEST && result) {
      updateDemoTestSession((current) => ({
        ...current,
        activeClipId: result.activeClipId || current.activeClipId,
        localCurrentTime: result.localCurrentTime ?? current.localCurrentTime,
        currentTime: result.currentTime ?? current.currentTime,
        raw: result.raw || current.raw
      }))
    }
    setStats((current) => ({
      ...current,
      canAsk: true,
      firstFrameUploaded: true,
      frameUploadedCount: Math.max(current.frameUploadedCount + 1, current.frameUploadedCount),
      lastFrameAt: Date.now()
    }))
    setStatusSnapshot((current) => result || current)
    return result
  }, [getDemoPlaybackState, updateDemoTestSession])

  const startDemoTickLoop = useCallback((epoch, sessionId) => {
    clearDemoTickLoop()
    demoTickTimerRef.current = window.setInterval(() => {
      if (!isCurrentEpoch(epoch) || pausedRef.current) return
      sendDemoTick({ sessionId }).catch((error) => {
        if (!isCurrentEpoch(epoch)) return
        setStatusError(error.message || 'Demo tick failed')
      })
    }, DEMO_TICK_INTERVAL_MS)
  }, [clearDemoTickLoop, isCurrentEpoch, sendDemoTick])

  const startDemoVideo = useCallback(async () => {
    if (!['idle', 'stopped', 'error', 'failed'].includes(status)) return
    const session = demoSessionRef.current
    if (!session.sessionId || !session.videoUrl) {
      setStreamError('Please upload a demo video in Advanced / Tools first.')
      return
    }

    const epoch = prepareStart(INPUT_MODES.DEMO_VIDEO)

    try {
      inputModeRef.current = INPUT_MODES.DEMO_VIDEO
      setInputModeState(INPUT_MODES.DEMO_VIDEO)
      const video = bindDemoVideo(session)
      if (!video) throw new Error('Demo video element is unavailable.')

      await waitForMediaMetadata(video)
      setVideoCurrentTime(video, 0)
      video.playbackRate = 1
      await startDemoPlayback(session.sessionId, {
        currentTime: 0,
        playbackSpeed: video.playbackRate || 1
      })
      if (epochRef.current !== epoch) return

      activateSession({
        ...session,
        inputMode: INPUT_MODES.DEMO_VIDEO,
        status: 'demo_started',
        streamInfo: session
      })
      streamStartTimeRef.current = Date.now()
      activeRef.current = true
      pausedRef.current = false
      setStats((current) => ({
        ...current,
        canAsk: true,
        firstFrameUploaded: true,
        frameUploadedCount: session.frameCount || current.frameUploadedCount
      }))
      setStatus('running')
      await nextAnimationFrame()
      if (!isCurrentEpoch(epoch)) return
      await video.play()
      await sendDemoTick({ sessionId: session.sessionId, currentTime: 0, paused: false })
      startDemoTickLoop(epoch, session.sessionId)
    } catch (error) {
      epochRef.current += 1
      activeRef.current = false
      clearDemoTickLoop()
      setStatus('failed')
      setStreamError(formatDemoError(error, 'Demo playback failed'))
    }
  }, [activateSession, bindDemoVideo, clearDemoTickLoop, isCurrentEpoch, prepareStart, sendDemoTick, startDemoTickLoop, status])

  const startDemoTestVideo = useCallback(async () => {
    if (!['idle', 'stopped', 'error', 'failed'].includes(status)) return
    const session = demoTestSessionRef.current
    const clip = getDemoTestActiveClip(session)
    if (!session.sessionId || !clip?.videoUrlResolved) {
      setStreamError('Please upload demo-test day1 and day2 videos in Advanced / Tools first.')
      return
    }

    const epoch = prepareStart(INPUT_MODES.DEMO_TEST)

    try {
      inputModeRef.current = INPUT_MODES.DEMO_TEST
      setInputModeState(INPUT_MODES.DEMO_TEST)
      const video = bindDemoTestVideo(session)
      if (!video) throw new Error('Demo-test video element is unavailable.')

      setVideoCurrentTime(video, 0)
      video.playbackRate = 1

      activateSession({
        ...session,
        inputMode: INPUT_MODES.DEMO_TEST,
        deviceKind: 'demo',
        deviceType: 'demo-test',
        askStreamUrl: session.askStreamUrl || `/ask/${session.sessionId}/stream`,
        status: 'demo_test_started',
        streamInfo: session
      })
      streamStartTimeRef.current = Date.now()
      activeRef.current = true
      pausedRef.current = false
      setStats((current) => ({
        ...current,
        canAsk: true,
        firstFrameUploaded: true,
        frameUploadedCount: Math.max(current.frameUploadedCount, 1)
      }))
      setStatus('running')
      await nextAnimationFrame()
      if (!isCurrentEpoch(epoch)) return
      await video.play()
      startDemoClockLoop(epoch)

      ;(async () => {
        try {
          await startDemoTestPlayback(session.sessionId, {
            clipId: clip.clipId,
            currentTime: 0,
            playbackSpeed: video.playbackRate || 1,
            baseUrl: session.demoApiBaseUrl || DEMO_API_BASE_URL
          })
          if (!isCurrentEpoch(epoch)) return
          await sendDemoTick({
            sessionId: session.sessionId,
            inputMode: INPUT_MODES.DEMO_TEST,
            clipId: clip.clipId,
            currentTime: video.currentTime || 0,
            paused: false
          })
          if (!isCurrentEpoch(epoch)) return
          startDemoTickLoop(epoch, session.sessionId)
        } catch (error) {
          if (!isCurrentEpoch(epoch)) return
          const message = formatDemoError(error, 'Demo-test backend sync failed')
          setStatusError(message)
          setStreamError(message)
        }
      })()
    } catch (error) {
      epochRef.current += 1
      activeRef.current = false
      clearDemoTickLoop()
      setStatus('failed')
      setStreamError(formatDemoError(error, 'Demo-test playback failed'))
    }
  }, [activateSession, bindDemoTestVideo, clearDemoTickLoop, isCurrentEpoch, prepareStart, sendDemoTick, startDemoClockLoop, startDemoTickLoop, status])

  const startWebRtcWhip = useCallback(async () => {
    if (!['idle', 'stopped', 'error', 'failed'].includes(status)) return
    const epoch = prepareStart(INPUT_MODES.WEBRTC_WHIP)
    let whipSignalingComplete = false

    try {
      assertWebRtcSecureContext()

      const mediaPromise = requestLocalMedia(epoch, { allowVideoOnly: false })
      const identity = clientIdentityRef.current
      const startPromise = startWebRtcWhipStream({
        ownerId: identity.ownerId,
        deviceId: identity.webDeviceId,
        deviceType: 'web',
        metadata: {
          client: 'web_frontend'
        }
      })
      const [stream, info] = await Promise.all([mediaPromise, startPromise])
      if (epochRef.current !== epoch) return
      if (!info.sessionId) throw new Error('Stream start failed: session_id missing')

      const currentSession = activateSession(info)
      activeRef.current = true
      setWebrtcState('ready')

      if (!currentSession.whipUrl) {
        throw new Error('Backend did not return a WHIP URL')
      }

      setStatus('publishing')
      setWebrtcState('publishing')
      const publication = await publishWhip(currentSession.whipUrl, stream, {
        onPeerConnection: (peerConnection) => {
          peerConnectionRef.current = peerConnection
          bindPeerConnectionState(peerConnection, epoch)
        }
      })
      if (!isCurrentEpoch(epoch)) return
      whipSignalingComplete = true
      setWebrtcState('connecting')

      await waitForPeerConnectionConnected(publication.peerConnection, 20000)
      if (!isCurrentEpoch(epoch)) return
      console.info('[LightMem-Ego WHIP] peer connected; starting live ingest', {
        sessionId: currentSession.sessionId,
        iceConnectionState: publication.peerConnection.iceConnectionState,
        connectionState: publication.peerConnection.connectionState
      })

      await startLiveIngest(currentSession.sessionId)
      if (!isCurrentEpoch(epoch)) return

      liveIngestStartedRef.current = true
      setStatus('live')
      setWebrtcState('live')
      startStatusPolling(epoch)
    } catch (error) {
      if (epochRef.current !== epoch) return

      if (whipSignalingComplete) {
        setStatus('publishing')
        setWebrtcState('failed')
        setStreamError(formatStartError(error))
        startStatusPolling(epoch)
        return
      }

      epochRef.current += 1
      activeRef.current = false
      stopLocalMedia()
      setStatus('failed')
      setWebrtcState('failed')
      setStreamError(formatStartError(error))
    }
  }, [
    activateSession,
    bindPeerConnectionState,
    isCurrentEpoch,
    prepareStart,
    requestLocalMedia,
    startStatusPolling,
    status,
    stopLocalMedia
  ])

  const stop = useCallback((options = {}) => {
    const currentSession = activeSessionRef.current
    const currentSessionId = currentSession.sessionId
    const currentInputMode = currentSession.inputMode || inputModeRef.current
    const stopEpoch = epochRef.current + 1
    epochRef.current = stopEpoch
    activeRef.current = false
    pausedRef.current = false
    const finalStatus = options.clearSession ? 'idle' : 'stopped'
    setStatus((current) => (current === 'idle' && !currentSessionId ? 'idle' : finalStatus))
    if (currentInputMode === INPUT_MODES.WEBRTC_WHIP || currentInputMode === INPUT_MODES.ROKID_LIVE_RTMP) {
      setWebrtcState(options.clearSession ? 'idle' : 'stopped')
    }
    clearDemoTickLoop()
    clearDemoClockLoop()
    if (currentInputMode === INPUT_MODES.DEMO_VIDEO && currentSessionId) {
      const playback = getDemoPlaybackState()
      stopDemoPlayback(currentSessionId, playback).catch((error) => {
        if (epochRef.current !== stopEpoch) return
        setStreamError(error.message || 'Demo stop failed')
      })
    }
    if (currentInputMode === INPUT_MODES.DEMO_TEST && currentSessionId) {
      const playback = getDemoPlaybackState()
      tickDemoTestPlayback(currentSessionId, {
        ...playback,
        paused: true,
        clipId: demoTestSessionRef.current.activeClipId || 'day1',
        baseUrl: demoTestSessionRef.current.demoApiBaseUrl || DEMO_API_BASE_URL
      }).catch((error) => {
        if (epochRef.current !== stopEpoch) return
        setStreamError(error.message || 'Demo-test stop tick failed')
      })
    }
    stopLocalMedia()

    if (currentInputMode === INPUT_MODES.WEBRTC_WHIP && currentSessionId) {
      stopLiveIngest(currentSessionId).catch((error) => {
        if (epochRef.current !== stopEpoch || sessionIdRef.current !== currentSessionId) return
        setStreamError(error.message || 'Live ingest stop failed')
      })
    }
    if (currentInputMode === INPUT_MODES.ROKID_LIVE_RTMP && currentSessionId) {
      stopRokidLiveIngest(currentSessionId).catch((error) => {
        if (epochRef.current !== stopEpoch || sessionIdRef.current !== currentSessionId) return
        setStreamError(error.message || 'Rokid live ingest stop failed')
      })
    }

    if (options.clearSession) {
      clearActiveSession()
      demoSessionRef.current = EMPTY_DEMO_SESSION
      demoTestSessionRef.current = EMPTY_DEMO_TEST_SESSION
      setDemoSession(EMPTY_DEMO_SESSION)
      setDemoTestSession(EMPTY_DEMO_TEST_SESSION)
      setStatusSnapshot(null)
      setStatusError('')
      setDemoClockText('')
      lastDemoClockTextRef.current = ''
      setStats(initialStats)
      frameIndexRef.current = 0
      audioIndexRef.current = 0
    }
  }, [clearActiveSession, clearDemoTickLoop, getDemoPlaybackState, stopLocalMedia])

  const pause = useCallback(() => {
    if (status === 'running' && inputModeRef.current === INPUT_MODES.DEMO_VIDEO) {
      pausedRef.current = true
      clearDemoTickLoop()
      videoRef.current?.pause?.()
      const currentSessionId = activeSessionRef.current.sessionId
      if (currentSessionId) {
        pauseDemoPlayback(currentSessionId, getDemoPlaybackState()).catch((error) => {
          setStatusError(error.message || 'Demo pause failed')
        })
      }
      setStatus('paused')
      return
    }

    if (status === 'running' && inputModeRef.current === INPUT_MODES.DEMO_TEST) {
      pausedRef.current = true
      clearDemoTickLoop()
      clearDemoClockLoop({ keepText: true })
      updateDemoClock()
      videoRef.current?.pause?.()
      const currentSessionId = activeSessionRef.current.sessionId
      if (currentSessionId) {
        sendDemoTick({
          sessionId: currentSessionId,
          inputMode: INPUT_MODES.DEMO_TEST,
          clipId: demoTestSessionRef.current.activeClipId,
          paused: true
        }).catch((error) => {
          setStatusError(error.message || 'Demo-test pause tick failed')
        })
      }
      setStatus('paused')
      return
    }

    if (status !== 'running' || inputModeRef.current !== INPUT_MODES.FRAME_AUDIO) return
    pausedRef.current = true
    window.clearInterval(frameTimerRef.current)
    frameTimerRef.current = null

    const recorder = audioRecorderRef.current
    if (recorder && recorder.state === 'recording') {
      try {
        recorder.pause()
      } catch (error) {
        setAudioError(error.message || 'Could not pause audio recorder')
      }
    }

    setStatus('paused')
  }, [clearDemoClockLoop, clearDemoTickLoop, getDemoPlaybackState, sendDemoTick, status, updateDemoClock])

  const resume = useCallback(async () => {
    if (status === 'paused' && inputModeRef.current === INPUT_MODES.DEMO_VIDEO) {
      const epoch = epochRef.current
      const currentSessionId = activeSessionRef.current.sessionId
      if (!currentSessionId) return
      pausedRef.current = false
      setStatus('running')
      await startDemoPlayback(currentSessionId, getDemoPlaybackState())
      await videoRef.current?.play?.()
      await sendDemoTick({ sessionId: currentSessionId, paused: false })
      startDemoTickLoop(epoch, currentSessionId)
      return
    }

    if (status === 'paused' && inputModeRef.current === INPUT_MODES.DEMO_TEST) {
      const epoch = epochRef.current
      const currentSessionId = activeSessionRef.current.sessionId
      const clipId = demoTestSessionRef.current.activeClipId || 'day1'
      if (!currentSessionId) return
      pausedRef.current = false
      setStatus('running')
      await startDemoTestPlayback(currentSessionId, {
        ...getDemoPlaybackState(),
        clipId,
        baseUrl: demoTestSessionRef.current.demoApiBaseUrl || DEMO_API_BASE_URL
      })
      await videoRef.current?.play?.()
      startDemoClockLoop(epoch)
      await sendDemoTick({
        sessionId: currentSessionId,
        inputMode: INPUT_MODES.DEMO_TEST,
        clipId,
        paused: false
      })
      startDemoTickLoop(epoch, currentSessionId)
      return
    }

    if (status !== 'paused' || inputModeRef.current !== INPUT_MODES.FRAME_AUDIO) return
    const epoch = epochRef.current
    pausedRef.current = false
    setStatus('running')

    const recorder = audioRecorderRef.current
    if (recorder && recorder.state === 'paused') {
      try {
        recorder.resume()
      } catch (error) {
        setAudioError(error.message || 'Could not resume audio recorder')
      }
    }

    startFrameLoop(epoch)
    await captureFrame(epoch)
  }, [captureFrame, getDemoPlaybackState, sendDemoTick, startDemoClockLoop, startDemoTickLoop, startFrameLoop, status])

  const reset = useCallback(() => {
    stop({ clearSession: true })
    setStreamError('')
    setAudioError('')
  }, [stop])

  const startRokidSession = useCallback(async (options = {}) => {
    if (['starting', 'running', 'publishing', 'live', 'paused', 'stopping'].includes(status)) return null
    const requestedInputMode = options.inputMode || INPUT_MODES.ROKID
    const epoch = prepareStart(requestedInputMode)

    try {
      const identity = clientIdentityRef.current
      let info = null
      if (requestedInputMode === INPUT_MODES.ROKID_LIVE_RTMP) {
        try {
          info = await getActiveRokidStream()
        } catch (activeError) {
          throw new Error(`Active Rokid session API is unavailable: ${activeError.message || 'request failed'}. Deploy and restart the backend with /rokid/stream/active first.`)
        }
        if (!info.active || !info.sessionId) {
          throw new Error('No active Rokid RTMP session found. Start recording on the Rokid Glass app first, then start Rokid RTMP here.')
        }
        const activeMetadata = info.metadata || info.raw?.metadata || {}
        const activeSource = activeMetadata.source || info.raw?.metadata?.source || ''
        const activeClient = activeMetadata.client || info.raw?.metadata?.client || ''
        const activeDeviceId = info.deviceId || activeMetadata.deviceId || info.raw?.device_id || info.raw?.metadata?.device_id || ''
        if (activeSource === 'web_frontend' || activeClient === 'web_frontend' || String(activeDeviceId).startsWith('rokid_web')) {
          throw new Error('Active Rokid RTMP session was created by the web frontend, not the Rokid Glass app. Stop that stale session or start recording on the glasses again, then retry Rokid RTMP.')
        }
        const activeLiveIngest = info.liveIngest || info.streamInfo?.liveIngest || info.raw?.live_ingest || info.raw?.stream?.live_ingest || {}
        const activeLiveIngestStatus = String(activeLiveIngest.status || '').toLowerCase()
        if (LIVE_INGEST_FAILED_STATES.has(activeLiveIngestStatus)) {
          const reason = activeLiveIngest.lastError || info.message || 'Rokid RTMP live ingest is not running.'
          throw new Error(`Active Rokid RTMP session is stale or failed. Start recording on the Rokid Glass app again, then retry Rokid RTMP. ${reason}`)
        }
      } else {
        const activeInfo = await getActiveRokidStream().catch(() => null)
        info = activeInfo?.active && activeInfo.sessionId
          ? activeInfo
          : await startRokidStream({
              inputMode: requestedInputMode,
              ownerId: identity.ownerId,
              deviceId: identity.rokidDeviceId,
              deviceType: 'rokid',
              metadata: {
                client: 'web_frontend',
                flow: requestedInputMode
              }
            })
      }
      if (epochRef.current !== epoch) return null
      if (!info.sessionId) {
        throw new Error('Rokid start failed: session_id missing')
      }
      const attachedInputMode = info.inputMode || requestedInputMode
      const isRokidLive = attachedInputMode === INPUT_MODES.ROKID_LIVE_RTMP
      if (attachedInputMode !== INPUT_MODES.ROKID && attachedInputMode !== INPUT_MODES.ROKID_LIVE_RTMP) {
        throw new Error(`Rokid API returned ${attachedInputMode}, not Rokid Glass`)
      }
      if (requestedInputMode === INPUT_MODES.ROKID_LIVE_RTMP && !isRokidLive) {
        throw new Error(`Active Rokid session is ${attachedInputMode || 'unknown'}, not Rokid RTMP. Start RTMP recording on the Rokid Glass app first.`)
      }
      const live = info.live || info.streamInfo?.live || info.raw?.live || {}
      const webrtcPlayUrl = live.webrtcPlayUrlPublic || live.webrtcPlayUrl || info.webrtcPlayUrl || ''

      inputModeRef.current = attachedInputMode
      setInputModeState(attachedInputMode)
      activeRef.current = true
      activateSession({
        ...info,
        inputMode: attachedInputMode,
        webrtcPlayUrl,
        streamName: info.streamName || live.streamName || live.stream_name || '',
        status: 'stream_attached',
        streamInfo: info.streamInfo || info
      })
      setStats((current) => ({
        ...current,
        canAsk: info.canAsk,
        firstFrameUploaded: (info.rokid?.framesReceived || info.raw?.rokid?.frames_received || 0) > 0,
        frameUploadedCount: info.rokid?.framesReceived || info.raw?.rokid?.frames_received || 0,
        audioUploadedCount: info.rokid?.audioChunksReceived || info.raw?.rokid?.audio_chunks_received || 0
      }))
      setStatusSnapshot(info.streamInfo || info)
      if (isRokidLive) {
        setStatus('publishing')
        setWebrtcState('ingest_starting')
        const ingestResult = await startRokidLiveIngest(info.sessionId)
        if (!isCurrentEpoch(epoch)) return null
        liveIngestStartedRef.current = true
        applyStatusSnapshot(ingestResult, { updateLiveState: true })
        setWebrtcState('live')
        setStatus('live')
        // Frontend RTMP playback is intentionally disabled; backend ingest above still receives the glasses video.
        // playbackRef.current = await playSrsWebRtc(webrtcPlayUrl, videoRef.current, ...)
      } else {
        setStatus('running')
        setWebrtcState('idle')
      }
      startStatusPolling(epoch)
      if (!isRokidLive) {
        startRokidPreviewPolling(epoch)
      }
      return info
    } catch (error) {
      const message = formatStartError(error)
      epochRef.current += 1
      activeRef.current = false
      stopLocalMedia()
      setWebrtcState('idle')
      if (options.silent && (message.includes('Rokid start failed') || message.includes('not Rokid Glass'))) {
        setStatus('idle')
        setStreamError('')
      } else {
        setStatus('failed')
        setStreamError(message)
      }
      return null
    }
  }, [activateSession, applyStatusSnapshot, isCurrentEpoch, prepareStart, startRokidPreviewPolling, startStatusPolling, stopLocalMedia])

  const start = useCallback(() => {
    if (inputModeRef.current === INPUT_MODES.DEMO_TEST) {
      return startDemoTestVideo()
    }
    if (inputModeRef.current === INPUT_MODES.DEMO_VIDEO) {
      return startDemoVideo()
    }
    if (inputModeRef.current === INPUT_MODES.ROKID || inputModeRef.current === INPUT_MODES.ROKID_LIVE_RTMP) {
      return startRokidSession({ inputMode: inputModeRef.current })
    }
    if (inputModeRef.current === INPUT_MODES.WEBRTC_WHIP) {
      return startWebRtcWhip()
    }
    return startFrameAudio()
  }, [startRokidSession, startDemoTestVideo, startDemoVideo, startFrameAudio, startWebRtcWhip])

  const refreshStatus = useCallback(async () => {
    const currentSessionId = sessionIdRef.current
    if (!currentSessionId) return null
    setDiagnosticsLoading(true)
    setStatusError('')

    try {
      const useRokidStatus = [INPUT_MODES.ROKID, INPUT_MODES.ROKID_LIVE_RTMP].includes(activeSessionRef.current.inputMode)
      const useDemoTestStatus = (activeSessionRef.current.inputMode || inputModeRef.current) === INPUT_MODES.DEMO_TEST
      const result = useDemoTestStatus
        ? await getDemoTestStatus(currentSessionId, { baseUrl: demoTestSessionRef.current.demoApiBaseUrl || DEMO_API_BASE_URL })
        : (useRokidStatus ? await getRokidStatus(currentSessionId) : await getStreamStatus(currentSessionId))
      if (useDemoTestStatus) {
        updateDemoTestSession((current) => ({
          ...current,
          memoryReady: !!(result.memoryReady || result.memory?.ready || current.memoryReady),
          status: result,
          raw: result.raw || result
        }))
      }
      applyStatusSnapshot(result, { updateLiveState: activeRef.current })
      return result
    } catch (error) {
      setStatusError(error.message || 'Status refresh failed')
      return null
    } finally {
      setDiagnosticsLoading(false)
    }
  }, [applyStatusSnapshot, updateDemoTestSession])

  const uploadDemo = useCallback(async (file, options = {}) => {
    if (!file || ['starting', 'running', 'publishing', 'live', 'paused', 'stopping'].includes(status)) return null
    setDemoUploading(true)
    setDemoError('')
    setStreamError('')

    try {
      const identity = clientIdentityRef.current
      const info = await uploadDemoVideo(file, {
        sampleFps: options.sampleFps ?? 1,
        autoPrepare: options.autoPrepare !== false,
        enqueuePreprocess: !!options.enqueuePreprocess,
        ownerId: identity.ownerId,
        deviceId: `demo_${identity.webDeviceId}`,
        deviceType: 'demo',
        metadata: {
          client: 'web_frontend',
          flow: INPUT_MODES.DEMO_VIDEO
        }
      })
      if (!info.sessionId || !info.videoUrl) {
        throw new Error('Demo upload did not return session_id or video_url.')
      }

      demoSessionRef.current = info
      setDemoSession(info)
      inputModeRef.current = INPUT_MODES.DEMO_VIDEO
      setInputModeState(INPUT_MODES.DEMO_VIDEO)
      clearActiveSession()
      activeSessionRef.current = EMPTY_ACTIVE_SESSION
      sessionIdRef.current = info.sessionId
      setStatus('idle')
      setStats((current) => ({
        ...initialStats,
        canAsk: !!info.prepared,
        firstFrameUploaded: !!info.prepared,
        frameUploadedCount: info.frameCount || current.frameUploadedCount
      }))
      setStatusSnapshot(info)
      await new Promise((resolve) => window.requestAnimationFrame(resolve))
      bindDemoVideo(info)
      return info
    } catch (error) {
      const message = formatDemoError(error, 'Demo upload failed')
      setDemoError(message)
      setStreamError(message)
      return null
    } finally {
      setDemoUploading(false)
    }
  }, [bindDemoVideo, clearActiveSession, status])

  const uploadDemoTest = useCallback(async (day1File, day2File, options = {}) => {
    if (!day1File || !day2File || ['starting', 'running', 'publishing', 'live', 'paused', 'stopping'].includes(status)) return null
    setDemoTestUploading(true)
    setDemoError('')
    setStreamError('')

    try {
      const identity = clientIdentityRef.current
      const info = await uploadDemoTestVideos(day1File, day2File, {
        sampleFps: options.sampleFps ?? 1,
        autoPrepare: options.autoPrepare !== false,
        enqueueOffline: !!options.enqueueOffline,
        ownerId: identity.ownerId,
        deviceId: `demo_test_${identity.webDeviceId}`,
        deviceType: 'demo-test',
        metadata: {
          client: 'web_frontend',
          flow: INPUT_MODES.DEMO_TEST
        }
      })
      if (!info.sessionId || !info.clips.length) {
        throw new Error('Demo-test upload did not return session_id or clips.')
      }

      demoTestSessionRef.current = info
      setDemoTestSession(info)
      inputModeRef.current = INPUT_MODES.DEMO_TEST
      setInputModeState(INPUT_MODES.DEMO_TEST)
      clearActiveSession()
      activeSessionRef.current = EMPTY_ACTIVE_SESSION
      sessionIdRef.current = info.sessionId
      setStatus('idle')
      setStats({
        ...initialStats,
        canAsk: !!info.prepared,
        firstFrameUploaded: !!info.prepared
      })
      setStatusSnapshot(info)
      clearDemoClockLoop()
      window.requestAnimationFrame(() => bindDemoTestVideo(info))
      return info
    } catch (error) {
      const message = formatDemoError(error, 'Demo-test upload failed')
      setDemoError(message)
      setStreamError(message)
      return null
    } finally {
      setDemoTestUploading(false)
    }
  }, [bindDemoTestVideo, clearActiveSession, clearDemoClockLoop, status])

  const setDemoTestActiveClip = useCallback((clipId) => {
    if (!clipId || ['starting', 'running', 'publishing', 'live', 'stopping'].includes(status)) return
    const current = demoTestSessionRef.current
    const clip = current.clips.find((item) => item.clipId === clipId)
    if (!clip) return
    const nextSession = updateDemoTestSession({
      activeClipId: clipId,
      displayDate: clip.displayDate || '',
      displayTime: clip.startTime || '',
      displayDatetime: '',
      localCurrentTime: 0,
      currentTime: 0
    })
    sessionIdRef.current = nextSession.sessionId
    clearDemoClockLoop()
    window.requestAnimationFrame(() => {
      if (inputModeRef.current === INPUT_MODES.DEMO_TEST) bindDemoTestVideo(nextSession)
      if (videoRef.current) setVideoCurrentTime(videoRef.current, 0)
    })
  }, [bindDemoTestVideo, clearDemoClockLoop, status, updateDemoTestSession])

  const refreshDemoTestStatus = useCallback(async () => {
    const currentSessionId = demoTestSessionRef.current.sessionId
    if (!currentSessionId) return null
    setDemoTestBusy(true)
    setStatusError('')
    try {
      const result = await getDemoTestStatus(currentSessionId, {
        baseUrl: demoTestSessionRef.current.demoApiBaseUrl || DEMO_API_BASE_URL
      })
      updateDemoTestSession((current) => ({
        ...current,
        memoryReady: !!(result.memoryReady || result.memory?.ready || current.memoryReady),
        status: result,
        raw: result.raw || result
      }))
      setStatusSnapshot(result)
      return result
    } catch (error) {
      setStatusError(error.message || 'Demo-test status refresh failed')
      return null
    } finally {
      setDemoTestBusy(false)
    }
  }, [updateDemoTestSession])

  const enqueueDemoTestMemory = useCallback(async (options = {}) => {
    const currentSessionId = demoTestSessionRef.current.sessionId
    if (!currentSessionId) return null
    setDemoTestBusy(true)
    setStatusError('')
    try {
      const result = await enqueueDemoTestOffline(currentSessionId, {
        forcePreprocess: !!options.forcePreprocess,
        enqueueEvidence: !!options.enqueueEvidence,
        forceEvidence: !!options.forceEvidence,
        baseUrl: demoTestSessionRef.current.demoApiBaseUrl || DEMO_API_BASE_URL
      })
      updateDemoTestSession((current) => ({
        ...current,
        status: result,
        raw: result.raw || result
      }))
      setStatusSnapshot(result)
      return result
    } catch (error) {
      setStatusError(error.message || 'Demo-test offline queue failed')
      return null
    } finally {
      setDemoTestBusy(false)
    }
  }, [updateDemoTestSession])

  const buildDemoTestParentMemory = useCallback(async (options = {}) => {
    const currentSessionId = demoTestSessionRef.current.sessionId
    if (!currentSessionId) return null
    setDemoTestBusy(true)
    setStatusError('')
    try {
      const result = await buildDemoTestMemory(currentSessionId, {
        force: options.force !== false,
        allowManifestFallback: !!options.allowManifestFallback,
        skipSemantic: !!options.skipSemantic,
        baseUrl: demoTestSessionRef.current.demoApiBaseUrl || DEMO_API_BASE_URL
      })
      updateDemoTestSession((current) => ({
        ...current,
        memoryReady: !!(result.memoryReady || result.memory?.ready || result.status === 'done' || current.memoryReady),
        status: result,
        raw: result.raw || result
      }))
      setStatusSnapshot(result)
      return result
    } catch (error) {
      setStatusError(error.message || 'Demo-test memory build failed')
      return null
    } finally {
      setDemoTestBusy(false)
    }
  }, [updateDemoTestSession])

  const syncBeforeAsk = useCallback(async () => {
    const activeInputMode = activeSessionRef.current.inputMode || inputModeRef.current
    const isRokid = activeInputMode === INPUT_MODES.ROKID || activeInputMode === INPUT_MODES.ROKID_LIVE_RTMP
    if (isRokid) {
      const currentSessionId = activeSessionRef.current.sessionId
      if (!currentSessionId) return null
      try {
        const result = await getRokidStatus(currentSessionId)
        applyStatusSnapshot(result, { updateLiveState: activeRef.current })
        return result
      } catch (error) {
        return null
      }
    }
    if (activeInputMode !== INPUT_MODES.DEMO_VIDEO && activeInputMode !== INPUT_MODES.DEMO_TEST) return null
    const currentSessionId = activeSessionRef.current.sessionId ||
      (activeInputMode === INPUT_MODES.DEMO_TEST ? demoTestSessionRef.current.sessionId : demoSessionRef.current.sessionId)
    if (!currentSessionId) return null
    return sendDemoTick({
      sessionId: currentSessionId,
      inputMode: activeInputMode,
      clipId: activeInputMode === INPUT_MODES.DEMO_TEST ? demoTestSessionRef.current.activeClipId : undefined
    })
  }, [sendDemoTick, applyStatusSnapshot, getRokidStatus])

  const setInputMode = useCallback((nextInputMode) => {
    if (!Object.values(INPUT_MODES).includes(nextInputMode)) return
    if (['starting', 'running', 'publishing', 'live', 'paused', 'stopping'].includes(status)) return

    inputModeRef.current = nextInputMode
    setInputModeState(nextInputMode)
    setWebrtcState('idle')
    clearActiveSession()
    if (nextInputMode === INPUT_MODES.DEMO_VIDEO) {
      sessionIdRef.current = demoSessionRef.current.sessionId
      window.requestAnimationFrame(() => bindDemoVideo(demoSessionRef.current))
    } else if (nextInputMode === INPUT_MODES.DEMO_TEST) {
      sessionIdRef.current = demoTestSessionRef.current.sessionId
      clearDemoClockLoop()
      window.requestAnimationFrame(() => bindDemoTestVideo(demoTestSessionRef.current))
    } else {
      sessionIdRef.current = ''
    }
    setStatus('idle')
    setStatusSnapshot(null)
    setStats(initialStats)
    setStreamError('')
    setAudioError('')
  }, [bindDemoTestVideo, bindDemoVideo, clearActiveSession, clearDemoClockLoop, status])

  useEffect(() => {
    return () => {
      const currentSession = activeSessionRef.current
      const currentSessionId = currentSession.sessionId
      const currentInputMode = currentSession.inputMode
      epochRef.current += 1
      activeRef.current = false
      stopLocalMedia()
      if (currentInputMode === INPUT_MODES.WEBRTC_WHIP && currentSessionId) {
        stopLiveIngest(currentSessionId).catch(() => {})
      }
      if (currentInputMode === INPUT_MODES.ROKID_LIVE_RTMP && currentSessionId) {
        stopRokidLiveIngest(currentSessionId).catch(() => {})
      }
      if (currentInputMode === INPUT_MODES.DEMO_VIDEO && currentSessionId) {
        stopDemoPlayback(currentSessionId, getDemoPlaybackState()).catch(() => {})
      }
      if (currentInputMode === INPUT_MODES.DEMO_TEST && currentSessionId) {
        tickDemoTestPlayback(currentSessionId, {
          ...getDemoPlaybackState(),
          paused: true,
          clipId: demoTestSessionRef.current.activeClipId || 'day1',
          baseUrl: demoTestSessionRef.current.demoApiBaseUrl || DEMO_API_BASE_URL
        }).catch(() => {})
      }
    }
  }, [getDemoPlaybackState, stopLocalMedia])

  const isWebRtcMode = inputMode === INPUT_MODES.WEBRTC_WHIP
  const activeSessionId = activeSession.sessionId
  const activeInputMode = activeSession.inputMode
  const activeIsWebRtc = activeInputMode === INPUT_MODES.WEBRTC_WHIP
  const activeIsRokidLive = activeInputMode === INPUT_MODES.ROKID_LIVE_RTMP
  const activeIsRokid = activeInputMode === INPUT_MODES.ROKID || activeIsRokidLive
  const activeIsDemo = activeInputMode === INPUT_MODES.DEMO_VIDEO
  const activeIsDemoTest = activeInputMode === INPUT_MODES.DEMO_TEST
  const isLegacyDemoMode = inputMode === INPUT_MODES.DEMO_VIDEO || activeIsDemo
  const isDemoTestMode = inputMode === INPUT_MODES.DEMO_TEST || activeIsDemoTest
  const isDemoMode = isLegacyDemoMode || isDemoTestMode
  const isRokidLiveMode = inputMode === INPUT_MODES.ROKID_LIVE_RTMP || activeIsRokidLive
  const isRokidMode = inputMode === INPUT_MODES.ROKID || inputMode === INPUT_MODES.ROKID_LIVE_RTMP || activeIsRokid
  const canStart = ['idle', 'stopped', 'error', 'failed'].includes(status)
  const isLive = ['running', 'publishing', 'live', 'preview_fallback'].includes(status)
  const isPaused = status === 'paused'
  const isBusy = status === 'starting' || status === 'stopping'
  const canUseMediaDevices = typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia
  const canFlipCamera = !activeIsRokid && !isDemoMode && !isBusy && status !== 'publishing' && !cameraSwitching && canUseMediaDevices
  const canPause = !isWebRtcMode && !activeIsRokid && status === 'running'
  const memoryReady = !!(stats.canAsk || stats.firstFrameUploaded || statusSnapshot?.canAsk)
  const demoSessionId = isDemoTestMode ? demoTestSession.sessionId : demoSession.sessionId
  const effectiveCanAsk = !!(activeSessionId || (isDemoMode && demoSessionId)) && (activeIsWebRtc || activeIsRokidLive || isDemoMode || memoryReady)
  const effectiveSessionId = activeSessionId || (isDemoMode ? demoSessionId : '')
  const activeDemoTestClip = getDemoTestActiveClip(demoTestSession)
  const demoVideoUrl = isDemoTestMode ? (activeRef.current ? (activeDemoTestClip?.videoUrlResolved || '') : '') : demoSession.videoUrl
  const askBaseUrl = isDemoTestMode ? (demoTestSession.demoApiBaseUrl || DEMO_API_BASE_URL) : undefined
  const askStreamEndpoint = isDemoTestMode ? (demoTestSession.askStreamUrl || (effectiveSessionId ? `/ask/${effectiveSessionId}/stream` : '')) : ''
  const streamInfo = activeSession.streamInfo
  const liveIngest = statusSnapshot?.liveIngest || streamInfo?.liveIngest || {}
  const live = statusSnapshot?.live || streamInfo?.live || {}
  const liveIngestStatus = liveIngest.status || live.ingestStatus || 'not_started'
  const streamName = activeSession.streamName
  const whipUrl = activeSession.whipUrl
  const whipUrlAvailable = !!whipUrl
  const webrtcPlayUrl = activeSession.webrtcPlayUrl || live.webrtcPlayUrlPublic || live.webrtcPlayUrl || ''
  const webrtcPlayUrlAvailable = !!webrtcPlayUrl
  const frameStream = statusSnapshot?.frameStream || streamInfo?.frameStream || {}
  const rokidPreviewPath = activeIsRokid
    ? (stats.latestFramePath || frameStream.latestFramePath || frameStream.latestCurrentFramePath || '')
    : ''
  const rokidPreviewBaseUrl = rokidPreviewPath
    ? buildEvidenceFrameUrl(activeSessionId, rokidPreviewPath)
    : ''
  const rokidPreviewVersion = stats.lastFrameAt || frameStream.latestFrameAt || frameStream.latestFrameIndex || stats.frameUploadedCount || Date.now()
  const rokidPreviewUrl = rokidPreviewBaseUrl && rokidPreviewVersion
    ? `${rokidPreviewBaseUrl}&v=${encodeURIComponent(rokidPreviewVersion)}`
    : rokidPreviewBaseUrl

  return useMemo(() => ({
    videoRef,
    canvasRef,
    inputMode,
    setInputMode,
    isWebRtcMode,
    modeLabel: isDemoTestMode ? 'Demo Test' : (isLegacyDemoMode ? 'Demo Video' : (isRokidMode ? (isRokidLiveMode ? 'Rokid RTMP Live' : 'Rokid Glass') : (isWebRtcMode ? 'WebRTC Live Stream' : 'Frame · Audio HTTP'))),
    isDemoMode,
    isLegacyDemoMode,
    isDemoTestMode,
    demoSession,
    demoTestSession,
    demoVideoUrl,
    demoClockText,
    askBaseUrl,
    askStreamEndpoint,
    demoUploading,
    demoTestUploading,
    demoTestBusy,
    demoError,
    isRokidMode,
    isRokidLiveMode,
    isPreviewFallback: status === 'preview_fallback',
    status,
    webrtcState,
    cameraFacingMode,
    cameraFacingLabel: cameraFacingMode === CAMERA_FACING.FRONT ? 'Front camera' : 'Rear camera',
    cameraSwitching,
    activeSession,
    activeSessionId,
    activeInputMode,
    sessionId: effectiveSessionId,
    streamInfo,
    streamName,
    whipUrl,
    whipUrlAvailable,
    webrtcPlayUrl,
    webrtcPlayUrlAvailable,
    rokidPreviewUrl,
    rokidPreviewPath,
    rokidPreviewUpdatedAt: frameStream.latestFrameAt || '',
    rokidPreviewFps: stats.previewFps ?? frameStream.previewFps ?? null,
    rokidMemoryFps: stats.memoryFps ?? frameStream.memoryFps ?? null,
    rokidMemoryTargetFps: stats.memoryTargetFps ?? frameStream.memoryTargetFps ?? frameStream.targetFps ?? 1,
    streamError,
    audioError,
    stats,
    statusSnapshot,
    statusError,
    diagnosticsLoading,
    liveIngest,
    liveIngestStatus,
    liveIngestFrames: liveIngest.framesIngested ?? live.framesIngested ?? 0,
    liveIngestAudioChunks: liveIngest.audioChunksIngested ?? live.audioChunksIngested ?? 0,
    liveIngestLastError: liveIngest.lastError || live.lastError || '',
    canStart,
    isLive,
    isPaused,
    isBusy,
    canFlipCamera,
    canPause,
    memoryReady,
    canAsk: effectiveCanAsk,
    start,
    stop,
    pause,
    resume,
    uploadDemo,
    uploadDemoTest,
    setDemoTestActiveClip,
    refreshDemoTestStatus,
    enqueueDemoTestMemory,
    buildDemoTestParentMemory,
    syncBeforeAsk,
    startRokidSession,
    toggleCamera,
    reset,
    refreshStatus
  }), [
    inputMode,
    setInputMode,
    isWebRtcMode,
    status,
    webrtcState,
    cameraFacingMode,
    cameraSwitching,
    activeSession,
    activeSessionId,
    effectiveSessionId,
    activeInputMode,
    streamInfo,
    streamName,
    whipUrl,
    whipUrlAvailable,
    webrtcPlayUrl,
    webrtcPlayUrlAvailable,
    rokidPreviewUrl,
    rokidPreviewPath,
    frameStream,
    stats.previewFps,
    stats.memoryFps,
    stats.memoryTargetFps,
    streamError,
    audioError,
    stats,
    statusSnapshot,
    statusError,
    diagnosticsLoading,
    liveIngest,
    liveIngestStatus,
    live,
    canStart,
    isLive,
    isPaused,
    isBusy,
    canFlipCamera,
    canPause,
    memoryReady,
    effectiveCanAsk,
    isDemoMode,
    isLegacyDemoMode,
    isDemoTestMode,
    demoSession,
    demoTestSession,
    demoVideoUrl,
    demoClockText,
    askBaseUrl,
    askStreamEndpoint,
    demoUploading,
    demoTestUploading,
    demoTestBusy,
    demoError,
    isRokidMode,
    isRokidLiveMode,
    start,
    stop,
    pause,
    resume,
    uploadDemo,
    uploadDemoTest,
    setDemoTestActiveClip,
    refreshDemoTestStatus,
    enqueueDemoTestMemory,
    buildDemoTestParentMemory,
    syncBeforeAsk,
    startRokidSession,
    toggleCamera,
    reset,
    refreshStatus
  ])
}

async function requestCameraAndMic(options = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia is not supported by this browser')
  }

  try {
    return await navigator.mediaDevices.getUserMedia({
      video: buildVideoConstraints(options.facingMode),
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    })
  } catch (error) {
    if (!options.allowVideoOnly) {
      throw new Error(formatMediaError(error))
    }

    try {
      return await navigator.mediaDevices.getUserMedia({
        video: buildVideoConstraints(options.facingMode),
        audio: false
      })
    } catch (videoError) {
      throw new Error(formatMediaError(videoError))
    }
  }
}

async function requestCameraOnly(facingMode) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia is not supported by this browser')
  }

  return navigator.mediaDevices.getUserMedia({
    video: buildVideoConstraints(facingMode),
    audio: false
  })
}

function buildVideoConstraints(facingMode = DEFAULT_CAMERA_FACING) {
  return {
    facingMode: { ideal: facingMode },
    width: { ideal: 1280 },
    height: { ideal: 720 }
  }
}

function getClientIdentity() {
  const fallbackId = `web_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
  if (typeof window === 'undefined') {
    return {
      ownerId: 'web_user',
      webDeviceId: fallbackId,
      rokidDeviceId: `rokid_${fallbackId}`
    }
  }

  const storage = window.localStorage
  const ownerId = storage.getItem('lightmem_owner_id') || 'web_user'
  let webDeviceId = storage.getItem('lightmem_web_device_id')
  let rokidDeviceId = storage.getItem('lightmem_rokid_device_id')

  if (!webDeviceId) {
    webDeviceId = fallbackId
    storage.setItem('lightmem_web_device_id', webDeviceId)
  }

  if (!rokidDeviceId) {
    rokidDeviceId = `rokid_${webDeviceId}`
    storage.setItem('lightmem_rokid_device_id', rokidDeviceId)
  }

  return {
    ownerId,
    webDeviceId,
    rokidDeviceId
  }
}

function toFiniteNumber(value, fallback = 0) {
  const number = Number(value)
  return Number.isFinite(number) ? number : fallback
}

function getDemoTestActiveClip(session = EMPTY_DEMO_TEST_SESSION) {
  const clips = Array.isArray(session.clips) ? session.clips : []
  return clips.find((clip) => clip.clipId === session.activeClipId) || clips[0] || null
}

function formatDemoTestClockAt(clip = null, currentTime = 0) {
  const start = parseDemoTestStart(clip)
  if (!start) return ''
  const elapsedSeconds = Math.max(0, Math.floor(Number(currentTime) || 0))
  const date = new Date(start.getTime() + elapsedSeconds * 1000)
  return formatEnglishDateTime(date)
}

function parseDemoTestStart(clip = null) {
  const rawDate = String(clip?.displayDate || '').trim()
  const rawTime = String(clip?.startTime || '').trim()
  const dateMatch = rawDate.match(/(\d{4})\D+(\d{1,2})\D+(\d{1,2})/)
  const timeMatch = rawTime.match(/(\d{1,2}):(\d{2})(?::(\d{2}))?/)
  if (!dateMatch || !timeMatch) return null
  const [, year, month, day] = dateMatch
  const [, hour, minute, second = '0'] = timeMatch
  return new Date(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second)
  )
}

function formatEnglishDateTime(date) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return ''
  const month = date.toLocaleString('en-US', { month: 'long' })
  const day = date.getDate()
  const year = date.getFullYear()
  const time = [
    date.getHours(),
    date.getMinutes(),
    date.getSeconds()
  ].map((value) => String(value).padStart(2, '0')).join(':')
  return `${month} ${day}, ${year} ${time}`
}

function withTimeout(promise, timeoutMs, onTimeout) {
  return new Promise((resolve, reject) => {
    let settled = false
    const timer = window.setTimeout(() => {
      if (settled) return
      settled = true
      onTimeout?.()
      const error = new Error('timed out waiting for SRS WebRTC playback')
      error.name = 'TimeoutError'
      reject(error)
    }, timeoutMs)

    promise.then(
      (value) => {
        if (settled) return
        settled = true
        window.clearTimeout(timer)
        resolve(value)
      },
      (error) => {
        if (settled) return
        settled = true
        window.clearTimeout(timer)
        reject(error)
      }
    )
  })
}

function getNextCameraFacingMode(facingMode) {
  return facingMode === CAMERA_FACING.FRONT ? CAMERA_FACING.BACK : CAMERA_FACING.FRONT
}

function bindVideoElementSource(video, sourceUrl) {
  if (!video || !sourceUrl) return null
  const resolvedSource = resolveMediaUrl(sourceUrl)
  video.srcObject = null
  video.controls = false
  video.loop = false
  video.muted = true
  video.playsInline = true

  if (video.src !== resolvedSource) {
    video.pause?.()
    video.src = resolvedSource
    video.load?.()
  }

  return video
}

function resolveMediaUrl(sourceUrl) {
  const rawSource = String(sourceUrl || '')
  if (!rawSource) return ''
  try {
    return new URL(rawSource, window.location.href).href
  } catch (error) {
    return rawSource
  }
}

function waitForMediaMetadata(video, timeoutMs = 8000) {
  if (video.readyState >= 1) return Promise.resolve()

  return new Promise((resolve, reject) => {
    let settled = false
    let timer = null
    const cleanup = () => {
      window.clearTimeout(timer)
      video.removeEventListener('loadedmetadata', handleLoaded)
      video.removeEventListener('error', handleError)
    }
    const finish = (callback) => {
      if (settled) return
      settled = true
      cleanup()
      callback()
    }
    const handleLoaded = () => finish(resolve)
    const handleError = () => finish(() => reject(new Error('Demo video failed to load.')))

    timer = window.setTimeout(() => {
      finish(() => reject(new Error('Demo video metadata was not ready in time.')))
    }, timeoutMs)

    video.addEventListener('loadedmetadata', handleLoaded)
    video.addEventListener('error', handleError)
  })
}

function setVideoCurrentTime(video, currentTime) {
  try {
    video.currentTime = currentTime
  } catch (error) {
    // Some browsers reject seeking before enough metadata is available.
  }
}

function nextAnimationFrame() {
  return new Promise((resolve) => {
    window.requestAnimationFrame(() => resolve())
  })
}

function waitForVideoReady(video) {
  return new Promise((resolve, reject) => {
    let settled = false
    const finish = (callback) => {
      if (settled) return
      settled = true
      window.clearTimeout(timer)
      video.onloadedmetadata = null
      callback()
    }
    const play = () => {
      video.play()
        .then(() => finish(resolve))
        .catch((error) => finish(() => reject(error)))
    }
    const timer = window.setTimeout(() => {
      if (video.videoWidth) {
        play()
      } else {
        finish(() => reject(new Error('Camera preview is not ready yet.')))
      }
    }, 5000)

    if (video.readyState >= 2 && video.videoWidth) {
      play()
      return
    }

    video.onloadedmetadata = play
  })
}

function attachPreviewStream(video, stream) {
  if (!video) return
  video.srcObject = stream
  const playPromise = video.play?.()
  if (playPromise?.catch) {
    playPromise.catch(() => {
      // waitForVideoReady handles the blocking preview check where needed.
    })
  }
}

async function captureVideoBlob(video, canvas) {
  const sourceWidth = video.videoWidth
  const sourceHeight = video.videoHeight
  const targetWidth = Math.min(960, sourceWidth)
  const targetHeight = Math.round(targetWidth * (sourceHeight / sourceWidth))
  canvas.width = targetWidth
  canvas.height = targetHeight

  const context = canvas.getContext('2d')
  context.drawImage(video, 0, 0, targetWidth, targetHeight)

  const webp = await canvasToBlob(canvas, 'image/webp', 0.72)
  if (webp && webp.size && webp.type.includes('webp')) {
    return { blob: webp, format: 'webp', width: targetWidth, height: targetHeight }
  }

  const jpg = await canvasToBlob(canvas, 'image/jpeg', 0.78)
  return { blob: jpg, format: 'jpg', width: targetWidth, height: targetHeight }
}

function canvasToBlob(canvas, mimeType, quality) {
  return new Promise((resolve) => {
    canvas.toBlob(resolve, mimeType, quality)
  })
}

function pickAudioMimeType() {
  const options = [
    'audio/mp4',
    'audio/mp4;codecs=mp4a.40.2',
    'audio/aac',
    'audio/mpeg',
    'audio/wav',
    'audio/webm;codecs=opus',
    'audio/webm'
  ]

  return options.find((type) => window.MediaRecorder?.isTypeSupported(type)) || ''
}

function inferAudioFormat(mimeType = '') {
  if (mimeType.includes('mp4')) return 'm4a'
  if (mimeType.includes('aac')) return 'aac'
  if (mimeType.includes('mpeg')) return 'mp3'
  if (mimeType.includes('wav')) return 'wav'
  return 'webm'
}

function formatMediaError(error) {
  if (error?.name === 'NotAllowedError' || error?.name === 'SecurityError') {
    return 'Camera/microphone permission denied'
  }
  if (error?.name === 'NotFoundError') {
    return 'Camera or microphone not found'
  }
  return error?.message || 'Could not access camera and microphone'
}

function formatCameraSwitchError(error) {
  const message = formatMediaError(error)
  return message === 'Could not access camera and microphone'
    ? 'Could not switch camera'
    : message
}

function formatStartError(error) {
  const message = error?.message || 'Could not start realtime stream'
  if (message.includes('WHIP URL missing')) return 'WHIP URL missing'
  if (message.includes('SDP answer invalid')) return message
  if (message.includes('WHIP publish failed')) return message
  if (message.includes('ICE failed')) return message
  return message
}

function formatDemoError(error, fallback) {
  const status = error?.status || error?.raw?.status
  const message = error?.message || error?.raw?.message || fallback
  if (status === 404) {
    return 'Demo routes are unavailable on the backend. Restart the normal API after the demo-route update and confirm POST /demo/upload is registered.'
  }
  if (status === 409 || String(message).includes('not_ready')) {
    return 'Demo memory is not ready yet. Start playback and let at least one tick reach the backend, then retry.'
  }
  return message || fallback
}
