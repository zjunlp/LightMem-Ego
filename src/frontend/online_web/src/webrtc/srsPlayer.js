const DEFAULT_ICE_TIMEOUT_MS = 10000

export async function playSrsWebRtc(playUrl, videoElement, options = {}) {
  if (!playUrl) {
    throw new Error('SRS WebRTC play URL missing')
  }
  if (!videoElement) {
    throw new Error('Video element is unavailable')
  }
  if (!window.RTCPeerConnection) {
    throw new Error('WebRTC is not supported by this browser')
  }

  const peerConnection = new RTCPeerConnection(options.rtcConfiguration || {})
  const remoteStream = new MediaStream()
  videoElement.srcObject = remoteStream
  videoElement.autoplay = true
  videoElement.muted = true
  videoElement.playsInline = true

  peerConnection.addTransceiver('audio', { direction: 'recvonly' })
  peerConnection.addTransceiver('video', { direction: 'recvonly' })
  peerConnection.ontrack = (event) => {
    const streams = event.streams || []
    const stream = streams[0]
    if (stream) {
      stream.getTracks().forEach((track) => remoteStream.addTrack(track))
    } else if (event.track) {
      remoteStream.addTrack(event.track)
    }
  }

  try {
    const offer = await peerConnection.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: true })
    await peerConnection.setLocalDescription(offer)
    await waitForIceGatheringComplete(peerConnection, options.iceTimeoutMs || DEFAULT_ICE_TIMEOUT_MS)

    const localDescription = peerConnection.localDescription
    if (!localDescription?.sdp) {
      throw new Error('SRS WebRTC play failed: local SDP offer is missing')
    }

    const payload = {
      api: playUrl,
      streamurl: buildSrsStreamUrl(playUrl),
      sdp: localDescription.sdp
    }
    const response = await fetch(playUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: options.signal
    })
    const responseText = await response.text()
    const result = parseSrsResponse(responseText)
    if (!response.ok) {
      throw new Error(('SRS WebRTC play failed: ' + response.status + ' ' + formatSrsError(result, responseText || response.statusText)).trim())
    }
    if (result && Number(result.code) !== 0) {
      throw new Error('SRS WebRTC play failed: ' + formatSrsError(result, responseText))
    }

    const answerSdp = result?.sdp || ''
    if (!answerSdp || !answerSdp.includes('v=')) {
      throw new Error(('SRS WebRTC play failed: SDP answer invalid ' + (responseText || '')).trim())
    }

    await peerConnection.setRemoteDescription({ type: 'answer', sdp: answerSdp })
    await videoElement.play().catch(() => {})
    return {
      peerConnection,
      close() {
        closeSrsWebRtcPlayer(peerConnection, videoElement)
      }
    }
  } catch (error) {
    closeSrsWebRtcPlayer(peerConnection, videoElement)
    throw error
  }
}

export function closeSrsWebRtcPlayer(peerConnection, videoElement) {
  try {
    peerConnection.ontrack = null
    peerConnection.getReceivers?.().forEach((receiver) => receiver.track?.stop?.())
    peerConnection.close?.()
  } catch (error) {
    // Closing is best-effort during page teardown.
  }
  if (videoElement) {
    videoElement.pause?.()
    videoElement.srcObject = null
  }
}

function buildSrsStreamUrl(playUrl) {
  const url = new URL(playUrl, window.location.href)
  const app = url.searchParams.get('app') || 'live'
  const stream = url.searchParams.get('stream') || ''
  const host = url.host || window.location.host
  if (!stream) {
    throw new Error('SRS WebRTC play stream missing')
  }
  return 'webrtc://' + host + '/' + encodeURIComponent(app) + '/' + encodeURIComponent(stream)
}

function parseSrsResponse(text) {
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch (error) {
    return null
  }
}

function formatSrsError(result, fallback) {
  if (!result) return String(fallback || 'unknown error')
  const parts = []
  if (result.code !== undefined) parts.push('code=' + result.code)
  if (result.msg) parts.push(result.msg)
  if (result.message) parts.push(result.message)
  if (result.error) parts.push(result.error)
  return parts.join(' ') || String(fallback || 'unknown error')
}

function waitForIceGatheringComplete(peerConnection, timeoutMs) {
  if (peerConnection.iceGatheringState === 'complete') return Promise.resolve()
  return new Promise((resolve) => {
    const timer = window.setTimeout(finish, timeoutMs)
    function finish() {
      window.clearTimeout(timer)
      peerConnection.removeEventListener('icegatheringstatechange', onChange)
      resolve()
    }
    function onChange() {
      if (peerConnection.iceGatheringState === 'complete') finish()
    }
    peerConnection.addEventListener('icegatheringstatechange', onChange)
  })
}
