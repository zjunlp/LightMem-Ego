const DEFAULT_ICE_TIMEOUT_MS = 10000
const DEFAULT_CONNECTION_TIMEOUT_MS = 20000

export async function publishWhip(whipUrl, localStream, options = {}) {
  if (!whipUrl) {
    throw new Error('WHIP URL missing')
  }

  if (!localStream || !localStream.getTracks().length) {
    throw new Error('Local media stream is unavailable')
  }

  if (!window.RTCPeerConnection) {
    throw new Error('WebRTC is not supported by this browser')
  }

  const peerConnection = new RTCPeerConnection(options.rtcConfiguration || {})
  bindPeerConnectionLogging(peerConnection)
  options.onPeerConnection?.(peerConnection)

  try {
    localStream.getTracks().forEach((track) => {
      peerConnection.addTrack(track, localStream)
    })

    const offer = await peerConnection.createOffer()
    await peerConnection.setLocalDescription(offer)
    logPeerConnectionState(peerConnection, 'setLocalDescription')
    await waitForIceGatheringComplete(peerConnection, options.iceTimeoutMs)
    logPeerConnectionState(peerConnection, 'ice-gathering-complete')

    const localDescription = peerConnection.localDescription
    if (!localDescription?.sdp) {
      throw new Error('WHIP publish failed: local SDP offer is missing')
    }
    console.info('[LightMem-Ego WHIP] posting SDP offer', {
      whipUrl,
      candidates: countSdpCandidates(localDescription.sdp),
      sdpLength: localDescription.sdp.length
    })

    let response
    try {
      response = await fetch(whipUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/sdp'
        },
        body: localDescription.sdp,
        signal: options.signal
      })
    } catch (error) {
      console.error('[LightMem-Ego WHIP] HTTP request failed before response', {
        message: error.message || 'network request failed'
      })
      throw new Error(`WHIP publish failed: ${error.message || 'network request failed'}`)
    }

    const answerSdp = await response.text()
    if (!response.ok) {
      console.error('[LightMem-Ego WHIP] HTTP failed', {
        status: response.status,
        statusText: response.statusText,
        body: answerSdp
      })
      throw new Error(`WHIP publish failed: ${response.status} ${answerSdp || response.statusText}`.trim())
    }
    console.info('[LightMem-Ego WHIP] received SDP answer', {
      status: response.status,
      candidates: countSdpCandidates(answerSdp),
      sdpLength: answerSdp.length
    })

    if (!answerSdp || !answerSdp.includes('v=')) {
      throw new Error('SDP answer invalid')
    }

    try {
      await peerConnection.setRemoteDescription({
        type: 'answer',
        sdp: answerSdp
      })
      logPeerConnectionState(peerConnection, 'setRemoteDescription')
    } catch (error) {
      throw new Error(`SDP answer invalid: ${error.message || 'setRemoteDescription failed'}`)
    }

    return {
      peerConnection,
      resourceUrl: response.headers.get('Location') || '',
      answerSdp
    }
  } catch (error) {
    peerConnection.close()
    throw error
  }
}

export function waitForIceGatheringComplete(peerConnection, timeoutMs = DEFAULT_ICE_TIMEOUT_MS) {
  if (peerConnection.iceGatheringState === 'complete') {
    return Promise.resolve()
  }

  return new Promise((resolve, reject) => {
    let settled = false
    const finish = () => {
      if (settled) return
      settled = true
      window.clearTimeout(timer)
      peerConnection.removeEventListener('icegatheringstatechange', onStateChange)
      resolve()
    }
    const timeout = () => {
      if (settled) return
      settled = true
      peerConnection.removeEventListener('icegatheringstatechange', onStateChange)
      logPeerConnectionState(peerConnection, 'ice-gathering-timeout')
      reject(new Error('WHIP publish failed: ICE gathering did not complete before SDP POST'))
    }
    const onStateChange = () => {
      logPeerConnectionState(peerConnection, 'icegatheringstatechange')
      if (peerConnection.iceGatheringState === 'complete') {
        finish()
      }
    }
    const timer = window.setTimeout(timeout, timeoutMs)

    peerConnection.addEventListener('icegatheringstatechange', onStateChange)
  })
}

export function waitForPeerConnectionConnected(peerConnection, timeoutMs = DEFAULT_CONNECTION_TIMEOUT_MS) {
  if (isPeerConnectionConnected(peerConnection)) {
    return Promise.resolve()
  }

  return new Promise((resolve, reject) => {
    let settled = false

    const cleanup = () => {
      window.clearTimeout(timer)
      peerConnection.removeEventListener('iceconnectionstatechange', onStateChange)
      peerConnection.removeEventListener('connectionstatechange', onStateChange)
    }
    const finish = () => {
      if (settled) return
      settled = true
      cleanup()
      logPeerConnectionState(peerConnection, 'connected')
      resolve()
    }
    const fail = (message) => {
      if (settled) return
      settled = true
      cleanup()
      logPeerConnectionState(peerConnection, 'connection-failed')
      reject(new Error(message))
    }
    const onStateChange = () => {
      if (isPeerConnectionConnected(peerConnection)) {
        finish()
        return
      }

      if (peerConnection.iceConnectionState === 'failed' || peerConnection.connectionState === 'failed') {
        fail('WebRTC ICE connection failed')
      }
    }
    const timer = window.setTimeout(() => {
      fail('WebRTC ICE connecting timeout')
    }, timeoutMs)

    peerConnection.addEventListener('iceconnectionstatechange', onStateChange)
    peerConnection.addEventListener('connectionstatechange', onStateChange)
    onStateChange()
  })
}

function isPeerConnectionConnected(peerConnection) {
  return (
    ['connected', 'completed'].includes(peerConnection.iceConnectionState) ||
    peerConnection.connectionState === 'connected'
  )
}

function bindPeerConnectionLogging(peerConnection) {
  peerConnection.addEventListener('icegatheringstatechange', () => {
    logPeerConnectionState(peerConnection, 'icegatheringstatechange')
  })
  peerConnection.addEventListener('iceconnectionstatechange', () => {
    logPeerConnectionState(peerConnection, 'iceconnectionstatechange')
  })
  peerConnection.addEventListener('connectionstatechange', () => {
    logPeerConnectionState(peerConnection, 'connectionstatechange')
  })
}

function logPeerConnectionState(peerConnection, label) {
  console.info('[LightMem-Ego WHIP] peer state', {
    label,
    iceConnectionState: peerConnection.iceConnectionState,
    connectionState: peerConnection.connectionState,
    iceGatheringState: peerConnection.iceGatheringState
  })
}

function countSdpCandidates(sdp = '') {
  return String(sdp).split('\n').filter((line) => line.startsWith('a=candidate:')).length
}

export function assertWebRtcSecureContext() {
  const hostname = window.location.hostname
  const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1'

  if (!window.isSecureContext && !isLocalhost) {
    throw new Error('WebRTC requires HTTPS or localhost')
  }
}
