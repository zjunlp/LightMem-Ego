import { useEffect, useMemo, useRef, useState } from 'react'
import AskPanel from './components/AskPanel.jsx'
import LiveView from './components/LiveView.jsx'
import { buildEvidenceFrameUrl } from './api/worldmmApi.js'
import { useAskWorldMM } from './hooks/useAskWorldMM.js'
import { useRealtimeStream } from './hooks/useRealtimeStream.js'

const DEMO_TEST_AUTO_QUESTIONS = [
  {
    id: 'day1-defense-prep',
    clipId: 'day1',
    triggerTime: 3,
    question: "Do you know what I'm preparing now?",
    answer: 'It looks like you are preparing for a project defense. I can see slides about an “AI Legal Assistant for Contract Revision,” including the title page and the contents page. The presentation seems to focus on an intelligent contract review and rewriting system based on RAG and large language models.',
    evidenceClipId: 'day1',
    evidenceSeconds: [1, 2, 3],
    answerInitialDelayMs: 1000,
    answerTokenDelayMs: 48,
    typingIntervalMs: 45,
    sendDelayMs: 220
  },
  {
    id: 'day2-defense-memory',
    clipId: 'day2',
    triggerTime: 1,
    question: 'Do you still remember why I was preparing the PPT today?',
    answer: 'Yes. Yesterday, you mentioned that you were preparing for today’s project defense. The retrieved slides show your presentation about an AI legal contract review and revision system, so you were preparing this PPT for that defense.',
    evidenceClipId: 'day1',
    evidenceSeconds: [1, 2, 3],
    answerInitialDelayMs: 1500,
    answerTokenDelayMs: 48,
    typingIntervalMs: 45,
    sendDelayMs: 220
  },
  {
    id: 'day2-usb-memory',
    clipId: 'day2',
    triggerTime: 8,
    question: 'Do you still remember where I put my USB drive?',
    answer: 'Yes. From yesterday’s retrieved evidence, you first placed the USB drive inside a beige notebook, and then put that notebook into the light gray laptop sleeve next to your laptop. So the USB drive should be in the beige notebook inside the light gray laptop sleeve.',
    evidenceClipId: 'day1',
    evidenceSeconds: [13, 14, 22],
    evidenceFramePaths: [
      'demo/demo_test/day1/frames/day1_frame_000014.jpg',
      'demo/demo_test/day1/frames/day1_frame_000015.jpg',
      'demo/demo_test/day1/frames/day1_frame_000023.jpg'
    ],
    answerInitialDelayMs: 1500,
    answerTokenDelayMs: 48,
    typingIntervalMs: 45,
    sendDelayMs: 220
  }
]

export default function App() {
  const [activePanel, setActivePanel] = useState('live')
  const stream = useRealtimeStream()
  const ask = useAskWorldMM(stream.sessionId, {
    syncBeforeAsk: stream.syncBeforeAsk,
    isDemoMode: stream.isDemoMode,
    askBaseUrl: stream.askBaseUrl,
    askStreamEndpoint: stream.askStreamEndpoint
  })

  useDemoTestAutoQuestion(stream, ask, setActivePanel)

  const resetAll = useMemo(() => {
    return () => {
      stream.reset()
      ask.reset()
      setActivePanel('live')
    }
  }, [ask, stream])

  return (
    <main className="app-shell">
      <div className="mobile-tabs" role="tablist" aria-label="LightMem-Ego sections">
        <button className={activePanel === 'live' ? 'active' : ''} type="button" onClick={() => setActivePanel('live')}>Live View</button>
        <button className={activePanel === 'ask' ? 'active' : ''} type="button" onClick={() => setActivePanel('ask')}>Ask & Results</button>
      </div>

      <div className="product-layout">
        <div className={`panel-slot live-slot ${activePanel === 'live' ? 'active' : ''}`}>
          <LiveView
            stream={stream}
            onOpenAsk={() => setActivePanel('ask')}
            onReset={resetAll}
          />
        </div>
        <div className={`panel-slot ask-slot ${activePanel === 'ask' ? 'active' : ''}`}>
          <AskPanel
            stream={stream}
            ask={ask}
            onOpenLive={() => setActivePanel('live')}
            onReset={resetAll}
          />
        </div>
      </div>
    </main>
  )
}

function useDemoTestAutoQuestion(stream, ask, setActivePanel) {
  const askRef = useRef(ask)
  const triggeredPlaybackRef = useRef(new Set())
  const lastPlaybackRef = useRef('')
  const typingTimerRef = useRef(null)
  const autoQuestionBusyRef = useRef(false)

  useEffect(() => {
    askRef.current = ask
  }, [ask])

  useEffect(() => {
    return () => {
      window.clearTimeout(typingTimerRef.current)
      autoQuestionBusyRef.current = false
    }
  }, [])

  useEffect(() => {
    const activeClipId = stream.demoTestSession?.activeClipId || ''
    const playbackKey = `${stream.sessionId || ''}:${activeClipId}`
    const activeScripts = DEMO_TEST_AUTO_QUESTIONS
      .filter((script) => script.clipId === activeClipId)
      .sort((left, right) => left.triggerTime - right.triggerTime)

    if (lastPlaybackRef.current !== playbackKey) {
      lastPlaybackRef.current = playbackKey
      triggeredPlaybackRef.current = new Set()
      autoQuestionBusyRef.current = false
      window.clearTimeout(typingTimerRef.current)
    }

    if (
      !stream.isDemoTestMode ||
      !stream.isLive ||
      stream.isPaused ||
      !stream.sessionId ||
      !activeScripts.length
    ) {
      window.clearTimeout(typingTimerRef.current)
      autoQuestionBusyRef.current = false
      return undefined
    }

    const pollTimer = window.setInterval(() => {
      const video = stream.videoRef?.current
      if (!video) return

      const currentTime = Number(video.currentTime) || 0
      if (currentTime < 0.75 && !askRef.current.loading && !autoQuestionBusyRef.current) {
        triggeredPlaybackRef.current = new Set()
      }

      if (askRef.current.loading || autoQuestionBusyRef.current) return

      const nextScript = activeScripts.find((script) => (
        !triggeredPlaybackRef.current.has(script.id) &&
        currentTime >= script.triggerTime
      ))
      if (!nextScript) return

      triggeredPlaybackRef.current.add(nextScript.id)
      const evidenceFramesPromise = buildDemoTestEvidenceFrames(stream, nextScript.evidenceSeconds, {
        clipId: nextScript.evidenceClipId,
        framePaths: nextScript.evidenceFramePaths
      })
      runTypingQuestion(nextScript.question, {
        askRef,
        setActivePanel,
        timerRef: typingTimerRef,
        busyRef: autoQuestionBusyRef,
        scriptedAnswer: nextScript.answer,
        evidenceFramesPromise,
        sessionId: stream.sessionId,
        answerInitialDelayMs: nextScript.answerInitialDelayMs,
        answerTokenDelayMs: nextScript.answerTokenDelayMs,
        typingIntervalMs: nextScript.typingIntervalMs,
        sendDelayMs: nextScript.sendDelayMs
      })
    }, 150)

    return () => {
      window.clearInterval(pollTimer)
    }
  }, [
    stream.demoTestSession?.activeClipId,
    stream.isDemoTestMode,
    stream.isLive,
    stream.isPaused,
    stream.sessionId,
    stream.demoTestSession?.clips,
    stream.videoRef,
    setActivePanel
  ])
}

function runTypingQuestion(question, options) {
  const {
    askRef,
    setActivePanel,
    timerRef,
    busyRef,
    scriptedAnswer,
    evidenceFramesPromise,
    sessionId,
    answerInitialDelayMs,
    answerTokenDelayMs,
    typingIntervalMs,
    sendDelayMs
  } = options

  window.clearTimeout(timerRef.current)
  if (busyRef) busyRef.current = true
  setActivePanel('ask')
  askRef.current.setQuestion('')

  let index = 0
  const typeNextCharacter = () => {
    index += 1
    askRef.current.setQuestion(question.slice(0, index))

    if (index < question.length) {
      timerRef.current = window.setTimeout(typeNextCharacter, typingIntervalMs)
      return
    }

    timerRef.current = window.setTimeout(() => {
      const finishBusy = () => {
        if (busyRef) busyRef.current = false
      }

      if (scriptedAnswer && askRef.current.runScriptedStream) {
        Promise.resolve(askRef.current.runScriptedStream({
          sessionId,
          question,
          answer: scriptedAnswer,
          initialDelayMs: answerInitialDelayMs,
          tokenDelayMs: answerTokenDelayMs,
          getEvidenceFrames: () => evidenceFramesPromise
        })).finally(finishBusy)
        return
      }
      Promise.resolve(askRef.current.ask(question)).finally(finishBusy)
    }, sendDelayMs)
  }

  timerRef.current = window.setTimeout(typeNextCharacter, typingIntervalMs)
}

async function buildDemoTestEvidenceFrames(stream, seconds = [], options = {}) {
  const session = stream.demoTestSession || {}
  const targetClipId = options.clipId || session.activeClipId
  const clip = (session.clips || []).find((item) => item.clipId === targetClipId) || (session.clips || [])[0]
  if (!clip?.videoUrlResolved || !seconds.length) return []

  const startDate = parseDemoClipStart(clip)
  const framePaths = Array.isArray(options.framePaths) ? options.framePaths : []
  const images = framePaths.length
    ? framePaths.map((path) => buildEvidenceFrameUrl(stream.sessionId, path, { baseUrl: stream.askBaseUrl }))
    : await captureVideoFrames(clip.videoUrlResolved, seconds)

  return seconds.map((second, index) => {
    const timestamp = startDate ? new Date(startDate.getTime() + second * 1000) : null
    const timestampText = timestamp ? formatDemoTimestamp(timestamp) : `${second}s`
    return {
      id: `scripted-${clip.clipId}-${second}`,
      imageUrl: images[index] || '',
      caption: 'Evidence frame',
      timestampText,
      timeRangeText: timestampText,
      scoreText: ''
    }
  })
}

async function captureVideoFrames(videoUrl, seconds = []) {
  const video = document.createElement('video')
  video.crossOrigin = 'anonymous'
  video.muted = true
  video.playsInline = true
  video.preload = 'auto'
  video.src = videoUrl
  video.load?.()

  await waitForVideoMetadata(video)

  const canvas = document.createElement('canvas')
  const width = video.videoWidth || 640
  const height = video.videoHeight || 360
  canvas.width = Math.min(width, 960)
  canvas.height = Math.round(canvas.width * (height / width))
  const context = canvas.getContext('2d')
  if (!context) return seconds.map(() => '')

  const frames = []
  for (const second of seconds) {
    await seekVideo(video, Math.max(0, Math.min(second, Math.max(0, (video.duration || second) - 0.05))))
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    frames.push(canvas.toDataURL('image/jpeg', 0.82))
  }

  video.removeAttribute('src')
  video.load?.()
  return frames
}

function waitForVideoMetadata(video) {
  if (video.readyState >= 1) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      window.clearTimeout(timer)
      video.removeEventListener('loadedmetadata', handleLoaded)
      video.removeEventListener('error', handleError)
    }
    const handleLoaded = () => {
      cleanup()
      resolve()
    }
    const handleError = () => {
      cleanup()
      reject(new Error('Scripted evidence video failed to load.'))
    }
    const timer = window.setTimeout(() => {
      cleanup()
      reject(new Error('Scripted evidence video metadata timed out.'))
    }, 8000)
    video.addEventListener('loadedmetadata', handleLoaded)
    video.addEventListener('error', handleError)
  })
}

function seekVideo(video, second) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      window.clearTimeout(timer)
      video.removeEventListener('seeked', handleSeeked)
      video.removeEventListener('error', handleError)
    }
    const handleSeeked = () => {
      cleanup()
      resolve()
    }
    const handleError = () => {
      cleanup()
      reject(new Error('Scripted evidence video seek failed.'))
    }
    const timer = window.setTimeout(() => {
      cleanup()
      resolve()
    }, 1600)
    video.addEventListener('seeked', handleSeeked)
    video.addEventListener('error', handleError)
    try {
      video.currentTime = second
    } catch (error) {
      cleanup()
      reject(error)
    }
  })
}

function parseDemoClipStart(clip = {}) {
  const rawDate = String(clip.displayDate || '').trim()
  const rawTime = String(clip.startTime || '').trim()
  const dateMatch = rawDate.match(/(\d{4})\D+(\d{1,2})\D+(\d{1,2})/)
  const timeMatch = rawTime.match(/(\d{1,2}):(\d{2})(?::(\d{2}))?/)
  if (!dateMatch || !timeMatch) return null
  const [, year, month, day] = dateMatch
  const [, hour, minute, second = '0'] = timeMatch
  return new Date(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second))
}

function formatDemoTimestamp(date) {
  const values = [
    date.getFullYear(),
    date.getMonth() + 1,
    date.getDate(),
    date.getHours(),
    date.getMinutes(),
    date.getSeconds()
  ]
  const [year, month, day, hour, minute, second] = values.map((value) => String(value).padStart(2, '0'))
  return `${year}-${month}-${day} ${hour}:${minute}:${second}`
}
