// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LanguageContext, translateText } from '../i18n.jsx'
import { INPUT_MODES } from '../hooks/useRealtimeStream.js'
import { ANSWER_MODES, LONG_TERM_RETRIEVAL_SCHEMES } from '../hooks/useAskLightMemEgo.js'
import AdvancedTools from './AdvancedTools.jsx'
import LiveView from './LiveView.jsx'
import AnswerCard from './AnswerCard.jsx'
import { MemoryRecordCard } from './MemoryDialog.jsx'
import GraphToolbar from './memory-graph/GraphToolbar.jsx'
import GraphDetailsPanel from './memory-graph/GraphDetailsPanel.jsx'

afterEach(cleanup)
const zh = { language: 'zh', t: (text, params) => translateText('zh', text, params) }
function localized(element, context = zh) {
  return <LanguageContext.Provider value={context}>{element}</LanguageContext.Provider>
}
function makeStream(mode = INPUT_MODES.FRAME_AUDIO) {
  return {
    inputMode: mode, modeLabel: 'HTTP Frames', status: 'idle', activeSession: {}, stats: {},
    demoTestSession: { clips: [] }, demoSession: {}, imageUploadQueue: [], imageUploadStatus: 'idle',
    cameraFacingLabel: 'Rear camera', canChangeInputMode: true, canStart: true,
    isDemoTestMode: mode === INPUT_MODES.DEMO_TEST,
    isLegacyDemoMode: mode === INPUT_MODES.DEMO_VIDEO,
    isRokidMode: [INPUT_MODES.ROKID, INPUT_MODES.ROKID_LIVE_RTMP].includes(mode),
    isRokidLiveMode: mode === INPUT_MODES.ROKID_LIVE_RTMP,
    isWebRtcMode: mode === INPUT_MODES.WEBRTC_WHIP,
    isImageUploadMode: mode === INPUT_MODES.IMAGE_UPLOAD,
    setInputMode: vi.fn(), toggleCamera: vi.fn(), uploadDemo: vi.fn(), setImageUploadFiles: vi.fn(),
    uploadDemoTest: vi.fn().mockResolvedValue({ sessionId: 'new-session' })
  }
}
const ask = {
  answerMode: ANSWER_MODES.STREAM, longTermRetrievalScheme: LONG_TERM_RETRIEVAL_SCHEMES.EM2MEMORY,
  setAnswerMode: vi.fn(), setLongTermRetrievalScheme: vi.fn()
}
function openTools(stream, answer = ask) {
  render(localized(<AdvancedTools stream={stream} ask={answer} onReset={vi.fn()} />))
  fireEvent.click(screen.getByRole('button', { name: '高级工具' }))
}

describe('Chinese UI workflows', () => {
  it.each([
    [INPUT_MODES.FRAME_AUDIO, /稳定的基础模式/],
    [INPUT_MODES.WEBRTC_WHIP, /摄像头和麦克风通过 WHIP/],
    [INPUT_MODES.ROKID, /Rokid 眼镜使用/],
    [INPUT_MODES.ROKID_LIVE_RTMP, /连接活跃的 Rokid 眼镜 RTMP 会话/],
    [INPUT_MODES.DEMO_VIDEO, /上传本地视频/],
    [INPUT_MODES.DEMO_TEST, /上传第 1 天和第 2 天的视频/],
    [INPUT_MODES.IMAGE_UPLOAD, /上传本地图片和音频/]
  ])('translates the %s description and keeps mode selection values', (mode, description) => {
    const stream = makeStream(mode)
    openTools(stream)
    expect(screen.getByText(description)).toBeTruthy()
    expect(screen.getByText(/逐步显示回答草稿/)).toBeTruthy()
    expect(screen.getByText('使用 EM2Memory 方案检索长期记忆。')).toBeTruthy()
    fireEvent.click(screen.getByRole('radio', { name: 'HTTP 帧' }))
    expect(stream.setInputMode).toHaveBeenCalledWith(INPUT_MODES.FRAME_AUDIO)
  })

  it('translates legacy answer and retrieval explanations', () => {
    openTools(makeStream(), { ...ask, answerMode: ANSWER_MODES.LEGACY, longTermRetrievalScheme: LONG_TERM_RETRIEVAL_SCHEMES.LIGHTMEM_EGO_LEGACY })
    expect(screen.getByText(/使用稳定的任务排队流程/)).toBeTruthy()
    expect(screen.getByText('使用原始 LightMemEgo 方案检索长期记忆。')).toBeTruthy()
  })

  it('keeps file objects and upload actions intact in both demo flows', async () => {
    const file1 = new File(['video1'], 'day1.mp4', { type: 'video/mp4' })
    const file2 = new File(['video2'], 'day2.mp4', { type: 'video/mp4' })
    const stream = makeStream(INPUT_MODES.DEMO_VIDEO)
    openTools(stream)
    fireEvent.change(screen.getByLabelText('上传视频'), { target: { files: [file1] } })
    expect(stream.uploadDemo).toHaveBeenCalledWith(file1)
    cleanup()
    const testStream = makeStream(INPUT_MODES.DEMO_TEST)
    openTools(testStream)
    fireEvent.change(screen.getByLabelText('选择第 1 天视频'), { target: { files: [file1] } })
    fireEvent.change(screen.getByLabelText('选择第 2 天视频'), { target: { files: [file2] } })
    fireEvent.click(screen.getByRole('button', { name: '上传演示测试视频' }))
    expect(testStream.uploadDemoTest).toHaveBeenCalledWith(file1, file2)
  })

  it('translates photo selection and validation while preserving file names', () => {
    const stream = { ...makeStream(INPUT_MODES.IMAGE_UPLOAD), imageUploadError: 'conference.mp3 exceeds the 50 MiB limit.' }
    openTools(stream)
    const file = new File(['audio'], 'conference.mp3', { type: 'audio/mpeg' })
    fireEvent.change(screen.getByLabelText('选择图片或音频'), { target: { files: [file] } })
    expect(stream.setImageUploadFiles).toHaveBeenCalledWith([file])
    expect(screen.getByText('conference.mp3 超过 50 MiB 大小限制。')).toBeTruthy()
    expect(screen.getAllByText('待命').length).toBeGreaterThan(0)
  })

  it('updates camera labels with the selected language without changing camera actions', () => {
    const stream = { ...makeStream(), canFlipCamera: true }
    const { rerender } = render(localized(<LiveView stream={stream} />))
    fireEvent.click(screen.getByRole('button', { name: '后置摄像头' }))
    expect(stream.toggleCamera).toHaveBeenCalledOnce()
    rerender(localized(<LiveView stream={{ ...stream, cameraFacingLabel: 'Front camera' }} />))
    expect(screen.getByRole('button', { name: '前置摄像头' })).toBeTruthy()
    rerender(localized(<LiveView stream={stream} />, { language: 'en', t: (text, params) => translateText('en', text, params) }))
    expect(screen.getByRole('button', { name: 'Rear camera' })).toBeTruthy()
  })

  it('translates answer state and export controls but preserves backend content', () => {
    const answer = 'The scene contains a laptop. Backend detail: failed to read source.'
    render(localized(<AnswerCard answer={answer} loading queryStatus="queued" error="Unrecognized server message · trace-42" />))
    expect(screen.getByText('排队中')).toBeTruthy()
    expect(screen.getByText(answer)).toBeTruthy()
    expect(screen.getByText('Unrecognized server message · trace-42')).toBeTruthy()
    const onExport = vi.fn()
    const memory = { id: 'memory-42', content: 'Original memory in English.' }
    render(localized(<MemoryRecordCard memory={memory} sessionId="session-42" onExport={onExport} />))
    fireEvent.click(screen.getByRole('button', { name: '将记忆 memory-42 导出为 JSON' }))
    expect(onExport).toHaveBeenCalledWith(memory, 'session-42')
    expect(screen.getByText(memory.content)).toBeTruthy()
  })

  it('translates graph filters while preserving server relation labels and filter values', () => {
    const onChange = vi.fn()
    render(localized(<GraphToolbar filters={{ query: '', relations: [], episodicMinConfidence: 0, semanticMinConfidence: 0, semanticMinSupportCount: 1 }} timeDomain={{ start: 0, end: 60 }} relationOptions={['works_on']} onChange={onChange} onReset={vi.fn()} />))
    expect(screen.getByText('works_on')).toBeTruthy()
    fireEvent.change(screen.getByPlaceholderText('事件、实体、关系…'), { target: { value: 'project' } })
    expect(onChange).toHaveBeenCalledWith({ query: 'project' })
  })

  it('renders localized event details while leaving event descriptions untouched', () => {
    const event = { id: 'event-1', title: 'Original event title', docId: 'doc-1', confidence: 0.9, status: 'ready', counts: { actions: 2 }, eventTime: {} }
    const model = {
      eventById: new Map([[event.id, event]]),
      documentByDocId: new Map([['doc-1', { text: 'Original backend description.' }]]),
      episodicEdgesByEventId: new Map()
    }
    render(localized(<GraphDetailsPanel selection={{ kind: 'event', id: event.id }} model={model} onClear={vi.fn()} />))
    expect(screen.getByText('置信度')).toBeTruthy()
    expect(screen.getByText('描述')).toBeTruthy()
    expect(screen.getByText('就绪')).toBeTruthy()
    expect(screen.getByText('Original event title')).toBeTruthy()
    expect(screen.getByText('Original backend description.')).toBeTruthy()
  })

  it('localizes validator messages and diagnostic prefixes without modifying identifiers', () => {
    expect(translateText('zh', 'Maximum 2000 characters.')).toBe('最多允许 2000 个字符。')
    expect(translateText('zh', 'The memory service is temporarily unavailable. Please retry. · server_code · Request ID: abc-123')).toBe('记忆服务暂时不可用，请重试。 · server_code · 请求 ID：abc-123')
    expect(translateText('zh', 'Unexpected remote failure: database unavailable')).toBe('Unexpected remote failure: database unavailable')
    expect(translateText('en', 'Remove {value1}', { value1: '$&-photo.png' })).toBe('Remove $&-photo.png')
  })
})
