import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  appendQaHistory,
  askQuestion,
  normalizeEvidenceFrames,
  pollQueryTask,
  streamAskQuestion
} from '../api/lightmem_egoApi.js'

export const ANSWER_MODES = {
  LEGACY: 'legacy',
  STREAM: 'stream'
}

export const LONG_TERM_RETRIEVAL_SCHEMES = {
  EM2MEMORY: 'em2memory',
  LIGHTMEM_EGO_LEGACY: 'lightmem_ego_legacy'
}

const initialAskState = {
  question: '',
  answer: '',
  answerMode: ANSWER_MODES.LEGACY,
  longTermRetrievalScheme: LONG_TERM_RETRIEVAL_SCHEMES.EM2MEMORY,
  answerPhase: 'idle',
  loading: false,
  queryStatus: 'idle',
  queryTaskId: '',
  error: '',
  latency: null,
  evidenceFrames: [],
  usedMemorySources: [],
  rawDebug: null,
  lastAskedAt: null
}

export function useAskLightMemEgo(sessionId, options = {}) {
  const queryEpochRef = useRef(0)
  const sessionIdRef = useRef(sessionId)
  const answerModeRef = useRef(ANSWER_MODES.LEGACY)
  const syncBeforeAskRef = useRef(options.syncBeforeAsk)
  const askBaseUrlRef = useRef(options.askBaseUrl)
  const askStreamEndpointRef = useRef(options.askStreamEndpoint)
  const [state, setState] = useState(initialAskState)

  useEffect(() => {
    syncBeforeAskRef.current = options.syncBeforeAsk
  }, [options.syncBeforeAsk])

  useEffect(() => {
    askBaseUrlRef.current = options.askBaseUrl
    askStreamEndpointRef.current = options.askStreamEndpoint
  }, [options.askBaseUrl, options.askStreamEndpoint])

  useEffect(() => {
    sessionIdRef.current = sessionId
    queryEpochRef.current += 1
    setState((current) => ({
      ...initialAskState,
      answerMode: current.answerMode,
      longTermRetrievalScheme: current.longTermRetrievalScheme,
      question: current.question
    }))
  }, [sessionId])

  const setQuestion = useCallback((question) => {
    setState((current) => ({ ...current, question }))
  }, [])

  const reset = useCallback(() => {
    queryEpochRef.current += 1
    setState((current) => ({
      ...initialAskState,
      answerMode: current.answerMode,
      longTermRetrievalScheme: current.longTermRetrievalScheme
    }))
  }, [])

  const setAnswerMode = useCallback((answerMode) => {
    const nextMode = answerMode === ANSWER_MODES.STREAM ? ANSWER_MODES.STREAM : ANSWER_MODES.LEGACY
    answerModeRef.current = nextMode
    setState((current) => ({ ...current, answerMode: nextMode }))
  }, [])

  const setLongTermRetrievalScheme = useCallback((scheme) => {
    const nextScheme = scheme === LONG_TERM_RETRIEVAL_SCHEMES.LIGHTMEM_EGO_LEGACY
      ? LONG_TERM_RETRIEVAL_SCHEMES.LIGHTMEM_EGO_LEGACY
      : LONG_TERM_RETRIEVAL_SCHEMES.EM2MEMORY
    setState((current) => ({ ...current, longTermRetrievalScheme: nextScheme }))
  }, [])

  const runScriptedStream = useCallback(async (script = {}) => {
    const targetSessionId = script.sessionId || sessionId
    const finalQuestion = String(script.question || '').trim()
    const scriptedAnswer = String(script.answer || '')
    if (!targetSessionId || !finalQuestion || !scriptedAnswer) return

    const epoch = queryEpochRef.current + 1
    const startedAt = performance.now()
    queryEpochRef.current = epoch

    setState((current) => ({
      ...current,
      question: finalQuestion,
      loading: true,
      queryStatus: 'streaming',
      answerPhase: 'starting',
      queryTaskId: '',
      error: '',
      answer: '',
      latency: null,
      evidenceFrames: [],
      rawDebug: {
        scripted: true,
        sessionId: targetSessionId,
        question: finalQuestion
      }
    }))

    const evidenceFramesPromise = typeof script.getEvidenceFrames === 'function'
      ? script.getEvidenceFrames()
      : Promise.resolve(script.evidenceFrames || [])

    try {
      await sleep(script.initialDelayMs ?? 1000)
      if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return

      let streamedAnswer = ''
      const tokens = tokenizeScriptedAnswer(scriptedAnswer)
      const tokenDelayMs = script.tokenDelayMs ?? 48

      for (const token of tokens) {
        streamedAnswer += token
        setState((current) => ({
          ...current,
          answer: streamedAnswer,
          answerPhase: 'streaming',
          queryStatus: 'streaming'
        }))
        await sleep(tokenDelayMs)
        if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return
      }

      const answerLatency = (performance.now() - startedAt) / 1000
      const evidenceFrames = await evidenceFramesPromise.catch(() => [])
      if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return

      appendQaHistory(targetSessionId, {
        question: finalQuestion,
        answer: scriptedAnswer,
        clientSource: 'frontend',
        inputMethod: 'scripted_demo',
        status: 'done',
        metadata: {
          scripted: true,
          evidenceFrames
        }
      }, {
        baseUrl: askBaseUrlRef.current
      }).catch((error) => {
        console.warn('[LightMem-Ego ask] scripted QA history append failed', error)
      })

      setState((current) => ({
        ...current,
        answer: scriptedAnswer,
        answerPhase: 'done',
        loading: false,
        queryStatus: 'done',
        latency: answerLatency,
        evidenceFrames,
        usedMemorySources: [],
        rawDebug: {
          scripted: true,
          sessionId: targetSessionId,
          question: finalQuestion,
          answer: scriptedAnswer,
          evidenceFrames
        },
        lastAskedAt: Date.now()
      }))
    } catch (error) {
      if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return
      const message = error?.message || 'Scripted demo answer failed.'
      setState((current) => ({
        ...current,
        loading: false,
        queryStatus: 'failed',
        answerPhase: 'error',
        error: message,
        answer: current.answer || message,
        rawDebug: error
      }))
    }
  }, [sessionId])

  const applyFinalResult = useCallback((result, options = {}) => {
    const targetSessionId = options.sessionId || sessionId
    if (!targetSessionId || sessionIdRef.current !== targetSessionId) return

    const finalResult = result || {}
    const resultSessionId = firstDefined(
      finalResult.sessionId,
      finalResult.session_id,
      finalResult.raw?.sessionId,
      finalResult.raw?.session_id,
      ''
    )
    if (resultSessionId && resultSessionId !== targetSessionId) return

    const answer = firstDefined(
      finalResult.answer,
      finalResult.finalAnswer,
      finalResult.final_answer,
      finalResult.response,
      finalResult.text,
      ''
    )
    const evidenceFrames = normalizeEvidenceFrames(finalResult, targetSessionId, {
      baseUrl: askBaseUrlRef.current
    })

    setState((current) => ({
      ...current,
      answer: answer || 'No answer returned.',
      answerPhase: 'done',
      loading: false,
      queryStatus: options.queryStatus || 'done',
      queryTaskId: options.taskId || current.queryTaskId,
      error: '',
      latency: firstDefined(finalResult.latency, finalResult.queryLatency, finalResult.query_latency, null),
      evidenceFrames,
      usedMemorySources: firstDefined(finalResult.usedMemorySources, finalResult.used_memory_sources, []),
      rawDebug: finalResult,
      lastAskedAt: Date.now()
    }))
  }, [sessionId])

  const askStream = useCallback(async (targetSessionId, finalQuestion, epoch, longTermRetrievalScheme, inputMethod = 'manual') => {
    let streamAnswer = ''
    let receivedTerminalEvent = false
    let serverErrorMessage = ''
    let lastStreamEvent = null
    let streamDiagnostics = null

    const requestOptions = {
      priority: 'high',
      preferCurrent: true,
      memoryMode: 'auto',
      longTermRetrievalScheme,
      useImageEvidence: options.isDemoMode ? true : 'auto',
      maxImageEvidence: options.isDemoMode ? 3 : 6,
      topK: 5,
      baseUrl: askBaseUrlRef.current,
      endpoint: askStreamEndpointRef.current || undefined,
      inputMethod,
      ...(options.isDemoMode
        ? {
            useCurrent: true,
            useShortTerm: false,
            useLongTerm: false
          }
        : {}),
      onEvent: ({ event, type, data }) => {
        if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return
        const eventType = event === 'message' ? (type || data?.type || event) : event
        lastStreamEvent = { event: eventType, data }

        if (eventType === 'start') {
          setState((current) => ({
            ...current,
            queryStatus: 'streaming',
            answerPhase: 'starting',
            rawDebug: data || current.rawDebug
          }))
          return
        }

        if (eventType === 'ping') {
          return
        }

        if (eventType === 'evidence') {
          setState((current) => ({
            ...current,
            queryStatus: 'streaming',
            rawDebug: data || current.rawDebug
          }))
          return
        }

        if (eventType === 'draft') {
          const text = firstDefined(data?.text, data?.answer, data?.delta, '')
          setState((current) => ({
            ...current,
            answer: text || current.answer,
            answerPhase: 'draft',
            queryStatus: 'streaming',
            rawDebug: data || current.rawDebug
          }))
          return
        }

        if (eventType === 'delta') {
          streamAnswer += data?.delta || ''
          setState((current) => ({
            ...current,
            answer: streamAnswer,
            answerPhase: 'streaming',
            queryStatus: 'streaming',
            rawDebug: data || current.rawDebug
          }))
          return
        }

        if (eventType === 'final') {
          const text = firstDefined(data?.answer, data?.text, '')
          if (!text) return
          setState((current) => ({
            ...current,
            answer: text,
            answerPhase: 'final',
            queryStatus: 'streaming',
            rawDebug: data || current.rawDebug
          }))
          return
        }

        if (eventType === 'done') {
          receivedTerminalEvent = true
          const isError = data?.status === 'error'
          const result = data?.result || data

          if (result && Object.keys(result).length) {
            applyFinalResult(result, {
              queryStatus: isError ? 'failed' : 'done',
              sessionId: targetSessionId
            })
            if (isError || serverErrorMessage) {
              setState((current) => ({
                ...current,
                loading: false,
                queryStatus: isError ? 'failed' : current.queryStatus,
                answerPhase: isError ? 'error' : current.answerPhase,
                error: data?.message || serverErrorMessage || current.error,
                rawDebug: result
              }))
            }
          } else {
            setState((current) => ({
              ...current,
              loading: false,
              queryStatus: isError ? 'failed' : 'done',
              answerPhase: isError ? 'error' : 'done',
              error: data?.message || serverErrorMessage || '',
              rawDebug: data || current.rawDebug
            }))
          }
          return
        }

        if (eventType === 'error') {
          serverErrorMessage = data?.message || data?.error || 'Streaming ask request failed.'
          setState((current) => ({
            ...current,
            queryStatus: 'streaming',
            answerPhase: 'error',
            error: serverErrorMessage,
            rawDebug: data || current.rawDebug
          }))
        }
      }
    }

    try {
      streamDiagnostics = await streamAskQuestion(targetSessionId, finalQuestion, requestOptions)
    } catch (error) {
      if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return
      const diagnostic = {
        ...(error.raw || {}),
        sessionId: targetSessionId,
        lastEvent: error.raw?.lastEvent || lastStreamEvent,
        requestPayload: error.raw?.requestPayload || buildAskDebugPayload(finalQuestion, requestOptions)
      }
      console.warn('[LightMem-Ego ask stream] request failed or closed before done', diagnostic)
      error.raw = diagnostic
      throw error
    }

    if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return
    if (!receivedTerminalEvent) {
      const diagnostic = {
        ...(streamDiagnostics || {}),
        sessionId: targetSessionId,
        lastEvent: streamDiagnostics?.lastEvent || lastStreamEvent,
        requestPayload: streamDiagnostics?.requestPayload || buildAskDebugPayload(finalQuestion, requestOptions)
      }
      console.warn('[LightMem-Ego ask stream] connection closed before done', diagnostic)
      const error = new Error('Streaming answer ended before a done event was received. Please retry.')
      error.raw = diagnostic
      error.retryable = true
      throw error
    }
  }, [applyFinalResult, options.isDemoMode])

  const ask = useCallback(async (overrideQuestion, askOptions = {}) => {
    const finalQuestion = String(overrideQuestion ?? state.question ?? '').trim()
    const targetSessionId = sessionId
    const inputMethod = askOptions.inputMethod || 'manual'

    if (!targetSessionId) {
      setState((current) => ({
        ...current,
        error: 'Please start Live View first.',
        answer: current.answer || 'Start a live session before asking.'
      }))
      return
    }

    if (!finalQuestion || state.loading) return

    const epoch = queryEpochRef.current + 1
    const longTermRetrievalScheme = state.longTermRetrievalScheme || LONG_TERM_RETRIEVAL_SCHEMES.EM2MEMORY
    queryEpochRef.current = epoch

    setState((current) => ({
      ...current,
      question: finalQuestion,
      loading: true,
      queryStatus: 'submitting',
      answerPhase: answerModeRef.current === ANSWER_MODES.STREAM ? 'starting' : 'idle',
      queryTaskId: '',
      error: '',
      answer: '',
      latency: null,
      evidenceFrames: [],
      rawDebug: null
    }))

    try {
      if (syncBeforeAskRef.current) {
        await syncBeforeAskRef.current()
        if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return
      }

      if (answerModeRef.current === ANSWER_MODES.STREAM) {
        await askStream(targetSessionId, finalQuestion, epoch, longTermRetrievalScheme, inputMethod)
        return
      }

      const response = await askQuestion(targetSessionId, finalQuestion, {
        mode: 'async',
        responseMode: ANSWER_MODES.LEGACY,
        priority: 'high',
        preferCurrent: true,
        memoryMode: 'auto',
        longTermRetrievalScheme,
        useImageEvidence: options.isDemoMode ? true : 'auto',
        maxImageEvidence: options.isDemoMode ? 3 : 6,
        topK: 5,
        baseUrl: askBaseUrlRef.current,
        inputMethod,
        ...(options.isDemoMode
          ? {
              useCurrent: true,
              useShortTerm: false,
              useLongTerm: false
            }
          : {})
      })

      if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return

      if (response.queued) {
        const taskId = response.taskId
        setState((current) => ({
          ...current,
          queryStatus: 'running',
          queryTaskId: taskId,
          answer: response.message || ''
        }))

        if (!taskId) {
          throw new Error('Query queued but no task id returned.')
        }

        const task = await pollQueryTask(taskId, {
          baseUrl: askBaseUrlRef.current,
          intervalMs: 900,
          timeoutMs: 120000,
          shouldContinue: () => queryEpochRef.current === epoch && sessionIdRef.current === targetSessionId,
          onPoll: (info) => {
            if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return
            setState((current) => ({
              ...current,
              queryStatus: info.status || 'running'
            }))
          }
        })

        if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return

        if (task.done) {
          if (task.sessionId && task.sessionId !== targetSessionId) return
          applyFinalResult(task.result, {
            taskId,
            queryStatus: 'done',
            sessionId: targetSessionId
          })
          return
        }

        throw new Error(task.message || task.error?.message || 'Query task failed or timed out.')
      }

      applyFinalResult(response.result || response.raw || response, {
        taskId: response.taskId || '',
        queryStatus: response.status || 'done',
        sessionId: targetSessionId
      })
    } catch (error) {
      if (queryEpochRef.current !== epoch || sessionIdRef.current !== targetSessionId) return
      const message = formatAskError(error)
      setState((current) => ({
        ...current,
        loading: false,
        queryStatus: error.retryable ? 'idle' : 'failed',
        answerPhase: error.retryable ? 'idle' : 'error',
        error: message,
        answer: current.answer || message,
        rawDebug: error.raw || error
      }))
    }
  }, [applyFinalResult, askStream, options.isDemoMode, sessionId, state.loading, state.longTermRetrievalScheme, state.question])

  return useMemo(() => ({
    ...state,
    setQuestion,
    setAnswerMode,
    setLongTermRetrievalScheme,
    runScriptedStream,
    ask,
    reset
  }), [ask, reset, runScriptedStream, setAnswerMode, setLongTermRetrievalScheme, setQuestion, state])
}

function firstDefined(...values) {
  return values.find((item) => item !== undefined && item !== null && item !== '')
}

function buildAskDebugPayload(question, options = {}) {
  return {
    question,
    response_mode: 'stream',
    retrieval_mode: options.retrievalMode || 'auto',
    memory_mode: options.memoryMode || 'auto',
    use_image_evidence: options.useImageEvidence || 'auto',
    client_source: options.clientSource || 'frontend',
    input_method: options.inputMethod || 'manual',
    ...(options.longTermRetrievalScheme ? { long_term_retrieval_scheme: options.longTermRetrievalScheme } : {}),
    ...(options.useCurrent !== undefined ? { use_current: options.useCurrent } : {}),
    ...(options.useShortTerm !== undefined ? { use_short_term: options.useShortTerm } : {}),
    ...(options.useLongTerm !== undefined ? { use_long_term: options.useLongTerm } : {})
  }
}

function tokenizeScriptedAnswer(answer) {
  return String(answer || '').match(/\S+\s*/g) || []
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

function formatAskError(error) {
  const status = error?.status || error?.raw?.httpStatus
  const message = error?.message || 'Ask request failed.'
  if (status === 409 || String(message).includes('not_ready')) {
    return 'Demo memory is not ready yet. The frontend sent a fresh tick; please retry in a moment.'
  }
  if (error?.retryable) {
    return 'Streaming connection closed before the final done event. Please retry.'
  }
  return message
}
