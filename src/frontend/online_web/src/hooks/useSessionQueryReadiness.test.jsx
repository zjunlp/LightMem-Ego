// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useSessionQueryReadiness } from './useSessionQueryReadiness.js'
import { getMemorySessionStatus, getSessionQueryWarmup } from '../api/memoryImportApi.js'

vi.mock('../api/memoryImportApi.js', async (original) => ({
  ...await original(), getMemorySessionStatus: vi.fn(), getSessionQueryWarmup: vi.fn()
}))
const session = { sessionId: 'imported-1', memoryVersion: 1 }
const status = (version = 1, ready = false) => ({ sessionId: 'imported-1', memoryVersion: version, queryAllowed: true, queryReady: ready })
const warmup = (version = 1, state = 'warming_query') => ({ sessionId: 'imported-1', memoryVersion: version, status: state })
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done }); return { promise, resolve } }

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(document, 'hidden', 'get').mockReturnValue(false)
  getMemorySessionStatus.mockReset().mockResolvedValue(status())
  getSessionQueryWarmup.mockReset().mockResolvedValue(warmup())
})
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers() })

describe('query preparation after publication', () => {
  it('polls until ready and stops without overlapping requests', async () => {
    const { result } = renderHook(() => useSessionQueryReadiness(session))
    await act(async () => {})
    expect(result.current.ready).toBe(false)
    await act(() => vi.advanceTimersByTimeAsync(4999))
    expect(getMemorySessionStatus).toHaveBeenCalledTimes(1)
    getMemorySessionStatus.mockResolvedValue(status(1, true))
    getSessionQueryWarmup.mockResolvedValue(warmup(1, 'ready'))
    await act(() => vi.advanceTimersByTimeAsync(1))
    expect(result.current.ready).toBe(true)
    await act(() => vi.advanceTimersByTimeAsync(30000))
    expect(getMemorySessionStatus).toHaveBeenCalledTimes(2)
  })

  it('rejects old query cache versions and ignores the previous publication response', async () => {
    const oldStatus = deferred()
    const oldWarmup = deferred()
    getMemorySessionStatus.mockReturnValueOnce(oldStatus.promise)
    getSessionQueryWarmup.mockReturnValueOnce(oldWarmup.promise)
    const first = { publicationId: 'first', memory: { memoryVersion: 1 } }
    const second = { publicationId: 'second', memory: { memoryVersion: 2 }, queryReady: false }
    const { result, rerender } = renderHook(({ publication }) => useSessionQueryReadiness(session, { publication }), { initialProps: { publication: first } })
    getMemorySessionStatus.mockResolvedValue(status(2, true))
    getSessionQueryWarmup.mockResolvedValue(warmup(1, 'ready'))
    rerender({ publication: second })
    await act(async () => { oldStatus.resolve(status(1, true)); oldWarmup.resolve(warmup(1, 'ready')) })
    expect(result.current.ready).toBe(false)
    expect(getMemorySessionStatus.mock.calls[0][0].signal.aborted).toBe(true)
    getSessionQueryWarmup.mockResolvedValue(warmup(2, 'ready'))
    await act(() => vi.advanceTimersByTimeAsync(5000))
    expect(result.current.ready).toBe(true)
    // A graph notification for the same publication must not restart readiness.
    const calls = getMemorySessionStatus.mock.calls.length
    rerender({ publication: { ...second, graphStatus: 'ready', revision: 10 } })
    await act(async () => {})
    expect(getMemorySessionStatus).toHaveBeenCalledTimes(calls)
    expect(result.current.ready).toBe(true)
  })

  it('stops after warmup failure and can check a later backend retry', async () => {
    getSessionQueryWarmup.mockResolvedValue(warmup(1, 'failed'))
    const { result } = renderHook(() => useSessionQueryReadiness(session))
    await act(async () => {})
    expect(result.current.failed).toBe(true)
    await act(() => vi.advanceTimersByTimeAsync(30000))
    expect(getSessionQueryWarmup).toHaveBeenCalledTimes(1)
    getMemorySessionStatus.mockResolvedValue(status(1, true))
    getSessionQueryWarmup.mockResolvedValue(warmup(1, 'ready'))
    await act(async () => result.current.refresh())
    expect(result.current.ready).toBe(true)
  })

  it('uses Retry-After and then the normal polling interval', async () => {
    getMemorySessionStatus.mockRejectedValueOnce({ status: 429, retryAfter: '3' })
    renderHook(() => useSessionQueryReadiness(session))
    await act(async () => {})
    await act(() => vi.advanceTimersByTimeAsync(2999))
    expect(getMemorySessionStatus).toHaveBeenCalledTimes(1)
    await act(() => vi.advanceTimersByTimeAsync(1))
    expect(getMemorySessionStatus).toHaveBeenCalledTimes(2)
    await act(() => vi.advanceTimersByTimeAsync(5000))
    expect(getMemorySessionStatus).toHaveBeenCalledTimes(3)
  })

  it('pauses while hidden, immediately refreshes on return, and cancels on leave', async () => {
    const { unmount } = renderHook(() => useSessionQueryReadiness(session))
    await act(async () => {})
    vi.spyOn(document, 'hidden', 'get').mockReturnValue(true)
    await act(async () => document.dispatchEvent(new Event('visibilitychange')))
    await act(() => vi.advanceTimersByTimeAsync(30000))
    expect(getMemorySessionStatus).toHaveBeenCalledTimes(1)
    vi.spyOn(document, 'hidden', 'get').mockReturnValue(false)
    await act(async () => document.dispatchEvent(new Event('visibilitychange')))
    expect(getMemorySessionStatus).toHaveBeenCalledTimes(2)
    unmount()
    await act(() => vi.advanceTimersByTimeAsync(30000))
    expect(getMemorySessionStatus).toHaveBeenCalledTimes(2)
  })

  it('disables readiness synchronously on Session switch even before the new status returns', async () => {
    getMemorySessionStatus.mockResolvedValue(status(1, true))
    getSessionQueryWarmup.mockResolvedValue(warmup(1, 'ready'))
    const { result, rerender } = renderHook(({ current }) => useSessionQueryReadiness(current), { initialProps: { current: session } })
    await act(async () => {})
    expect(result.current.ready).toBe(true)
    getMemorySessionStatus.mockReturnValue(new Promise(() => {}))
    getSessionQueryWarmup.mockReturnValue(new Promise(() => {}))
    rerender({ current: { ...session, sessionId: 'different' } })
    expect(result.current.ready).toBe(false)
  })
})
