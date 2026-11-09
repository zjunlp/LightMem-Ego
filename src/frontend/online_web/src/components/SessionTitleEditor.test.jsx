// @vitest-environment jsdom

import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import SessionTitleEditor from './SessionTitleEditor.jsx'

const api = vi.hoisted(() => ({
  getSessionTitle: vi.fn(),
  updateSessionTitle: vi.fn(),
  validateSessionTitle: vi.fn((value) => {
    const title = String(value ?? '').trim()
    if (!title) return 'session_title_required'
    if (/[\u0000-\u001f\u007f-\u009f]/u.test(title)) return 'session_title_control_character'
    if (Array.from(title).length > 100) return 'session_title_too_long'
    return ''
  })
}))

vi.mock('../api/lightmem_egoApi.js', () => api)

beforeEach(() => {
  Object.values(api).forEach((mock) => mock.mockReset())
  api.validateSessionTitle.mockImplementation((value) => {
    const title = String(value ?? '').trim()
    if (!title) return 'session_title_required'
    if (/[\u0000-\u001f\u007f-\u009f]/u.test(title)) return 'session_title_control_character'
    if (Array.from(title).length > 100) return 'session_title_too_long'
    return ''
  })
})

afterEach(() => cleanup())

describe('SessionTitleEditor', () => {
  it('loads the latest title and saves only after a successful response', async () => {
    const user = userEvent.setup()
    const onUpdated = vi.fn()
    api.getSessionTitle.mockResolvedValue(snapshot('session-1', 'Old name', 2))
    api.updateSessionTitle.mockResolvedValue(snapshot('session-1', '项目 🧠', 3))
    render(<SessionTitleEditor sessionId="session-1" title="stale" onUpdated={onUpdated} />)

    await user.click(screen.getByRole('button', { name: 'Rename Session' }))
    const input = await screen.findByRole('textbox', { name: 'Session name' })
    expect(input.value).toBe('Old name')
    await user.clear(input)
    await user.type(input, '  项目 🧠  ')
    await user.keyboard('{Enter}')

    await waitFor(() => expect(api.updateSessionTitle).toHaveBeenCalledWith(expect.objectContaining({
      sessionId: 'session-1', title: '项目 🧠', expectedRevision: 2
    })))
    expect((await screen.findByRole('status')).textContent).toContain('Session name updated.')
    expect(onUpdated).toHaveBeenLastCalledWith(snapshot('session-1', '项目 🧠', 3))
  })

  it('restores the Session ID as the default name', async () => {
    const user = userEvent.setup()
    api.getSessionTitle.mockResolvedValue(snapshot('session-1', 'Custom name', 4))
    api.updateSessionTitle.mockResolvedValue(snapshot('session-1', 'session-1', 5))
    render(<SessionTitleEditor sessionId="session-1" />)

    await user.click(screen.getByRole('button', { name: 'Rename Session' }))
    await screen.findByRole('textbox', { name: 'Session name' })
    await user.click(screen.getByRole('button', { name: 'Restore default name' }))

    await waitFor(() => expect(api.updateSessionTitle).toHaveBeenCalledWith(expect.objectContaining({
      sessionId: 'session-1', title: 'session-1', expectedRevision: 4
    })))
  })

  it('cancels with Escape without submitting a draft', async () => {
    const user = userEvent.setup()
    api.getSessionTitle.mockResolvedValue(snapshot('session-1', 'Current name', 1))
    render(<SessionTitleEditor sessionId="session-1" />)

    await user.click(screen.getByRole('button', { name: 'Rename Session' }))
    const input = await screen.findByRole('textbox', { name: 'Session name' })
    await user.clear(input)
    await user.type(input, 'Unsaved name')
    await user.keyboard('{Escape}')

    expect(screen.queryByRole('textbox', { name: 'Session name' })).toBeNull()
    expect(api.updateSessionTitle).not.toHaveBeenCalled()
  })

  it('keeps the draft after a conflict and retries against the returned revision', async () => {
    const user = userEvent.setup()
    api.getSessionTitle.mockResolvedValue(snapshot('session-1', 'Old name', 1))
    api.updateSessionTitle
      .mockRejectedValueOnce(Object.assign(new Error('conflict'), {
        code: 'session_title_conflict',
        details: { current: snapshot('session-1', 'Server name', 2) }
      }))
      .mockResolvedValueOnce(snapshot('session-1', 'My draft', 3))
    render(<SessionTitleEditor sessionId="session-1" />)

    await user.click(screen.getByRole('button', { name: 'Rename Session' }))
    const input = await screen.findByRole('textbox', { name: 'Session name' })
    await user.clear(input)
    await user.type(input, 'My draft')
    await user.click(screen.getByRole('button', { name: 'Save name' }))

    expect((await screen.findByRole('alert')).textContent).toContain('Current name: Server name')
    expect(input.value).toBe('My draft')
    await user.click(screen.getByRole('button', { name: 'Save name' }))
    await waitFor(() => expect(api.updateSessionTitle).toHaveBeenCalledTimes(2))
    expect(api.updateSessionTitle.mock.calls[1][0].expectedRevision).toBe(2)
  })

  it('confirms an ambiguous save with GET and does not resend PATCH', async () => {
    const user = userEvent.setup()
    const onUpdated = vi.fn()
    api.getSessionTitle
      .mockResolvedValueOnce(snapshot('session-1', 'Old name', 1))
      .mockResolvedValueOnce(snapshot('session-1', 'Saved name', 2))
    api.updateSessionTitle.mockRejectedValueOnce(new TypeError('network failed'))
    render(<SessionTitleEditor sessionId="session-1" onUpdated={onUpdated} />)

    await user.click(screen.getByRole('button', { name: 'Rename Session' }))
    const input = await screen.findByRole('textbox', { name: 'Session name' })
    await user.clear(input)
    await user.type(input, 'Saved name')
    await user.click(screen.getByRole('button', { name: 'Save name' }))

    await waitFor(() => expect(api.getSessionTitle).toHaveBeenCalledTimes(2))
    expect(api.updateSessionTitle).toHaveBeenCalledTimes(1)
    expect((await screen.findByRole('status')).textContent).toContain('Session name updated.')
    expect(onUpdated).toHaveBeenLastCalledWith(snapshot('session-1', 'Saved name', 2))
  })

  it('blocks duplicate saves while one request is pending', async () => {
    const user = userEvent.setup()
    let resolveUpdate
    api.getSessionTitle.mockResolvedValue(snapshot('session-1', 'Old name', 1))
    api.updateSessionTitle.mockImplementation(() => new Promise((resolve) => { resolveUpdate = resolve }))
    render(<SessionTitleEditor sessionId="session-1" />)

    await user.click(screen.getByRole('button', { name: 'Rename Session' }))
    const input = await screen.findByRole('textbox', { name: 'Session name' })
    await user.clear(input)
    await user.type(input, 'New name')
    const save = screen.getByRole('button', { name: 'Save name' })
    await user.click(save)
    await waitFor(() => expect(api.updateSessionTitle).toHaveBeenCalledTimes(1))
    expect(save.disabled).toBe(true)
    await user.click(save)
    expect(api.updateSessionTitle).toHaveBeenCalledTimes(1)

    await act(async () => resolveUpdate(snapshot('session-1', 'New name', 2)))
    expect(await screen.findByRole('status')).toBeTruthy()
  })

  it('rejects invalid drafts and ignores a late response after switching Sessions', async () => {
    const user = userEvent.setup()
    const onUpdated = vi.fn()
    let resolveOld
    api.getSessionTitle.mockImplementation(({ sessionId }) => {
      if (sessionId === 'session-1') return new Promise((resolve) => { resolveOld = resolve })
      return Promise.resolve(snapshot('session-2', 'Second', 1))
    })
    const view = render(<SessionTitleEditor sessionId="session-1" onUpdated={onUpdated} />)
    await user.click(screen.getByRole('button', { name: 'Rename Session' }))
    view.rerender(<SessionTitleEditor sessionId="session-2" onUpdated={onUpdated} />)
    await user.click(screen.getByRole('button', { name: 'Rename Session' }))
    const input = await screen.findByRole('textbox', { name: 'Session name' })
    await user.clear(input)
    await user.type(input, `${'😀'.repeat(101)}`)
    await user.click(screen.getByRole('button', { name: 'Save name' }))
    expect((await screen.findByRole('alert')).textContent).toContain('100 Unicode characters')
    expect(api.updateSessionTitle).not.toHaveBeenCalled()

    await act(async () => resolveOld(snapshot('session-1', 'Late old name', 9)))
    expect(onUpdated).not.toHaveBeenCalledWith(expect.objectContaining({ sessionId: 'session-1' }))
    expect(input.value).toBe('😀'.repeat(101))
  })
})

function snapshot(sessionId, title, titleRevision) {
  return { status: 'ok', sessionId, title, titleRevision, titleUpdatedAt: null }
}
