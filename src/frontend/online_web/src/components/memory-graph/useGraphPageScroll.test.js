// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { bindGraphPageScroll } from './useGraphPageScroll.js'

let container, canvas, cy, dispose, ungrabify
function touch(type, count, clientY = 50) {
  const event = new Event(type, { bubbles: true, cancelable: true })
  Object.defineProperty(event, 'touches', { value: Array.from({ length: count }, (_, identifier) => ({ identifier, clientX: 30, clientY })) })
  canvas.dispatchEvent(event)
  return event
}
beforeEach(() => {
  container = document.createElement('div')
  canvas = document.createElement('canvas')
  container.append(canvas)
  document.body.append(container)
  ungrabify = false
  cy = {
    container: () => container,
    autoungrabify: (value) => {
      if (value !== undefined) ungrabify = value
      return ungrabify
    }
  }
  dispose = bindGraphPageScroll(cy)
})
afterEach(() => {
  dispose()
  container.remove()
})

describe('Native scrolling over memory graphs', () => {
  it('keeps single-finger moves available to native scrolling before the renderer can cancel them', () => {
    const renderer = vi.fn((event) => event.preventDefault())
    container.addEventListener('touchmove', renderer)
    touch('touchstart', 1)
    expect(ungrabify).toBe(true)
    expect(touch('touchmove', 1).defaultPrevented).toBe(false)
    expect(renderer).not.toHaveBeenCalled()
    touch('touchcancel', 0)
    expect(ungrabify).toBe(false)
  })

  it('preserves tap delivery, mouse interaction and existing node-grab settings', () => {
    const tap = vi.fn()
    const mouse = vi.fn()
    container.addEventListener('touchend', tap)
    container.addEventListener('mousedown', mouse)
    ungrabify = true
    touch('touchstart', 1)
    touch('touchend', 0)
    expect(tap).toHaveBeenCalledOnce()
    expect(ungrabify).toBe(true)
    canvas.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }))
    expect(mouse).toHaveBeenCalledOnce()
  })

  it('cancels the pending node tap when a finger swipes instead of tapping', () => {
    let pendingTap = false
    const select = vi.fn()
    container.addEventListener('touchstart', () => { pendingTap = true })
    container.addEventListener('touchcancel', () => { pendingTap = false })
    container.addEventListener('touchend', () => { if (pendingTap) select() })
    touch('touchstart', 1)
    expect(touch('touchmove', 1, 100).defaultPrevented).toBe(false)
    touch('touchend', 0)
    expect(select).not.toHaveBeenCalled()
    expect(ungrabify).toBe(false)
    touch('touchstart', 1)
    touch('touchmove', 1, 52)
    touch('touchend', 0)
    expect(select).toHaveBeenCalledOnce()
  })

  it('leaves a two-finger gesture with the graph until every finger is released', () => {
    const renderer = vi.fn((event) => event.preventDefault())
    container.addEventListener('touchmove', renderer)
    touch('touchstart', 1)
    touch('touchstart', 2)
    expect(ungrabify).toBe(false)
    expect(touch('touchmove', 2).defaultPrevented).toBe(true)
    touch('touchend', 1)
    expect(touch('touchmove', 1).defaultPrevented).toBe(true)
    touch('touchend', 0)
    renderer.mockClear()
    touch('touchstart', 1)
    expect(touch('touchmove', 1).defaultPrevented).toBe(false)
    expect(renderer).not.toHaveBeenCalled()
  })

  it('allows plain wheel scrolling while preserving modifier-wheel graph zoom', () => {
    const renderer = vi.fn((event) => event.preventDefault())
    container.addEventListener('wheel', renderer)
    const scroll = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 80 })
    canvas.dispatchEvent(scroll)
    expect(scroll.defaultPrevented).toBe(false)
    expect(renderer).not.toHaveBeenCalled()
    const zoom = new WheelEvent('wheel', { bubbles: true, cancelable: true, ctrlKey: true, deltaY: 80 })
    canvas.dispatchEvent(zoom)
    expect(zoom.defaultPrevented).toBe(true)
    expect(renderer).toHaveBeenCalledOnce()
  })

  it('removes interception and restores graph settings on cleanup', () => {
    const renderer = vi.fn((event) => event.preventDefault())
    container.addEventListener('touchmove', renderer)
    touch('touchstart', 1)
    dispose()
    expect(ungrabify).toBe(false)
    expect(touch('touchmove', 1).defaultPrevented).toBe(true)
    expect(renderer).toHaveBeenCalledOnce()
  })
})
