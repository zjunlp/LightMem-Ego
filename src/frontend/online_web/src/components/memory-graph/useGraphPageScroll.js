import { useEffect } from 'react'

// Keep taps and two-finger graph gestures, but let a single finger scroll the
// surrounding dialog natively. Cytoscape otherwise cancels that default action
// in its window touchmove handler, even when the gesture starts on a node.
export default function useGraphPageScroll(cy) {
  useEffect(() => {
    if (!cy) return undefined
    return bindGraphPageScroll(cy)
  }, [cy])
}

export function bindGraphPageScroll(cy) {
  const container = cy.container()
  let multiTouch = false
  let origin = null
  let previousUngrabify = null
  const restore = () => {
    if (previousUngrabify !== null) {
      cy.autoungrabify(previousUngrabify)
      previousUngrabify = null
    }
  }
  const start = (event) => {
    if (event.touches.length > 1) {
      multiTouch = true
      origin = null
      restore()
    } else if (!multiTouch && previousUngrabify === null) {
      origin = event.touches[0]
      previousUngrabify = cy.autoungrabify()
      cy.autoungrabify(true)
    }
  }
  const move = (event) => {
    if (event.touches.length !== 1 || multiTouch) return
    event.stopPropagation()
    const finger = event.touches[0]
    if (origin && Math.hypot(finger.clientX - origin.clientX, finger.clientY - origin.clientY) > 6) {
      // Some browsers keep delivering touchend during native scrolling. Cancel
      // the renderer's pending tap so lifting a scrolling finger selects nothing.
      origin = null
      const cancel = new Event('touchcancel', { bubbles: true })
      Object.defineProperty(cancel, 'touches', { value: [] })
      container.dispatchEvent(cancel)
    }
  }
  const end = (event) => {
    if (!event.touches.length) {
      multiTouch = false
      origin = null
      restore()
    }
  }
  // Plain wheel/trackpad scrolling should work over the graph too. Keep the
  // modifier-wheel gesture available for zooming and the explicit zoom controls.
  const wheel = (event) => {
    if (!event.ctrlKey && !event.metaKey) event.stopPropagation()
  }
  const bindings = { touchstart: start, touchmove: move, touchend: end, touchcancel: end, wheel }
  for (const [type, handler] of Object.entries(bindings)) {
    container.addEventListener(type, handler, { capture: true, passive: true })
  }
  return () => {
    for (const [type, handler] of Object.entries(bindings)) container.removeEventListener(type, handler, true)
    restore()
  }
}
