import { useEffect, useState } from 'react'

// Cytoscape draws text on canvas, so pass it the same resolved scale as CSS text.
function readTextScale() {
  return Number.parseFloat(getComputedStyle(document.body).fontSize) / 16 || 1
}

export default function useGraphTextScale() {
  const [scale, setScale] = useState(readTextScale)
  useEffect(() => {
    const sync = () => setScale(readTextScale())
    const observer = new MutationObserver(sync)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-font-scale'] })
    window.addEventListener('resize', sync)
    sync()
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', sync)
    }
  }, [])
  return scale
}
