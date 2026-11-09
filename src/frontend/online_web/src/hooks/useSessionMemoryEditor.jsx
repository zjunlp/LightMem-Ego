import { createContext, useCallback, useContext, useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { createMemoryEditStore } from './memoryEditStore.js'

const MemoryEditContext = createContext(null)

export function MemoryEditProvider({ children, onPublication }) {
  const publicationRef = useRef(onPublication)
  publicationRef.current = onPublication
  const [store] = useState(() => createMemoryEditStore({ onPublication: (update) => publicationRef.current?.(update) }))
  // Defer disposal so React StrictMode's effect replay keeps the same live store.
  const lifetime = useRef(0)
  useEffect(() => {
    const generation = ++lifetime.current
    return () => queueMicrotask(() => { if (lifetime.current === generation) store.dispose() })
  }, [store])
  useEffect(() => {
    const beforeUnload = (event) => {
      if (store.hasUnsaved()) { event.preventDefault(); event.returnValue = '' }
    }
    window.addEventListener('beforeunload', beforeUnload)
    return () => window.removeEventListener('beforeunload', beforeUnload)
  }, [store])
  return <MemoryEditContext.Provider value={store}>{children}</MemoryEditContext.Provider>
}

export function useSessionMemoryEditor(sessionId, { initialMemory, onPublication } = {}) {
  const shared = useContext(MemoryEditContext)
  const callbackRef = useRef(onPublication)
  callbackRef.current = onPublication
  const [local] = useState(() => createMemoryEditStore({ onPublication: (update) => callbackRef.current?.(update) }))
  const store = shared || local
  const subscribe = useCallback((listener) => store.subscribe(sessionId, listener), [store, sessionId])
  const snapshot = useCallback(() => store.getSnapshot(sessionId), [store, sessionId])
  const state = useSyncExternalStore(subscribe, snapshot, snapshot)
  useEffect(() => { store.seed(sessionId, initialMemory) }, [store, sessionId, initialMemory])
  const lifetime = useRef(0)
  useEffect(() => {
    const generation = ++lifetime.current
    return () => queueMicrotask(() => { if (!shared && lifetime.current === generation) local.dispose() })
  }, [local, shared])
  return { state, store }
}
