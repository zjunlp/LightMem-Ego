import {
  fetchLongTermMemories,
  fetchLongTermMemoryGraph,
  updateThirtySecondMemories,
  rollbackThirtySecondMemories
} from '../api/lightmem_egoApi.js'

const RETRY_DELAYS = [2000, 5000, 10000, 30000]
const REBUILD_ERRORS = new Set(['semantic_rebuild_unavailable', 'memory_derivation_failed'])
export const codePointLength = (value) => Array.from(String(value ?? '')).length

export function changedMemoryRecords(state) {
  return (state.memory?.episodic?.['30sec'] || [])
    .filter((record) => state.memory.canEdit && record.editable && state.drafts[record.id] !== record.content)
    .map((record) => ({ id: record.id, content: state.drafts[record.id] ?? '' }))
}

function initialState() {
  return { memory: null, drafts: {}, loading: false, submitting: false, pendingOperation: null,
    lastOperation: null, checkingResult: false, error: null, fieldErrors: {}, notice: '', graphState: { status: 'idle', error: null } }
}

function key() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
  const bytes = globalThis.crypto.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 15) | 64
  bytes[8] = (bytes[8] & 63) | 128
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

// One store per mounted application. Writes outlive individual detail/editor views.
export function createMemoryEditStore({ onPublication = () => {}, api = {} } = {}) {
  const services = { fetchLongTermMemories, fetchLongTermMemoryGraph, updateThirtySecondMemories,
    rollbackThirtySecondMemories, ...api }
  const entries = new Map()
  let revision = 0
  let disposed = false
  const entry = (id) => {
    if (!entries.has(id)) entries.set(id, { state: initialState(), listeners: new Set(), readGeneration: 0,
      graphGeneration: 0, checkGeneration: 0, latestVersion: 0 })
    return entries.get(id)
  }
  const patch = (id, update) => {
    if (disposed) return
    const item = entry(id)
    item.state = { ...item.state, ...update }
    item.listeners.forEach((listener) => listener())
  }
  const publish = (update) => {
    if (!disposed) onPublication({ ...update, revision: ++revision })
  }
  const install = (id, memory) => {
    const item = entry(id)
    if (memory.sessionId !== id || memory.memoryVersion < item.latestVersion) return false
    item.latestVersion = memory.memoryVersion
    patch(id, { memory, drafts: Object.fromEntries(memory.episodic['30sec'].map((record) => [record.id, record.content])),
      fieldErrors: {} })
    return true
  }
  const installPreservingDrafts = (id, memory) => {
    const item = entry(id)
    if (memory.sessionId !== id || memory.memoryVersion < item.latestVersion) return false
    item.latestVersion = memory.memoryVersion
    patch(id, { memory, fieldErrors: {} })
    return true
  }
  const load = async (id, { discard = false } = {}) => {
    if (!id || disposed) return null
    const item = entry(id)
    if (item.state.pendingOperation || item.state.submitting) return null
    item.readController?.abort()
    const controller = new AbortController()
    item.readController = controller
    const generation = ++item.readGeneration
    const timer = setTimeout(() => controller.abort(), 15000)
    patch(id, { loading: true, error: null })
    try {
      const result = await services.fetchLongTermMemories({ sessionId: id, signal: controller.signal })
      if (disposed || controller.signal.aborted || generation !== item.readGeneration || result.sessionId !== id) return null
      if (discard || !changedMemoryRecords(item.state).length) install(id, result)
      else if (result.memoryVersion !== item.state.memory?.memoryVersion) {
        patch(id, { error: { code: 'memory_version_conflict' } })
      }
      return result
    } catch (error) {
      if (!disposed && generation === item.readGeneration) patch(id, { error })
      return null
    } finally {
      clearTimeout(timer)
      if (generation === item.readGeneration) patch(id, { loading: false })
    }
  }

  const refreshGraph = async (id, target = entry(id).graphTarget, attempt = 0) => {
    if (!target || disposed) return
    const item = entry(id)
    clearTimeout(item.graphTimer)
    item.graphController?.abort()
    const controller = new AbortController()
    item.graphController = controller
    const generation = ++item.graphGeneration
    const timer = setTimeout(() => controller.abort(), 15000)
    patch(id, { graphState: { status: 'refreshing', error: null } })
    const common = { sessionId: id, memory: target.memory, propagation: target.propagation,
      queryReady: target.queryReady, publicationId: target.publicationId }
    try {
      const result = await services.fetchLongTermMemoryGraph({ sessionId: id, etag: '', signal: controller.signal })
      if (disposed || generation !== item.graphGeneration || target.memory.memoryVersion !== item.latestVersion) return
      const graph = result.graph
      const version = target.memory.memoryVersion
      const expectedComponents = target.graphUpdate?.componentVersions
      const expectedGraphs = target.graphUpdate?.graphVersions
      if (graph?.sessionId !== id || graph?.memoryVersion !== version
        || (expectedComponents && ['episodic', 'graph', 'semantic'].some((part) => graph.componentVersions?.[part] !== expectedComponents[part]))
        || (expectedGraphs && ['episodicGraph', 'semanticGraph'].some((part) => graph.graphVersions?.[part] !== expectedGraphs[part]))) {
        throw Object.assign(new Error('Graph version is still changing.'), { code: 'memory_component_lagging' })
      }
      patch(id, { graphState: { status: 'ready', error: null } })
      publish({ ...common, graph, graphEtag: result.etag, graphStatus: 'ready' })
    } catch (error) {
      if (disposed || generation !== item.graphGeneration) return
      patch(id, { graphState: { status: 'error', error } })
      publish({ ...common, graph: null, graphEtag: '', graphStatus: 'error', graphError: error })
      if (['memory_component_lagging', 'memory_graph_not_ready', 'memory_snapshot_changing'].includes(error.code) && attempt < RETRY_DELAYS.length) {
        item.graphTimer = setTimeout(() => refreshGraph(id, target, attempt + 1), RETRY_DELAYS[attempt])
      }
    } finally {
      clearTimeout(timer)
    }
  }

  const reconcileLatest = async (id, operation, failure, { explicit = false } = {}) => {
    if (disposed) return null
    const item = entry(id)
    item.checkController?.abort()
    const controller = new AbortController()
    item.checkController = controller
    const generation = ++item.checkGeneration
    const timer = setTimeout(() => controller.abort(), 15000)
    if (explicit) patch(id, { loading: true })
    try {
      const memory = await services.fetchLongTermMemories({ sessionId: id, signal: controller.signal })
      if (disposed || controller.signal.aborted || generation !== item.checkGeneration || memory.sessionId !== id
        || memory.memoryVersion < item.latestVersion) return null

      const baseVersion = operation
        ? operation.type === 'update' ? operation.body.baseVersion : operation.body.currentVersion
        : null
      const acceptedVersion = failure?.acceptedVersion
      const operationEffectIsVisible = operation && (operation.type === 'update'
        ? operation.body.records.every(({ id: recordId, content }) => (
          memory.episodic?.['30sec']?.some((record) => record.id === recordId && record.content === content)
        ))
        : memory.canRollback === false)
      const operationIsVisible = operation && memory.memoryVersion > baseVersion && (
        failure?.serverAccepted === true
          ? (Number.isInteger(acceptedVersion) && memory.memoryVersion >= acceptedVersion) || operationEffectIsVisible
          : operationEffectIsVisible
      )

      if (operationIsVisible) {
        const queryReady = memory.queryReady ?? failure?.queryReady
        const publishedMemory = queryReady === undefined ? memory : { ...memory, queryReady }
        if (!install(id, publishedMemory)) return null
        const notice = queryReady === false ? 'Memory saved. Preparing query service…'
          : operation.type === 'rollback'
            ? 'Memory rollback succeeded. The latest version is displayed.'
            : 'Memory changes saved successfully. The latest version is displayed.'
        const publicationId = operation.key
        item.graphTarget = { memory: publishedMemory, propagation: publishedMemory.propagation, queryReady, publicationId }
        patch(id, { pendingOperation: null, lastOperation: null, checkingResult: false, error: null, notice,
          graphState: { status: 'refreshing', error: null } })
        publish({ sessionId: id, memory: publishedMemory, propagation: publishedMemory.propagation, queryReady,
          publicationId, graph: null, graphEtag: '', graphStatus: 'refreshing' })
        void refreshGraph(id, item.graphTarget)
        return 'confirmed'
      }

      if (explicit && operation && item.state.error?.code === 'memory_version_conflict'
        && memory.memoryVersion > (item.state.memory?.memoryVersion || 0)) {
        if (!installPreservingDrafts(id, memory)) return null
        patch(id, { pendingOperation: null, lastOperation: null,
          checkingResult: false, error: { code: 'memory_version_conflict' }, notice: 'Latest memory refreshed. Your draft is preserved.',
          graphState: { status: 'refreshing', error: null } })
        item.graphTarget = { memory, propagation: memory.propagation, publicationId: `refresh-${memory.memoryVersion}` }
        publish({ sessionId: id, memory, propagation: memory.propagation, queryReady: memory.queryReady,
          publicationId: item.graphTarget.publicationId, graph: null, graphEtag: '', graphStatus: 'refreshing' })
        void refreshGraph(id, item.graphTarget)
        return 'refreshed-conflict'
      }

      if (explicit && operation) {
        if (!installPreservingDrafts(id, memory)) return null
        patch(id, { checkingResult: false, notice: 'Latest memory loaded. The save result is still unconfirmed.' })
        item.graphTarget = { memory, propagation: memory.propagation, publicationId: `refresh-${memory.memoryVersion}` }
        publish({ sessionId: id, memory, propagation: memory.propagation, queryReady: memory.queryReady,
          publicationId: item.graphTarget.publicationId, graph: null, graphEtag: '', graphStatus: 'refreshing' })
        void refreshGraph(id, item.graphTarget)
        return 'loaded-unconfirmed'
      }

      if (explicit && !operation) {
        if (!install(id, memory)) return null
        patch(id, { checkingResult: false, error: null, notice: 'Latest memory refreshed.' })
        publish({ sessionId: id, memory, queryReady: memory.queryReady, graph: null,
          graphEtag: '', graphStatus: 'refreshing' })
        item.graphTarget = { memory, propagation: memory.propagation, publicationId: `refresh-${memory.memoryVersion}` }
        void refreshGraph(id, item.graphTarget)
        return 'refreshed'
      }

      return 'unconfirmed'
    } catch (error) {
      if (explicit && !disposed && generation === item.checkGeneration && error?.name !== 'AbortError') {
        patch(id, { error: { ...item.state.error, refreshError: error } })
      }
      return null
    } finally {
      clearTimeout(timer)
      if (item.checkController === controller) item.checkController = null
      if (explicit && !disposed && generation === item.checkGeneration) patch(id, { loading: false, checkingResult: false })
    }
  }

  const run = async (id) => {
    const item = entry(id)
    const operation = item.state.pendingOperation
    if (!operation || item.state.submitting || disposed) return
    clearTimeout(item.retryTimer)
    item.retryTimer = null
    item.readController?.abort()
    item.readGeneration += 1
    const controller = new AbortController()
    item.writeController = controller
    let timedOut = false
    const timer = setTimeout(() => { timedOut = true; controller.abort() }, 300000)
    patch(id, { submitting: true, loading: false, error: null, fieldErrors: {} })
    try {
      const options = { sessionId: id, ...operation.body, idempotencyKey: operation.key, signal: controller.signal }
      const result = await (operation.type === 'update' ? services.updateThirtySecondMemories(options) : services.rollbackThirtySecondMemories(options))
      if (disposed) return
      if (result.sessionId !== id || result.memory?.sessionId !== id) {
        throw Object.assign(new Error('Wrong Session in mutation response.'), { code: 'invalid_memory_mutation_response' })
      }
      if (result.queryReady !== undefined) result.memory = { ...result.memory, queryReady: result.queryReady }
      if (!install(id, result.memory)) throw Object.assign(new Error('Stale mutation response.'), { code: 'invalid_memory_mutation_response' })
      const publicationId = operation.key
      item.graphTarget = { ...result, publicationId }
      patch(id, { pendingOperation: null, lastOperation: null, error: null,
        notice: result.queryReady === false ? 'Memory saved. Preparing query service…'
        : operation.type === 'rollback' ? 'Memory rollback succeeded. The latest version is displayed.'
          : 'Memory changes saved successfully. The latest version is displayed.',
        graphState: { status: 'refreshing', error: null } })
      publish({ sessionId: id, memory: result.memory, propagation: result.propagation,
        queryReady: result.queryReady, publicationId, graph: null, graphEtag: '', graphStatus: 'refreshing' })
      void refreshGraph(id, item.graphTarget)
    } catch (failure) {
      if (disposed) return
      const outcomeUnknown = timedOut || failure?.name === 'TypeError' || failure?.name === 'AbortError'
        || failure?.code === 'invalid_memory_mutation_response' || [502, 503, 504].includes(failure?.status)
      const error = outcomeUnknown ? { ...failure, code: timedOut ? 'request_timeout'
        : failure?.name === 'TypeError' ? 'network_error' : failure?.code || 'request_timeout', outcomeUnknown: true } : failure
      if (failure?.code === 'memory_update_in_progress') {
        const attempt = operation.attempt || 0
        patch(id, { pendingOperation: Object.freeze({ ...operation, attempt: attempt + 1 }), error })
        if (attempt < RETRY_DELAYS.length) {
          const retryAfter = Number(failure.retryAfter) * 1000
          item.retryTimer = setTimeout(() => run(id), retryAfter > 0 ? retryAfter : RETRY_DELAYS[attempt])
        }
      } else if (failure?.code === 'memory_version_conflict') {
        patch(id, { pendingOperation: null, lastOperation: operation, checkingResult: false, error })
      } else if (REBUILD_ERRORS.has(failure?.code)) {
        patch(id, { lastOperation: operation, error: { ...failure, retryableRebuild: true } })
      } else if (outcomeUnknown) {
        patch(id, { lastOperation: operation, checkingResult: true, error })
        const reconciliation = await reconcileLatest(id, operation, error)
        if (reconciliation !== 'confirmed' && !disposed) patch(id, { checkingResult: false })
      } else {
        patch(id, { pendingOperation: null, lastOperation: null, error,
          fieldErrors: failure?.details?.recordId ? { [failure.details.recordId]: failure.message } : {} })
      }
    } finally {
      clearTimeout(timer)
      patch(id, { submitting: false })
    }
  }

  const start = (id, type) => {
    const item = entry(id)
    const state = item.state
    if (state.submitting || state.pendingOperation || state.loading || state.error?.code === 'memory_version_conflict') return
    const memory = state.memory
    if (!memory || (type === 'update' ? !memory.canEdit : !memory.canRollback)) return
    const records = changedMemoryRecords(state)
    const errors = {}
    if (type === 'update') {
      for (const record of records) {
        if (!record.content.trim()) errors[record.id] = 'Content cannot be empty.'
        else if (codePointLength(record.content) > memory.editPolicy.maxCharsPerRecord) errors[record.id] = 'The content exceeds the per-record limit.'
      }
      if (records.reduce((total, record) => total + codePointLength(record.content), 0) > memory.editPolicy.maxTotalChars) errors._global = 'This update exceeds the total character limit.'
      if (!records.length) errors._global = 'There are no changes to save.'
      if (Object.keys(errors).length) { patch(id, { fieldErrors: errors }); return }
    }
    try {
      const body = type === 'update'
        ? Object.freeze({ baseVersion: memory.memoryVersion, records: Object.freeze(records.map((record) => Object.freeze(record))) })
        : Object.freeze({ currentVersion: memory.memoryVersion })
      patch(id, { lastOperation: null, pendingOperation: Object.freeze({ type, sessionId: id, body, key: key(), attempt: 0 }) })
      void run(id)
    } catch (error) { patch(id, { error }) }
  }

  return {
    getSnapshot: (id) => entry(id).state,
    hasUnsaved: () => Array.from(entries.values()).some(({ state }) => !!state.pendingOperation || changedMemoryRecords(state).length > 0),
    subscribe: (id, listener) => { entry(id).listeners.add(listener); return () => entry(id).listeners.delete(listener) },
    seed(id, memory) {
      const item = entry(id)
      if (!memory || memory.sessionId !== id || memory.memoryVersion < item.latestVersion || item.state.pendingOperation) return
      if (changedMemoryRecords(item.state).length) {
        if (memory.memoryVersion > (item.state.memory?.memoryVersion || 0)) patch(id, { error: { code: 'memory_version_conflict' } })
      } else if (memory !== item.state.memory) install(id, memory)
    },
    load,
    setDraft(id, recordId, content) {
      const state = entry(id).state
      if (!state.pendingOperation && !state.submitting && !state.loading) patch(id, {
        drafts: { ...state.drafts, [recordId]: content }, fieldErrors: { ...state.fieldErrors, [recordId]: '', _global: '' }
      })
    },
    discard(id) {
      const item = entry(id)
      if (item.state.submitting || (item.state.pendingOperation && !item.state.error?.retryableRebuild)) return false
      clearTimeout(item.retryTimer)
      patch(id, { pendingOperation: null, lastOperation: null, error: null, notice: '' })
      if (item.state.memory) install(id, item.state.memory)
      return true
    },
    submit: (id) => start(id, 'update'),
    rollback: (id) => start(id, 'rollback'),
    retry: (id) => run(id),
    refreshLatest: (id) => {
      const item = entry(id)
      if (item.state.loading || item.state.submitting) return null
      const operation = item.state.pendingOperation || item.state.lastOperation
      patch(id, { checkingResult: true })
      return reconcileLatest(id, operation, item.state.error, { explicit: true })
    },
    retryGraph: (id) => refreshGraph(id),
    setQueryState(id, ready, failed) {
      const state = entry(id).state
      if (state.notice === 'Memory saved. Preparing query service…' || state.notice === 'Memory saved. Query preparation failed.') {
        patch(id, { notice: ready ? 'Memory updated. Query service ready.' : failed ? 'Memory saved. Query preparation failed.' : state.notice })
      }
    },
    dispose() {
      disposed = true
      entries.forEach((item) => {
        clearTimeout(item.retryTimer)
        clearTimeout(item.graphTimer)
        item.readController?.abort()
        item.writeController?.abort()
        item.checkController?.abort()
        item.graphController?.abort()
      })
    }
  }
}
