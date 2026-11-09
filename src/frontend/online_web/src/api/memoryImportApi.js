import { snakeToCamelDeep } from './lightmem_egoApi.js'

export const IMPORT_MAX_BYTES = 10485760
const STATES = new Set(['queued', 'processing', 'parsing', 'normalizing', 'building_memory', 'building_retrieval', 'warming_query', 'ready', 'failed'])

export function validateImportFile(file) {
  if (!file) return 'Choose a JSON or TXT file.'
  if (!/\.(json|txt)$/i.test(file.name)) return 'Only .json and .txt files are supported.'
  if (!file.size) return 'The file is empty.'
  if (file.size > IMPORT_MAX_BYTES) return 'The file must be 10 MiB or smaller.'
  return ''
}

export function retryAfterMs(value, fallback = 10000) {
  if (!value) return fallback
  const seconds = Number(value)
  const delay = Number.isFinite(seconds) ? seconds * 1000 : Date.parse(value) - Date.now()
  return Number.isFinite(delay) ? Math.max(1000, delay) : fallback
}

const ERROR_MESSAGES = {
  invalid_import_format: 'The import format is invalid.',
  unsupported_import_extension: 'Only .json and .txt files are supported.',
  memory_import_too_large: 'The file exceeds the upload size limit.',
  memory_import_invalid_utf8: 'Save the file as UTF-8 and upload it again.',
  memory_import_invalid_json: 'The JSON file is invalid.',
  memory_import_empty: 'The file contains no valid memory content.',
  memory_import_record_limit: 'The file contains too many memory records.',
  memory_import_task_not_found: 'This import task no longer exists.',
  memory_import_failed: 'Memory import failed. Choose a file to try again.',
  memory_import_memory_not_ready: 'Long-term memory is not ready yet. Refresh the query status.',
  memory_import_query_not_ready: 'The query service is not ready yet. Refresh the query status.',
  memory_import_rate_limited: 'Too many import requests. Please wait before trying again.',
  invalid_memory_import_response: 'The import service returned an inconsistent response. Refresh the task status.'
}

export function formatMemoryImportError(error) {
  const message = ERROR_MESSAGES[error?.code]
    || (error?.status === 413 ? ERROR_MESSAGES.memory_import_too_large : '')
    || (error?.status === 429 ? ERROR_MESSAGES.memory_import_rate_limited : '')
    || (error?.status === 404 || error?.status === 410 ? ERROR_MESSAGES.memory_import_task_not_found : '')
    || (error?.status >= 500 ? 'The memory service is temporarily unavailable. Please retry.' : '')
    || 'Unable to contact the memory service. Please retry.'
  return [message, error?.code, error?.requestId && `Request ID: ${error.requestId}`].filter(Boolean).join(' · ')
}

function invalidResponse() {
  return Object.assign(new Error('Invalid memory import response'), { code: 'invalid_memory_import_response' })
}

async function request(path, { body, signal, expectedStatus } = {}) {
  const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')
  const response = await fetch(`${base}${path}`, {
    method: body ? 'POST' : 'GET',
    headers: { Accept: 'application/json' },
    body,
    signal,
    credentials: 'same-origin'
  })
  let raw
  try { raw = await response.json() } catch { raw = null }
  const requestId = response.headers.get('X-Request-ID') || ''
  if (!response.ok) {
    const backendError = raw?.error && typeof raw.error === 'object' ? raw.error : {}
    throw Object.assign(new Error('Memory service request failed'), {
      status: response.status,
      code: backendError.code || raw?.error_code || '',
      requestId,
      retryAfter: response.headers.get('Retry-After') || ''
    })
  }
  if (!raw || (expectedStatus && response.status !== expectedStatus)) throw Object.assign(invalidResponse(), { requestId })
  return { ...snakeToCamelDeep(raw), requestId }
}

export async function createMemoryImport({ file, title = '', signal }) {
  const validation = validateImportFile(file)
  if (validation) throw new Error(validation)
  const body = new FormData()
  body.append('file', file)
  body.append('format', 'auto')
  if (title.trim()) body.append('title', title.trim())
  const result = await request('/memory-imports', { body, signal, expectedStatus: 202 })
  if (result.status !== 'queued' || result.state !== 'queued' || !validId(result.taskId) || !validId(result.sessionId)) throw invalidResponse()
  return result
}

function validId(value) { return typeof value === 'string' && value.trim() !== '' }

export async function getMemoryImportTask({ taskId, sessionId, signal }) {
  const result = await request(`/memory-imports/${encodeURIComponent(taskId)}`, { signal })
  if (result.status !== 'ok' || result.taskId !== taskId || result.sessionId !== sessionId || !STATES.has(result.state)
    || (result.state === 'processing' && !STATES.has(result.stage))) throw invalidResponse()
  return {
    ...result,
    taskId,
    sessionId,
    progress: result.progress ?? result.percent ?? null,
    recordCount: result.recordCount ?? result.parsedRecordCount ?? null,
    queryReady: result.queryReady === true
  }
}

export async function getMemorySessionStatus({ sessionId, signal }) {
  const result = await request(`/session/${encodeURIComponent(sessionId)}/status`, { signal })
  if (result.sessionId && result.sessionId !== sessionId) throw invalidResponse()
  return {
    ...result,
    sessionId,
    sessionKind: result.sessionKind || result.kind || '',
    interactionMode: result.interactionMode || '',
    queryAllowed: result.queryAllowed === true,
    queryReady: result.queryReady === true,
    title: result.title || sessionId
  }
}

export async function getSessionQueryWarmup({ sessionId, signal }) {
  const result = await request(`/session/${encodeURIComponent(sessionId)}/query_warmup`, { signal })
  if (result.sessionId && result.sessionId !== sessionId) throw invalidResponse()
  return { ...result, sessionId }
}
