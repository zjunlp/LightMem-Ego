// Synthetic contract fixtures; contains no live historical content.
export function buildMemory({ version = 1, content = 'original', canRollback = false, sessionId = 'session-1', kind = 'ordinary', canEdit = true } = {}) {
  return {
    status: 'ok',
    schemaVersion: 1,
    sessionId,
    memorySource: 'M_lt',
    canEdit,
    editDisabledReason: canEdit ? null : 'no_editable_records',
    sessionKind: kind === 'imported' ? 'imported_memory' : undefined,
    memoryVersion: version,
    componentVersions: { episodic: version, semantic: version, visual: version },
    activeRootKind: kind === 'legacy' ? 'lightmem_ego' : 'em2mem',
    qaAligned: kind !== 'legacy',
    updatedAt: null,
    completeRecordSet: true,
    fieldTruncation: false,
    truncatedFieldCount: 0,
    availability: { episodic: 'ready', semantic: 'ready', visual: 'ready' },
    counts: {
      total: 1,
      episodic: 1,
      semantic: 0,
      visual: 0,
      episodicByGranularity: { '30sec': 1, '3min': 0, '10min': 0, '1h': 0 }
    },
    episodic: {
      '30sec': [{ id: 'record-1', content, memoryType: 'episodic', granularity: '30sec', editable: canEdit, editDisabledReason: canEdit ? null : 'unsupported_source_record' }],
      '3min': [],
      '10min': [],
      '1h': []
    },
    semantic: [],
    visual: [],
    editPolicy: {
      field: 'episodic.30sec[].content',
      maxCharsPerRecord: 4000,
      maxTotalChars: 32000,
      lengthUnit: 'unicode_code_points'
    },
    propagation: buildPropagation(version),
    canRollback
  }
}

export function buildPropagation(version) {
  return {
    status: 'ready',
    sourceScale: '30sec',
    targetMemoryVersion: version,
    derivedScales: {
      '3min': { status: 'ready', recomputedRecords: 0 },
      '10min': { status: 'ready', recomputedRecords: 0 },
      '1h': { status: 'ready', recomputedRecords: 0 }
    },
    semantic: { status: 'ready', version, changedFacts: 0 },
    graphs: {
      episodic: { status: 'ready', version },
      semantic: { status: 'ready', version }
    },
    backends: {
      multiscale: 'llm',
      triplets: 'llm_openie',
      semantic: 'llm_semantic_extraction_consolidation'
    }
  }
}

export function buildGraphResponse(version) {
  return {
    schema_version: 1,
    status: 'ok',
    session_id: 'session-1',
    memory_source: 'M_lt',
    scale: '30sec',
    memory_version: version,
    component_versions: { episodic: version, graph: version, semantic: version },
    graph_versions: { episodic_graph: version, semantic_graph: version },
    episodic: { nodes: [], edges: [] },
    semantic: { nodes: [], edges: [], timeline: [] },
    documents: {},
    indexes: {},
    warnings: {}
  }
}

export function buildMixedImport(options = {}) {
  const memory = buildMemory({ kind: 'imported', ...options })
  memory.episodic['3min'] = [{ id: 'summary-1', content: 'Original imported summary', memoryType: 'episodic', granularity: '3min', editable: false, editDisabledReason: 'derived_memory' }]
  memory.semantic = [{ id: 'semantic-1', content: 'Original imported fact', memoryType: 'semantic', editable: false, editDisabledReason: 'semantic_memory' }]
  memory.visual = [{ id: 'visual-1', content: 'Original visual record', memoryType: 'visual', editable: false, editDisabledReason: 'visual_memory', eventTime: { date: '2026-10-01', startSeconds: 12 } }]
  memory.counts = { total: 4, episodic: 2, semantic: 1, visual: 1, episodicByGranularity: { '30sec': 1, '3min': 1, '10min': 0, '1h': 0 } }
  return memory
}

export function buildNoEditable(options = {}) {
  const memory = buildMixedImport({ ...options, canEdit: false })
  memory.episodic['30sec'] = []
  memory.counts.total -= 1
  memory.counts.episodic -= 1
  memory.counts.episodicByGranularity['30sec'] = 0
  return memory
}

export function buildMutation(memory = buildMemory({ version: 2, canRollback: true }), options = {}) {
  const version = memory.memoryVersion
  return { status: 'ok', success: true, message: 'Memory saved.', sessionId: memory.sessionId,
    memoryVersion: version, canRollback: memory.canRollback, memory,
    propagation: buildPropagation(version),
    graphUpdate: { status: 'ready', scale: '30sec', memoryVersion: version,
      componentVersions: { episodic: version, graph: version, semantic: version },
      graphVersions: { episodicGraph: version, semanticGraph: version } }, ...options }
}

export function buildGraph(version = 2, sessionId = 'session-1') {
  return { sessionId, memoryVersion: version, componentVersions: { episodic: version, graph: version, semantic: version },
    graphVersions: { episodicGraph: version, semanticGraph: version }, episodic: { nodes: [], edges: [] },
    semantic: { nodes: [], edges: [], timeline: [] }, documents: {}, indexes: {}, warnings: {} }
}
