import { useLanguage } from '../../i18n.jsx'
import cytoscape from 'cytoscape'
import coseBilkent from 'cytoscape-cose-bilkent'
import CytoscapeComponent from 'react-cytoscapejs'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import GraphLegend from './GraphLegend.jsx'
import GraphZoomControls from './GraphZoomControls.jsx'
import useGraphPageScroll from './useGraphPageScroll.js'
import { connectedGraphContext, focusGraphElements } from './graphViewportUtils.js'
import { NODE_COLOR_STYLES } from './nodeColorPalette.js'
import { useTheme } from '../ThemeProvider.jsx'
import { buildGraphAppearance } from './graphAppearance.js'
import useGraphTextScale from './useGraphTextScale.js'

cytoscape.use(coseBilkent)

const SEMANTIC_STYLES = [
  {
    selector: 'node',
    style: {
      shape: 'ellipse',
      width: 'mapData(visualWeight, 1, 30, 46, 92)',
      height: 'mapData(visualWeight, 1, 30, 46, 92)',
      'background-color': '#cbd5e1',
      'background-opacity': 1,
      'border-color': '#64748b',
      'border-width': 'mapData(supportWeight, 0, 50, 1.5, 5)',
      'border-opacity': 'mapData(supportWeight, 0, 50, 0.55, 1)',
      label: 'data(label)',
      color: '#1e293b',
      'font-size': 9.5,
      'font-weight': 700,
      'text-wrap': 'wrap',
      'text-max-width': 78,
      'text-valign': 'center',
      'min-zoomed-font-size': 7,
      'shadow-blur': 11,
      'shadow-color': '#1d4ed8',
      'shadow-opacity': 0.15,
      'shadow-offset-y': 3
    }
  },
  ...NODE_COLOR_STYLES,
  {
    selector: 'edge',
    style: {
      width: 'mapData(supportCount, 0, 20, 1.4, 7)',
      opacity: 'mapData(confidence, 0, 1, 0.34, 0.96)',
      'curve-style': 'bezier',
      'control-point-step-size': 42,
      'line-color': '#64748b',
      'target-arrow-color': '#64748b',
      'target-arrow-shape': 'triangle',
      'arrow-scale': 0.9,
      label: 'data(label)',
      color: '#334155',
      'font-size': 8.5,
      'font-weight': 700,
      'text-background-color': '#ffffff',
      'text-background-opacity': 0.96,
      'text-background-padding': 3,
      'text-background-shape': 'roundrectangle',
      'text-border-width': 1,
      'text-border-color': '#e2e8f0',
      'text-border-opacity': 0.9,
      'text-rotation': 'autorotate',
      'min-zoomed-font-size': 7
    }
  },
  { selector: 'edge[relationGroup = "spatial"]', style: { 'line-color': '#14b8a6', 'target-arrow-color': '#14b8a6' } },
  { selector: 'edge[relationGroup = "action"]', style: { 'line-color': '#8b5cf6', 'target-arrow-color': '#8b5cf6' } },
  { selector: 'edge[relationGroup = "state"]', style: { 'line-color': '#f59e0b', 'target-arrow-color': '#f59e0b' } },
  { selector: 'edge[relationGroup = "dialogue"]', style: { 'line-color': '#ec4899', 'target-arrow-color': '#ec4899' } },
  { selector: 'edge[habitStrength = "low"]', style: { 'line-style': 'dotted' } },
  { selector: 'edge[habitStrength = "medium"]', style: { 'line-style': 'dashed' } },
  { selector: 'edge[habitStrength = "high"]', style: { 'line-style': 'solid' } },
  { selector: 'node.is-linked', style: { opacity: 1, 'z-index': 15 } },
  { selector: 'edge.is-linked', style: { width: 7, opacity: 1, 'underlay-color': '#0f172a', 'underlay-padding': 3, 'underlay-opacity': 0.1, 'z-index': 15 } },
  { selector: '.is-selected', style: { 'border-color': '#1d4ed8', 'border-width': 5, 'shadow-color': '#2563eb', 'shadow-opacity': 0.48, 'z-index': 20 } },
  { selector: '.is-dimmed', style: { opacity: 0.14 } },
  { selector: '.is-hover-dimmed', style: { opacity: 0.08 } },
  { selector: '.is-hover-context', style: { opacity: 1, 'z-index': 25 } },
  { selector: 'node.is-hovered', style: { 'border-color': '#0f766e', 'border-width': 4, 'shadow-color': '#14b8a6', 'shadow-opacity': 0.5, 'z-index': 30 } },
  { selector: 'edge.is-hovered', style: { width: 8, opacity: 1, 'underlay-color': '#0f172a', 'underlay-padding': 4, 'underlay-opacity': 0.16, 'z-index': 30 } },
  { selector: 'edge.is-selected', style: { width: 9, opacity: 1, 'underlay-color': '#0f172a', 'underlay-padding': 5, 'underlay-opacity': 0.24, 'z-index': 35 } }
]

export default function SemanticGraph({ elements, selection, linkedFactIds, linkedEntityId, focusFactIds, onSelect, onClear }) {
  const { t } = useLanguage()
  const { theme } = useTheme()
  const textScale = useGraphTextScale()
  const stylesheet = useMemo(() => buildGraphAppearance(SEMANTIC_STYLES, theme, textScale), [theme, textScale])
  const [cy, setCy] = useState(null)
  useGraphPageScroll(cy)
  const containerRef = useRef(null)
  const hasLaidOutRef = useRef(false)
  const previousNodeIdsRef = useRef(new Set())
  const elementSignature = useMemo(() => elements.map((element) => element.data.id).sort().join('|'), [elements])
  const focusSignature = JSON.stringify([...(focusFactIds || [])].sort())
  const registerCy = useCallback((instance) => setCy(instance), [])

  useEffect(() => {
    if (!cy) return undefined
    const handleNode = (event) => onSelect('entity', event.target.id())
    const handleEdge = (event) => onSelect('fact', event.target.id())
    const handleBackground = (event) => {
      if (event.target === cy) onClear()
    }
    const handleHover = (event) => {
      clearHover(cy)
      const target = event.target
      const context = target.isNode()
        ? target.union(target.connectedEdges()).union(target.neighborhood('node'))
        : target.union(target.connectedNodes())
      target.addClass('is-hovered')
      context.addClass('is-hover-context')
      cy.elements().difference(context).addClass('is-hover-dimmed')
    }
    const handleHoverOut = () => clearHover(cy)
    cy.on('tap', 'node', handleNode)
    cy.on('tap', 'edge[graphKind = "semantic_fact"]', handleEdge)
    cy.on('tap', handleBackground)
    cy.on('mouseover', 'node, edge', handleHover)
    cy.on('mouseout', 'node, edge', handleHoverOut)
    return () => {
      cy.off('tap', 'node', handleNode)
      cy.off('tap', 'edge[graphKind = "semantic_fact"]', handleEdge)
      cy.off('tap', handleBackground)
      cy.off('mouseover', 'node, edge', handleHover)
      cy.off('mouseout', 'node, edge', handleHoverOut)
    }
  }, [cy, onClear, onSelect])

  useEffect(() => {
    if (!cy || !elementSignature) return
    const nodeIds = new Set(cy.nodes().map((node) => node.id()))
    const preservesContext = hasLaidOutRef.current
      && [...nodeIds].some((id) => previousNodeIdsRef.current.has(id))
    cy.layout({
      name: 'cose-bilkent',
      quality: 'proof',
      nodeDimensionsIncludeLabels: true,
      randomize: !preservesContext,
      fit: true,
      padding: 60,
      animate: reducedMotion() ? false : 'end',
      animationDuration: 480,
      nodeRepulsion: 12000,
      idealEdgeLength: 155,
      edgeElasticity: 0.18,
      gravity: 0.18,
      numIter: 3200,
      nestingFactor: 0.1,
      tile: true,
      tilingPaddingVertical: 30,
      tilingPaddingHorizontal: 30,
      initialEnergyOnIncremental: 0.25
    }).run()
    hasLaidOutRef.current = true
    previousNodeIdsRef.current = nodeIds
  }, [cy, elementSignature])

  useEffect(() => {
    if (!cy) return
    cy.batch(() => {
      cy.elements().removeClass('is-selected is-linked is-dimmed')
      const selected = selection?.id ? cy.getElementById(selection.id) : cy.collection()
      if (selected.length) selected.addClass('is-selected')
      const linkedIds = linkedEntityId ? [linkedEntityId] : linkedFactIds || []
      if (linkedIds.length) {
        let highlighted = cy.collection()
        for (const id of linkedIds) highlighted = highlighted.union(cy.getElementById(id))
        highlighted = connectedGraphContext(highlighted)
        highlighted.addClass('is-linked')
        cy.elements().difference(highlighted).difference(selected).addClass('is-dimmed')
      } else if (selection?.kind === 'entity' && selected.length) {
        const highlighted = connectedGraphContext(selected)
        cy.elements().difference(highlighted).addClass('is-dimmed')
      }
    })
  }, [cy, linkedEntityId, linkedFactIds, selection])

  useEffect(() => {
    if (!cy || !focusSignature) return
    let targets = cy.collection()
    for (const id of JSON.parse(focusSignature)) targets = targets.union(cy.getElementById(id))
    const context = connectedGraphContext(targets)
    if (context.length) focusGraphElements(cy, context, reducedMotion() ? 0 : 260)
  }, [cy, focusSignature])

  useEffect(() => {
    if (!cy || !containerRef.current || typeof ResizeObserver === 'undefined') return undefined
    const observer = new ResizeObserver(() => cy.resize())
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [cy])

  return (
    <div ref={containerRef} className="memory-graph-canvas" role="region" aria-label={t("Semantic graph with {value1} visible facts", { value1: elements.filter((element) => element.group === 'edges').length })}>
      {!elements.length && <div className="memory-graph-canvas-empty">{t("No semantic facts match the current filters.")}</div>}
      <CytoscapeComponent
        className="memory-graph-surface"
        elements={elements}
        stylesheet={stylesheet}
        layout={{ name: 'preset' }}
        cy={registerCy}
        style={{ width: '100%', height: '100%' }}
        wheelSensitivity={0.8}
        minZoom={0.12}
        maxZoom={2.5}
      />
      <GraphLegend mode="semantic" />
      <GraphZoomControls cy={cy} disabled={!elements.length} label={t("Semantic graph")} />
    </div>
  )
}

function clearHover(cy) {
  cy.elements().removeClass('is-hovered is-hover-context is-hover-dimmed')
}

function reducedMotion() {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}
