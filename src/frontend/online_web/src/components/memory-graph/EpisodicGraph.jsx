import { useLanguage } from '../../i18n.jsx'
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

const EPISODIC_STYLES = [
  {
    selector: 'node[graphKind = "event"]',
    style: {
      shape: 'round-rectangle',
      width: 150,
      height: 64,
      'background-color': '#cbd5e1',
      'background-opacity': 1,
      'border-color': '#64748b',
      'border-width': 2,
      color: '#1e293b',
      label: 'data(label)',
      'font-size': 9,
      'font-weight': 700,
      'text-wrap': 'wrap',
      'text-max-width': 134,
      'text-valign': 'center',
      'text-halign': 'center',
      'min-zoomed-font-size': 7,
      'shadow-blur': 12,
      'shadow-color': '#64748b',
      'shadow-opacity': 0.2,
      'shadow-offset-y': 4
    }
  },
  {
    selector: 'node[graphKind = "entity"]',
    style: {
      shape: 'ellipse',
      width: 62,
      height: 62,
      'background-color': '#cbd5e1',
      'background-opacity': 1,
      'border-color': '#64748b',
      'border-width': 2,
      color: '#1e293b',
      label: 'data(label)',
      'font-size': 9,
      'font-weight': 700,
      'text-wrap': 'wrap',
      'text-max-width': 56,
      'text-valign': 'center',
      'min-zoomed-font-size': 7,
      'shadow-blur': 9,
      'shadow-color': '#0f766e',
      'shadow-opacity': 0.14,
      'shadow-offset-y': 3
    }
  },
  ...NODE_COLOR_STYLES,
  {
    selector: 'edge',
    style: {
      width: 2.2,
      'curve-style': 'bezier',
      'control-point-step-size': 40,
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
  {
    selector: 'edge[graphKind = "before"]',
    style: { 'curve-style': 'straight', 'line-color': '#2563eb', 'target-arrow-color': '#2563eb', width: 3.2 }
  },
  {
    selector: 'edge[graphKind = "mentions"]',
    style: { 'line-style': 'dashed', 'line-color': '#94a3b8', 'target-arrow-color': '#94a3b8', width: 1.5 }
  },
  { selector: 'edge[relationGroup = "spatial"]', style: { 'line-color': '#14b8a6', 'target-arrow-color': '#14b8a6' } },
  { selector: 'edge[relationGroup = "action"]', style: { 'line-color': '#8b5cf6', 'target-arrow-color': '#8b5cf6' } },
  { selector: 'edge[relationGroup = "state"]', style: { 'line-color': '#f59e0b', 'target-arrow-color': '#f59e0b' } },
  { selector: 'edge[relationGroup = "dialogue"]', style: { 'line-color': '#ec4899', 'target-arrow-color': '#ec4899' } },
  { selector: 'node.is-linked', style: { opacity: 1, 'z-index': 15 } },
  { selector: 'edge.is-linked', style: { width: 4.5, opacity: 1, 'underlay-color': '#0f172a', 'underlay-padding': 3, 'underlay-opacity': 0.1, 'z-index': 15 } },
  { selector: '.is-selected', style: { 'border-color': '#1d4ed8', 'border-width': 5, 'shadow-color': '#2563eb', 'shadow-opacity': 0.48, 'z-index': 20 } },
  { selector: '.is-dimmed', style: { opacity: 0.16 } },
  { selector: '.is-hover-dimmed', style: { opacity: 0.1 } },
  { selector: '.is-hover-context', style: { opacity: 1, 'z-index': 25 } },
  { selector: 'node.is-hovered', style: { 'border-color': '#0f766e', 'border-width': 4, 'shadow-color': '#14b8a6', 'shadow-opacity': 0.48, 'z-index': 30 } },
  { selector: 'edge.is-hovered', style: { width: 5, opacity: 1, 'underlay-color': '#0f172a', 'underlay-padding': 4, 'underlay-opacity': 0.16, 'z-index': 30 } },
  { selector: 'edge.is-selected', style: { width: 5.5, opacity: 1, 'underlay-color': '#0f172a', 'underlay-padding': 5, 'underlay-opacity': 0.24, 'z-index': 35 } }
]

export default function EpisodicGraph({ elements, selection, linkedEventIds, linkedEntityId, focusEventIds, onSelect, onClear }) {
  const { t } = useLanguage()
  const { theme } = useTheme()
  const textScale = useGraphTextScale()
  const stylesheet = useMemo(() => buildGraphAppearance(EPISODIC_STYLES, theme, textScale), [theme, textScale])
  const [cy, setCy] = useState(null)
  useGraphPageScroll(cy)
  const containerRef = useRef(null)
  const elementSignature = useMemo(() => elements.map((element) => element.data.id).join('|'), [elements])
  const focusSignature = JSON.stringify([...(focusEventIds || [])].sort())
  const registerCy = useCallback((instance) => setCy(instance), [])

  useEffect(() => {
    if (!cy) return undefined
    const handleNode = (event) => {
      const node = event.target
      const graphKind = node.data('graphKind')
      if (graphKind === 'event' || graphKind === 'entity') onSelect(graphKind, node.id())
    }
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
    cy.on('tap', handleBackground)
    cy.on('mouseover', 'node, edge', handleHover)
    cy.on('mouseout', 'node, edge', handleHoverOut)
    return () => {
      cy.off('tap', 'node', handleNode)
      cy.off('tap', handleBackground)
      cy.off('mouseover', 'node, edge', handleHover)
      cy.off('mouseout', 'node, edge', handleHoverOut)
    }
  }, [cy, onClear, onSelect])

  useEffect(() => {
    if (!cy) return
    cy.layout({ name: 'preset', fit: true, padding: 42, animate: false }).run()
  }, [cy, elementSignature])

  useEffect(() => {
    if (!cy) return
    cy.batch(() => {
      cy.elements().removeClass('is-selected is-linked is-dimmed')
      const selected = selection?.id ? cy.getElementById(selection.id) : cy.collection()
      if (selected.length) selected.addClass('is-selected')
      const linkedIds = linkedEntityId ? [linkedEntityId] : linkedEventIds || []
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
  }, [cy, linkedEntityId, linkedEventIds, selection])

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
    <div ref={containerRef} className="memory-graph-canvas" role="region" aria-label={t("Episodic graph with {value1} visible nodes", { value1: elements.filter((element) => element.group === 'nodes').length })}>
      {!elements.length && <div className="memory-graph-canvas-empty">{t("No episodic events match the current filters.")}</div>}
      <CytoscapeComponent
        className="memory-graph-surface"
        elements={elements}
        stylesheet={stylesheet}
        layout={{ name: 'preset', fit: true, padding: 42 }}
        cy={registerCy}
        style={{ width: '100%', height: '100%' }}
        wheelSensitivity={0.8}
        minZoom={0.12}
        maxZoom={2.5}
      />
      <GraphLegend mode="episodic" />
      <GraphZoomControls cy={cy} disabled={!elements.length} label={t("Episodic graph")} />
    </div>
  )
}

function clearHover(cy) {
  cy.elements().removeClass('is-hovered is-hover-context is-hover-dimmed')
}

function reducedMotion() {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}
