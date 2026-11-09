import { describe, expect, it } from 'vitest'
import { buildGraphAppearance } from './graphAppearance.js'
import { NODE_COLOR_STYLES } from './nodeColorPalette.js'

describe('canvas theme compatibility', () => {
  it('scales canvas text without changing node geometry, zoom thresholds or the source rules', () => {
    const source = [{ selector: 'node', style: { label: 'data(label)', 'font-size': 9, width: 150, 'min-zoomed-font-size': 7 } }]
    const result = buildGraphAppearance(source, 'light', 1.25)[0].style
    expect(result['font-size']).toBe(11.25)
    expect(result.width).toBe(150)
    expect(result['min-zoomed-font-size']).toBe(7)
    expect(source[0].style['font-size']).toBe(9)
  })
  it('keeps categorical node colors and selectors stable while making dark edge labels readable', () => {
    const edge = { selector: 'edge', style: { label: 'data(label)', color: '#334155', 'text-background-color': '#ffffff', 'line-color': '#64748b' } }
    const source = [edge, ...NODE_COLOR_STYLES]
    const themed = buildGraphAppearance(source, 'dark')
    expect(themed.map((rule) => rule.selector)).toEqual(source.map((rule) => rule.selector))
    expect(themed[0].style.color).toBe('#d0d4df')
    expect(themed[0].style['text-background-color']).toBe('#1e222c')
    expect(themed[1].style['background-color']).toBe(NODE_COLOR_STYLES[0].style['background-color'])
    expect(themed[1].style.color).toBe(NODE_COLOR_STYLES[0].style.color)
    expect(edge.style.color).toBe('#334155')
    expect(buildGraphAppearance(source, 'light')[0].style['text-background-color']).toBe('#ffffff')
  })
})
