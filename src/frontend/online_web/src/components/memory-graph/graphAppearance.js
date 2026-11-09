// Canvas labels use resolved colors; CSS variables cannot be read by Cytoscape's renderer.
const FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Helvetica Neue", "Segoe UI", "PingFang SC", sans-serif'
const DARK_COLORS = {
  '#cbd5e1': '#c0cada',
  '#64748b': '#98a6be',
  '#1e293b': '#202733',
  '#334155': '#d0d4df',
  '#ffffff': '#1e222c',
  '#e2e8f0': '#59657a',
  '#0f172a': '#edf2fb',
  '#1d4ed8': '#8bbbff',
  '#2563eb': '#8bbbff',
  '#94a3b8': '#b2bed1',
  '#14b8a6': '#5fd8c9',
  '#8b5cf6': '#b099ff',
  '#f59e0b': '#f5be61',
  '#ec4899': '#fa92ca'
}

export function buildGraphAppearance(styles, theme, textScale = 1) {
  const dark = theme === 'dark'
  return styles.map(({ selector, style }) => {
    const themed = { ...style }
    for (const [property, value] of Object.entries(themed)) {
      if (dark && typeof value === 'string' && DARK_COLORS[value]) themed[property] = DARK_COLORS[value]
    }
    if (themed.label) {
      themed['font-family'] = FONT_FAMILY
      themed['font-weight'] = 500
    }
    if (typeof themed['font-size'] === 'number') {
      themed['font-size'] *= textScale
    }
    if (themed['shadow-opacity'] !== undefined) {
      themed['shadow-opacity'] = dark ? 0.24 : 0.12
    }
    if (themed['shadow-blur'] !== undefined) {
      themed['shadow-blur'] = dark ? 18 : 16
      themed['shadow-offset-y'] = 4
    }
    return { selector, style: themed }
  })
}
