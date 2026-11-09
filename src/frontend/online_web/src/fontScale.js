export const FONT_SCALE_STORAGE_KEY = 'lightmem-ego.ui-font-size.v1'
export const FONT_SCALE_VALUES = ['small', 'normal', 'large']

export function readFontScalePreference() {
  try {
    const value = window.localStorage.getItem(FONT_SCALE_STORAGE_KEY)
    return FONT_SCALE_VALUES.includes(value) ? value : null
  } catch {
    return null
  }
}

export function applyFontScale(scale) {
  document.documentElement.dataset.fontScale = scale
}
