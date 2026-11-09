/* Set the theme before styles paint, including when browser storage is unavailable. */
;(function () {
  var preference = null
  try { preference = localStorage.getItem('lightmem-ego.ui-theme.v1') } catch (_) {}
  var systemDark = Boolean(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
  var theme = preference === 'dark' || preference === 'light' ? preference : systemDark ? 'dark' : 'light'
  document.documentElement.dataset.theme = theme
  document.documentElement.style.colorScheme = theme
  var meta = document.querySelector('meta[name="theme-color"]')
  if (meta) meta.content = theme === 'dark' ? '#0e1015' : '#f5f5f7'

  var fontPreference = null
  try { fontPreference = localStorage.getItem('lightmem-ego.ui-font-size.v1') } catch (_) {}
  var fontScale = fontPreference === 'small' || fontPreference === 'normal' || fontPreference === 'large'
    ? fontPreference : 'normal'
  document.documentElement.dataset.fontScale = fontScale
})()
