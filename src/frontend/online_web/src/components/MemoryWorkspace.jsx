import { useEffect, useMemo, useState, useCallback } from 'react'
import { Activity, BookOpenText, Database, Languages } from 'lucide-react'
import { LanguageContext, translateText } from '../i18n.jsx'
import { MemoryEditProvider } from '../hooks/useSessionMemoryEditor.jsx'
import MemoryPanel from './MemoryPanel.jsx'
import ThemeToggle from './ThemeToggle.jsx'
import FontSizeToggle from './FontSizeToggle.jsx'
import { AdminModeButton } from './AdminMode.jsx'

const LANGUAGE_KEY = 'lightmem-ego.ui-language.v1'
function initialLanguage() {
  try { return localStorage.getItem(LANGUAGE_KEY) === 'zh' ? 'zh' : 'en' }
  catch { return 'en' }
}

// This entry does not mount the capture hooks or request camera permissions.
export default function MemoryWorkspace() {
  const [language, setLanguage] = useState(initialLanguage)
  const [publications, setPublications] = useState(() => new Map())
  const [titles, setTitles] = useState({})
  const t = useMemo(() => (text, params) => translateText(language, text, params), [language])
  useEffect(() => {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en'
    document.title = `${t('Memories')} · LightMem-Ego`
    try { localStorage.setItem(LANGUAGE_KEY, language) } catch {}
  }, [language, t])
  useEffect(() => {
    const sync = event => { if (event.key === LANGUAGE_KEY || event.key === null) setLanguage(initialLanguage()) }
    window.addEventListener('storage', sync)
    return () => window.removeEventListener('storage', sync)
  }, [])
  const publish = useCallback(update => setPublications(current => {
    const previous = current.get(update.sessionId)
    if ((previous?.memory?.memoryVersion || 0) > (update.memory?.memoryVersion || 0)) return current
    return new Map(current).set(update.sessionId, update)
  }), [])
  const rename = useCallback(snapshot => setTitles(current => ({ ...current, [snapshot.sessionId]: snapshot })), [])
  return <LanguageContext.Provider value={{ language, t }}>
    <div className="app-shell">
      <header className="app-topbar">
        <a className="app-brand" href="/#/collect" aria-label={t('LightMem-Ego home')}>
          <span className="app-brand-mark"><Database size={18} /></span>
          <span className="app-brand-copy"><strong>LightMem-Ego</strong><small>{t('Realtime memory workspace')}</small></span>
        </a>
        <nav className="app-primary-nav" aria-label={t('Main navigation')}>
          <a href="/#/collect"><Activity size={16} /><span>{t('Collect & Ask')}</span></a>
          <a className="active" href="/memories/" aria-current="page"><BookOpenText size={16} /><span>{t('Memories')}</span></a>
          <a href="/history-demo/" data-history-demo-nav="true">{language === 'zh' ? '视频提问' : 'Video & Ask'}</a>
        </nav>
        <div className="app-toolbar">
          <ThemeToggle />
          <button className="language-toggle" type="button" aria-label={t('Switch language')}
            onClick={() => setLanguage(value => value === 'zh' ? 'en' : 'zh')}>
            <Languages size={16} /><span>{language === 'zh' ? 'English' : '中文'}</span>
          </button>
          <FontSizeToggle />
          <AdminModeButton />
        </div>
      </header>
      <main className="app-page is-memory-page">
        <section className="memory-page" aria-label={t('Memories')}>
          <header className="page-heading">
            <div><span className="page-heading-icon"><BookOpenText size={19} /></span><div><p>{t('Session memory')}</p><h1>{t('Memories')}</h1></div></div>
          </header>
          <MemoryEditProvider onPublication={publish}>
            <MemoryPanel standalone active authoritativeUpdates={publications} sessionTitles={titles} onSessionTitleUpdated={rename} />
          </MemoryEditProvider>
        </section>
      </main>
    </div>
  </LanguageContext.Provider>
}
