import { createContext, useContext, useState, useRef } from 'react'
import { ShieldCheck } from 'lucide-react'
import { AdminMode } from '../../../history_demo/core.js'
import { useLanguage } from '../i18n.jsx'

const Context=createContext(null)
export const useAdminMode=()=>useContext(Context)
export function AdminModeProvider({children}) {
  const [mode]=useState(()=>new AdminMode())
  const [unlocked,setUnlocked]=useState(()=>mode.isUnlocked())
  const value={unlocked,
    unlock(code){const ok=mode.unlock(code);setUnlocked(mode.isUnlocked());return ok},
    lock(){mode.lock();setUnlocked(false)}
  }
  return <Context.Provider value={value}>{children}</Context.Provider>
}
export function AdminModeButton() {
  const admin=useAdminMode(),{t}=useLanguage(),dialog=useRef(null)
  const [code,setCode]=useState(''),[error,setError]=useState('')
  if(!admin)return null
  const close=()=>{dialog.current.close();setCode('');setError('')}
  return <>
    <button className="language-toggle" type="button" aria-pressed={admin.unlocked}
      onClick={()=>admin.unlocked?admin.lock():dialog.current.showModal()}>
      <ShieldCheck size={16}/><span>{t(admin.unlocked?'Exit admin mode':'Admin mode')}</span>
    </button>
    <dialog ref={dialog} className="admin-access-dialog" onCancel={event=>{event.preventDefault();close()}}>
      <form onSubmit={event=>{event.preventDefault();if(admin.unlock(code))close();else setError(t('Incorrect admin code.'))}}>
        <h2>{t('Admin mode')}</h2>
        <p>{t('Enter admin mode to edit memories.')}</p>
        <label htmlFor="memory-admin-code">{t('Admin code')}</label>
        <input id="memory-admin-code" type="password" required autoComplete="off" value={code} onChange={event=>setCode(event.target.value)}/>
        <p role="status">{error}</p>
        <div className="admin-access-actions"><button type="submit">{t('Unlock')}</button><button type="button" onClick={close}>{t('Cancel')}</button></div>
      </form>
    </dialog>
  </>
}
