(() => {
  if (window.__lightmemHistoryDemoNav) return;
  window.__lightmemHistoryDemoNav = true;
  const storageKey = 'lightmem-ego.history-session.v1';
  const valid = value => typeof value === 'string' && /^[a-zA-Z0-9_-]{1,200}$/.test(value);
  const params = new URLSearchParams(location.search);
  const requested = params.get('memory_session');
  let target = valid(requested) ? requested : null;
  const editRequested = params.get('edit') === '1';
  let opened = false, scheduled = false, lastPage = '';
  const chinese = () => document.documentElement.lang.toLowerCase().startsWith('zh');
  function remembered() { try {const value=localStorage.getItem(storageKey);return valid(value)?value:null;} catch {return null;} }
  function remember(sid) { if(valid(sid))try{localStorage.setItem(storageKey,sid);}catch{} }
  function videoURL(sid) { return '/history-demo/' + (valid(sid)?`?session=${encodeURIComponent(sid)}`:''); }
  function label(node, text) {if(node.textContent!==text)node.textContent=text;}
  function link(parent, sid, dialog=false) {
    let node=parent.querySelector('a[data-history-demo-nav]');
    if(!node){node=document.createElement('a');node.dataset.historyDemoNav='true';if(dialog)node.className='icon-button secondary';parent.append(node);}
    const href=videoURL(sid);if(node.getAttribute('href')!==href)node.setAttribute('href',href);
    label(node,chinese()?(dialog?'对此会话提问':'视频提问'):(dialog?'Ask about this session':'Video & Ask'));
  }
  function finishTarget() {
    target=null;
    const url=new URL(location.href);
    if(!/^\/memories\/?$/.test(location.pathname))url.searchParams.delete('memory_session');
    url.searchParams.delete('edit');
    history.replaceState(history.state,'',url);
  }
  function update() {
    scheduled=false;
    const dialog=document.querySelector('.memory-dialog');
    const sid=dialog?.querySelector('.memory-dialog-session-id')?.textContent?.trim();
    if(valid(sid))remember(sid);
    const nav=document.querySelector('nav.app-primary-nav');
    if(nav)link(nav,remembered());
    const actions=dialog?.querySelector('.session-memory-actions');
    if(actions&&valid(sid))link(actions,sid,true);
    const memoryPage=location.hash==='#/memory'||/^\/memories\/?$/.test(location.pathname);
    if(!target||!memoryPage)return;
    if(dialog&&sid===target){
      if(!editRequested){finishTarget();return;}
      const edit=dialog.querySelector('.current-memory-edit-button');
      if(edit&&!edit.disabled){finishTarget();edit.click();}
      return;
    }
    if(opened)return;
    const cards=[...document.querySelectorAll('button.memory-session-card')];
    const card=cards.find(node=>node.querySelector('.memory-session-id')?.textContent?.trim()===target);
    if(card){opened=true;remember(target);card.click();return;}
    // The original React catalog owns pagination and the editor. Use its own
    // controls to locate a session requested from the video page.
    const pageKey=cards.map(node=>node.querySelector('.memory-session-id')?.textContent).join('|');
    const next=document.querySelector('.memory-catalog-page-actions button:last-of-type');
    if(pageKey&&pageKey!==lastPage&&next&&!next.disabled){lastPage=pageKey;next.click();}
  }
  function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(update);}}
  new MutationObserver(schedule).observe(document.documentElement,{childList:true,subtree:true,attributes:true,attributeFilter:['lang','disabled']});
  window.addEventListener('hashchange',schedule);
  window.addEventListener('storage',event=>{if(event.key===storageKey)schedule();});
  update();
})();
