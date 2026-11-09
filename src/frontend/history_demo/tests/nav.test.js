import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {launchBrowser} from './browser-env.js';
const browser = await launchBrowser();
try {
  const page = await browser.newPage();
  await page.setContent('<html lang="zh-CN"><body><header><nav class="app-primary-nav"><a href="#/collect">采集</a><a href="#/memory">记忆</a></nav></header></body></html>');
  const path = fileURLToPath(new URL('../main-nav-entry.js',import.meta.url));
  await page.addScriptTag({path});
  await page.locator('nav a[href="/history-demo/"]').waitFor();
  await page.addScriptTag({path});
  assert.equal(await page.locator('nav a[href="/history-demo/"]').count(),1);
  assert.equal(await page.locator('nav a[href="/history-demo/"]').innerText(),'视频提问');
  await page.evaluate(()=>{document.querySelector('nav').replaceChildren(document.createElement('a'));document.documentElement.lang='en';});
  await page.waitForFunction(()=>document.querySelector('nav a[href="/history-demo/"]')?.textContent==='Video & Ask');
  assert.equal(await page.locator('nav a[href="/history-demo/"]').count(),1);
  await page.route('http://nav.test/**',route=>route.fulfill({contentType:'text/html',body:'<html lang="zh-CN"><body><nav class="app-primary-nav"></nav><div id="catalog"><button class="memory-session-card"><span class="memory-session-id">first</span></button></div><nav class="memory-catalog-page-actions"><button>Previous</button><button id="next">Next</button></nav></body></html>'}));
  await page.goto('http://nav.test/?memory_session=target&edit=1#/memory');
  await page.evaluate(()=>{
    document.querySelector('#next').onclick=()=>{document.querySelector('#catalog').innerHTML='<button class="memory-session-card"><span class="memory-session-id">target</span></button>';document.querySelector('#next').disabled=true;};
    document.querySelector('#catalog').onclick=event=>{
      if(event.target.closest('.memory-session-card')?.textContent!=='target')return;
      document.body.insertAdjacentHTML('beforeend','<section class="memory-dialog"><small class="memory-dialog-session-id">target</small><div class="session-memory-actions"><button class="current-memory-edit-button">Edit</button></div></section>');
      document.querySelector('.current-memory-edit-button').onclick=()=>{window.editOpened=(window.editOpened||0)+1;};
    };
  });
  await page.addScriptTag({path});
  await page.waitForFunction(()=>window.editOpened===1);
  assert.equal(await page.locator('nav.app-primary-nav a').getAttribute('href'),'/history-demo/?session=target');
  assert.equal(await page.locator('.memory-dialog a').innerText(),'对此会话提问');
  assert.equal(await page.evaluate(()=>localStorage.getItem('lightmem-ego.history-session.v1')),'target');
  assert.ok(!page.url().includes('memory_session'),'handled links must not reopen an editor after refresh');
  await page.evaluate(()=>{document.querySelector('.memory-dialog').remove();document.body.append(document.createElement('div'));});
  assert.equal(await page.locator('.memory-dialog').count(),0);
  assert.equal(await page.evaluate(()=>window.editOpened),1);
  await page.goto('http://nav.test/memories/?memory_session=target');
  await page.evaluate(()=>{
    document.querySelector('#next').onclick=()=>{document.querySelector('#catalog').innerHTML='<button class="memory-session-card"><span class="memory-session-id">target</span></button>';document.querySelector('#next').disabled=true;};
    document.querySelector('#catalog').onclick=event=>{
      if(event.target.closest('.memory-session-card')?.textContent!=='target')return;
      document.body.insertAdjacentHTML('beforeend','<section class="memory-dialog"><small class="memory-dialog-session-id">target</small><div class="session-memory-actions"></div></section>');
    };
  });
  await page.addScriptTag({path});
  await page.locator('.memory-dialog-session-id').waitFor();
  assert.equal(new URL(page.url()).pathname,'/memories/');
  assert.equal(new URL(page.url()).searchParams.get('memory_session'),'target','standalone links retain the session after refresh');
  assert.equal(await page.locator('.memory-dialog a').innerText(),'对此会话提问');
  console.log('PASS nav entry: language, replacement, remembered session, native pagination/editor and return link');
} finally {await browser.close();}
