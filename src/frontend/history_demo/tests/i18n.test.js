import test from 'node:test';
import assert from 'node:assert/strict';
import {translate, readLanguage, LANGUAGE_STORAGE_KEY} from '../i18n.js';

test('UI translations preserve dynamic identifiers, timestamps and raw errors', () => {
  assert.equal(translate('en','已载入 12 个会话'),'12 sessions loaded');
  assert.equal(translate('en','跳转视频 · 01:23–01:26'),'Go to video · 01:23–01:26');
  assert.equal(translate('zh','上传视频'),'上传视频');
  assert.equal(translate('en','Unrecognized remote message: trace-42'),'Unrecognized remote message: trace-42');
});
test('language preference shares the React storage key and tolerates unavailable storage', () => {
  assert.equal(LANGUAGE_STORAGE_KEY,'lightmem-ego.ui-language.v1');
  assert.equal(readLanguage({getItem:()=> 'en'}),'en');
  assert.equal(readLanguage({getItem:()=> 'zh'}),'zh');
  assert.equal(readLanguage({getItem:()=>{throw new Error('blocked');}}),'zh');
});
