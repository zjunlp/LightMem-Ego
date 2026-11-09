// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { BackendFeaturesProvider, detectBackendFeatures } from './BackendFeatures.jsx'
import SessionTitleEditor from './SessionTitleEditor.jsx'
import { MemoryRecordCard } from './MemoryRecordCard.jsx'

afterEach(() => { cleanup(); vi.restoreAllMocks() })

it('enables optional features only when the required HTTP routes exist', () => {
  expect(detectBackendFeatures({paths:{}})).toEqual({memoryImport:false,memoryExport:false,titleEditing:false})
  expect(detectBackendFeatures({paths:{
    '/memory-imports':{post:{}},'/memory-imports/{task_id}':{get:{}},
    '/session/{sid}/title':{get:{},patch:{}},'/session/{sid}/memories/{id}/export':{get:{}}
  }})).toEqual({memoryImport:true,memoryExport:true,titleEditing:true})
  expect(detectBackendFeatures({paths:{'/session/{sid}/title':{get:{}}}}).titleEditing).toBe(false)
})

it('keeps unsupported rename/export actions out of an otherwise readable memory card', async () => {
  const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({paths:{}})))
  render(<BackendFeaturesProvider>
    <SessionTitleEditor sessionId="session" title="Session" />
    <MemoryRecordCard memory={{id:'record',content:'Saved memory',eventTime:{},raw:{}}} onExport={vi.fn()} />
  </BackendFeaturesProvider>)
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1))
  expect(screen.getByText('Saved memory')).toBeTruthy()
  expect(screen.queryByRole('button', {name:/Rename|Export/})).toBeNull()
})
