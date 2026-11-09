import asyncio
import io
import json
import subprocess
import threading
from pathlib import Path

import pytest

from online_video_upload import VideoUploadError, VideoUploadService, prepare_uploaded_video

UPLOAD_ID = '123456781234123412341234567890ab'

def publish(session, *_args):
    memory = session / 'em2mem'
    memory.mkdir(exist_ok=True)
    (memory / 'memory_config.json').write_text(json.dumps({
        'status':'memory_ready', 'memory_version':1, 'retrieval_artifacts_ready':True,
    }))
    return {'duration_seconds':2, 'memory_version':1}

def test_upload_is_idempotent_and_conflicting_content_does_not_replace_video(tmp_path):
    service = VideoUploadService(tmp_path, tmp_path, builder=publish, schedule=False)
    first = service.receive(io.BytesIO(b'recording'), '视频.mp4', UPLOAD_ID)
    second = service.receive(io.BytesIO(b'recording'), '视频.mp4', UPLOAD_ID)
    assert first['session_id'] == second['session_id']
    assert len(list(tmp_path.glob('video_*'))) == 1
    assert first['query_ready'] is False
    with pytest.raises(VideoUploadError) as exc:
        service.receive(io.BytesIO(b'other recording'), '视频.mp4', UPLOAD_ID)
    assert exc.value.status_code == 409
    assert (tmp_path / first['session_id'] / 'source.mp4').read_bytes() == b'recording'

def test_failed_or_incomplete_build_is_never_ready(tmp_path):
    service = VideoUploadService(tmp_path, tmp_path, builder=lambda *a: {}, schedule=False)
    service.receive(io.BytesIO(b'recording'), 'video.mp4', UPLOAD_ID)
    service.run(UPLOAD_ID)
    state = service.status(UPLOAD_ID)
    assert state['status'] == 'failed'
    assert state['query_ready'] is False

def test_only_published_memory_and_prepared_indexes_unlock_the_uploaded_session(tmp_path):
    service = VideoUploadService(tmp_path, tmp_path, builder=publish, schedule=False)
    state = service.receive(io.BytesIO(b'recording'), 'video.mp4', UPLOAD_ID)
    assert state['status'] == 'queued'
    service.run(UPLOAD_ID)
    state = service.status(UPLOAD_ID)
    assert state['status'] == 'ready' and state['query_ready'] is True
    assert state['history_url'] == f"/history-demo/?session={state['session_id']}"
    assert state['memory_url'] == f"/memories/?memory_session={state['session_id']}"
    metadata=json.loads((tmp_path / state['session_id'] / 'metadata.json').read_text())
    assert metadata['demo_media']['time_origin'] == 'session_relative_seconds'
    assert metadata['demo_media']['video_start_seconds'] == 0
    assert metadata['demo_media']['memory_until_seconds'] == 2

@pytest.mark.parametrize('name,data', [('bad.txt',b'video'),('empty.mp4',b''),('large.mp4',b'12345')])
def test_bad_extension_empty_and_oversized_uploads_are_rejected(tmp_path,name,data):
    service = VideoUploadService(tmp_path,tmp_path,schedule=False,max_bytes=4)
    with pytest.raises(VideoUploadError): service.receive(io.BytesIO(data),name,UPLOAD_ID)
    assert not list(tmp_path.glob('video_*'))

def test_upload_ids_cannot_select_an_existing_or_external_session(tmp_path):
    service = VideoUploadService(tmp_path,tmp_path,schedule=False)
    for invalid in ['../other','existing-session','/tmp/x','x'*200]:
        with pytest.raises(VideoUploadError): service.receive(io.BytesIO(b'video'),'a.mp4',invalid)

def test_actual_video_validation_and_browser_mp4_conversion(tmp_path):
    source=tmp_path/'source.webm'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=160x90:r=5','-t','2','-c:v','libvpx-vp9',str(source)],check=True)
    result=prepare_uploaded_video(tmp_path,source)
    assert (tmp_path/'input.mp4').is_file()
    assert result['duration_seconds'] == pytest.approx(2,abs=.1)
    probe=subprocess.run(['ffprobe','-v','error','-show_streams','-of','json',str(tmp_path/'input.mp4')],capture_output=True,text=True,check=True)
    assert json.loads(probe.stdout)['streams'][0]['codec_name'] == 'h264'
    source.write_bytes(b'not a video')
    with pytest.raises(Exception): prepare_uploaded_video(tmp_path,source)

def test_api_returns_a_receipt_and_does_not_wait_for_memory_build(tmp_path,monkeypatch):
    import httpx
    import api_server
    import online_video_upload as uploads
    started,release=threading.Event(),threading.Event()
    def slow_builder(*args):
        started.set()
        assert release.wait(5)
        return publish(*args)
    monkeypatch.setattr(api_server,'ONLINE_SESSIONS_DIR',tmp_path)
    monkeypatch.setattr(uploads,'build_uploaded_video',slow_builder)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_server.app),base_url='http://test') as client:
            response=await asyncio.wait_for(client.post('/video-uploads',data={'upload_id':UPLOAD_ID},files={'video':('video.mp4',b'upload','video/mp4')}),2)
            assert response.status_code == 202
            assert response.json()['upload_id'] == UPLOAD_ID
            state=await client.get(f'/video-uploads/{UPLOAD_ID}')
            assert state.status_code == 200 and state.json()['query_ready'] is False
            blocked=await client.post(f"/session/{response.json()['session_id']}/ask",json={'question':'What happened?','mode':'async'})
            assert blocked.status_code == 409 and blocked.json()['status'] == 'not_ready'
    try: asyncio.run(run())
    finally: release.set()


def test_catalog_omits_uploaded_sessions_until_the_entire_build_is_ready(tmp_path):
    from online_memory_view.catalog import LongTermMemoryCatalogService
    service=VideoUploadService(tmp_path,tmp_path,builder=publish,schedule=False)
    state=service.receive(io.BytesIO(b'recording'),'test.mp4',UPLOAD_ID)
    publish(tmp_path/state['session_id'])
    assert LongTermMemoryCatalogService(tmp_path).load().payload['items'] == []


def test_upload_pipeline_uses_configured_model_and_builds_indexes_after_memory(tmp_path,monkeypatch):
    import online_processor
    import online_memory
    import online_preprocess.evidence_builder as evidence_builder
    import online_video_upload as uploads
    from types import SimpleNamespace

    service=VideoUploadService(tmp_path,tmp_path,schedule=False)
    state=service.receive(io.BytesIO(b'recording'),'test.mp4',UPLOAD_ID)
    session=tmp_path/state['session_id']
    calls=[]
    monkeypatch.delenv('EM2MEM_VLM_MODEL',raising=False)
    monkeypatch.setenv('EM2MEM_EXTERNAL_OPENAI_MODEL','configured-vision-model')
    monkeypatch.setenv('EM2MEM_MEMORY_MODEL','configured-memory-model')
    monkeypatch.setattr(uploads,'prepare_uploaded_video',lambda *_: {'duration_seconds':2})
    monkeypatch.setattr(online_processor,'process_session',lambda *a,**kw: calls.append('preprocess'))
    def evidence(*args,**kwargs):
        assert kwargs['model'] == 'configured-vision-model'
        calls.append('evidence')
        path=session/'evidence.json'
        path.write_text(json.dumps([{'status':'ok'}]))
        return path,path
    def memory(*args,**kwargs):
        assert kwargs['source'] == 'legacy_evidence'
        assert kwargs['model_name'] == 'configured-memory-model'
        assert kwargs['skip_semantic'] is False
        calls.append('memory')
        publish(session)
        return session/'em2mem/memory_config.json'
    def indexes(command,**kwargs):
        assert 'build_retrieval_artifacts' in command
        calls.append('indexes')
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(evidence_builder,'build_session_evidence',evidence)
    monkeypatch.setattr(online_memory,'build_online_em2mem_memory',memory)
    monkeypatch.setattr(uploads.subprocess,'run',indexes)
    stages=[]
    result=uploads.build_uploaded_video(session,tmp_path,tmp_path,lambda stage,_:stages.append(stage))
    assert calls == ['preprocess','evidence','memory','indexes']
    assert stages[-1] == 'indexing' and result['duration_seconds'] == 2
