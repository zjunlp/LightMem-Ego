"""Real frame inputs and deterministic schedules for the October memory audit."""
from __future__ import annotations

import sys
import threading
import multiprocessing

import cv2
import numpy as np
import pytest

from online_current.mcur_selector import MCurFrameSelector
from online_current.mcur_store import MCurStore
from online_pipeline.realtime_ingest import ingest_frame
from online_preprocess.io_utils import write_json_atomic
from online_query.evidence_packer import EvidencePacker


def png(color):
    ok, data = cv2.imencode('.png', np.full((64, 96, 3), color, dtype=np.uint8))
    assert ok
    return data.tobytes()


@pytest.mark.parametrize('with_short_term', [True, False])
def test_current_change_score_uses_actual_frames(tmp_path, monkeypatch, with_short_term):
    monkeypatch.setenv('EM2MEM_FRAME_STREAM_ENQUEUE_REFINE', '0')
    session = tmp_path / 'online_sessions' / 'changes'
    write_json_atomic(session / 'stream/stream_state.json', {
        'session_id': session.name, 'input_mode': 'frame_audio_stream', 'status': 'recording',
    })
    responses = []
    for t in range(11):
        responses.append(ingest_frame(
            tmp_path, session.name, png((0, 0, 255) if t == 5 else (0, 255, 0)),
            frame_index=t, relative_ts_ms=t * 1000, update_mcur=True,
            update_mst=with_short_term, async_memory_updates=False,
        ))
    assert all(not r.get('mcur_error') and not r.get('frame_mst_error') for r in responses)
    context = MCurStore(session).get_current_context()
    at_five = next(f for f in context['frames'] if f['timestamp'] == 5)
    assert at_five['diff_score'] >= .4
    if with_short_term:
        assert at_five['diff_score'] == responses[5]['frame_mst']['diff_score']
    selection = MCurFrameSelector().select_frames_for_query(context, '当前画面是什么颜色？')
    packed = EvidencePacker(session).pack('当前画面', {
        'use_image_evidence': True, 'max_image_evidence': 3,
    }, {'current_context': context, 'current_selection': selection})
    assert at_five['path'] in packed['selected_image_paths_for_mllm']


@pytest.mark.parametrize('budget', [1, 3])
def test_only_newest_frame_is_latest_and_is_sent(tmp_path, budget):
    store = MCurStore(tmp_path)
    for t, score in [(2, .6), (4, .6), (6, .6), (8, .01)]:
        path = tmp_path / f'input_{t}.png'
        path.write_bytes(png((0, 255, 0)))
        store.update_from_chunk({'end_time': t}, [
            {'timestamp': t, 'path': path.name, 'diff_score': score},
        ], [])
    context = store.get_current_context()
    assert [f['timestamp'] for f in context['frames'] if f['role'] == 'latest'] == [8]
    latest = context['frames'][-1]['path']
    selection = MCurFrameSelector().select_frames_for_query(context, '当前画面', max_images=budget)
    assert selection['selected_image_paths_for_mllm'][0] == latest
    packed = EvidencePacker(tmp_path).pack('当前画面', {
        'use_image_evidence': True, 'max_image_evidence': budget,
    }, {'current_context': context, 'current_selection': selection})
    assert packed['selected_image_paths_for_mllm'][0] == latest


def test_frame_update_does_not_erase_interleaved_asr(tmp_path):
    path = tmp_path / 'input.png'
    path.write_bytes(png((0, 255, 0)))
    store = MCurStore(tmp_path)
    store.update_from_frame_stream(frame_index=1, frame_path=path.name, relative_ts_ms=1000)
    reached, release = threading.Event(), threading.Event()
    errors = []
    target = MCurStore.update_from_chunk.__code__

    def checkpoint(frame, event, arg):
        if event == 'call' and frame.f_code is target:
            reached.set()
            assert release.wait(5)
        return None

    def frame_job():
        sys.settrace(checkpoint)
        try:
            MCurStore(tmp_path).update_from_frame_stream(
                frame_index=2, frame_path=path.name, relative_ts_ms=2000,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=frame_job)
    thread.start()
    try:
        assert reached.wait(5)
        store.update_transcript_segments([
            {'segment_id': 'speech', 'start': .2, 'end': .8, 'text': '钥匙放在副驾'},
        ])
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    context = store.get_current_context()
    assert context['transcript'] == '钥匙放在副驾'
    assert context['open_event']['transcript'] == '钥匙放在副驾'
    assert context['state']['transcript_segment_count'] == 1
    store.update_from_frame_stream(frame_index=3, frame_path=path.name, relative_ts_ms=3000)
    assert store.get_current_context()['transcript'] == '钥匙放在副驾'


def test_preview_cleanup_preserves_memory_evidence(tmp_path, monkeypatch):
    from online_pipeline.frame_stream import FrameStreamStore, sha256_path
    from online_short_term.frame_stream_event_builder import FrameStreamMicroEventBuilder

    monkeypatch.setenv('EM2MEM_FRAME_STREAM_PREVIEW_HISTORY', '1')
    monkeypatch.setenv('EM2MEM_FRAME_STREAM_MEMORY_HISTORY', '2')
    monkeypatch.setenv('EM2MEM_FRAME_STREAM_STATE_HISTORY', '500')
    store = FrameStreamStore(tmp_path / 'session')
    store.initialize(stream_id='evidence', target_fps=1.)
    builder = FrameStreamMicroEventBuilder(store.session_dir)

    def register(i, timestamp=None):
        p = tmp_path / f'upload_{i}.png'
        p.write_bytes(png((0, 255, 0)))
        return store.register_frame(
            tmp_path=p, frame_index=i, checksum=sha256_path(p), suffix='.png',
            size_bytes=p.stat().st_size, client_ts_ms=None, relative_ts_ms=i * 1000 if timestamp is None else timestamp,
            source_ts_ms=None, timestamp_source='test', width=96, height=64, source='test',
        )['frame']

    for i in (0, 1):
        builder.process_frame(frame_record=register(i), enqueue_refine=False)
    builder.close_open_event(enqueue_refine=False)
    paths = [f['path'] for f in builder.store.load_events()[0]['keyframes']]
    for i in range(2, 8):
        register(i)
    assert paths and all((store.session_dir / p).is_file() for p in paths)
    preview_only = register(8, 7100)
    assert not preview_only['memory_accepted']
    register(9, 8000)
    assert not (store.session_dir / preview_only['saved_path']).exists()


def test_frame_and_asr_mutations_serialize_across_processes(tmp_path):
    """Separate worker processes must share the same session mutation lock."""
    ctx = multiprocessing.get_context('fork')
    path = tmp_path / 'input.png'
    path.write_bytes(png((0, 255, 0)))
    MCurStore(tmp_path).update_from_chunk({'end_time': 1}, [
        {'timestamp': 1., 'path': path.name, 'diff_score': 0.},
    ], [])
    writing, release, started, completed = (ctx.Event() for _ in range(4))

    def frame_job():
        store = MCurStore(tmp_path)
        original = store._write_jsonl
        def checkpoint(target, rows):
            if target == store.frames_path:
                writing.set()
                assert release.wait(5)
            return original(target, rows)
        store._write_jsonl = checkpoint
        store.update_from_chunk({'end_time': 2}, [
            {'timestamp': 2., 'path': path.name, 'diff_score': 0.},
        ], [])

    def asr_job():
        started.set()
        MCurStore(tmp_path).update_transcript_segments([
            {'segment_id': 'speech', 'start': .2, 'end': .8, 'text': '保留语音'},
        ])
        completed.set()

    frame = ctx.Process(target=frame_job)
    asr = ctx.Process(target=asr_job)
    frame.start()
    try:
        assert writing.wait(5)
        asr.start()
        assert started.wait(5)
        assert not completed.wait(.25), 'ASR wrote while the frame mutation was still in progress'
    finally:
        release.set()
        frame.join(5)
        if asr.pid is not None:
            asr.join(5)
        for process in (frame, asr):
            if process.is_alive():
                process.terminate()
                process.join()
    assert frame.exitcode == asr.exitcode == 0
    assert MCurStore(tmp_path).get_current_context()['transcript'] == '保留语音'
