"""Offline correctness reproductions. No services, credentials or model calls.

Run from the repository root with .venv/bin/python and PYTHONDONTWRITEBYTECODE=1.
All session data is generated in TemporaryDirectory and deleted afterwards.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def test_captionless_timestamps():
    from online_short_term.episodic_window_builder import MSTEpisodicWindowBuilder
    path = 'stream/frames/frame_000040.jpg'
    event = {'event_id': 'cross', 'start_time': 0.0, 'end_time': 10.0,
             'status': 'refined', 'event_caption_refined': 'A phone is visible.',
             'keyframes': [{'path': path, 'timestamp': 5.0, 'role': 'latest'}]}
    with tempfile.TemporaryDirectory() as tmp:
        episode = MSTEpisodicWindowBuilder(backend='rule').build_episode(
            {'start_time': 0, 'end_time': 30}, [event], session_id='x', session_dir=Path(tmp))
    assert episode['keyframe_captions'] == [{'timestamp': 5.0, 'path': path, 'caption': ''}]


def test_readable_chunk_crash():
    import cv2
    import numpy as np
    from online_current.mcur_updater import MCurUpdater
    from online_short_term.micro_event_builder import MicroEventBuilder

    with tempfile.TemporaryDirectory() as tmp:
        session = Path(tmp) / 'chunk_session'
        session.mkdir()
        video = session / 'tiny.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 5.0, (64, 64))
        assert writer.isOpened()
        for _ in range(5):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        chunk = {'chunk_id': 'chunk_000000_000001', 'path': video.name,
                 'start_time': 0.0, 'end_time': 1.0, 'duration': 1.0}
        errors = {}
        for name, call in [
            ('MicroEventBuilder.process_chunk', lambda: MicroEventBuilder(session).process_chunk(chunk)),
            ('MCurUpdater.update_from_stream_chunk', lambda: MCurUpdater(session).update_from_stream_chunk(chunk_info=chunk)),
        ]:
            try:
                call()
            except Exception as exc:
                errors[name] = f'{type(exc).__name__}: {exc}'
        assert errors == {}


def test_audio_current_window():
    from online_current.mcur_store import MCurStore

    with tempfile.TemporaryDirectory() as tmp:
        store = MCurStore(Path(tmp) / 'audio_current')
        store.window_seconds = 30.0
        store.update_transcript_segments([{'segment_id': 'old', 'start': 0, 'end': 1, 'text': 'OLD_SPEECH'}])
        store.update_transcript_segments([{'segment_id': 'new', 'start': 40, 'end': 41, 'text': 'NEW_SPEECH'}])
        context = store.get_current_context()
        state = context['state']
        assert state['current_time'] == 41.0
        assert state['window_start_time'] == 11.0
        assert context['transcript'] == 'NEW_SPEECH'
        # Delayed old ASR must neither rewind the window nor restore old text.
        store.update_transcript_segments([{'start': 0, 'end': 1, 'text': 'LATE_OLD'}])
        replayed = store.get_current_context()
        assert replayed['state']['current_time'] == 41.0
        assert replayed['state']['window_start_time'] == 11.0
        assert replayed['transcript'] == 'NEW_SPEECH'


def test_cross_window_backfill():
    from online_short_term.mst_store import MSTStore
    from online_short_term.mst_to_episodic import build_episodic_from_mst
    from online_streaming.transcript_backfill import TranscriptBackfiller
    from refine_mst_micro_events import refine_session

    with tempfile.TemporaryDirectory() as tmp:
        sessions = Path(tmp) / 'online_sessions'
        session_id = 'cross_window'
        session = sessions / session_id
        store = MSTStore(session)
        store.append_events([
            {'event_id': name, 'start_time': start, 'end_time': end,
             'status': 'refined', 'needs_refine': False,
             'event_caption_refined': 'Prior visual event.', 'transcript': ''}
            for name, start, end in [('before', 0, 25), ('crossing', 25, 35), ('after', 35, 60)]
        ])
        initial = build_episodic_from_mst(session_id=session_id, sessions_root=sessions, backend='rule')
        assert initial['generated_episode_count'] == 2
        result = TranscriptBackfiller(session, project_root=Path(tmp), enqueue_refine=False,
                                      enqueue_consolidation=False).backfill_segments([
            {'segment_id': 'new_transcript', 'start': 32, 'end': 33, 'text': 'NEW_CROSSING_SPEECH'}
        ], reason='audio_asr_backfill')
        dirty = result['dirty_windows_due_to_transcript']
        assert [w['window_id'] for w in dirty] == ['win_000000_000030', 'win_000030_000060']
        crossing = next(event for event in store.load_archive_events() if event['event_id'] == 'crossing')
        assert crossing['dirty_window_ids'] == ['win_000000_000030', 'win_000030_000060']
        assert crossing['dirty_time_range'] == [0, 60]
        refine_session(session_id=session_id, sessions_root=sessions, backend='mock', limit_events=10,
                       event_ids=['crossing'], force_refine=True)
        for window in dirty:
            build_episodic_from_mst(session_id=session_id, sessions_root=sessions, backend='rule',
                                    force=True, window_start=window['start_time'], window_end=window['end_time'])
        episodes = json.loads((session / 'em2mem/mst_episodic/mst_30sec_episodes.json').read_text())
        transcripts = {episode['episode_id']: episode['transcript'] for episode in episodes}
        assert transcripts['mst_ep_000000_000030'] == 'NEW_CROSSING_SPEECH'
        assert transcripts['mst_ep_000030_000060'] == 'NEW_CROSSING_SPEECH'


def test_open_frame_event_backfill(monkeypatch):
    import cv2
    import numpy as np
    from online_short_term.frame_stream_event_builder import FrameStreamMicroEventBuilder
    from online_short_term.mst_store import MSTStore
    from online_short_term.mst_to_episodic import build_episodic_from_mst
    from online_streaming.partial_transcript_store import PartialTranscriptStore
    from online_streaming.transcript_backfill import TranscriptBackfiller
    from online_short_term.transcript_aligner import TranscriptAligner
    from refine_mst_micro_events import refine_session

    # Identical valid images and an explicit maximum duration isolate timing,
    # rather than relying on a particular visual-difference heuristic.
    monkeypatch.setenv('EM2MEM_FRAME_STREAM_MAX_EVENT_SECONDS', '10')
    monkeypatch.setenv('EM2MEM_FRAME_STREAM_DIFF_THRESHOLD', '2')
    with tempfile.TemporaryDirectory() as tmp:
        sessions = Path(tmp) / 'online_sessions'
        session_id = 'open_frame'
        session = sessions / session_id
        frames_dir = session / 'stream/frames'
        frames_dir.mkdir(parents=True)
        builder = FrameStreamMicroEventBuilder(session)

        def process(ts):
            path = frames_dir / f'frame_{ts:06d}.jpg'
            assert cv2.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8))
            return builder.process_frame(frame_record={
                'saved_path': str(path.relative_to(session)), 'frame_index': ts,
                'frame_id': f'frame_{ts:06d}', 'relative_ts_ms': ts * 1000,
            }, enqueue_refine=False)

        for ts in [0, 10, 15]:
            process(ts)
        segments = [
            {'segment_id': 'closed_speech', 'start': 5, 'end': 6, 'text': 'CLOSED_SPEECH'},
            {'segment_id': 'open_speech', 'start': 12, 'end': 13, 'text': 'OPEN_SPEECH'},
        ]
        PartialTranscriptStore(session).append_segments(segments)
        result = TranscriptBackfiller(session, project_root=Path(tmp), enqueue_refine=False,
                                      enqueue_consolidation=False).backfill_segments(segments, reason='audio_asr_backfill')
        assert result['backfilled_event_count'] == 1 and result['audio_only_event_count'] == 0
        for ts in [20, 30]:
            process(ts)
        archive = MSTStore(session).load_archive_events()
        at_12 = next(e for e in archive if e['start_time'] == 10 and e['end_time'] == 20)
        assert at_12['transcript'] == 'OPEN_SPEECH'
        assert TranscriptAligner(session).align(10, 20)['transcript'] == 'OPEN_SPEECH'
        assert any('OPEN_SPEECH' in event['transcript'] for event in archive)
        refine_session(session_id=session_id, sessions_root=sessions, backend='mock', limit_events=10)
        build_episodic_from_mst(session_id=session_id, sessions_root=sessions, backend='rule')
        episodes = json.loads((session / 'em2mem/mst_episodic/mst_30sec_episodes.json').read_text())
        assert len(episodes) == 1 and episodes[0]['transcript'] == 'CLOSED_SPEECH OPEN_SPEECH'


@pytest.mark.parametrize("initially_merged", [False, True])
def test_aggregation_overwrites_backfill(initially_merged):
    from unittest.mock import patch
    from online_short_term.episodic_window_builder import MSTEpisodicWindowBuilder
    from online_short_term.mst_store import MSTStore
    from online_short_term.mst_to_episodic import build_episodic_from_mst
    from online_streaming.transcript_backfill import TranscriptBackfiller

    with tempfile.TemporaryDirectory() as tmp:
        sessions = Path(tmp) / 'online_sessions'
        session_id = 'inflight_backfill'
        session = sessions / session_id
        store = MSTStore(session)
        store.append_events([{'event_id': 'ready', 'start_time': 0, 'end_time': 30,
                              'status': 'refined', 'needs_refine': False,
                              'event_caption_refined': 'A visual event.'}])
        if initially_merged:
            build_episodic_from_mst(session_id=session_id, sessions_root=sessions, backend='rule')
        original_build = MSTEpisodicWindowBuilder.build_episode
        during = {}

        def interleaved_build(builder, *args, **kwargs):
            # Simulate the ASR worker completing after consolidation snapshots
            # its events and before it commits its model result. This is a legal
            # deterministic interleaving; no implementation behavior is altered.
            TranscriptBackfiller(session, project_root=Path(tmp), enqueue_refine=False,
                                  enqueue_consolidation=False).backfill_segments([
                {'segment_id': 'late', 'start': 15, 'end': 16, 'text': 'LATE_SPEECH'}
            ], reason='audio_asr_backfill')
            current = store.load_archive_events()[0]
            during.update({'transcript': current['transcript'], 'version': current['version'],
                           'needs_refine': current['needs_refine'],
                           'transcript_version': current['transcript_version']})
            return original_build(builder, *args, **kwargs)

        with patch.object(MSTEpisodicWindowBuilder, 'build_episode', interleaved_build):
            result = build_episodic_from_mst(session_id=session_id, sessions_root=sessions, backend='rule', force=True)
        assert result['generated_episode_count'] == 0
        assert result['generated_episode_ids'] == []
        persisted_episodes = json.loads((session / 'em2mem/mst_episodic/mst_30sec_episodes.json').read_text())
        assert len(persisted_episodes) == int(initially_merged)
        if persisted_episodes:
            assert persisted_episodes[0]['version'] == 1
        after = store.load_archive_events()[0]
        assert during['transcript'] == 'LATE_SPEECH' and during['needs_refine']
        assert after['transcript'] == 'LATE_SPEECH' and after['needs_refine']
        assert after['merged_to_long_term'] is initially_merged
        assert after['needs_reconsolidation']
        assert after['dirty_reason']
        assert after['version'] >= during['version']


def test_audio_current_window_advances_for_overlapping_segments(tmp_path):
    from online_current.mcur_store import MCurStore
    store = MCurStore(tmp_path / 'audio_overlap')
    store.window_seconds = 30
    store.update_transcript_segments([{'start': 0, 'end': 1, 'text': 'OLD'}])
    store.update_transcript_segments([{'start': 1, 'end': 41, 'text': 'NEW'}])
    current = store.get_current_context()
    assert current['state']['current_time'] == 41
    assert current['state']['window_start_time'] == 11
    assert current['transcript'] == 'NEW'


def test_dirty_windows_use_half_open_event_ranges():
    from online_short_term.time_windows import event_windows
    assert [w['start_time'] for w in event_windows({'start_time': 0, 'end_time': 30})] == [0]
    assert [w['start_time'] for w in event_windows({'start_time': 25, 'end_time': 65})] == [0, 30, 60]
    assert [w['start_time'] for w in event_windows({'start_time': 30, 'end_time': 30})] == [30]


def test_dirty_completion_preserves_newer_asr_changes(tmp_path):
    from online_short_term.mst_store import MSTStore
    from online_streaming.transcript_backfill import (
        load_transcript_dirty_windows, mark_transcript_dirty_windows,
        mark_transcript_dirty_window_consolidated,
    )
    store = MSTStore(tmp_path / 'dirty_completion')
    store.append_events([{'event_id': 'late', 'start_time': 0, 'end_time': 30,
                         'status': 'refined', 'needs_refine': True,
                         'needs_reconsolidation': True, 'refined_stale': True}])
    mark_transcript_dirty_windows(store.session_dir, store.load_archive_events())
    mark_transcript_dirty_window_consolidated(store.session_dir, 0, 30)
    assert load_transcript_dirty_windows(store.session_dir)['windows'][0]['status'] == 'dirty'
    # An idempotent completion can still clean up once newer work is merged.
    latest = store.load_archive_events()[0]
    latest.update(needs_refine=False, refined_stale=False, needs_reconsolidation=False,
                  merged_to_long_term=True)
    store.update_events([latest])
    mark_transcript_dirty_window_consolidated(store.session_dir, 0, 30)
    assert load_transcript_dirty_windows(store.session_dir)['windows'][0]['status'] == 'consolidated'


def test_episode_publication_serializes_newer_asr_commit(tmp_path, monkeypatch):
    """ASR arriving at publication must survive in both episode and M_st."""
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    import online_short_term.mst_to_episodic as module
    from online_short_term.mst_store import MSTStore, _process_mutation_lock
    from online_streaming.transcript_backfill import TranscriptBackfiller
    from refine_mst_micro_events import refine_session

    session = tmp_path / 'publication'
    store = MSTStore(session)
    store.append_events([{'event_id': 'event', 'start_time': 0, 'end_time': 30,
                          'status': 'refined', 'needs_refine': False,
                          'event_caption_refined': 'A visual event.'}])
    original_write = module._write_outputs
    probe_results = Queue()
    newer_future = None
    started = False

    def newer_build():
        # Probe from the other thread without blocking. This chooses the legal
        # schedule deterministically: if publication is unprotected, finish the
        # new build before the old write; otherwise ASR waits for that commit.
        lock = _process_mutation_lock(store.mutation_lock_path)
        available = lock.acquire(blocking=False)
        if available:
            lock.release()
        probe_results.put(available)
        TranscriptBackfiller(session, project_root=tmp_path, enqueue_refine=False,
                              enqueue_consolidation=False).backfill_segments([
            {'segment_id': 'late', 'start': 15, 'end': 16, 'text': 'LATE_SPEECH'}
        ], reason='audio_asr_backfill')
        refine_session(session_id=session.name, sessions_root=tmp_path, backend='mock',
                       force_refine=True, event_ids=['event'], limit_events=1)
        return module.build_episodic_from_mst(session_id=session.name, sessions_root=tmp_path,
                                             backend='rule', force=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        def interleaved_write(*args, **kwargs):
            nonlocal started, newer_future
            if not started:
                started = True
                newer_future = executor.submit(newer_build)
                if probe_results.get(timeout=10):
                    newer_future.result(timeout=10)
            return original_write(*args, **kwargs)

        monkeypatch.setattr(module, '_write_outputs', interleaved_write)
        module.build_episodic_from_mst(session_id=session.name, sessions_root=tmp_path, backend='rule')
        assert newer_future.result(timeout=10)['generated_episode_count'] == 1

    episodes = json.loads((session / 'em2mem/mst_episodic/mst_30sec_episodes.json').read_text())
    assert episodes[0]['transcript'] == 'LATE_SPEECH'
    latest = store.load_archive_events()[0]
    assert latest['transcript'] == 'LATE_SPEECH'
    assert latest['merged_to_long_term'] and not latest['needs_reconsolidation']
    assert episodes[0]['version'] == latest['merged_version'] == 2


def test_concurrent_window_publication_merges_latest_episode_and_state(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import online_short_term.mst_to_episodic as module
    from online_short_term.episodic_window_builder import MSTEpisodicWindowBuilder
    from online_short_term.mst_store import MSTStore
    from online_short_term.consolidation_state import load_consolidation_state

    session = tmp_path / 'separate_windows'
    store = MSTStore(session)
    store.append_events([
        {'event_id': f'event_{start}', 'start_time': start, 'end_time': start + 30,
         'status': 'refined', 'needs_refine': False, 'event_caption_refined': f'Event {start}.'}
        for start in (0, 30)
    ])
    barrier = Barrier(2)
    original_build = MSTEpisodicWindowBuilder.build_episode

    def synchronized_build(builder, *args, **kwargs):
        episode = original_build(builder, *args, **kwargs)
        barrier.wait(timeout=10)
        return episode

    monkeypatch.setattr(MSTEpisodicWindowBuilder, 'build_episode', synchronized_build)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(module.build_episodic_from_mst, session_id=session.name,
                                   sessions_root=tmp_path, backend='rule',
                                   window_start=start, window_end=start + 30)
                   for start in (0, 30)]
        assert [future.result(timeout=20)['generated_episode_count'] for future in futures] == [1, 1]

    episodes = json.loads((session / 'em2mem/mst_episodic/mst_30sec_episodes.json').read_text())
    assert [episode['episode_id'] for episode in episodes] == ['mst_ep_000000_000030', 'mst_ep_000030_000060']
    state = load_consolidation_state(session, session.name)
    assert set(state['window_to_episode']) == {'win_000000_000030', 'win_000030_000060'}
    assert state['generated_episode_count'] == 2
    assert state['last_consolidated_window_end'] == 60
    assert all(event['merged_to_long_term'] for event in store.load_archive_events())


def test_frame_closure_serializes_asr_after_alignment(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    import cv2
    import numpy as np
    from online_short_term.frame_stream_event_builder import FrameStreamMicroEventBuilder
    from online_short_term.mst_store import MSTStore, _process_mutation_lock
    from online_short_term.transcript_aligner import TranscriptAligner
    from online_streaming.partial_transcript_store import PartialTranscriptStore
    from online_streaming.transcript_backfill import TranscriptBackfiller

    monkeypatch.setenv('EM2MEM_FRAME_STREAM_MAX_EVENT_SECONDS', '10')
    monkeypatch.setenv('EM2MEM_FRAME_STREAM_DIFF_THRESHOLD', '2')
    session = tmp_path / 'close_race'
    frames = session / 'stream/frames'
    frames.mkdir(parents=True)
    builder = FrameStreamMicroEventBuilder(session)

    def process(timestamp):
        path = frames / f'frame_{timestamp:06d}.jpg'
        assert cv2.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8))
        builder.process_frame(frame_record={'saved_path': str(path.relative_to(session)),
                                            'frame_index': timestamp, 'frame_id': path.stem,
                                            'relative_ts_ms': timestamp * 1000}, enqueue_refine=False)

    for timestamp in (0, 10, 15):
        process(timestamp)
    original_align = TranscriptAligner.align
    probes = Queue()
    asr_future = None

    def late_asr():
        segments = [{'segment_id': 'closed', 'start': 5, 'end': 6, 'text': 'CLOSED'},
                    {'segment_id': 'open', 'start': 12, 'end': 13, 'text': 'OPEN'}]
        # Production writes partial transcripts before trying the M_st lock.
        PartialTranscriptStore(session).append_segments(segments)
        lock = _process_mutation_lock(builder.store.mutation_lock_path)
        available = lock.acquire(blocking=False)
        if available:
            lock.release()
        probes.put(available)
        return TranscriptBackfiller(session, project_root=tmp_path, enqueue_refine=False,
                                    enqueue_consolidation=False).backfill_segments(segments)

    with ThreadPoolExecutor(max_workers=1) as executor:
        def after_alignment(aligner, start, end):
            nonlocal asr_future
            aligned = original_align(aligner, start, end)
            if start == 10 and asr_future is None:
                asr_future = executor.submit(late_asr)
                if probes.get(timeout=10):
                    asr_future.result(timeout=10)
            return aligned

        monkeypatch.setattr(TranscriptAligner, 'align', after_alignment)
        process(20)
        assert asr_future.result(timeout=10)['audio_only_event_count'] == 0
    event = next(event for event in builder.store.load_archive_events() if event['start_time'] == 10)
    assert event['transcript'] == 'OPEN'


@pytest.mark.parametrize("initially_merged", [False, True])
@pytest.mark.parametrize("refine_before_commit", [False, True])
def test_episode_commit_rejects_new_window_members(tmp_path, monkeypatch, initially_merged, refine_before_commit):
    import online_short_term.mst_to_episodic as module
    from online_short_term.episodic_window_builder import MSTEpisodicWindowBuilder
    from online_short_term.mst_store import MSTStore
    from online_streaming.transcript_backfill import TranscriptBackfiller, load_transcript_dirty_windows
    from online_mst_refine_worker import _run_refine_task

    session = tmp_path / 'new_member'
    store = MSTStore(session)
    store.append_events([
        {'event_id': name, 'start_time': start, 'end_time': end, 'status': 'refined',
         'needs_refine': False, 'event_caption_refined': 'A visual event.'}
        for name, start, end in [('first', 0, 10), ('next', 30, 60)]
    ])
    if initially_merged:
        module.build_episodic_from_mst(session_id=session.name, sessions_root=tmp_path,
                                      backend='rule', window_start=0, window_end=30)
    original_build = MSTEpisodicWindowBuilder.build_episode
    inserted = []
    ready_tasks = []

    def run_refine():
        return _run_refine_task(
            project_root=tmp_path, sessions_root=tmp_path, session_id=session.name,
            backend='mock', limit_events=10, event_id=None, event_ids=inserted,
            force_refine=True, task_reason='audio_asr_backfill', task_id=None,
            task_queued_at=None, task_worker_started_at=None, verbose=False,
        )

    def add_audio_after_model(builder, *args, **kwargs):
        episode = original_build(builder, *args, **kwargs)
        result = TranscriptBackfiller(session, project_root=tmp_path, enqueue_refine=False,
                                      enqueue_consolidation=False).backfill_segments([
            {'segment_id': 'new', 'start': 15, 'end': 16, 'text': 'NEW_AUDIO'}
        ], reason='audio_asr_backfill')
        assert result['audio_only_event_count'] == 1
        inserted.extend(result['audio_only_event_ids'])
        if refine_before_commit:
            ready_tasks.append(run_refine()['consolidation_task_path'])
        return episode

    with monkeypatch.context() as patch:
        patch.setattr(MSTEpisodicWindowBuilder, 'build_episode', add_audio_after_model)
        stale = module.build_episodic_from_mst(session_id=session.name, sessions_root=tmp_path,
                                              backend='rule', window_start=0, window_end=30, force=True)
    assert stale['generated_episode_count'] == 0
    assert load_transcript_dirty_windows(session)['windows'][0]['status'] == 'dirty'
    added = next(event for event in store.load_archive_events() if event['event_id'] in inserted)
    assert added['needs_reconsolidation'] and not added['merged_to_long_term']
    if refine_before_commit:
        # The already queued ordinary ready task must recover too, even when
        # an older episode previously made this window look completed.
        task = json.loads(Path(ready_tasks[0]).read_text())
        assert not task['force']
    else:
        refined = run_refine()
        tasks = [json.loads(Path(path).read_text()) for path in refined['dirty_consolidation_task_paths']]
        task = next(task for task in tasks if task['window_start'] == 0 and task['window_end'] == 30)
        assert task['force']
    retry = module.build_episodic_from_mst(session_id=session.name, sessions_root=tmp_path,
                                          backend='rule', window_start=task.get('window_start'),
                                          window_end=task.get('window_end'), force=task['force'])
    assert retry['generated_episode_count'] >= 1
    episodes = json.loads((session / 'em2mem/mst_episodic/mst_30sec_episodes.json').read_text())
    assert episodes[0]['transcript'] == 'NEW_AUDIO'
