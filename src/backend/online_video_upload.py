"""Durable video upload receipts and the existing offline memory build path."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO, Callable
from uuid import uuid4

from online_preprocess.io_utils import ffmpeg_bin, ffprobe_bin, read_json, utc_now_iso, write_json_atomic, write_status

MAX_UPLOAD_BYTES = 95 * 1024 * 1024
SUFFIXES = {'.mp4', '.mov', '.m4v', '.webm', '.mkv', '.avi'}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='video-upload')
_jobs: dict[tuple[str, str], object] = {}
_jobs_lock = threading.Lock()


class VideoUploadError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def canonical_id(value: str) -> str:
    normalized = str(value or '').replace('-', '').lower()
    if not re.fullmatch(r'[a-f0-9]{32}', normalized):
        raise VideoUploadError(400, 'Invalid upload ID.')
    return normalized


def _run_media(command: list[str], *, timeout=3600) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise VideoUploadError(422, 'Video could not be processed: ' + result.stderr[-1200:])
    return result


def prepare_uploaded_video(session: Path, source: Path) -> dict:
    probe = _run_media([ffprobe_bin(), '-v', 'error', '-protocol_whitelist', 'file,pipe',
                        '-show_streams', '-show_format', '-of', 'json', str(source)], timeout=60)
    payload = json.loads(probe.stdout)
    video = next((stream for stream in payload.get('streams', []) if stream.get('codec_type') == 'video'), None)
    duration = float(payload.get('format', {}).get('duration') or (video or {}).get('duration') or 0)
    if not video or not math.isfinite(duration) or duration <= 0:
        raise VideoUploadError(422, 'The file does not contain a readable video stream.')
    target = session / 'input.mp4'
    if not target.exists():
        staged = session / 'input.preparing.mp4'
        _run_media([ffmpeg_bin(), '-v', 'error', '-nostdin', '-y', '-protocol_whitelist', 'file,pipe',
                    '-i', str(source), '-map', '0:v:0', '-map', '0:a:0?', '-sn',
                    '-vf', r'scale=trunc(min(1280\,iw)/2)*2:-2', '-c:v', 'libx264',
                    '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-threads', '2',
                    '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(staged)])
        staged.replace(target)
    return {'duration_seconds': duration, 'has_audio': any(s.get('codec_type') == 'audio' for s in payload.get('streams', []))}


def build_uploaded_video(session: Path, sessions_root: Path, project_root: Path, report: Callable) -> dict:
    """Use the repository's full-video preprocess/evidence/memory functions.

    This is the existing offline path, not camera replay and not a synthetic
    live session. No active-camera session or retrieval policy is changed.
    """
    from online_processor import process_session
    from online_preprocess.evidence_builder import build_session_evidence
    from online_memory import build_online_em2mem_memory
    from online_llm_config import external_openai_model

    state = read_json(session / 'video_upload.json', {})
    report('validating', 3)
    report('transcoding', 8)
    media = prepare_uploaded_video(session, session / state['source_file'])
    report('preprocessing', 15)
    process_session(session.name, sessions_root, whisperx_model=os.getenv('EM2MEM_WHISPERX_MODEL', 'large-v3'),
                    device=os.getenv('EM2MEM_ASR_DEVICE', 'cpu'), compute_type='auto',
                    language=None, model_dir=None, align_model_dir=None, skip_asr=False, force=False)
    report('evidence', 60)
    previous_evidence = read_json(session/'evidence/session_evidence.json', [])
    evidence, _ = build_session_evidence(session.name, sessions_root,
                    backend=os.getenv('EM2MEM_EVIDENCE_CAPTION_BACKEND', 'openai'),
                    model=os.getenv('EM2MEM_VLM_MODEL') or external_openai_model(),
                    max_keyframes=int(os.getenv('EM2MEM_VLM_MAX_KEYFRAMES', '8')),
                    force=any(row.get('status') == 'fallback' for row in previous_evidence))
    records = read_json(evidence, [])
    if not records:
        raise RuntimeError('No video events were generated.')
    failed = [row for row in records if row.get('status') == 'fallback']
    if failed:
        raise RuntimeError(f"Event descriptions failed for {len(failed)} segments: {failed[0].get('error', 'generation failed')}")
    report('memory', 85)
    config_path = build_online_em2mem_memory(session.name, sessions_root, source='legacy_evidence',
                    model_name=os.getenv('EM2MEM_MEMORY_MODEL') or external_openai_model(),
                    skip_visual_embedding=os.getenv('EM2MEM_SKIP_VISUAL_EMBEDDING', '1').lower() in {'1','true','yes'},
                    skip_semantic=False)
    report('indexing', 95)
    # Index initialization changes process-wide model settings internally;
    # keep it in the same isolated subprocess used by the existing worker.
    result = subprocess.run([sys.executable, str(project_root/'memory_worker_subprocess.py'),
                    'build_retrieval_artifacts', '--session-id', session.name,
                    '--sessions-root', str(sessions_root)], cwd=project_root,
                    capture_output=True, text=True, timeout=3600, check=False)
    if result.returncode:
        raise RuntimeError('Retrieval index initialization failed: ' + result.stderr[-1200:])
    config = read_json(config_path, {})
    return {**media, 'memory_version': config.get('memory_version')}


class VideoUploadService:
    def __init__(self, sessions_root: Path, project_root: Path, *, builder=None,
                 schedule=True, max_bytes=MAX_UPLOAD_BYTES):
        self.root = Path(sessions_root).resolve()
        self.project = Path(project_root).resolve()
        self.builder = builder or build_uploaded_video
        self.schedule = schedule
        self.max_bytes = max_bytes

    def _session(self, upload_id: str) -> Path:
        path = self.root / ('video_' + canonical_id(upload_id))
        if path.resolve().parent != self.root:
            raise VideoUploadError(400, 'Invalid upload session.')
        return path

    def _write(self, session: Path, state: dict):
        state['updated_at'] = utc_now_iso()
        write_json_atomic(session/'video_upload.json', state)

    def receive(self, file: BinaryIO, filename: str, upload_id: str) -> dict:
        upload_id = canonical_id(upload_id)
        filename = str(filename or 'video.mp4').replace('\\', '/').split('/')[-1]
        suffix = Path(filename).suffix.lower()
        if suffix not in SUFFIXES:
            raise VideoUploadError(415, 'Choose an MP4, MOV, WebM, MKV or AVI video.')
        staging = self.root/'.video-upload-staging'
        staging.mkdir(parents=True, exist_ok=True)
        temporary = staging/(uuid4().hex+'.part')
        digest, size = hashlib.sha256(), 0
        try:
            with temporary.open('wb') as output:
                while block := file.read(1024*1024):
                    size += len(block)
                    if size > self.max_bytes:
                        raise VideoUploadError(413, 'Video exceeds the 95 MiB upload limit.')
                    digest.update(block)
                    output.write(block)
            if not size:
                raise VideoUploadError(400, 'The uploaded video is empty.')
            session = self._session(upload_id)
            # Different API processes use the same receipt lock.
            with (staging/(upload_id+'.lock')).open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                previous = read_json(session/'video_upload.json', {})
                if previous:
                    if previous.get('sha256') != digest.hexdigest() or previous.get('filename') != filename:
                        raise VideoUploadError(409, 'This upload ID belongs to a different file.')
                else:
                    session.mkdir(exist_ok=False)
                    source = session/('source'+suffix)
                    temporary.replace(source)
                    now = utc_now_iso()
                    write_json_atomic(session/'metadata.json', {
                        'session_id': session.name, 'source': 'video_upload', 'original_filename': filename,
                        'title': Path(filename).stem[:100], 'title_revision': 1, 'size_bytes':size,
                        'created_at':now, 'updated_at':now, 'memory_pipeline':'offline_evidence',
                    })
                    self._write(session, {
                        'upload_id':upload_id, 'session_id':session.name, 'filename':filename,
                        'sha256':digest.hexdigest(), 'source_file':source.name, 'size_bytes':size,
                        'status':'queued', 'stage':'queued', 'progress':0, 'error':None,
                        'created_at':now, 'query_ready':False,
                    })
            self.ensure_scheduled(upload_id)
            return self.status(upload_id)
        finally:
            temporary.unlink(missing_ok=True)

    def ensure_scheduled(self, upload_id: str):
        if not self.schedule or self.status(upload_id)['status'] in {'ready','failed'}:
            return
        key = (str(self.root), canonical_id(upload_id))
        with _jobs_lock:
            previous = _jobs.get(key)
            if previous is None or previous.done():
                _jobs[key] = _executor.submit(self.run, upload_id)

    def status(self, upload_id: str) -> dict:
        session = self._session(upload_id)
        state = read_json(session/'video_upload.json', {})
        if not state:
            raise VideoUploadError(404, 'Upload not found.')
        keys = ['upload_id','session_id','filename','size_bytes','status','stage','progress','error','created_at','updated_at','query_ready','memory_version','duration_seconds']
        result = {key:state[key] for key in keys if key in state}
        result['history_url'] = f'/history-demo/?session={session.name}'
        result['memory_url'] = f'/memories/?memory_session={session.name}'
        return result

    def run(self, upload_id: str):
        session = self._session(upload_id)
        # A repeated receipt, resume request or server recovery cannot run the
        # same build twice, even with multiple API processes.
        with (session/'video_upload.lock').open('a+') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: return
            state = read_json(session/'video_upload.json', {})
            if state.get('status') == 'ready': return
            def report(stage, progress):
                state.update(status='processing',stage=stage,progress=progress,query_ready=False,error=None)
                self._write(session,state)
            try:
                result = self.builder(session,self.root,self.project,report)
                config = read_json(session/'em2mem/memory_config.json', {})
                if config.get('status') != 'memory_ready' or not config.get('retrieval_artifacts_ready'):
                    raise RuntimeError('Memory publication and retrieval indexes are not ready.')
                metadata = read_json(session/'metadata.json', {})
                metadata['demo_media'] = {
                    'path':'input.mp4', 'duration_seconds':result.get('duration_seconds'),
                    'video_start_seconds':0, 'time_origin':'session_relative_seconds',
                    'memory_until_seconds':result.get('duration_seconds'),
                }
                metadata['updated_at'] = utc_now_iso()
                write_json_atomic(session/'metadata.json',metadata)
                write_json_atomic(session/'em2mem/query_warmup_state.json', {
                    'status':'ready','state':'ready','session_id':session.name,
                    'memory_version':config.get('memory_version'),'query_ready':True,'updated_at':utc_now_iso(),
                })
                write_status(session,session.name,status='ready',stage='memory_ready',progress=100,error=None)
                state.update(result,status='ready',stage='ready',progress=100,query_ready=True,error=None)
            except Exception as exc:
                state.update(status='failed',error=str(exc),query_ready=False)
            self._write(session,state)

    def resume(self, upload_id: str) -> dict:
        session = self._session(upload_id)
        state = read_json(session/'video_upload.json', {})
        if not state: raise VideoUploadError(404, 'Upload not found.')
        if state['status'] == 'failed':
            state.update(status='queued',stage='queued',error=None)
            self._write(session,state)
        self.ensure_scheduled(upload_id)
        return self.status(upload_id)

    def recover(self):
        for path in self.root.glob('video_*/video_upload.json'):
            state = read_json(path,{})
            if state.get('status') in {'queued','processing'}:
                self.ensure_scheduled(state['upload_id'])
