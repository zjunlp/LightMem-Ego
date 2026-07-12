import asyncio
import json
import os
import threading
import subprocess
import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote
from uuid import uuid4

from fastapi import Body, FastAPI, File, Form, Header, UploadFile, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic, write_status
from online_preprocess.text_normalization import normalize_user_visible_text_fields, simplify_chinese_text
from online_preprocess.task_queue import (
    enqueue_evidence_task,
    enqueue_memory_task,
    enqueue_preprocess_task,
    enqueue_query_task,
    enqueue_query_warmup_task,
    ensure_queue_dirs,
    get_queue_dirs,
)
from online_pipeline.stream_timeline import append_timeline_event
from online_retrieval_scheme import normalize_long_term_retrieval_scheme
from online_qa_history import append_qa_history, load_qa_history, qa_history_path


PROJECT_ROOT = Path(__file__).resolve().parent
ONLINE_SESSIONS_DIR = PROJECT_ROOT / "online_sessions"
CHUNK_SIZE_BYTES = 8 * 1024 * 1024
ALLOWED_SESSION_FILE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ROKID_ACTIVE_SESSION_PATH = Path("runtime") / "active_rokid_session.json"


app = FastAPI(title="Em2Mem Upload API")


def _cors_origins() -> list[str]:
    raw = os.getenv("EM2MEM_CORS_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


EM2MEM_CORS_ORIGINS = _cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=EM2MEM_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _voice_question_asr_python() -> Path:
    configured = os.getenv("EM2MEM_VOICE_QUESTION_ASR_PYTHON") or os.getenv("EM2MEM_WHISPERX_PYTHON")
    return Path(configured) if configured else PROJECT_ROOT / ".venv_whisperx" / "bin" / "python"


def _voice_question_model_name_or_path() -> str:
    configured = os.getenv("EM2MEM_VOICE_QUESTION_MODEL") or os.getenv("EM2MEM_WHISPERX_MODEL") or "faster-whisper-medium"
    configured_path = Path(configured)
    if configured_path.is_absolute() and configured_path.exists():
        return str(configured_path)
    local_model = PROJECT_ROOT / "models" / "whisperx" / configured
    if local_model.exists():
        return str(local_model)
    return configured


def _voice_question_language_arg() -> str:
    configured = os.getenv("EM2MEM_VOICE_QUESTION_LANGUAGE", "auto")
    language = str(configured or "").strip()
    if language.lower() in {"", "auto", "detect", "auto_detect", "auto-detect", "none", "null"}:
        return ""
    return language


def _voice_question_initial_prompt(language: str) -> str:
    configured = os.getenv("EM2MEM_VOICE_QUESTION_INITIAL_PROMPT")
    if not language:
        return ""
    if configured is not None:
        return str(configured)
    if language.lower() in {"zh", "cn", "chinese", "zh-cn", "zh_hans", "zh-hans", "simplified_chinese"}:
        return "以下是普通话简体中文语音，请输出简体中文。"
    return ""


def _run_voice_question_asr(audio_path: Path, output_srt: Path, output_json: Path) -> list[dict[str, Any]]:
    asr_python = _voice_question_asr_python()
    if not asr_python.exists():
        raise RuntimeError(f"voice question ASR python not found: {asr_python}")

    script = r'''
import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def parse_device(value):
    if value.startswith("cuda:"):
        return "cuda", int(value.split(":", 1)[1])
    return value, 0


def srt_time(seconds):
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1_000
    millis = total_ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


audio_path = Path(sys.argv[1])
output_json = Path(sys.argv[2])
output_srt = Path(sys.argv[3])
model_name_or_path = sys.argv[4]
device_arg = sys.argv[5]
compute_type = sys.argv[6]
language = sys.argv[7] or None
beam_size = int(sys.argv[8])
vad_filter = sys.argv[9].lower() in {"1", "true", "yes", "on"}
initial_prompt = sys.argv[10] or None

device, device_index = parse_device(device_arg)
model = WhisperModel(
    model_name_or_path,
    device=device,
    device_index=device_index,
    compute_type=compute_type,
)
segments_iter, _info = model.transcribe(
    str(audio_path),
    language=language,
    beam_size=beam_size,
    vad_filter=vad_filter,
    initial_prompt=initial_prompt,
)
segments = [
    {"start": float(seg.start), "end": float(seg.end), "text": str(seg.text).strip()}
    for seg in segments_iter
    if str(seg.text).strip()
]
output_json.parent.mkdir(parents=True, exist_ok=True)
output_json.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
srt_lines = []
for idx, seg in enumerate(segments, start=1):
    srt_lines.extend([
        str(idx),
        f"{srt_time(seg['start'])} --> {srt_time(seg['end'])}",
        seg["text"],
        "",
    ])
output_srt.write_text("\n".join(srt_lines), encoding="utf-8")
print(json.dumps({"segments": segments}, ensure_ascii=False))
'''
    device = os.getenv("EM2MEM_VOICE_QUESTION_DEVICE") or os.getenv("EM2MEM_WHISPERX_DEVICE") or "cuda"
    compute_type = os.getenv("EM2MEM_VOICE_QUESTION_COMPUTE_TYPE") or os.getenv("EM2MEM_WHISPERX_COMPUTE_TYPE") or "float16"
    language = _voice_question_language_arg()
    beam_size = os.getenv("EM2MEM_VOICE_QUESTION_BEAM_SIZE", "1")
    vad_filter = os.getenv("EM2MEM_VOICE_QUESTION_VAD_FILTER", "0")
    initial_prompt = _voice_question_initial_prompt(language)
    cmd = [
        str(asr_python),
        "-c",
        script,
        str(audio_path),
        str(output_json),
        str(output_srt),
        _voice_question_model_name_or_path(),
        device,
        compute_type,
        language,
        beam_size,
        vad_filter,
        initial_prompt,
    ]

    timeout_seconds = int(os.getenv("EM2MEM_VOICE_QUESTION_ASR_TIMEOUT", "60") or 60)
    env = os.environ.copy()
    ffmpeg_bin_dir = os.getenv("EM2MEM_FFMPEG_BIN_DIR", "/zjunlp/chenyijun/miniconda3/bin")
    env["PATH"] = f"{ffmpeg_bin_dir}:{env.get('PATH', '')}"
    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=env,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "voice question ASR failed").strip()
        raise RuntimeError(detail[-1200:])

    segments = read_json(output_json, default=[])
    if not isinstance(segments, list):
        return []
    for segment in segments:
        if isinstance(segment, dict) and isinstance(segment.get("text"), str):
            segment["text"] = simplify_chinese_text(segment["text"])
    return segments


def _start_query_warmup_thread(session_id: str, *, reason: str = "stream_start", wait_for_memory: bool = False) -> None:
    if not _env_bool("EM2MEM_QUERY_WARMUP_ON_STREAM_START", True):
        return
    if _env_bool("EM2MEM_QUERY_WARMUP_VIA_WORKER", True):
        try:
            task_path = enqueue_query_warmup_task(
                PROJECT_ROOT,
                session_id,
                reason=reason,
                wait_for_memory=wait_for_memory,
                long_term_retrieval_scheme=normalize_long_term_retrieval_scheme(None),
            )
            print(f"[api] query warmup queued session={session_id} task={task_path.name} reason={reason}", flush=True)
            return
        except Exception as exc:
            print(f"[api] query warmup enqueue failed session={session_id}: {exc}", flush=True)

    def _run() -> None:
        try:
            from online_query.warmup import warm_query_session

            long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(None)
            result = warm_query_session(
                session_id=session_id,
                sessions_root=ONLINE_SESSIONS_DIR,
                wait_for_memory=wait_for_memory,
                reason=reason,
                long_term_retrieval_scheme=long_term_retrieval_scheme,
            )
            print(
                f"[api] query warmup session={session_id} status={result.get('status')} "
                f"long_term_retrieval_scheme={long_term_retrieval_scheme} total_ms={result.get('total_ms')}",
                flush=True,
            )
        except Exception as exc:
            print(f"[api] query warmup session={session_id} failed: {exc}", flush=True)

    threading.Thread(target=_run, name=f"query-warmup-{session_id}", daemon=True).start()


def _json_response_content(response: JSONResponse) -> dict[str, Any]:
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _rokid_day_session_enabled() -> bool:
    return _env_bool("EM2MEM_ROKID_DAY_SESSION_ENABLED", True)


def _sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _stream_error_response(
    session_id: str,
    question: str,
    message: str,
    *,
    result: dict[str, Any] | None = None,
    status_code: int = 200,
) -> StreamingResponse:
    result_payload = {
        "status": "error",
        "response_mode": "stream",
        "session_id": session_id,
        "question": question,
        "message": message,
        "answer": "",
    }
    if result:
        result_payload.update(result)
        result_payload.setdefault("response_mode", "stream")
        result_payload.setdefault("session_id", session_id)
        result_payload.setdefault("question", question)
        result_payload.setdefault("answer", "")
        result_payload.setdefault("message", message)

    async def _events():
        yield _sse_event(
            "start",
            {
                "type": "start",
                "session_id": session_id,
                "question": question,
                "response_mode": "stream",
            },
        )
        yield _sse_event(
            "error",
            {
                "type": "error",
                "status": "error",
                "message": message,
                "session_id": session_id,
            },
        )
        yield _sse_event(
            "done",
            {
                "type": "done",
                "status": "error",
                "message": message,
                "result": normalize_user_visible_text_fields(result_payload),
            },
        )

    return StreamingResponse(
        _events(),
        status_code=status_code,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _new_stream_query_task(
    session_id: str,
    request: Any,
    long_term_retrieval_scheme: str,
    *,
    allow_inactive_session: bool = False,
    task_source: str = "api_stream",
) -> dict[str, Any]:
    task_id = f"{session_id}_{uuid4().hex[:8]}"
    now = utc_now_iso()
    return {
        "task_id": task_id,
        "task_type": "query",
        "task_source": task_source,
        "session_id": session_id,
        "question": request.question,
        "top_k": request.top_k,
        "retrieval_mode": request.retrieval_mode,
        "use_image_evidence": request.use_image_evidence,
        "max_image_frames": request.max_image_frames,
        "max_image_evidence": request.max_image_evidence if request.max_image_evidence is not None else 3,
        "text_top_k": request.text_top_k,
        "visual_top_k": request.visual_top_k,
        "final_evidence_k": request.final_evidence_k,
        "memory_mode": request.memory_mode,
        "use_interaction_cache": request.use_interaction_cache,
        "cache_mode": request.cache_mode,
        "use_current": request.use_current,
        "use_short_term": request.use_short_term,
        "use_long_term": request.use_long_term,
        "debug_router": request.debug_router,
        "long_term_retrieval_scheme": long_term_retrieval_scheme,
        "retrieval_scheme": long_term_retrieval_scheme,
        "client_source": request.client_source,
        "input_method": request.input_method,
        "allow_inactive_session": bool(allow_inactive_session),
        "response_mode": "stream",
        "priority": 0,
        "query_priority_reason": "stream",
        "status": "in_progress",
        "created_at": now,
        "claimed_at": now,
        "updated_at": now,
    }


def _new_direct_query_task(
    session_id: str,
    request: Any,
    long_term_retrieval_scheme: str,
    *,
    allow_inactive_session: bool = False,
    task_source: str = "api_sync",
    response_mode: str = "legacy",
) -> dict[str, Any]:
    task = _new_stream_query_task(
        session_id,
        request,
        long_term_retrieval_scheme,
        allow_inactive_session=allow_inactive_session,
        task_source=task_source,
    )
    task["response_mode"] = response_mode
    task["query_priority_reason"] = response_mode
    return task


def _finish_stream_query_task(
    task: dict[str, Any],
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> Path | None:
    if status not in {"done", "failed"}:
        raise ValueError(f"Unsupported stream query task status: {status}")
    try:
        dirs = ensure_queue_dirs(PROJECT_ROOT)
        output_dir = dirs["query_done"] if status == "done" else dirs["query_failed"]
        task = dict(task)
        task["status"] = status
        task["error"] = error
        if result is not None:
            task["result"] = result
        task["updated_at"] = utc_now_iso()
        output_path = output_dir / f"{task['task_id']}.json"
        write_json_atomic(output_path, task)
        return output_path
    except Exception as exc:
        print(f"[stream_query_task] finish failed task_id={task.get('task_id')}: {exc}", flush=True)
        return None


def _finish_stream_query_task_failure(
    task: dict[str, Any],
    message: str,
    *,
    result: dict[str, Any] | None = None,
) -> Path | None:
    response_mode = str(task.get("response_mode") or "stream")
    error_result = {
        "status": "failed",
        "response_mode": response_mode,
        "session_id": task.get("session_id"),
        "question": task.get("question"),
        "message": message,
        "answer": "",
    }
    if result:
        error_result.update(result)
        error_result.setdefault("response_mode", response_mode)
        error_result.setdefault("session_id", task.get("session_id"))
        error_result.setdefault("question", task.get("question"))
        error_result.setdefault("message", message)
        error_result.setdefault("answer", "")
    return _finish_stream_query_task(task, "failed", result=error_result, error=message)


def _record_failed_stream_query_task(
    session_id: str,
    request: Any,
    message: str,
    *,
    result: dict[str, Any] | None = None,
    allow_inactive_session: bool = False,
    task_source: str = "api_stream",
) -> str:
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(
        request.long_term_retrieval_scheme or request.retrieval_scheme
    )
    stream_task = _new_stream_query_task(
        session_id,
        request,
        long_term_retrieval_scheme,
        allow_inactive_session=allow_inactive_session,
        task_source=task_source,
    )
    _finish_stream_query_task_failure(stream_task, message, result=result)
    return str(stream_task["task_id"])


async def _ask_streaming_response(
    session_id: str,
    request: "AskRequest",
    *,
    api_base_url: str = "",
    allow_inactive_session: bool = False,
    task_source: str = "api_stream",
) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    terminal_sent = threading.Event()
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(
        request.long_term_retrieval_scheme or request.retrieval_scheme
    )
    stream_task = _new_stream_query_task(
        session_id,
        request,
        long_term_retrieval_scheme,
        allow_inactive_session=allow_inactive_session,
        task_source=task_source,
    )

    def _put(event: str, data: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    def _stream_handler(event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "message")
        _put(event_type, event)

    def _terminal_done(result: dict[str, Any] | None = None, *, status: str = "ok", message: str | None = None) -> None:
        if terminal_sent.is_set():
            return
        terminal_sent.set()
        enhanced_result = _augment_evidence_frames_for_response(
            {"result": result or {}},
            session_id,
            api_base_url=api_base_url,
        ).get("result", {})
        payload = {
            "type": "done",
            "status": status,
            "result": normalize_user_visible_text_fields(enhanced_result),
        }
        if message:
            payload["message"] = message
        _put("done", payload)

    def _worker() -> None:
        try:
            from online_query import query_session
            from online_query.stream_query_context import load_stream_query_context

            _put(
                "start",
                {
                    "type": "start",
                    "task_id": stream_task["task_id"],
                    "session_id": session_id,
                    "question": request.question,
                    "response_mode": "stream",
                    "long_term_retrieval_scheme": long_term_retrieval_scheme,
                },
            )
            result = query_session(
                session_id=session_id,
                question=request.question,
                sessions_root=ONLINE_SESSIONS_DIR,
                top_k=request.top_k,
                retrieval_mode=request.retrieval_mode,
                use_image_evidence=request.use_image_evidence,
                max_image_frames=request.max_image_frames,
                max_image_evidence=request.max_image_evidence,
                text_top_k=request.text_top_k,
                visual_top_k=request.visual_top_k,
                final_evidence_k=request.final_evidence_k,
                memory_mode=request.memory_mode,
                use_interaction_cache=request.use_interaction_cache,
                cache_mode=request.cache_mode,
                use_current=request.use_current,
                use_short_term=request.use_short_term,
                use_long_term=request.use_long_term,
                debug_router=request.debug_router,
                long_term_retrieval_scheme=long_term_retrieval_scheme,
                stream_handler=_stream_handler,
            )
            try:
                stream_context = load_stream_query_context(
                    session_id,
                    sessions_root=ONLINE_SESSIONS_DIR,
                    project_root=PROJECT_ROOT,
                    question=request.question,
                )
                if stream_context and not result.get("stream_context"):
                    result["stream_context"] = stream_context
            except Exception:
                pass
            result["status"] = "ok"
            result["response_mode"] = "stream"
            _finish_stream_query_task(stream_task, "done", result=result)
            _append_qa_history_safe(
                session_id,
                question=request.question,
                answer=str(result.get("answer") or result.get("answer_text") or ""),
                client_source=request.client_source,
                input_method=request.input_method,
                status="done",
                task_id=stream_task["task_id"],
                response_mode="stream",
                metadata={"long_term_retrieval_scheme": long_term_retrieval_scheme},
            )
            _terminal_done(result, status="ok")
        except Exception as exc:
            message = str(exc)
            _finish_stream_query_task_failure(stream_task, message)
            _append_qa_history_safe(
                session_id,
                question=request.question,
                client_source=request.client_source,
                input_method=request.input_method,
                status="failed",
                error=message,
                task_id=stream_task["task_id"],
                response_mode="stream",
                metadata={"long_term_retrieval_scheme": long_term_retrieval_scheme},
            )
            _put("error", {"type": "error", "status": "error", "message": message, "session_id": session_id})
            _terminal_done(
                {
                    "status": "error",
                    "response_mode": "stream",
                    "session_id": session_id,
                    "question": request.question,
                    "message": message,
                    "answer": "",
                },
                status="error",
                message=message,
            )
        finally:
            _put("__end__", None)

    threading.Thread(target=_worker, name=f"ask-stream-{session_id}", daemon=True).start()

    async def _events():
        while True:
            try:
                event, data = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield _sse_event(
                    "ping",
                    {
                        "type": "ping",
                        "session_id": session_id,
                        "ts": utc_now_iso(),
                    },
                )
                continue
            if event == "__end__":
                break
            yield _sse_event(event, data)

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: Optional[str] = None
    response_mode: str = "legacy"
    retrieval_mode: str = "auto"
    use_image_evidence: Any = "auto"
    max_image_frames: int = 4
    max_image_evidence: Optional[int] = 3
    text_top_k: Optional[int] = None
    visual_top_k: Optional[int] = None
    final_evidence_k: Optional[int] = None
    memory_mode: str = "auto"
    use_current: Optional[bool] = None
    use_short_term: Optional[bool] = None
    use_long_term: Optional[bool] = None
    use_interaction_cache: bool = True
    cache_mode: str = "auto"
    debug_router: bool = False
    long_term_retrieval_scheme: Optional[str] = None
    retrieval_scheme: Optional[str] = None
    client_source: str = "unknown"
    input_method: str = "unknown"


class QaHistoryAppendRequest(BaseModel):
    question: str
    answer: str = ""
    client_source: str = "unknown"
    input_method: str = "unknown"
    status: str = "done"
    error: str = ""
    metadata: Optional[dict[str, Any]] = None


class ClearShortTermRequest(BaseModel):
    clear_archive: bool = False


class RefineShortTermRequest(BaseModel):
    backend: str = "mock"
    limit_events: int = 5
    event_id: Optional[str] = None
    force_refine: bool = False


class ConsolidateShortTermRequest(BaseModel):
    backend: str = "openai"
    update_em2mem: bool = True
    force: bool = False
    limit_windows: Optional[int] = None


class BuildCurrentFromStreamRequest(BaseModel):
    force: bool = False
    limit_chunks: Optional[int] = None


class AppendMemoryIncrementalRequest(BaseModel):
    episode_ids: Optional[list[str]] = None
    append_ready_episodes: bool = True
    force: bool = False
    skip_graph_semantic: bool = False


class StreamStartRequest(BaseModel):
    session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    run_id: Optional[str] = None
    create_parent_session: bool = False
    input_mode: Optional[str] = None
    chunk_duration: float = 5.0
    metadata: Optional[dict[str, Any]] = None
    owner_id: Optional[str] = None
    device_id: Optional[str] = None
    device_type: Optional[str] = None


class StreamEndRequest(BaseModel):
    final_chunk_index: Optional[int] = None
    close_open_event: bool = True
    force_accept: bool = False


class StreamRetryChunkRequest(BaseModel):
    chunk_index: int
    force: bool = False


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _valid_session_id(session_id: str) -> bool:
    return bool(session_id) and all(ch.isalnum() or ch in {"-", "_"} for ch in session_id)


def _api_base_url(request: FastAPIRequest | None = None) -> str:
    if request is None:
        return ""
    try:
        return str(request.base_url).rstrip("/")
    except Exception:
        return ""


def _evidence_frame_path(path: Any) -> str:
    return str(path or "").replace("\\", "/").lstrip("/")


def _evidence_frame_uses_long_term_session(path: Any) -> bool:
    return _evidence_frame_path(path).startswith("stream/day_assets/")


def _session_image_file_available(session_id: str, rel_path: str) -> bool:
    if not _valid_session_id(session_id):
        return False
    path = Path(rel_path)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        return False
    if path.suffix.lower() not in ALLOWED_SESSION_FILE_SUFFIXES:
        return False
    try:
        session_root = (ONLINE_SESSIONS_DIR / session_id).resolve()
        target_path = (ONLINE_SESSIONS_DIR / session_id / path).resolve()
        target_path.relative_to(session_root)
    except Exception:
        return False
    return target_path.exists() and target_path.is_file()


def _evidence_frame_owner_session_id(frame: dict[str, Any], requested_session_id: str, stream_context: dict[str, Any]) -> str:
    existing_owner = str(frame.get("owner_session_id") or "").strip()
    if existing_owner:
        return existing_owner
    path = frame.get("path") or frame.get("image_path")
    if _evidence_frame_uses_long_term_session(path):
        for key in ("long_term_session_id", "parent_session_id"):
            owner = str(stream_context.get(key) or "").strip()
            if owner:
                return owner
    return requested_session_id


def _augment_evidence_frames_for_response(
    payload: dict[str, Any],
    requested_session_id: str,
    *,
    api_base_url: str = "",
) -> dict[str, Any]:
    """Add file ownership metadata to evidence frames without changing legacy paths."""
    output = dict(payload or {})
    result = output.get("result") if isinstance(output.get("result"), dict) else None
    stream_context = {}
    if result and isinstance(result.get("stream_context"), dict):
        stream_context = result.get("stream_context") or {}
    elif isinstance(output.get("stream_context"), dict):
        stream_context = output.get("stream_context") or {}

    def augment_frames(frames: Any) -> list[dict[str, Any]]:
        augmented: list[dict[str, Any]] = []
        for frame in list(frames or []):
            if not isinstance(frame, dict):
                continue
            item = dict(frame)
            rel_path = _evidence_frame_path(item.get("path") or item.get("image_path"))
            if rel_path:
                owner_session_id = _evidence_frame_owner_session_id(item, requested_session_id, stream_context)
                item.setdefault("owner_session_id", owner_session_id)
                relative_file_url = f"/session/{owner_session_id}/file?path={quote(rel_path, safe='')}"
                display_file_url = f"{api_base_url}{relative_file_url}" if api_base_url else relative_file_url
                item["relative_file_url"] = relative_file_url
                item["file_url"] = display_file_url
                item.setdefault("image_url", display_file_url)
                item.setdefault("url", display_file_url)
                item.setdefault("src", display_file_url)
                item.setdefault("thumbnail_url", display_file_url)
                item["file_available"] = _session_image_file_available(owner_session_id, rel_path)
            augmented.append(item)
        return augmented

    if result is not None:
        result = dict(result)
        if "evidence_frames" in result:
            result["evidence_frames"] = augment_frames(result.get("evidence_frames"))
        output["result"] = result
        if "evidence_frames" in output:
            output["evidence_frames"] = augment_frames(output.get("evidence_frames"))
        elif "evidence_frames" in result:
            output["evidence_frames"] = list(result.get("evidence_frames") or [])
    elif "evidence_frames" in output:
        output["evidence_frames"] = augment_frames(output.get("evidence_frames"))
    return output


def create_online_session(
    *,
    source: str,
    original_filename: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    sid = session_id or uuid4().hex[:12]
    if not _valid_session_id(sid):
        raise ValueError("invalid session_id")
    session_dir = ONLINE_SESSIONS_DIR / sid
    session_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "session_id": sid,
        "source": source,
        "original_filename": original_filename,
        "saved_video_path": str(session_dir / "input.mp4"),
        "size_bytes": 0,
        "upload_time": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    write_json_atomic(session_dir / "metadata.json", payload)
    write_status(
        session_dir=session_dir,
        session_id=sid,
        status="created",
        stage=f"{source}_created",
        progress=0,
        error=None,
    )
    return {
        "session_id": sid,
        "session_dir": session_dir,
        "metadata": payload,
        "status": "created",
        "stage": f"{source}_created",
    }


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _pipeline_mode() -> str:
    mode = os.getenv("EM2MEM_PIPELINE_MODE", "mst").strip().lower()
    return mode if mode in {"mst", "legacy", "hybrid"} else "mst"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _append_timeline_event_safe(session_dir: Path, event_type: str, **kwargs: Any) -> None:
    try:
        append_timeline_event(session_dir, event_type, **kwargs)
    except Exception as exc:
        session_id = session_dir.name if isinstance(session_dir, Path) else str(session_dir)
        print(f"[timeline] append failed session_id={session_id} event_type={event_type}: {exc}", flush=True)


def _append_qa_history_safe(
    session_id: str,
    *,
    question: str,
    answer: str = "",
    client_source: str = "unknown",
    input_method: str = "unknown",
    status: str = "done",
    error: str = "",
    task_id: str | None = None,
    response_mode: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        if not _valid_session_id(session_id):
            print(f"[qa_history] skip invalid session_id={session_id}", flush=True)
            return None
        session_dir = ONLINE_SESSIONS_DIR / session_id
        if not session_dir.exists():
            print(f"[qa_history] skip missing session_id={session_id}", flush=True)
            return None
        return append_qa_history(
            session_dir,
            session_id=session_id,
            question=question,
            answer=answer,
            client_source=client_source,
            input_method=input_method,
            status=status,
            error=error,
            task_id=task_id,
            response_mode=response_mode,
            metadata=metadata,
        )
    except Exception as exc:
        print(f"[qa_history] append failed session_id={session_id}: {exc}", flush=True)
        return None


def _admin_token_ok(x_em2mem_admin_token: str | None, authorization: str | None) -> bool:
    expected = os.getenv("EM2MEM_RUNTIME_ADMIN_TOKEN") or os.getenv("EM2MEM_API_ADMIN_TOKEN")
    if not expected:
        return False
    candidates = [x_em2mem_admin_token or ""]
    if authorization:
        candidates.append(authorization.removeprefix("Bearer ").strip())
    return any(candidate == expected for candidate in candidates)


def _inactive_session_response(session_id: str) -> JSONResponse | None:
    if not _env_bool("EM2MEM_SINGLE_ACTIVE_SESSION", False):
        return None
    try:
        from online_pipeline.active_session import read_active_session_id

        active_session_id = read_active_session_id(PROJECT_ROOT)
    except Exception:
        active_session_id = None
    if active_session_id and session_id != active_session_id:
        return JSONResponse(
            status_code=409,
            content={
                "status": "inactive_session",
                "message": "this session is no longer active; start a new stream or use the active session",
                "session_id": session_id,
                "active_session_id": active_session_id,
            },
        )
    return None


def _count_json(path: Path) -> int:
    return len(list(path.glob("*.json"))) if path.exists() else 0


def _record_missing_stream_chunk(session_dir: Path, chunk_index: int) -> None:
    try:
        from online_short_term.stream_chunk_manager import StreamChunkManager, probe_duration

        manager = StreamChunkManager(session_dir)
        state = manager.load_stream_state(default={})
        if not isinstance(state, dict):
            return
        missing = set()
        for item in state.get("missing_chunks", []) or []:
            try:
                missing.add(int(item))
            except Exception:
                continue
        missing.add(int(chunk_index))
        state["missing_chunks"] = sorted(missing)
        manager.save_stream_state(state)
    except Exception:
        return


def _find_active_stream_task(
    *,
    session_id: str,
    task_type: str,
    match_fields: dict[str, Any] | None = None,
) -> tuple[str, Path, dict[str, Any]] | None:
    dirs = ensure_queue_dirs(PROJECT_ROOT)
    if task_type == "stream_asr":
        keys = ("stream_asr_queued", "stream_asr_in_progress", "stream_asr_done")
    else:
        keys = ("stream_chunk_queued", "stream_chunk_in_progress", "stream_chunk_done")
    for key in keys:
        for path in sorted(dirs[key].glob(f"{session_id}_*.json")):
            payload = read_json(path, default={})
            if not isinstance(payload, dict):
                continue
            if str(payload.get("session_id") or "") != str(session_id):
                continue
            if str(payload.get("task_type") or "") != task_type:
                continue
            matched = True
            for field, expected in (match_fields or {}).items():
                value = payload.get(field)
                try:
                    matched = int(value) == int(expected)
                except Exception:
                    matched = value == expected
                if not matched:
                    break
            if matched:
                return key, path, payload
    return None


def _find_query_task(task_id: str) -> tuple[str, Path, dict] | None:
    dirs = get_queue_dirs(PROJECT_ROOT)
    for state_key in ("query_queued", "query_in_progress", "query_done", "query_failed"):
        path = dirs[state_key] / f"{task_id}.json"
        if path.exists():
            return state_key, path, read_json(path, default={})
    aborted_root = PROJECT_ROOT / "online_tasks_aborted"
    if aborted_root.exists():
        for path in sorted(aborted_root.glob(f"*/query*/{task_id}.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            payload = read_json(path, default={})
            return "query_aborted", path, payload if isinstance(payload, dict) else {}
    return None


@app.on_event("startup")
def startup_prepare_task_queue() -> None:
    ensure_queue_dirs(PROJECT_ROOT)


@app.get("/ping")
async def ping() -> dict[str, str]:
    return {"status": "ok", "message": "Em2Mem API is running"}


@app.get("/runtime")
async def runtime_status() -> dict[str, object]:
    queue_dirs = get_queue_dirs(PROJECT_ROOT)
    return {
        "status": "ok",
        "mode": "upload_api_only",
        "gpu_preprocess": "external_worker",
        "queued_tasks": _count_json(queue_dirs["queued"]),
        "in_progress_tasks": _count_json(queue_dirs["in_progress"]),
        "queued_evidence_tasks": _count_json(queue_dirs["evidence_queued"]),
        "in_progress_evidence_tasks": _count_json(queue_dirs["evidence_in_progress"]),
        "queued_memory_tasks": _count_json(queue_dirs["memory_queued"]),
        "in_progress_memory_tasks": _count_json(queue_dirs["memory_in_progress"]),
        "queued_query_tasks": _count_json(queue_dirs["query_queued"]),
        "in_progress_query_tasks": _count_json(queue_dirs["query_in_progress"]),
        "queued_visual_tasks": _count_json(queue_dirs["visual_queued"]),
        "in_progress_visual_tasks": _count_json(queue_dirs["visual_in_progress"]),
        "queued_mst_refine_tasks": _count_json(queue_dirs["mst_refine_queued"]),
        "in_progress_mst_refine_tasks": _count_json(queue_dirs["mst_refine_in_progress"]),
        "queued_mst_consolidation_tasks": _count_json(queue_dirs["mst_consolidation_queued"]),
        "in_progress_mst_consolidation_tasks": _count_json(queue_dirs["mst_consolidation_in_progress"]),
    }


@app.get("/query_runtime")
async def query_runtime() -> dict[str, object]:
    queue_dirs = get_queue_dirs(PROJECT_ROOT)
    runtime_path = PROJECT_ROOT / "online_tasks" / "query_runtime.json"
    worker_runtime = read_json(runtime_path, default=None)
    return {
        "status": "ok",
        "default_ask_mode": _env_str("EM2MEM_ASK_DEFAULT_MODE", "async"),
        "router": {
            "memory_router_enabled": True,
            "default_memory_mode": _env_str("EM2MEM_DEFAULT_MEMORY_MODE", "auto"),
        },
        "loaded_sessions": (
            ((worker_runtime or {}).get("loaded_sessions"))
            or (((worker_runtime or {}).get("cache") or {}).get("loaded_sessions"))
            or []
        ),
        "worker_runtime": worker_runtime,
        "queue_counts": {
            "query_queued": _count_json(queue_dirs["query_queued"]),
            "query_in_progress": _count_json(queue_dirs["query_in_progress"]),
            "query_done": _count_json(queue_dirs["query_done"]),
            "query_failed": _count_json(queue_dirs["query_failed"]),
        },
    }


@app.get("/pipeline_runtime")
async def pipeline_runtime(session_id: Optional[str] = None) -> dict[str, object]:
    from online_pipeline.runtime_state import collect_pipeline_runtime

    return collect_pipeline_runtime(PROJECT_ROOT, ONLINE_SESSIONS_DIR, session_id=session_id)


@app.get("/session/{session_id}/pipeline_state")
async def session_pipeline_state(session_id: str) -> JSONResponse:
    from online_pipeline.runtime_state import refresh_session_pipeline_state

    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    state = refresh_session_pipeline_state(session_dir)
    return JSONResponse(status_code=200, content=state)


@app.get("/session/{session_id}/memory_versions")
async def session_memory_versions(session_id: str) -> JSONResponse:
    from online_memory_incremental import IncrementalMemoryAppender

    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    appender = IncrementalMemoryAppender(session_id=session_id, sessions_root=ONLINE_SESSIONS_DIR, project_root=PROJECT_ROOT)
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "session_id": session_id,
            "component_versions": appender.load_versions(),
            "append_state": read_json(session_dir / "em2mem" / "incremental" / "append_state.json", default={}),
            "dirty_windows": read_json(session_dir / "em2mem" / "incremental" / "dirty_windows.json", default={}),
            "graph_state": read_json(session_dir / "em2mem" / "incremental" / "graph" / "graph_state.json", default={}),
            "semantic_state": read_json(session_dir / "em2mem" / "incremental" / "semantic" / "semantic_state.json", default={}),
        },
    )


@app.post("/session/{session_id}/append_memory_incremental")
async def append_memory_incremental(session_id: str, request: AppendMemoryIncrementalRequest) -> JSONResponse:
    from online_memory_incremental import IncrementalMemoryAppender
    from online_pipeline.runtime_state import refresh_session_pipeline_state

    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    if not request.append_ready_episodes and not request.episode_ids:
        return JSONResponse(status_code=400, content={"status": "error", "message": "append_ready_episodes=false requires episode_ids"})
    try:
        result = IncrementalMemoryAppender(
            session_id=session_id,
            sessions_root=ONLINE_SESSIONS_DIR,
            project_root=PROJECT_ROOT,
        ).append_ready_episodes(
            episode_ids=request.episode_ids,
            force=request.force,
            skip_graph_semantic=request.skip_graph_semantic,
        )
        refresh_session_pipeline_state(session_dir)
        return JSONResponse(status_code=200, content=result.to_dict())
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/session/{session_id}/refresh_pipeline_state")
async def refresh_pipeline_state_api(session_id: str) -> JSONResponse:
    from online_pipeline.runtime_state import refresh_session_pipeline_state

    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    state = refresh_session_pipeline_state(session_dir)
    return JSONResponse(status_code=200, content={"status": "ok", "session_id": session_id, "pipeline_state": state})


@app.get("/query_task/{task_id}")
async def query_task_status(task_id: str, http_request: FastAPIRequest = None) -> JSONResponse:
    found = _find_query_task(task_id)
    if found is None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "not_found",
                "task_id": task_id,
                "message": "query task not found or expired",
            },
        )
    state_key, path, payload = found
    payload = dict(payload or {})
    if state_key == "query_aborted":
        payload.setdefault("status", "cancelled" if payload.get("error_type") == "cancelled" else "aborted")
        payload.setdefault("message", "query task was cancelled or aborted")
    elif payload.get("status") == "cancelled":
        payload.setdefault("message", "query task was cancelled because a new active stream session started")
    elif state_key == "query_done" and "result" not in payload:
        payload["status"] = "not_found"
        payload["message"] = "query task result is missing or expired"
    elif state_key == "query_done" and isinstance(payload.get("result"), dict):
        result = payload["result"]
        for key in ("answer", "answer_text", "timestamps", "evidence_frames", "latency", "stream_context"):
            if key in result and key not in payload:
                payload[key] = result[key]
        payload.setdefault("result_status", result.get("status"))
        payload.setdefault("response_mode", result.get("response_mode", "legacy"))
    payload["queue_state"] = state_key
    payload["task_path"] = str(path)
    requested_session_id = str(payload.get("session_id") or ((payload.get("result") or {}).get("session_id") if isinstance(payload.get("result"), dict) else "") or "")
    if requested_session_id:
        payload = _augment_evidence_frames_for_response(
            payload,
            requested_session_id,
            api_base_url=_api_base_url(http_request),
        )
    return JSONResponse(status_code=200, content=normalize_user_visible_text_fields(payload))


@app.get("/task/query/{task_id}")
async def query_task_status_alias(task_id: str, http_request: FastAPIRequest = None) -> JSONResponse:
    return await query_task_status(task_id, http_request)


@app.get("/session/{session_id}/status")
async def session_status(session_id: str) -> JSONResponse:
    status_path = ONLINE_SESSIONS_DIR / session_id / "status.json"
    if not status_path.exists():
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"status.json not found for session {session_id}"},
        )
    return JSONResponse(status_code=200, content=read_json(status_path, default={}))


@app.get("/session/{session_id}/file", response_model=None)
async def session_file(session_id: str, path: str):
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    rel_path = Path(path)
    if rel_path.is_absolute() or any(part == ".." for part in rel_path.parts):
        return JSONResponse(status_code=400, content={"status": "error", "message": "path must be a safe relative path"})
    if rel_path.suffix.lower() not in ALLOWED_SESSION_FILE_SUFFIXES:
        return JSONResponse(status_code=403, content={"status": "error", "message": "file type is not allowed"})
    try:
        session_root = session_dir.resolve()
        target_path = (session_dir / rel_path).resolve()
        target_path.relative_to(session_root)
    except Exception:
        return JSONResponse(status_code=403, content={"status": "error", "message": "path escapes session directory"})
    if not target_path.exists() or not target_path.is_file():
        return JSONResponse(status_code=404, content={"status": "error", "message": "file not found"})
    media_type = mimetypes.guess_type(target_path.name)[0] or "application/octet-stream"
    return FileResponse(target_path, media_type=media_type)


@app.get("/session/{session_id}/interaction_cache")
async def interaction_cache_status(session_id: str) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_query.interaction_cache import get_interaction_cache_summary

        return JSONResponse(status_code=200, content=get_interaction_cache_summary(session_dir, session_id))
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/session/{session_id}/clear_interaction_cache")
async def clear_session_interaction_cache(session_id: str) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_query.interaction_cache import clear_interaction_cache

        summary = clear_interaction_cache(session_dir, session_id)
        return JSONResponse(status_code=200, content={"status": "ok", "cache": summary})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.get("/session/{session_id}/current")
async def current_status(session_id: str) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_current.mcur_store import MCurStore

        return JSONResponse(status_code=200, content={"status": "ok", **MCurStore(session_dir).summary(limit=5)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/session/{session_id}/clear_current")
async def clear_current(session_id: str) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_current.mcur_store import MCurStore

        state = MCurStore(session_dir).clear()
        return JSONResponse(status_code=200, content={"status": "ok", "session_id": session_id, "state": state})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/session/{session_id}/build_current_from_stream")
async def build_current_from_stream(session_id: str, request: Optional[BuildCurrentFromStreamRequest] = None) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_current.mcur_updater import MCurUpdater

        req = request or BuildCurrentFromStreamRequest()
        state = MCurUpdater(session_dir).update_from_existing_stream(
            force=bool(req.force),
            limit_chunks=req.limit_chunks,
        )
        return JSONResponse(status_code=200, content={"status": "ok", "session_id": session_id, "state": state})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.get("/session/{session_id}/short_term")
async def short_term_status(session_id: str) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_short_term.mst_store import MSTStore
        from online_short_term.refine_status import write_refine_status

        store = MSTStore(session_dir)
        state = store.get_state()
        archive_state = store.get_archive_state()
        _, refine_state_path = write_refine_status(store)
        refine_state = read_json(refine_state_path, default={})
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "session_id": session_id,
                "short_term_ready": store.is_ready(),
                "mst_version": state.get("mst_version"),
                "archive_version": state.get("archive_version"),
                "event_count": state.get("event_count"),
                "active_event_count": state.get("active_event_count"),
                "archive_event_count": state.get("archive_event_count"),
                "refined_event_count": archive_state.get("refined_event_count"),
                "pending_event_count": refine_state.get("pending_event_count"),
                "ready_30s_window_count": refine_state.get("ready_30s_window_count"),
                "last_processed_time": state.get("last_processed_time"),
                "recent_events": store.get_recent_events(limit=5),
                "state": state,
                "archive_state": archive_state,
                "refine_state": refine_state,
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/session/{session_id}/clear_short_term")
async def clear_short_term(session_id: str, request: Optional[ClearShortTermRequest] = None) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_short_term.mst_store import MSTStore

        clear_archive = bool(request.clear_archive) if request is not None else False
        store = MSTStore(session_dir)
        state = store.clear(clear_archive=clear_archive)
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "session_id": session_id,
                "clear_archive": clear_archive,
                "state": state,
                "archive_state": store.get_archive_state(),
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/session/{session_id}/refine_short_term")
async def refine_short_term(session_id: str, request: RefineShortTermRequest) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    backend = (request.backend or "mock").strip().lower()
    if backend not in {"mock", "openai"}:
        return JSONResponse(status_code=400, content={"status": "error", "message": "backend must be mock or openai"})
    try:
        from refine_mst_micro_events import refine_session

        result = refine_session(
            session_id=session_id,
            sessions_root=ONLINE_SESSIONS_DIR,
            backend=backend,
            limit_events=max(0, int(request.limit_events)),
            event_id=request.event_id,
            force_refine=bool(request.force_refine),
            verbose=False,
        )
        return JSONResponse(status_code=200, content={"status": "ok", **result})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.get("/session/{session_id}/short_term/refine_status")
async def short_term_refine_status(session_id: str) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_short_term.mst_store import MSTStore
        from online_short_term.refine_status import write_refine_status

        store = MSTStore(session_dir)
        windows_path, refine_state_path = write_refine_status(store)
        refine_state = read_json(refine_state_path, default={})
        windows = read_json(windows_path, default=[])
        ready_windows = [window for window in windows if isinstance(window, dict) and window.get("ready_for_30s_episodic")]
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "session_id": session_id,
                "mst_state": store.get_state(),
                "archive_state": store.get_archive_state(),
                "refine_state": refine_state,
                "ready_30s_window_count": len(ready_windows),
                "not_ready_30s_window_count": max(0, len(windows) - len(ready_windows)) if isinstance(windows, list) else 0,
                "ready_windows": ready_windows[:20],
                "refined_ready_windows_path": str(windows_path),
                "refine_state_path": str(refine_state_path),
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/session/{session_id}/consolidate_short_term")
async def consolidate_short_term(session_id: str, request: ConsolidateShortTermRequest) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    backend = (request.backend or "openai").strip().lower()
    if backend not in {"openai", "rule", "mock"}:
        return JSONResponse(status_code=400, content={"status": "error", "message": "backend must be openai, rule, or mock"})
    try:
        from online_mst_to_em2mem import consolidate_short_term_to_em2mem

        result = consolidate_short_term_to_em2mem(
            session_id=session_id,
            sessions_root=ONLINE_SESSIONS_DIR,
            backend=backend,
            update_em2mem=bool(request.update_em2mem),
            force=bool(request.force),
            limit_windows=request.limit_windows,
            verbose=False,
        )
        return JSONResponse(status_code=200, content={"status": "ok", **result})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/session/{session_id}/preprocess")
async def start_preprocess(session_id: str) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    input_video = session_dir / "input.mp4"
    if not input_video.exists():
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"input.mp4 not found for session {session_id}"},
        )

    task_path = enqueue_preprocess_task(
        project_root=PROJECT_ROOT,
        session_id=session_id,
        force=_env_bool("EM2MEM_FORCE_PREPROCESS", False),
    )
    write_status(
        session_dir=session_dir,
        session_id=session_id,
        status="processing",
        stage="preprocess_queued",
        progress=5,
        error=None,
    )
    return JSONResponse(
        status_code=202,
        content={"status": "queued", "session_id": session_id, "task_path": str(task_path)},
    )


@app.post("/session/{session_id}/build_evidence")
async def build_evidence(session_id: str) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    session_30sec_path = session_dir / "preprocess" / "session_30sec.json"
    if not session_30sec_path.exists():
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"preprocess/session_30sec.json not found for session {session_id}"},
        )

    task_path = enqueue_evidence_task(
        project_root=PROJECT_ROOT,
        session_id=session_id,
        force=_env_bool("EM2MEM_FORCE_EVIDENCE", False),
        backend=os.getenv("EM2MEM_EVIDENCE_CAPTION_BACKEND"),
    )
    write_status(
        session_dir=session_dir,
        session_id=session_id,
        status="processing",
        stage="evidence_queued",
        progress=60,
        error=None,
    )
    return JSONResponse(
        status_code=202,
        content={"status": "queued", "session_id": session_id, "task_path": str(task_path)},
    )


@app.post("/session/{session_id}/build_memory")
async def build_memory(session_id: str) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    mode = _pipeline_mode()
    mst_ready = (session_dir / "captions" / "mst_session_30sec_captioned.json").exists() and (session_dir / "evidence" / "mst_session_evidence.json").exists()
    legacy_ready = (session_dir / "captions" / "session_30sec_captioned.json").exists() and (session_dir / "evidence" / "session_evidence.json").exists()
    allow_legacy_fallback = _env_bool("EM2MEM_ALLOW_LEGACY_EVIDENCE_FALLBACK", False)
    if mode == "legacy":
        source = "legacy_evidence"
        required_ok = legacy_ready
        missing_msg = f"legacy captions/session_30sec_captioned.json or evidence/session_evidence.json not found for session {session_id}"
    elif mode == "hybrid":
        source = "auto"
        required_ok = mst_ready or legacy_ready
        missing_msg = f"neither MST nor legacy 30s evidence exists for session {session_id}"
    else:
        source = "auto"
        required_ok = mst_ready or (allow_legacy_fallback and legacy_ready)
        missing_msg = f"MST captions/mst_session_30sec_captioned.json or evidence/mst_session_evidence.json not found for session {session_id}; waiting for mst_consolidation"
    if not required_ok:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": missing_msg, "pipeline_mode": mode},
        )

    task_path = enqueue_memory_task(
        project_root=PROJECT_ROOT,
        session_id=session_id,
        force=_env_bool("EM2MEM_FORCE_MEMORY", False),
        skip_visual_embedding=_env_bool("EM2MEM_SKIP_VISUAL_EMBEDDING", True),
        skip_semantic=_env_bool("EM2MEM_SKIP_SEMANTIC_MEMORY", False),
        source=source,
        update_mode="incremental_append" if source in {"auto", "mst_episodic"} else None,
        append_ready_episodes=True if source in {"auto", "mst_episodic"} else None,
    )
    write_status(
        session_dir=session_dir,
        session_id=session_id,
        status="processing",
        stage="memory_queued",
        progress=90,
        error=None,
    )
    return JSONResponse(
        status_code=202,
        content={"status": "queued", "session_id": session_id, "task_path": str(task_path)},
    )


async def _prepare_audio_question_transcript(
    session_id: str,
    audio: Optional[UploadFile],
    *,
    duration_ms: Optional[int],
    sample_rate: Optional[int],
    channels: Optional[int],
    sample_width: Optional[int],
    encoding: Optional[str],
    audio_format: Optional[str],
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    if audio is None:
        return None, JSONResponse(status_code=400, content={"status": "error", "message": "No audio uploaded. Expected form field 'audio'."})
    if not _valid_session_id(session_id):
        return None, JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})

    inactive = _inactive_session_response(session_id)
    if inactive is not None:
        return None, inactive

    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return None, JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})

    from online_pipeline.frame_stream import frame_stream_input_mode
    from online_pipeline.rokid_ingest import RokidNormalizationError, normalize_rokid_audio_upload

    payload = await audio.read()
    if not payload:
        return None, JSONResponse(status_code=400, content={"status": "error", "message": "empty audio question"})
    if len(payload) > CHUNK_SIZE_BYTES:
        return None, JSONResponse(status_code=413, content={"status": "error", "message": "audio question is too large"})

    stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
    if not isinstance(stream_state, dict):
        stream_state = {}
    input_mode = frame_stream_input_mode(stream_state.get("input_mode") or "frame_audio_stream")
    try:
        payload, audio_format, _ = normalize_rokid_audio_upload(
            payload,
            format_hint=audio_format,
            filename=audio.filename,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            encoding=encoding,
        )
    except RokidNormalizationError as exc:
        return None, JSONResponse(status_code=400, content={"status": "error", "message": str(exc), "input_mode": input_mode})

    question_id = uuid4().hex
    question_dir = session_dir / "stream" / "audio_questions"
    question_dir.mkdir(parents=True, exist_ok=True)
    audio_path = question_dir / f"{question_id}.wav"
    transcript_json = question_dir / f"{question_id}.json"
    transcript_srt = question_dir / f"{question_id}.srt"
    audio_path.write_bytes(payload)

    segments = _run_voice_question_asr(
        audio_path=audio_path,
        output_srt=transcript_srt,
        output_json=transcript_json,
    )
    question = simplify_chinese_text(" ".join(str(item.get("text") or "").strip() for item in segments if str(item.get("text") or "").strip()).strip())
    if not question:
        return None, JSONResponse(status_code=422, content={"status": "no_speech", "message": "no speech recognized", "session_id": session_id})

    return {
        "status": "ok",
        "session_id": session_id,
        "question": question,
        "audio_question_id": question_id,
        "audio_question_duration_ms": duration_ms,
        "input_mode": input_mode,
        "audio_format": audio_format,
    }, None


def _json_response_payload(response: JSONResponse) -> dict[str, Any]:
    try:
        payload = json.loads(response.body.decode("utf-8")) if response.body else {}
        return payload if isinstance(payload, dict) else {"status": "error", "message": str(payload)}
    except Exception:
        return {"status": "error", "message": "request failed"}


@app.post("/stream/{session_id}/audio_question")
async def stream_audio_question(
    session_id: str,
    audio: Optional[UploadFile] = File(default=None),
    duration_ms: Optional[int] = Form(default=None),
    sample_rate: Optional[int] = Form(default=None),
    channels: Optional[int] = Form(default=None),
    sample_width: Optional[int] = Form(default=None),
    encoding: Optional[str] = Form(default=None),
    audio_format: Optional[str] = Form(default=None, alias="format"),
    mode: Optional[str] = Form(default="async"),
    retrieval_mode: str = Form(default="auto"),
    memory_mode: str = Form(default="auto"),
    top_k: int = Form(default=5),
    use_current: Optional[bool] = Form(default=True),
    use_image_evidence: Any = Form(default="auto"),
    max_image_evidence: Optional[int] = Form(default=6),
    use_interaction_cache: bool = Form(default=True),
    debug_router: bool = Form(default=True),
    long_term_retrieval_scheme: Optional[str] = Form(default=None),
    retrieval_scheme: Optional[str] = Form(default=None),
    client_source: str = Form(default="glasses"),
    input_method: str = Form(default="voice"),
) -> JSONResponse:
    try:
        prepared, error_response = await _prepare_audio_question_transcript(
            session_id=session_id,
            audio=audio,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            encoding=encoding,
            audio_format=audio_format,
        )
        if error_response is not None:
            return error_response
        assert prepared is not None
        question = str(prepared.get("question") or "")
        ask_response = await ask_session(
            session_id,
            AskRequest(
                question=question,
                top_k=top_k,
                mode=mode,
                retrieval_mode=retrieval_mode,
                use_image_evidence=use_image_evidence,
                max_image_evidence=max_image_evidence,
                memory_mode=memory_mode,
                use_current=use_current,
                use_interaction_cache=use_interaction_cache,
                debug_router=debug_router,
                long_term_retrieval_scheme=long_term_retrieval_scheme,
                retrieval_scheme=retrieval_scheme,
                client_source=client_source,
                input_method=input_method,
            ),
        )
        content = json.loads(ask_response.body.decode("utf-8")) if ask_response.body else {}
        if isinstance(content, dict):
            content.setdefault("status", "queued")
            content["question"] = question
            content["audio_question_id"] = prepared.get("audio_question_id")
            content["audio_question_duration_ms"] = duration_ms
            content = normalize_user_visible_text_fields(content)
        return JSONResponse(status_code=ask_response.status_code, content=content)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})
    finally:
        if audio is not None:
            await audio.close()


@app.post("/stream/{session_id}/audio_question/stream", response_model=None)
async def stream_audio_question_stream(
    session_id: str,
    audio: Optional[UploadFile] = File(default=None),
    duration_ms: Optional[int] = Form(default=None),
    sample_rate: Optional[int] = Form(default=None),
    channels: Optional[int] = Form(default=None),
    sample_width: Optional[int] = Form(default=None),
    encoding: Optional[str] = Form(default=None),
    audio_format: Optional[str] = Form(default=None, alias="format"),
    retrieval_mode: str = Form(default="auto"),
    memory_mode: str = Form(default="auto"),
    top_k: int = Form(default=5),
    use_current: Optional[bool] = Form(default=True),
    use_image_evidence: Any = Form(default="auto"),
    max_image_evidence: Optional[int] = Form(default=6),
    use_interaction_cache: bool = Form(default=True),
    debug_router: bool = Form(default=True),
    client_source: str = Form(default="glasses"),
    input_method: str = Form(default="voice"),
) -> StreamingResponse:
    async def _events():
        try:
            yield _sse_event(
                "transcribing",
                {
                    "type": "transcribing",
                    "status": "running",
                    "session_id": session_id,
                },
            )
            prepared, error_response = await _prepare_audio_question_transcript(
                session_id=session_id,
                audio=audio,
                duration_ms=duration_ms,
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
                encoding=encoding,
                audio_format=audio_format,
            )
            if error_response is not None:
                payload = _json_response_payload(error_response)
                message = str(payload.get("message") or payload.get("status") or "audio question failed")
                yield _sse_event(
                    "error",
                    {
                        "type": "error",
                        "status": "error",
                        "message": message,
                        "session_id": session_id,
                        "result": normalize_user_visible_text_fields(payload),
                    },
                )
                yield _sse_event(
                    "done",
                    {
                        "type": "done",
                        "status": "error",
                        "message": message,
                        "result": normalize_user_visible_text_fields(payload),
                    },
                )
                return

            assert prepared is not None
            question = str(prepared.get("question") or "")
            yield _sse_event(
                "transcript",
                {
                    "type": "transcript",
                    "status": "ok",
                    "session_id": session_id,
                    "question": question,
                    "audio_question_id": prepared.get("audio_question_id"),
                    "audio_question_duration_ms": duration_ms,
                },
            )
            ask_response = await ask_session(
                session_id,
                AskRequest(
                    question=question,
                    top_k=top_k,
                    response_mode="stream",
                    retrieval_mode=retrieval_mode,
                    use_image_evidence=use_image_evidence,
                    max_image_evidence=max_image_evidence,
                    memory_mode=memory_mode,
                    use_current=use_current,
                    use_interaction_cache=use_interaction_cache,
                    debug_router=debug_router,
                    client_source=client_source,
                    input_method=input_method,
                ),
            )
            if isinstance(ask_response, JSONResponse):
                payload = _json_response_payload(ask_response)
                message = str(payload.get("message") or payload.get("status") or "question failed")
                yield _sse_event("error", {"type": "error", "status": "error", "message": message, "session_id": session_id})
                yield _sse_event("done", {"type": "done", "status": "error", "message": message, "result": normalize_user_visible_text_fields(payload)})
                return

            async for chunk in ask_response.body_iterator:
                yield chunk
        except Exception as exc:
            message = str(exc)
            payload = {
                "status": "error",
                "response_mode": "stream",
                "session_id": session_id,
                "message": message,
                "answer": "",
            }
            yield _sse_event("error", {"type": "error", "status": "error", "message": message, "session_id": session_id})
            yield _sse_event("done", {"type": "done", "status": "error", "message": message, "result": payload})
        finally:
            if audio is not None:
                await audio.close()

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/ask/{session_id}", response_model=None)
async def ask_session(
    session_id: str,
    request: AskRequest,
    http_request: FastAPIRequest = None,
) -> JSONResponse | StreamingResponse:
    return await _handle_ask_session(
        session_id,
        request,
        http_request=http_request,
        allow_inactive_session=False,
        task_source="api",
    )


async def _handle_ask_session(
    session_id: str,
    request: AskRequest,
    *,
    http_request: FastAPIRequest = None,
    allow_inactive_session: bool = False,
    task_source: str = "api",
) -> JSONResponse | StreamingResponse:
    inactive = None if allow_inactive_session else _inactive_session_response(session_id)
    if inactive is not None:
        response_mode = str(getattr(request, "response_mode", "legacy") or "legacy").strip().lower()
        stream_task_id = None
        if response_mode in {"stream", "streaming", "progressive"}:
            stream_task_id = _record_failed_stream_query_task(session_id, request, "inactive_session")
        _append_qa_history_safe(
            session_id,
            question=request.question,
            client_source=request.client_source,
            input_method=request.input_method,
            status="failed",
            error="inactive_session",
            task_id=stream_task_id,
            response_mode=response_mode,
        )
        if response_mode in {"stream", "streaming", "progressive"}:
            return _stream_error_response(session_id, request.question, "inactive_session")
        return inactive
    try:
        from online_pipeline.rokid_day import resolve_query_session_context

        query_context = resolve_query_session_context(session_id, ONLINE_SESSIONS_DIR)
    except Exception:
        query_context = {
            "realtime_session_id": session_id,
            "short_term_session_id": session_id,
            "long_term_session_id": session_id,
            "interaction_cache_session_id": session_id,
            "is_rokid_day_child": False,
        }
    realtime_session_id = str(query_context.get("realtime_session_id") or session_id)
    short_term_session_id = str(query_context.get("short_term_session_id") or realtime_session_id)
    long_term_session_id = str(query_context.get("long_term_session_id") or session_id)
    session_dir = ONLINE_SESSIONS_DIR / realtime_session_id
    short_term_session_dir = ONLINE_SESSIONS_DIR / short_term_session_id
    long_term_session_dir = ONLINE_SESSIONS_DIR / long_term_session_id
    memory_config = long_term_session_dir / "em2mem" / "memory_config.json"
    short_term_ready = False
    current_ready = False
    current_text_ready = False
    partial_transcript_ready = False
    frame_open_event_ready = False
    memory_status_question = False
    if session_dir.exists():
        try:
            from online_short_term.mst_store import MSTStore

            short_term_ready = MSTStore(short_term_session_dir).is_ready()
        except Exception:
            short_term_ready = False
        try:
            from online_current.mcur_store import MCurStore

            current_store = MCurStore(session_dir)
            current_state = current_store.get_state()
            current_text_ready = bool(
                current_state.get("current_text_ready")
                or current_state.get("audio_current_ready")
                or int(current_state.get("transcript_segment_count", 0) or 0) > 0
            )
            current_ready = bool((current_state.get("mcur_ready") or current_text_ready) and not current_store.is_stale(current_state))
        except Exception:
            current_ready = False
            current_text_ready = False
        try:
            transcript_state = read_json(session_dir / "stream" / "transcript" / "partial_transcript_state.json", default={})
            partial_transcript_ready = bool(isinstance(transcript_state, dict) and int(transcript_state.get("segment_count", 0) or 0) > 0)
        except Exception:
            partial_transcript_ready = False
        try:
            frame_event_state = read_json(session_dir / "stream" / "frame_event_state.json", default={})
            frame_open_event_ready = bool(isinstance(frame_event_state, dict) and isinstance(frame_event_state.get("open_event"), dict) and frame_event_state.get("open_event"))
        except Exception:
            frame_open_event_ready = False
        try:
            from online_query.query_engine import _is_memory_status_question

            memory_status_question = _is_memory_status_question(request.question)
        except Exception:
            memory_status_question = False
    if not memory_config.exists() and not short_term_ready and not current_ready and not current_text_ready and not partial_transcript_ready and not frame_open_event_ready and not memory_status_question:
        status = read_json(session_dir / "status.json", default={}) if session_dir.exists() else {}
        content = {
            "status": "not_ready",
            "message": "memory is not ready and short-term/current memory is not ready",
            "stage": status.get("stage") if isinstance(status, dict) else None,
            "progress": status.get("progress") if isinstance(status, dict) else None,
            "long_term_ready": False,
            "short_term_ready": False,
            "current_ready": False,
            "current_text_ready": current_text_ready,
            "partial_transcript_ready": partial_transcript_ready,
            "frame_open_event_ready": False,
        }
        response_mode = str(request.response_mode or "legacy").strip().lower()
        stream_task_id = None
        if response_mode in {"stream", "streaming", "progressive"}:
            stream_task_id = _record_failed_stream_query_task(
                session_id,
                request,
                content["message"],
                result=content,
                allow_inactive_session=allow_inactive_session,
                task_source=f"{task_source}_stream" if task_source != "api" else "api_stream",
            )
        _append_qa_history_safe(
            session_id,
            question=request.question,
            client_source=request.client_source,
            input_method=request.input_method,
            status="failed",
            error=content["message"],
            task_id=stream_task_id,
            response_mode=response_mode,
            metadata={"stage": content.get("stage"), "progress": content.get("progress")},
        )
        if response_mode in {"stream", "streaming", "progressive"}:
            return _stream_error_response(session_id, request.question, content["message"], result=content)
        return JSONResponse(status_code=409, content=content)

    response_mode = str(request.response_mode or "legacy").strip().lower()
    if response_mode in {"stream", "streaming", "progressive"}:
        stream_task_source = f"{task_source}_stream" if task_source != "api" else "api_stream"
        return await _ask_streaming_response(
            session_id,
            request,
            api_base_url=_api_base_url(http_request),
            allow_inactive_session=allow_inactive_session,
            task_source=stream_task_source,
        )
    if response_mode not in {"legacy", "json", "polling"}:
        return JSONResponse(status_code=400, content={"status": "error", "message": "response_mode must be legacy or stream"})

    mode = (request.mode or _env_str("EM2MEM_ASK_DEFAULT_MODE", "async")).strip().lower()
    if mode not in {"sync", "async"}:
        return JSONResponse(status_code=400, content={"status": "error", "message": "mode must be sync or async"})
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(
        request.long_term_retrieval_scheme or request.retrieval_scheme
    )

    if mode == "async":
        task_path = enqueue_query_task(
            project_root=PROJECT_ROOT,
            session_id=session_id,
            question=request.question,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            use_image_evidence=request.use_image_evidence,
            max_image_frames=request.max_image_frames,
            max_image_evidence=request.max_image_evidence,
            text_top_k=request.text_top_k,
            visual_top_k=request.visual_top_k,
            final_evidence_k=request.final_evidence_k,
            memory_mode=request.memory_mode,
            use_interaction_cache=request.use_interaction_cache,
            cache_mode=request.cache_mode,
            use_current=request.use_current,
            use_short_term=request.use_short_term,
            use_long_term=request.use_long_term,
            debug_router=request.debug_router,
            long_term_retrieval_scheme=long_term_retrieval_scheme,
            retrieval_scheme=request.retrieval_scheme,
            client_source=request.client_source,
            input_method=request.input_method,
            allow_inactive_session=allow_inactive_session,
            task_source=task_source,
        )
        task_id = task_path.stem
        stream_context = None
        try:
            from online_query.stream_query_context import load_stream_query_context

            stream_context = load_stream_query_context(session_id, sessions_root=ONLINE_SESSIONS_DIR, project_root=PROJECT_ROOT, question=request.question)
        except Exception:
            stream_context = None
        return JSONResponse(
            status_code=202,
            content={
                "status": "queued",
                "mode": "async",
                "response_mode": "legacy",
                "session_id": session_id,
                "task_id": task_id,
                "task_path": str(task_path),
                "result_url": f"/query_task/{task_id}",
                "long_term_retrieval_scheme": long_term_retrieval_scheme,
                "stream_context": stream_context,
            },
        )

    if not _env_bool("EM2MEM_ENABLE_SYNC_ASK", True):
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "message": "Synchronous /ask is disabled. Use mode=async and run the query worker on the compute node.",
            },
        )

    sync_task = _new_direct_query_task(
        session_id,
        request,
        long_term_retrieval_scheme,
        allow_inactive_session=allow_inactive_session,
        task_source=f"{task_source}_sync" if task_source != "api" else "api_sync",
        response_mode="legacy",
    )

    try:
        from online_query import query_session

        result = query_session(
            session_id=session_id,
            question=request.question,
            sessions_root=ONLINE_SESSIONS_DIR,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            use_image_evidence=request.use_image_evidence,
            max_image_frames=request.max_image_frames,
            max_image_evidence=request.max_image_evidence,
            text_top_k=request.text_top_k,
            visual_top_k=request.visual_top_k,
            final_evidence_k=request.final_evidence_k,
            memory_mode=request.memory_mode,
            use_interaction_cache=request.use_interaction_cache,
            cache_mode=request.cache_mode,
            use_current=request.use_current,
            use_short_term=request.use_short_term,
            use_long_term=request.use_long_term,
            debug_router=request.debug_router,
            long_term_retrieval_scheme=long_term_retrieval_scheme,
        )
        try:
            from online_query.stream_query_context import load_stream_query_context

            stream_context = load_stream_query_context(session_id, sessions_root=ONLINE_SESSIONS_DIR, project_root=PROJECT_ROOT, question=request.question)
            if stream_context and not result.get("stream_context"):
                result["stream_context"] = stream_context
        except Exception:
            pass
        result["status"] = "ok"
        result["response_mode"] = "legacy"
        _finish_stream_query_task(sync_task, "done", result=result)
        result = _augment_evidence_frames_for_response(result, session_id, api_base_url=_api_base_url(http_request))
        _append_qa_history_safe(
            session_id,
            question=request.question,
            answer=str(result.get("answer") or result.get("answer_text") or ""),
            client_source=request.client_source,
            input_method=request.input_method,
            status="done",
            task_id=sync_task["task_id"],
            response_mode="legacy",
            metadata={"long_term_retrieval_scheme": long_term_retrieval_scheme},
        )
        return JSONResponse(status_code=200, content=result)
    except Exception as exc:
        _finish_stream_query_task_failure(sync_task, str(exc))
        _append_qa_history_safe(
            session_id,
            question=request.question,
            client_source=request.client_source,
            input_method=request.input_method,
            status="failed",
            error=str(exc),
            task_id=sync_task["task_id"],
            response_mode="legacy",
            metadata={"long_term_retrieval_scheme": long_term_retrieval_scheme},
        )
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


@app.post("/ask/{session_id}/stream", response_model=None)
async def ask_session_stream(session_id: str, request: AskRequest) -> StreamingResponse | JSONResponse:
    request.response_mode = "stream"
    response = await ask_session(session_id, request)
    return response


@app.post("/session/{session_id}/ask", response_model=None)
async def ask_historical_session(
    session_id: str,
    request: AskRequest,
    http_request: FastAPIRequest = None,
) -> JSONResponse | StreamingResponse:
    return await _handle_ask_session(
        session_id,
        request,
        http_request=http_request,
        allow_inactive_session=True,
        task_source="session_history_api",
    )


@app.post("/session/{session_id}/ask/stream", response_model=None)
async def ask_historical_session_stream(
    session_id: str,
    request: AskRequest,
    http_request: FastAPIRequest = None,
) -> StreamingResponse | JSONResponse:
    request.response_mode = "stream"
    return await _handle_ask_session(
        session_id,
        request,
        http_request=http_request,
        allow_inactive_session=True,
        task_source="session_history_api",
    )


@app.post("/session/{session_id}/qa_history")
async def append_session_qa_history(session_id: str, request: QaHistoryAppendRequest) -> JSONResponse:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    record = _append_qa_history_safe(
        session_id,
        question=request.question,
        answer=request.answer,
        client_source=request.client_source,
        input_method=request.input_method,
        status=request.status,
        error=request.error,
        response_mode="manual_record",
        metadata=request.metadata,
    )
    if record is None:
        return JSONResponse(status_code=500, content={"status": "error", "message": "qa history append failed"})
    return JSONResponse(status_code=200, content={"status": "ok", "session_id": session_id, "record": record})


@app.get("/session/{session_id}/qa_history")
async def get_session_qa_history(session_id: str, limit: Optional[int] = None) -> JSONResponse:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    history = load_qa_history(session_dir, limit=limit)
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "session_id": session_id,
            "count": len(history),
            "qa_history_path": str(qa_history_path(session_dir)),
            "items": history,
        },
    )

@app.get("/stream/active")
async def stream_active() -> JSONResponse:
    if not _env_bool("EM2MEM_SINGLE_ACTIVE_SESSION", False):
        return JSONResponse(
            status_code=200,
            content={
                "status": "multi_session_mode",
                "active": False,
                "session_id": None,
                "message": "single active session is disabled; clients must keep and use their own session_id",
            },
        )
    try:
        from online_pipeline.active_session import read_active_session_id
        from online_pipeline.stream_status import build_stream_status

        session_id = read_active_session_id(PROJECT_ROOT)
        if not session_id:
            return JSONResponse(status_code=200, content={"status": "not_found", "active": False, "session_id": None})
        if not _valid_session_id(session_id):
            return JSONResponse(status_code=200, content={"status": "invalid_active_session", "active": False, "session_id": session_id})
        session_dir = ONLINE_SESSIONS_DIR / session_id
        if not session_dir.exists():
            return JSONResponse(status_code=200, content={"status": "missing_session", "active": False, "session_id": session_id})
        stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
        if not isinstance(stream_state, dict):
            stream_state = {}
        stream_status = str(stream_state.get("stream_status") or stream_state.get("status") or "").strip().lower()
        if stream_status in {"ended", "stopped", "aborted", "cancelled", "canceled"}:
            return JSONResponse(
                status_code=200,
                content={
                    "status": stream_status,
                    "active": False,
                    "session_id": session_id,
                    "stream_status": stream_status,
                },
            )
        status = build_stream_status(PROJECT_ROOT, session_dir)
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "active": True,
                "session_id": session_id,
                "input_mode": stream_state.get("input_mode"),
                "stream_status": status.get("stream_status"),
                "can_ask": bool(status.get("can_ask")),
                "rokid": status.get("rokid"),
                "live": status.get("live"),
                "webrtc": status.get("webrtc"),
                "live_ingest": status.get("live_ingest"),
                "memory": status.get("memory"),
                "stream": status,
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "active": False, "message": str(exc)})


@app.get("/session/{session_id}/query_warmup")
async def query_warmup_status(session_id: str) -> JSONResponse:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    payload = read_json(session_dir / "em2mem" / "query_warmup_state.json", default={})
    if not isinstance(payload, dict) or not payload:
        return JSONResponse(status_code=200, content={"status": "not_started", "session_id": session_id})
    return JSONResponse(status_code=200, content=payload)


def _active_rokid_session_file() -> Path:
    return PROJECT_ROOT / ROKID_ACTIVE_SESSION_PATH


def _read_active_rokid_session() -> dict[str, Any]:
    payload = read_json(_active_rokid_session_file(), default={})
    return payload if isinstance(payload, dict) else {}


def _write_active_rokid_session(session_id: str, *, input_mode: str, reason: str = "rokid_stream_start") -> dict[str, Any]:
    payload = {
        "active_session_id": session_id,
        "session_id": session_id,
        "input_mode": input_mode,
        "updated_at": utc_now_iso(),
        "reason": reason,
    }
    write_json_atomic(_active_rokid_session_file(), payload)
    return payload


def _clear_active_rokid_session(session_id: str | None = None, *, reason: str = "rokid_stream_end") -> dict[str, Any]:
    previous = _read_active_rokid_session()
    previous_session_id = str(previous.get("active_session_id") or previous.get("session_id") or "").strip()
    if session_id and previous_session_id and previous_session_id != session_id:
        return {
            "active_session_id": previous_session_id,
            "cleared": False,
            "cleared_session_id": None,
            "updated_at": utc_now_iso(),
            "reason": "active_rokid_session_mismatch",
        }
    payload = {
        "active_session_id": None,
        "session_id": None,
        "cleared": True,
        "cleared_session_id": previous_session_id or None,
        "updated_at": utc_now_iso(),
        "reason": reason,
    }
    write_json_atomic(_active_rokid_session_file(), payload)
    return payload


def _is_terminal_stream_status(stream_state: dict[str, Any]) -> bool:
    status = str(stream_state.get("stream_status") or stream_state.get("status") or "").strip().lower()
    return status in {"ended", "stopped", "aborted", "cancelled", "canceled", "failed"}


def _is_rokid_stream_state(stream_state: dict[str, Any]) -> bool:
    try:
        from online_pipeline.rokid_ingest import is_rokid_input_mode

        return is_rokid_input_mode(stream_state.get("input_mode"))
    except Exception:
        return str(stream_state.get("input_mode") or "").strip().lower().startswith("rokid_")


def _is_rokid_glass_stream_state(stream_state: dict[str, Any]) -> bool:
    if not _is_rokid_stream_state(stream_state):
        return False
    metadata = stream_state.get("metadata") if isinstance(stream_state.get("metadata"), dict) else {}
    source = str(metadata.get("source") or "").strip().lower()
    client = str(metadata.get("client") or "").strip().lower()
    device_id = str(metadata.get("device_id") or "").strip().lower()
    if source == "web_frontend" or client == "web_frontend" or device_id.startswith("rokid_web"):
        return False
    return True


def _parse_utc_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _rokid_live_inactive_reason(session_dir: Path, input_mode: str, live_ingest: dict[str, Any] | None = None) -> str:
    if input_mode != "rokid_live_rtmp":
        return ""
    state = live_ingest if isinstance(live_ingest, dict) else {}
    if not state:
        state = read_json(session_dir / "stream" / "live_ingest_state.json", default={})
        if not isinstance(state, dict):
            state = {}
    status = str(state.get("status") or "").strip().lower()
    if status in {"failed", "aborted", "cancelled", "canceled"}:
        return f"live_ingest_{status}"
    try:
        frames_ingested = int(state.get("frames_ingested", 0) or 0)
    except Exception:
        frames_ingested = 0
    last_frame_at = _parse_utc_datetime(state.get("last_frame_at"))
    if frames_ingested > 0 and last_frame_at:
        stale_seconds = float(os.getenv("EM2MEM_ROKID_ACTIVE_STALE_SECONDS", "45") or 45)
        age_seconds = (datetime.now(timezone.utc) - last_frame_at).total_seconds()
        if age_seconds > stale_seconds:
            return "live_ingest_stale"
    return ""


def _find_latest_active_rokid_session_id() -> str | None:
    if not ONLINE_SESSIONS_DIR.exists():
        return None
    candidates: list[tuple[float, str]] = []
    for session_dir in ONLINE_SESSIONS_DIR.iterdir():
        if not session_dir.is_dir() or not _valid_session_id(session_dir.name):
            continue
        stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
        if not isinstance(stream_state, dict) or not _is_rokid_glass_stream_state(stream_state) or _is_terminal_stream_status(stream_state):
            continue
        input_mode = str(stream_state.get("input_mode") or "").strip()
        if _rokid_live_inactive_reason(session_dir, input_mode):
            continue
        try:
            updated = (session_dir / "stream" / "stream_state.json").stat().st_mtime
        except Exception:
            updated = session_dir.stat().st_mtime
        candidates.append((float(updated), session_dir.name))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def _build_active_rokid_response(session_id: str, *, source: str = "active_rokid_session") -> dict[str, Any]:
    if not _valid_session_id(session_id):
        return {"status": "invalid_active_session", "active": False, "session_id": session_id}
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return {"status": "missing_session", "active": False, "session_id": session_id}
    stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
    if not isinstance(stream_state, dict):
        stream_state = {}
    input_mode = str(stream_state.get("input_mode") or "").strip()
    if not _is_rokid_stream_state(stream_state):
        return {"status": "wrong_input_mode", "active": False, "session_id": session_id, "input_mode": input_mode or None}
    if not _is_rokid_glass_stream_state(stream_state):
        _clear_active_rokid_session(session_id, reason="frontend_created_session")
        return {"status": "frontend_created_session", "active": False, "session_id": session_id, "input_mode": input_mode or None}
    if _is_terminal_stream_status(stream_state):
        _clear_active_rokid_session(session_id, reason="terminal_stream_status")
        return {
            "status": str(stream_state.get("stream_status") or stream_state.get("status") or "ended"),
            "active": False,
            "session_id": session_id,
            "input_mode": input_mode,
        }
    try:
        from online_pipeline.stream_status import build_stream_status

        status_payload = build_stream_status(PROJECT_ROOT, session_dir)
    except Exception as exc:
        status_payload = {"status_error": str(exc), "stream_status": stream_state.get("status"), "input_mode": input_mode}
    metadata = stream_state.get("metadata") if isinstance(stream_state.get("metadata"), dict) else {}
    live = status_payload.get("live") if isinstance(status_payload.get("live"), dict) else {}
    webrtc = status_payload.get("webrtc") if isinstance(status_payload.get("webrtc"), dict) else {}
    live_ingest = status_payload.get("live_ingest") if isinstance(status_payload.get("live_ingest"), dict) else {}
    inactive_reason = _rokid_live_inactive_reason(session_dir, input_mode, live_ingest)
    if inactive_reason:
        _clear_active_rokid_session(session_id, reason=inactive_reason)
        return {
            "status": inactive_reason,
            "active": False,
            "session_id": session_id,
            "input_mode": input_mode,
            "stream_status": status_payload.get("stream_status") or stream_state.get("status"),
            "live_ingest": live_ingest,
            "message": "Rokid RTMP session is no longer producing live frames; start recording on the glasses again.",
        }
    stream_name = live.get("stream_name") or live.get("streamName") or webrtc.get("stream_name") or webrtc.get("streamName")
    push_url = live.get("push_url_public") or live.get("pushUrlPublic") or live.get("push_url") or live.get("pushUrl")
    webrtc_play_url = live.get("webrtc_play_url_public") or live.get("webrtcPlayUrlPublic") or webrtc.get("webrtc_play_url_public") or webrtc.get("webrtcPlayUrlPublic")
    return {
        "status": "ok",
        "active": True,
        "source": source,
        "session_id": session_id,
        "stream_id": stream_state.get("stream_id") or status_payload.get("stream_id"),
        "input_mode": input_mode,
        "stream_status": status_payload.get("stream_status") or stream_state.get("status"),
        "metadata": metadata,
        "owner_id": metadata.get("owner_id"),
        "device_id": metadata.get("device_id"),
        "device_type": metadata.get("device_type", "rokid"),
        "frame_upload_url": f"/rokid/{session_id}/frame",
        "audio_upload_url": f"/rokid/{session_id}/audio_chunk",
        "status_url": f"/rokid/{session_id}/status",
        "ask_stream_url": f"/ask/{session_id}/stream",
        "live_ingest_start_url": f"/rokid/{session_id}/live/ingest/start" if input_mode == "rokid_live_rtmp" else None,
        "live_ingest_stop_url": f"/rokid/{session_id}/live/ingest/stop" if input_mode == "rokid_live_rtmp" else None,
        "push_url": push_url,
        "stream_name": stream_name,
        "webrtc_play_url_public": webrtc_play_url,
        "can_ask": bool(status_payload.get("can_ask")),
        "rokid": status_payload.get("rokid"),
        "live": live,
        "webrtc": webrtc,
        "live_ingest": live_ingest,
        "frame_stream": status_payload.get("frame_stream"),
        "audio_stream": status_payload.get("audio_stream"),
        "memory": status_payload.get("memory"),
        "stream": status_payload,
    }


@app.get("/rokid/stream/active")
async def rokid_stream_active() -> JSONResponse:
    active_payload = _read_active_rokid_session()
    session_id = str(active_payload.get("active_session_id") or active_payload.get("session_id") or "").strip()
    source = "active_rokid_session"
    if not session_id:
        session_id = _find_latest_active_rokid_session_id() or ""
        source = "latest_rokid_session" if session_id else "none"
    if not session_id:
        return JSONResponse(status_code=200, content={"status": "not_found", "active": False, "session_id": None})
    response = _build_active_rokid_response(session_id, source=source)
    if not response.get("active") and response.get("status") == "frontend_created_session":
        fallback_session_id = _find_latest_active_rokid_session_id() or ""
        if fallback_session_id and fallback_session_id != session_id:
            response = _build_active_rokid_response(fallback_session_id, source="latest_rokid_session")
        else:
            response = {"status": "not_found", "active": False, "session_id": None, "ignored_session_id": session_id, "ignored_reason": "frontend_created_session"}
    if response.get("active"):
        _write_active_rokid_session(str(response.get("session_id") or session_id), input_mode=str(response.get("input_mode") or ""), reason=source)
    return JSONResponse(status_code=200, content=response)

async def _start_stream_session(
    request: StreamStartRequest,
    *,
    route_kind: str = "stream",
    forced_input_mode: Optional[str] = None,
    allow_rokid: bool = False,
    query_warmup_session_id: Optional[str] = None,
    query_warmup_reason: Optional[str] = None,
    query_warmup_wait_for_memory: bool = False,
) -> JSONResponse:
    try:
        from online_pipeline.live_rtmp import (
            build_webrtc_whip_live_source,
            build_live_source,
            live_rtmp_config,
            make_stream_name,
            public_webrtc_status_block,
            save_live_source,
            stream_state_live_block,
            webrtc_whip_config,
        )
        from online_pipeline.frame_stream import (
            FrameStreamStore,
            frame_stream_input_mode,
            is_frame_stream_mode,
            public_frame_stream_status_block,
        )
        from online_pipeline.audio_stream import AudioStreamStore, public_audio_stream_status_block
        from online_pipeline.rokid_ingest import (
            ROKID_LIVE_RTMP_INPUT_MODE,
            ROKID_INPUT_MODE,
            default_rokid_metadata,
            initialize_rokid_state,
            is_rokid_input_mode,
            public_rokid_start_block,
        )
        from online_short_term.stream_chunk_manager import StreamChunkManager

        input_mode = frame_stream_input_mode(forced_input_mode or request.input_mode)
        live_input_modes = {"live_pusher_rtmp", "web_webrtc_whip", ROKID_LIVE_RTMP_INPUT_MODE}
        stream_input_modes = {"chunk", "live_pusher_rtmp", "web_webrtc_whip", "frame_audio_stream"}
        rokid_input_modes = {ROKID_INPUT_MODE, ROKID_LIVE_RTMP_INPUT_MODE}
        supported_modes = stream_input_modes | (rokid_input_modes if allow_rokid else set())
        if input_mode in rokid_input_modes and not allow_rokid:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Rokid input modes are served by /rokid/stream/start. Use the dedicated Rokid API instead of /stream/start.",
                    "input_mode": input_mode,
                    "rokid_start_url": "/rokid/stream/start",
                },
            )
        if input_mode not in supported_modes:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"unsupported input_mode: {request.input_mode}",
                    "supported_input_modes": sorted(supported_modes | {"frame_stream"}),
                },
            )
        live_config = live_rtmp_config()
        whip_config = webrtc_whip_config()
        if input_mode in {"live_pusher_rtmp", ROKID_LIVE_RTMP_INPUT_MODE} and not bool(live_config.get("enabled")):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"{input_mode} is disabled; set EM2MEM_LIVE_RTMP_ENABLED=1 to enable RTMP push URL creation",
                    "input_mode": input_mode,
                    "live": {
                        "enabled": False,
                        "protocol": live_config.get("scheme", "rtmp"),
                        "domain": live_config.get("domain"),
                        "app": live_config.get("app"),
                    },
                },
            )
        if input_mode == "web_webrtc_whip" and not bool(whip_config.get("enabled")):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "web_webrtc_whip is disabled; set EM2MEM_WEBRTC_WHIP_ENABLED=1 to enable WHIP URL creation",
                    "input_mode": input_mode,
                    "webrtc": {
                        "enabled": False,
                        "protocol": "whip",
                        "domain": whip_config.get("domain"),
                        "app": whip_config.get("app"),
                    },
                },
            )

        metadata = default_rokid_metadata(request.metadata, input_mode=input_mode) if is_rokid_input_mode(input_mode) else dict(request.metadata or {})
        metadata["input_mode"] = input_mode
        metadata["api_namespace"] = route_kind
        if request.owner_id:
            metadata["owner_id"] = str(request.owner_id)
        if request.device_id:
            metadata["device_id"] = str(request.device_id)
        if request.device_type:
            metadata["device_type"] = str(request.device_type)
        elif route_kind == "stream" and input_mode == "frame_audio_stream":
            metadata.setdefault("device_type", "phone")
        requested_sid = (request.session_id or "").strip() or None
        if requested_sid is not None:
            if not _valid_session_id(requested_sid):
                return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
            session_dir = ONLINE_SESSIONS_DIR / requested_sid
            if not session_dir.exists():
                return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {requested_sid}"})
            session_id = requested_sid
        else:
            created = create_online_session(
                source="stream",
                original_filename=None,
                metadata=metadata,
            )
            session_id = str(created["session_id"])
            session_dir = Path(created["session_dir"])

        manager = StreamChunkManager(session_dir)
        processing_seconds = float(os.getenv("EM2MEM_STREAM_PROCESSING_CHUNK_SECONDS") or request.chunk_duration or 5.0)
        stream_state = manager.init_stream(
            chunk_duration=max(0.1, processing_seconds),
            metadata=metadata,
        )
        live_source = None
        if input_mode in live_input_modes:
            stream_name = make_stream_name(session_id)
            if input_mode == "web_webrtc_whip":
                live_source = build_webrtc_whip_live_source(
                    session_id=session_id,
                    stream_id=str(stream_state.get("stream_id") or ""),
                    stream_name=stream_name,
                    whip_config=whip_config,
                    rtmp_config=live_config,
                )
            else:
                live_source = build_live_source(
                    session_id=session_id,
                    stream_id=str(stream_state.get("stream_id") or ""),
                    stream_name=stream_name,
                    input_mode=input_mode,
                    config=live_config,
                )
            save_live_source(session_dir, live_source)
            try:
                from online_pipeline.live_ingest import initialize_live_ingest_state, public_live_ingest_status_block

                initialize_live_ingest_state(
                    session_dir,
                    stream_id=str(stream_state.get("stream_id") or ""),
                    source_url=str(live_source.get("pull_url_internal") or live_source.get("pull_url_public") or ""),
                    frame_fps=float(os.getenv("EM2MEM_LIVE_INGEST_FRAME_FPS", "1") or 1),
                    audio_segment_ms=int(os.getenv("EM2MEM_LIVE_INGEST_AUDIO_SEGMENT_MS", "1500") or 1500),
                    source="srs_webrtc_whip" if input_mode == "web_webrtc_whip" else "srs_rtmp",
                    input_mode=input_mode,
                    status="not_started",
                )
                live_ingest_response = public_live_ingest_status_block(session_dir)
            except Exception:
                live_ingest_response = {"status": "not_started"}

            def _set_live_state(state: dict[str, Any]) -> None:
                state["input_mode"] = input_mode
                state.setdefault("metadata", {}).update(metadata)
                state["live"] = stream_state_live_block(live_source or {})

            stream_state, _ = manager.update_stream_state_locked(_set_live_state)
        else:
            live_ingest_response = {"status": "not_started"}
            def _set_chunk_mode(state: dict[str, Any]) -> None:
                state["input_mode"] = input_mode
                state.setdefault("metadata", {}).update(metadata)

            stream_state, _ = manager.update_stream_state_locked(_set_chunk_mode)
        if is_frame_stream_mode(input_mode):
            frame_store = FrameStreamStore(session_dir)
            frame_store.initialize(
                stream_id=str(stream_state.get("stream_id") or ""),
                input_mode=input_mode,
            )
            frame_response = frame_store.public_status(input_mode=input_mode)
            audio_store = AudioStreamStore(session_dir)
            audio_store.initialize(
                stream_id=str(stream_state.get("stream_id") or ""),
                input_mode=input_mode,
            )
            audio_response = audio_store.public_status(input_mode=input_mode)

            def _set_frame_stream_state(state: dict[str, Any]) -> None:
                state["input_mode"] = input_mode
                state["frame_stream"] = frame_response
                state["audio_stream"] = audio_response

            stream_state, _ = manager.update_stream_state_locked(_set_frame_stream_state)
        else:
            frame_response = public_frame_stream_status_block(session_dir, input_mode=input_mode)
            audio_response = public_audio_stream_status_block(session_dir, input_mode=input_mode)
        rokid_response = None
        active_rokid_session = None
        if is_rokid_input_mode(input_mode):
            initialize_rokid_state(
                session_dir,
                stream_id=str(stream_state.get("stream_id") or ""),
                metadata=metadata,
                input_mode=input_mode,
            )
            rokid_response = public_rokid_start_block(metadata, input_mode=input_mode)
            if _is_rokid_glass_stream_state({"input_mode": input_mode, "metadata": metadata}):
                active_rokid_session = _write_active_rokid_session(session_id, input_mode=input_mode, reason="rokid_stream_start")
        active_cleanup = None
        if _env_bool("EM2MEM_SINGLE_ACTIVE_SESSION", False):
            from online_pipeline.active_session import clear_old_session_tasks, write_active_session

            write_active_session(PROJECT_ROOT, session_id, reason="stream_start")
            active_cleanup = clear_old_session_tasks(PROJECT_ROOT, keep_session_id=session_id, reason="stream_start")
        write_status(
            session_dir=session_dir,
            session_id=session_id,
            status="streaming",
            stage="stream_started",
            progress=0,
            error=None,
        )
        video_path = session_dir / "input.mp4"
        live_response = stream_state_live_block(live_source or {}) if live_source else {
            "enabled": False,
            "input_mode": "chunk",
            "push_url_available": False,
            "ingest_status": None,
            "last_frame_at": None,
            "last_audio_at": None,
            "srs_checked": False,
            "srs_online": None,
        }
        if input_mode not in live_input_modes:
            live_response["input_mode"] = input_mode
        push_url = live_source.get("push_url_public") if isinstance(live_source, dict) else None
        webrtc_response = public_webrtc_status_block(session_dir, stream_state=stream_state)
        route_prefix = "/rokid" if route_kind == "rokid" else "/stream"
        _start_query_warmup_thread(
            query_warmup_session_id or session_id,
            reason=query_warmup_reason or f"{route_kind}_start",
            wait_for_memory=query_warmup_wait_for_memory,
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "stream_started",
                "session_id": session_id,
                "stream_id": stream_state.get("stream_id"),
                "input_mode": input_mode,
                "message": "stream session created",
                "chunk_duration": stream_state.get("chunk_duration"),
                "processing_chunk_seconds": stream_state.get("processing_chunk_seconds", stream_state.get("chunk_duration")),
                "next_expected_chunk_index": stream_state.get("next_expected_chunk_index", 0),
                "next_expected_upload_chunk_index": stream_state.get("next_expected_upload_chunk_index", 0),
                "next_expected_proc_index": stream_state.get("next_expected_proc_index", 0),
                "stream_upload_url": f"{route_prefix}/{session_id}/chunk" if route_kind != "rokid" else None,
                "frame_upload_url": f"{route_prefix}/{session_id}/frame",
                "audio_upload_url": f"{route_prefix}/{session_id}/audio_chunk",
                "push_url": push_url,
                "stream_name": live_source.get("stream_name") if isinstance(live_source, dict) else None,
                "webrtc": webrtc_response,
                "live": live_response,
                "live_ingest": live_ingest_response,
                "live_ingest_available": input_mode in live_input_modes,
                "live_ingest_status": live_ingest_response.get("status"),
                "live_ingest_start_url": f"{route_prefix}/{session_id}/live/ingest/start" if input_mode in live_input_modes else None,
                "live_ingest_stop_url": f"{route_prefix}/{session_id}/live/ingest/stop" if input_mode in live_input_modes else None,
                "frame_stream": frame_response,
                "audio_stream": audio_response,
                "rokid": rokid_response,
                "frame_audio_stream": {"enabled": is_frame_stream_mode(input_mode), "input_mode": input_mode},
                "video_path": str(video_path),
                "size_bytes": video_path.stat().st_size if video_path.exists() else 0,
                "preprocess_queued": False,
                "query_warmup_started": _env_bool("EM2MEM_QUERY_WARMUP_ON_STREAM_START", True),
                "task_path": None,
                "single_active_session": _env_bool("EM2MEM_SINGLE_ACTIVE_SESSION", False),
                "active_session_cleanup": active_cleanup,
                "active_rokid_session": active_rokid_session,
                "can_ask": False,
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/stream/start")
async def stream_start(request: StreamStartRequest) -> JSONResponse:
    return await _start_stream_session(request, route_kind="stream", allow_rokid=False)


@app.post("/rokid/stream/start")
async def rokid_stream_start(request: StreamStartRequest) -> JSONResponse:
    return await _start_rokid_stream_session(request)


async def _start_plain_rokid_stream_session(request: StreamStartRequest, input_mode: str) -> JSONResponse:
    metadata = dict(request.metadata or {})
    for key in (
        "parent_session_id",
        "child_session_id",
        "is_rokid_day_child",
        "day_label",
        "day_index",
        "run_id",
    ):
        metadata.pop(key, None)
    metadata["rokid_session_mode"] = "plain"
    plain_request = StreamStartRequest(
        session_id=None,
        parent_session_id=None,
        run_id=None,
        create_parent_session=False,
        input_mode=input_mode,
        chunk_duration=request.chunk_duration,
        metadata=metadata,
        owner_id=request.owner_id,
        device_id=request.device_id,
        device_type=request.device_type,
    )
    response = await _start_stream_session(
        plain_request,
        route_kind="rokid",
        forced_input_mode=input_mode,
        allow_rokid=True,
    )
    content = _json_response_content(response)
    if content:
        content["rokid_session_mode"] = "plain"
        content["parent_session_id"] = content.get("session_id")
        content["child_session_id"] = content.get("session_id")
        content["day_context"] = {
            "enabled": False,
            "mode": "plain",
            "day_label": None,
            "day_index": None,
            "run_id": "",
        }
        return JSONResponse(status_code=response.status_code, content=content)
    return response


async def _start_rokid_stream_session(request: StreamStartRequest) -> JSONResponse:
    from online_pipeline.rokid_ingest import ROKID_INPUT_MODE, ROKID_INPUT_MODES

    input_mode = str(request.input_mode or ROKID_INPUT_MODE).strip().lower()
    if input_mode not in ROKID_INPUT_MODES:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"unsupported rokid input_mode: {request.input_mode}",
                "supported_input_modes": sorted(ROKID_INPUT_MODES),
            },
        )
    if not _rokid_day_session_enabled():
        return await _start_plain_rokid_stream_session(request, input_mode)

    from online_pipeline.rokid_day import (
        enrich_start_response_for_day,
        cleanup_failed_child_reservation,
        mark_rokid_day_failed,
        mark_rokid_day_started,
        normalize_run_id,
        reserve_rokid_day_child,
        update_child_metadata,
        valid_session_id,
    )

    metadata = dict(request.metadata or {})
    parent_session_id = (
        str(request.parent_session_id or "").strip()
        or str(metadata.get("parent_session_id") or "").strip()
        or str(request.session_id or "").strip()
    )
    create_parent = bool(request.create_parent_session)
    if create_parent or not parent_session_id:
        parent = create_online_session(
            source="rokid_parent",
            original_filename=None,
            metadata={
                "source": "rokid_glass",
                "device_type": "rokid",
                "is_rokid_parent": True,
                "created_by": "rokid_stream_start",
            },
        )
        parent_session_id = str(parent["session_id"])
    if not valid_session_id(parent_session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid parent_session_id"})
    parent_dir = ONLINE_SESSIONS_DIR / parent_session_id
    if not parent_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"parent session not found: {parent_session_id}"})

    run_id = normalize_run_id(request.run_id or metadata.get("run_id"))
    if not run_id:
        return JSONResponse(status_code=400, content={"status": "error", "message": "run_id is required for Rokid DAY start"})

    try:
        run, _ = reserve_rokid_day_child(
            sessions_root=ONLINE_SESSIONS_DIR,
            parent_session_id=parent_session_id,
            run_id=run_id,
            input_mode=input_mode,
            metadata=metadata,
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})

    if str(run.get("status") or "") == "started" and isinstance(run.get("start_response"), dict):
        return JSONResponse(status_code=200, content=enrich_start_response_for_day(run["start_response"], run))

    child_session_id = str(run.get("child_session_id") or "")
    child_metadata = {
        **metadata,
        "input_mode": input_mode,
        "parent_session_id": parent_session_id,
        "child_session_id": child_session_id,
        "is_rokid_day_child": True,
        "day_label": run.get("day_label"),
        "day_index": run.get("day_index"),
        "run_id": run_id,
    }
    if not (ONLINE_SESSIONS_DIR / child_session_id).exists():
        create_online_session(
            source="rokid_day_child",
            original_filename=None,
            metadata=child_metadata,
            session_id=child_session_id,
        )
    child_request = StreamStartRequest(
        session_id=child_session_id,
        parent_session_id=parent_session_id,
        run_id=run_id,
        input_mode=input_mode,
        chunk_duration=request.chunk_duration,
        metadata=child_metadata,
        owner_id=request.owner_id,
        device_id=request.device_id,
        device_type=request.device_type,
    )
    response = await _start_stream_session(
        child_request,
        route_kind="rokid",
        forced_input_mode=input_mode,
        allow_rokid=True,
        query_warmup_session_id=parent_session_id,
        query_warmup_reason="rokid_parent_start",
        query_warmup_wait_for_memory=False,
    )
    content = _json_response_content(response)
    if response.status_code < 200 or response.status_code >= 300:
        mark_rokid_day_failed(
            sessions_root=ONLINE_SESSIONS_DIR,
            parent_session_id=parent_session_id,
            run_id=run_id,
            error=str(content.get("message") or response.status_code),
        )
        cleanup_failed_child_reservation(ONLINE_SESSIONS_DIR, child_session_id)
        content = enrich_start_response_for_day(
            {
                **content,
                "status": content.get("status") or "error",
                "message": content.get("message") or "Rokid stream start failed",
            },
            run,
        )
        return JSONResponse(status_code=response.status_code, content=content)
    content = enrich_start_response_for_day(content, run)
    update_child_metadata(ONLINE_SESSIONS_DIR / child_session_id, run)
    started_run = mark_rokid_day_started(
        sessions_root=ONLINE_SESSIONS_DIR,
        parent_session_id=parent_session_id,
        run_id=run_id,
        response=content,
    )
    content = enrich_start_response_for_day(content, started_run)
    return JSONResponse(status_code=response.status_code, content=content)


def _require_rokid_session(session_id: str) -> JSONResponse | None:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
    if not isinstance(stream_state, dict):
        stream_state = {}
    try:
        from online_pipeline.rokid_ingest import is_rokid_input_mode

        input_mode = str(stream_state.get("input_mode") or "")
        if is_rokid_input_mode(input_mode):
            return None
    except Exception:
        input_mode = str(stream_state.get("input_mode") or "")
    return JSONResponse(
        status_code=409,
        content={
            "status": "wrong_input_mode",
            "message": "this session was not started through the Rokid API",
            "session_id": session_id,
            "input_mode": input_mode or None,
            "expected_input_modes": ["rokid_frame_audio", "rokid_live_rtmp"],
        },
    )


@app.post("/stream/{session_id}/live/ingest/start")
async def stream_live_ingest_start(session_id: str) -> JSONResponse:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    inactive = _inactive_session_response(session_id)
    if inactive is not None:
        return inactive
    try:
        from online_pipeline.live_ingest import (
            choose_live_source_url,
            initialize_live_ingest_state,
            load_live_ingest_state,
            public_live_ingest_status_block,
            update_live_source_ingest,
        )
        from online_pipeline.live_rtmp import load_live_source
        from online_preprocess.task_queue import enqueue_live_ingest_task

        live_source = load_live_source(session_dir)
        if not live_source:
            return JSONResponse(status_code=409, content={"status": "error", "message": "live_source.json not found; start stream with input_mode=live_pusher_rtmp or web_webrtc_whip"})
        source_url = choose_live_source_url(live_source)
        if not source_url:
            return JSONResponse(status_code=409, content={"status": "error", "message": "live source has no pull URL"})
        live_input_mode = str(live_source.get("input_mode") or "live_pusher_rtmp")
        live_task_source = "srs_webrtc_whip" if live_input_mode == "web_webrtc_whip" else "srs_rtmp"
        state = load_live_ingest_state(session_dir)
        if str(state.get("status") or "") in {
            "queued",
            "starting",
            "waiting_stream",
            "waiting_rtmp_output",
            "waiting_keyframe",
            "running",
            "stopping",
        } and not state.get("stop_requested"):
            return JSONResponse(
                status_code=200,
                content={
                    "status": "already_running",
                    "session_id": session_id,
                    "stream_id": live_source.get("stream_id"),
                    "source_url": source_url,
                    "live_ingest": public_live_ingest_status_block(session_dir),
                },
            )
        task_path = enqueue_live_ingest_task(
            PROJECT_ROOT,
            session_id=session_id,
            stream_id=str(live_source.get("stream_id") or ""),
            source_url=source_url,
            source=live_task_source,
            input_mode=live_input_mode,
            reason="api_live_ingest_start",
            force=str(state.get("status") or "") in {"failed", "stopped"},
        )
        state = initialize_live_ingest_state(
            session_dir,
            stream_id=str(live_source.get("stream_id") or ""),
            source_url=source_url,
            frame_fps=float(os.getenv("EM2MEM_LIVE_INGEST_FRAME_FPS", "1") or 1),
            audio_segment_ms=int(os.getenv("EM2MEM_LIVE_INGEST_AUDIO_SEGMENT_MS", "1500") or 1500),
            source=live_task_source,
            input_mode=live_input_mode,
            status="queued",
        )
        state["task_id"] = task_path.stem
        from online_preprocess.io_utils import write_json_atomic

        write_json_atomic(session_dir / "stream" / "live_ingest_state.json", state)
        update_live_source_ingest(session_dir, ingest_status="queued", last_error=None)
        return JSONResponse(
            status_code=200,
            content={
                "status": "queued",
                "session_id": session_id,
                "stream_id": live_source.get("stream_id"),
                "task_id": task_path.stem,
                "task_path": str(task_path),
                "source_url": source_url,
                "live_ingest": public_live_ingest_status_block(session_dir),
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})


@app.post("/stream/{session_id}/live/ingest/stop")
async def stream_live_ingest_stop(session_id: str) -> JSONResponse:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_pipeline.live_ingest import request_live_ingest_stop

        state = request_live_ingest_stop(session_dir, reason="api_stop")
        stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
        stream_end_payload = None
        if isinstance(stream_state, dict) and _is_rokid_stream_state(stream_state):
            end_status, end_payload = _execute_stream_end(
                session_id,
                session_dir,
                StreamEndRequest(),
            )
            stream_end_payload = {
                "status_code": end_status,
                **end_payload,
            }
        return JSONResponse(
            status_code=200,
            content={
                "status": "stop_requested",
                "session_id": session_id,
                "live_ingest": state,
                "stream_end": stream_end_payload,
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})


@app.post("/rokid/{session_id}/live/ingest/start")
async def rokid_live_ingest_start(session_id: str) -> JSONResponse:
    guard = _require_rokid_session(session_id)
    if guard is not None:
        return guard
    return await stream_live_ingest_start(session_id)


@app.post("/rokid/{session_id}/live/ingest/stop")
async def rokid_live_ingest_stop(session_id: str) -> JSONResponse:
    guard = _require_rokid_session(session_id)
    if guard is not None:
        return guard
    return await stream_live_ingest_stop(session_id)


@app.post("/stream/{session_id}/frame")
async def stream_upload_frame(
    session_id: str,
    frame: Optional[UploadFile] = File(default=None),
    frame_index: int = Form(...),
    client_ts_ms: Optional[int] = Form(default=None),
    relative_ts_ms: Optional[int] = Form(default=None),
    source_ts_ms: Optional[int] = Form(default=None),
    device_ts_ms: Optional[int] = Form(default=None),
    timestamp_source: Optional[str] = Form(default=None),
    width: Optional[int] = Form(default=None),
    height: Optional[int] = Form(default=None),
    format_hint: Optional[str] = Form(default=None, alias="format"),
    source: str = Form(default="camera_take_photo"),
) -> JSONResponse:
    if frame is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No image uploaded. Expected form field 'frame'."})
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    if int(frame_index) < 0:
        return JSONResponse(status_code=400, content={"status": "error", "message": "frame_index must be >= 0"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    inactive = _inactive_session_response(session_id)
    if inactive is not None:
        return inactive
    try:
        from online_pipeline.frame_stream import frame_stream_input_mode
        from online_pipeline.rokid_ingest import (
            ROKID_DEFAULT_VIDEO_SOURCE,
            RokidNormalizationError,
            is_rokid_input_mode,
            normalize_rokid_frame_upload,
        )
        from online_pipeline.realtime_ingest import ingest_frame

        stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
        if not isinstance(stream_state, dict):
            stream_state = {}
        input_mode = frame_stream_input_mode(stream_state.get("input_mode") or "frame_audio_stream")
        payload = await frame.read()
        effective_source = str(source or "").strip()
        effective_source_ts_ms = source_ts_ms if source_ts_ms is not None else device_ts_ms
        if is_rokid_input_mode(input_mode):
            if not effective_source or effective_source == "camera_take_photo":
                effective_source = ROKID_DEFAULT_VIDEO_SOURCE
            try:
                payload, format_hint, _ = normalize_rokid_frame_upload(
                    payload,
                    format_hint=format_hint,
                    filename=frame.filename,
                    width=width,
                    height=height,
                )
            except RokidNormalizationError as exc:
                return JSONResponse(status_code=400, content={"status": "error", "message": str(exc), "input_mode": input_mode})
        else:
            effective_source = effective_source or "camera_take_photo"
        result = ingest_frame(
            PROJECT_ROOT,
            session_id,
            payload,
            frame_index=int(frame_index),
            client_ts_ms=client_ts_ms,
            relative_ts_ms=relative_ts_ms,
            source_ts_ms=effective_source_ts_ms,
            timestamp_source=timestamp_source,
            width=width,
            height=height,
            format=format_hint,
            source=effective_source,
            input_mode=input_mode,
            filename_hint=frame.filename,
            allow_live_input=is_rokid_input_mode(input_mode),
        )
        status_code = int(result.pop("_http_status_code", 200))
        return JSONResponse(status_code=status_code, content=result)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    finally:
        await frame.close()


@app.post("/rokid/{session_id}/frame")
async def rokid_upload_frame(
    session_id: str,
    frame: Optional[UploadFile] = File(default=None),
    frame_index: int = Form(...),
    client_ts_ms: Optional[int] = Form(default=None),
    relative_ts_ms: Optional[int] = Form(default=None),
    source_ts_ms: Optional[int] = Form(default=None),
    device_ts_ms: Optional[int] = Form(default=None),
    timestamp_source: Optional[str] = Form(default=None),
    width: Optional[int] = Form(default=None),
    height: Optional[int] = Form(default=None),
    format_hint: Optional[str] = Form(default=None, alias="format"),
    source: str = Form(default="rokid_sdk_video"),
) -> JSONResponse:
    guard = _require_rokid_session(session_id)
    if guard is not None:
        if frame is not None:
            await frame.close()
        return guard
    return await stream_upload_frame(
        session_id=session_id,
        frame=frame,
        frame_index=frame_index,
        client_ts_ms=client_ts_ms,
        relative_ts_ms=relative_ts_ms,
        source_ts_ms=source_ts_ms,
        device_ts_ms=device_ts_ms,
        timestamp_source=timestamp_source,
        width=width,
        height=height,
        format_hint=format_hint,
        source=source,
    )


@app.post("/stream/{session_id}/audio_chunk")
async def stream_upload_audio_chunk(
    session_id: str,
    audio: Optional[UploadFile] = File(default=None),
    audio_index: int = Form(...),
    client_ts_ms: Optional[int] = Form(default=None),
    relative_ts_ms: Optional[int] = Form(default=None),
    source_ts_ms: Optional[int] = Form(default=None),
    device_ts_ms: Optional[int] = Form(default=None),
    timestamp_source: Optional[str] = Form(default=None),
    duration_ms: Optional[int] = Form(default=None),
    sample_rate: Optional[int] = Form(default=None),
    channels: Optional[int] = Form(default=None),
    sample_width: Optional[int] = Form(default=None),
    encoding: Optional[str] = Form(default=None),
    audio_format: Optional[str] = Form(default=None, alias="format"),
    source: str = Form(default="recorder_manager"),
) -> JSONResponse:
    if audio is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No audio uploaded. Expected form field 'audio'."})
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    if int(audio_index) < 0:
        return JSONResponse(status_code=400, content={"status": "error", "message": "audio_index must be >= 0"})
    if not _env_bool("EM2MEM_AUDIO_STREAM_ENABLED", True):
        return JSONResponse(
            status_code=409,
            content={
                "status": "audio_stream_disabled",
                "message": "audio stream is disabled by EM2MEM_AUDIO_STREAM_ENABLED=0",
                "session_id": session_id,
            },
        )
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    inactive = _inactive_session_response(session_id)
    if inactive is not None:
        return inactive
    try:
        from online_pipeline.frame_stream import frame_stream_input_mode
        from online_pipeline.rokid_ingest import (
            ROKID_DEFAULT_AUDIO_SOURCE,
            RokidNormalizationError,
            is_rokid_input_mode,
            normalize_rokid_audio_upload,
        )
        from online_pipeline.realtime_ingest import ingest_audio_chunk

        stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
        if not isinstance(stream_state, dict):
            stream_state = {}
        input_mode = frame_stream_input_mode(stream_state.get("input_mode") or "frame_audio_stream")
        payload = await audio.read()
        effective_source = str(source or "").strip()
        effective_source_ts_ms = source_ts_ms if source_ts_ms is not None else device_ts_ms
        if is_rokid_input_mode(input_mode):
            if not effective_source or effective_source == "recorder_manager":
                effective_source = ROKID_DEFAULT_AUDIO_SOURCE
            try:
                payload, audio_format, _ = normalize_rokid_audio_upload(
                    payload,
                    format_hint=audio_format,
                    filename=audio.filename,
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_width=sample_width,
                    encoding=encoding,
                )
            except RokidNormalizationError as exc:
                return JSONResponse(status_code=400, content={"status": "error", "message": str(exc), "input_mode": input_mode})
        else:
            effective_source = effective_source or "recorder_manager"
        result = ingest_audio_chunk(
            PROJECT_ROOT,
            session_id,
            payload,
            audio_index=int(audio_index),
            client_ts_ms=client_ts_ms,
            relative_ts_ms=relative_ts_ms,
            source_ts_ms=effective_source_ts_ms,
            timestamp_source=timestamp_source,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            channels=channels,
            format=audio_format,
            content_type=audio.content_type,
            source=effective_source,
            input_mode=input_mode,
            filename_hint=audio.filename,
            enqueue_asr=True,
            allow_live_input=is_rokid_input_mode(input_mode),
        )
        status_code = int(result.pop("_http_status_code", 200))
        return JSONResponse(status_code=status_code, content=result)
    except Exception as exc:
        _append_timeline_event_safe(
            session_dir,
            "audio_chunk_error",
            chunk_index=int(audio_index),
            chunk_id=f"audio_{int(audio_index):06d}",
            metadata={"error": str(exc)},
        )
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id, "audio_index": int(audio_index)})
    finally:
        await audio.close()


@app.post("/rokid/{session_id}/audio_chunk")
async def rokid_upload_audio_chunk(
    session_id: str,
    audio: Optional[UploadFile] = File(default=None),
    audio_index: int = Form(...),
    client_ts_ms: Optional[int] = Form(default=None),
    relative_ts_ms: Optional[int] = Form(default=None),
    source_ts_ms: Optional[int] = Form(default=None),
    device_ts_ms: Optional[int] = Form(default=None),
    timestamp_source: Optional[str] = Form(default=None),
    duration_ms: Optional[int] = Form(default=None),
    sample_rate: Optional[int] = Form(default=None),
    channels: Optional[int] = Form(default=None),
    sample_width: Optional[int] = Form(default=None),
    encoding: Optional[str] = Form(default=None),
    audio_format: Optional[str] = Form(default=None, alias="format"),
    source: str = Form(default="rokid_sdk_audio"),
) -> JSONResponse:
    guard = _require_rokid_session(session_id)
    if guard is not None:
        if audio is not None:
            await audio.close()
        return guard
    return await stream_upload_audio_chunk(
        session_id=session_id,
        audio=audio,
        audio_index=audio_index,
        client_ts_ms=client_ts_ms,
        relative_ts_ms=relative_ts_ms,
        source_ts_ms=source_ts_ms,
        device_ts_ms=device_ts_ms,
        timestamp_source=timestamp_source,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        encoding=encoding,
        audio_format=audio_format,
        source=source,
    )


@app.post("/stream/{session_id}/chunk")
async def stream_upload_chunk(
    session_id: str,
    file: Optional[UploadFile] = File(default=None),
    chunk_index: int = Form(...),
    chunk_start_time: Optional[float] = Form(default=None),
    chunk_duration: Optional[float] = Form(default=None),
    is_last: bool = Form(default=False),
    client_timestamp: Optional[str] = Form(default=None),
) -> JSONResponse:
    if file is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No file uploaded. Expected form field 'file'."})
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    inactive = _inactive_session_response(session_id)
    if inactive is not None:
        return inactive
    try:
        from online_short_term.stream_chunk_manager import StreamChunkManager, probe_duration
        from online_pipeline.stream_timeline import append_timeline_event

        manager = StreamChunkManager(session_dir)
        state = manager.load_stream_state(default={})
        if not state:
            return JSONResponse(status_code=409, content={"status": "error", "message": "stream not started"})
        if str(state.get("status") or "") in {"ending", "ended"}:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "stream_not_accepting_chunks",
                    "message": "stream is ending or ended; use retry_chunk for failed chunks",
                    "session_id": session_id,
                    "stream_status": state.get("status"),
                },
            )
        stream_id = str(state.get("stream_id") or "")
        processing_seconds = float(state.get("processing_chunk_seconds") or state.get("chunk_duration") or 5.0)
        upload_chunk_id = manager.make_upload_chunk_id(int(chunk_index))
        tmp_path = manager.tmp_dir / f"{upload_chunk_id}_{uuid4().hex[:8]}.tmp"
        manager.tmp_dir.mkdir(parents=True, exist_ok=True)
        manager.upload_chunks_dir.mkdir(parents=True, exist_ok=True)
        manager.chunks_dir.mkdir(parents=True, exist_ok=True)
        size_bytes = 0
        with tmp_path.open("wb") as output:
            while True:
                data = await file.read(CHUNK_SIZE_BYTES)
                if not data:
                    break
                size_bytes += len(data)
                output.write(data)
        if size_bytes <= 0:
            tmp_path.unlink(missing_ok=True)
            return JSONResponse(status_code=400, content={"status": "error", "message": "Uploaded chunk is empty."})
        checksum = _sha256_path(tmp_path)
        try:
            actual_duration = probe_duration(tmp_path)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            print(
                f"[stream_upload] duration probe failed session_id={session_id} chunk_index={chunk_index}: {exc}",
                flush=True,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "session_id": session_id,
                    "chunk_index": int(chunk_index),
                    "message": f"failed to probe uploaded chunk duration: {exc}",
                    "frontend_should_reupload_chunk": True,
                },
            )
        try:
            registered = manager.register_upload_chunk_transaction(
                tmp_upload_path=tmp_path,
                upload_chunk_index=int(chunk_index),
                checksum=checksum,
                size_bytes=size_bytes,
                actual_duration=actual_duration,
                client_timestamp=client_timestamp,
                is_last=bool(is_last),
            )
        except ValueError as exc:
            tmp_path.unlink(missing_ok=True)
            from online_pipeline.backpressure import compute_backpressure

            backpressure = compute_backpressure(project_root=PROJECT_ROOT, stream_latency=manager.load_stream_state(default={}).get("latency"))
            return JSONResponse(
                status_code=409,
                content={
                    "status": "conflict",
                    "message": str(exc),
                    "session_id": session_id,
                    "chunk_index": int(chunk_index),
                    "can_upload_next_chunk": backpressure.get("recommended_action") != "pause_upload",
                    "backpressure": backpressure,
                },
            )
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            print(f"[stream_upload] state registration failed session_id={session_id} chunk_index={chunk_index}: {exc}", flush=True)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"failed to register uploaded chunk: {exc}"},
            )
        task_path = None
        task_created = False
        if not registered.get("duplicate"):
            task_path = manager.enqueue_ready_upload_chunk(PROJECT_ROOT)
            task_created = task_path is not None
        updated_state = manager.load_stream_state(default={})
        upload_chunk = registered.get("upload_chunk") if isinstance(registered.get("upload_chunk"), dict) else {}
        rel_path = str(upload_chunk.get("path") or f"stream/upload_chunks/{upload_chunk_id}.mp4")
        from online_pipeline.backpressure import compute_backpressure

        backpressure = compute_backpressure(project_root=PROJECT_ROOT, stream_latency=updated_state.get("latency"))
        response_status = "duplicate_ignored" if registered.get("duplicate") else str(registered.get("status") or "chunk_received")
        if response_status == "received":
            response_status = "chunk_received"
        can_upload_next = str(updated_state.get("status") or "") == "running" and backpressure.get("recommended_action") != "pause_upload"
        video_path = session_dir / "input.mp4"
        append_timeline_event(
            session_dir,
            "chunk_received",
            chunk_index=int(chunk_index),
            chunk_id=upload_chunk_id,
            metadata={
                "status": response_status,
                "size_bytes": size_bytes,
                "task_created": task_created,
                "actual_duration": upload_chunk.get("actual_duration"),
            },
        )
        if task_created:
            append_timeline_event(
                session_dir,
                "stream_chunk_queued",
                chunk_index=int(chunk_index),
                chunk_id=upload_chunk_id,
                metadata={"task_id": task_path.stem if task_path else upload_chunk.get("task_id"), "upload_level": True},
            )
        return JSONResponse(
            status_code=200,
            content={
                "status": response_status,
                "session_id": session_id,
                "stream_id": stream_id,
                "chunk_id": upload_chunk_id,
                "upload_chunk_id": upload_chunk_id,
                "chunk_index": int(chunk_index),
                "upload_chunk_index": int(chunk_index),
                "size_bytes": size_bytes,
                "received_bytes": size_bytes,
                "chunk_path": rel_path,
                "upload_chunk_path": rel_path,
                "actual_duration": upload_chunk.get("actual_duration"),
                "upload_duration": upload_chunk.get("upload_duration"),
                "processing_chunk_seconds": round(float(processing_seconds), 3),
                "processing_chunk_count": len(upload_chunk.get("processing_chunks") or []),
                "processing_chunks": upload_chunk.get("processing_chunks") or [],
                "video_path": str(video_path),
                "message": "duplicate chunk ignored" if registered.get("duplicate") else "chunk saved",
                "task_created": task_created,
                "task_id": upload_chunk.get("task_id") or (task_path.stem if task_path else None),
                "stream_chunk_task_id": upload_chunk.get("task_id") or (task_path.stem if task_path else None),
                "stream_asr_task_id": upload_chunk.get("asr_task_id"),
                "asr_queued": bool(upload_chunk.get("asr_task_id")),
                "task_path": upload_chunk.get("task_path") or (str(task_path) if task_path else None),
                "duplicate": bool(registered.get("duplicate")),
                "next_expected_chunk_index": updated_state.get("next_expected_chunk_index", 0),
                "next_expected_upload_chunk_index": updated_state.get("next_expected_upload_chunk_index", 0),
                "next_expected_proc_index": updated_state.get("next_expected_proc_index", 0),
                "can_upload_next_chunk": can_upload_next,
                "backpressure": backpressure,
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    finally:
        await file.close()


def _register_optional_demo_routes() -> None:
    if not _env_bool("EM2MEM_ENABLE_DEMO_ROUTES", True):
        return
    try:
        import demo.api  # noqa: F401
    except ModuleNotFoundError as exc:
        if str(exc.name or "") != "demo":
            print(f"[demo] optional demo routes not registered: {exc}", flush=True)
    except RuntimeError as exc:
        if "already registered" not in str(exc):
            print(f"[demo] optional demo routes failed to register: {exc}", flush=True)
    except Exception as exc:
        print(f"[demo] optional demo routes failed to register: {exc}", flush=True)


_register_optional_demo_routes()


def _execute_stream_end(
    session_id: str,
    session_dir: Path,
    request: StreamEndRequest,
) -> tuple[int, dict[str, Any]]:
    if not _valid_session_id(session_id):
        return 400, {"status": "error", "message": "invalid session_id"}
    if not session_dir.exists():
        return 404, {"status": "error", "message": f"session not found: {session_id}"}
    try:
        from online_short_term.stream_chunk_manager import StreamChunkManager

        manager = StreamChunkManager(session_dir)
        existing_state = manager.load_stream_state(default={})
        if isinstance(existing_state, dict) and _is_terminal_stream_status(existing_state):
            summary = manager.summary()
            return 200, {
                "status": "stream_ended",
                "session_id": session_id,
                "stream_id": existing_state.get("stream_id"),
                "final_chunk_index": existing_state.get("final_chunk_index"),
                "close_open_event": bool(request.close_open_event),
                "task_created": False,
                "task_id": existing_state.get("stream_end_task_id"),
                "task_path": existing_state.get("stream_end_task_path"),
                "received_chunk_count": summary.get("received_chunk_count", 0),
                "processed_chunk_count": summary.get("processed_chunk_count", 0),
                "failed_chunk_count": len(summary.get("failed_chunks", []) or []),
                "open_event_closed": True,
                "frame_open_event_closed": False,
                "frame_mst": None,
                "frame_mst_error": None,
                "can_ask": True,
                "active_session_cleared": None,
                "active_rokid_session_cleared": None,
                "rokid_day_merge_task": None,
                "rokid_day_merge_error": None,
                "already_terminal": True,
                "stream_status": str(existing_state.get("status") or ""),
            }
        state = manager.mark_stream_ending(final_chunk_index=request.final_chunk_index, close_open_event=request.close_open_event)
        task_path = manager.enqueue_stream_end_if_ready(PROJECT_ROOT)
        frame_mst_result = None
        frame_mst_error = None
        try:
            from online_pipeline.frame_stream import frame_stream_input_mode, is_frame_stream_mode

            if request.close_open_event and is_frame_stream_mode(frame_stream_input_mode(state.get("input_mode"))) and _env_bool("EM2MEM_FRAME_STREAM_ENABLE_MST", True):
                from online_short_term.frame_stream_event_builder import FrameStreamMicroEventBuilder

                frame_mst_result = FrameStreamMicroEventBuilder(session_dir).close_open_event(
                    project_root=PROJECT_ROOT,
                    enqueue_refine=_env_bool("EM2MEM_FRAME_STREAM_ENQUEUE_REFINE", True),
                    reason="stream_end",
                )
                from online_pipeline.runtime_state import refresh_session_pipeline_state

                refresh_session_pipeline_state(session_dir)
                if (frame_mst_result or {}).get("status") == "ignored_short_event":
                    _append_timeline_event_safe(
                        session_dir,
                        "frame_mst_ignored_short_event",
                        metadata={"reason": "stream_end"},
                    )
                if int((frame_mst_result or {}).get("closed_event_count", 0) or 0) > 0:
                    _append_timeline_event_safe(
                        session_dir,
                        "frame_mst_closed",
                        metadata={
                            "reason": "stream_end",
                            "closed_event_ids": (frame_mst_result or {}).get("closed_event_ids", []),
                            "refine_task_paths": (frame_mst_result or {}).get("refine_task_paths", []),
                        },
                    )
                    if (frame_mst_result or {}).get("refine_task_paths"):
                        _append_timeline_event_safe(
                            session_dir,
                            "frame_mst_refine_queued",
                            metadata={"refine_task_paths": (frame_mst_result or {}).get("refine_task_paths", [])},
                        )
        except Exception as exc:
            frame_mst_error = str(exc)
            print(f"[frame_stream] stream_end M_st close failed session_id={session_id}: {exc}", flush=True)
        summary = manager.summary()

        _append_timeline_event_safe(
            session_dir,
            "stream_end_requested",
            metadata={"final_chunk_index": state.get("final_chunk_index"), "task_id": task_path.stem if task_path else state.get("stream_end_task_id")},
        )
        write_status(
            session_dir=session_dir,
            session_id=session_id,
            status="streaming",
            stage="stream_ending",
            progress=0,
            error=None,
        )
        active_session_cleared = None
        active_rokid_session_cleared = None
        rokid_day_merge_task = None
        rokid_day_merge_error = None
        if _is_rokid_stream_state(state):
            active_rokid_session_cleared = _clear_active_rokid_session(session_id, reason="stream_end")
            try:
                from online_pipeline.rokid_day import load_rokid_day_child_metadata
                from online_preprocess.task_queue import enqueue_rokid_day_merge_task

                day_meta = load_rokid_day_child_metadata(session_dir)
                if day_meta:
                    merge_path = enqueue_rokid_day_merge_task(
                        PROJECT_ROOT,
                        parent_session_id=str(day_meta["parent_session_id"]),
                        child_session_id=session_id,
                        day_label=str(day_meta["day_label"]),
                        day_index=int(day_meta["day_index"]),
                        run_id=str(day_meta.get("run_id") or ""),
                        reason="stream_end",
                    )
                    rokid_day_merge_task = {
                        "task_id": merge_path.stem,
                        "task_path": str(merge_path),
                        "parent_session_id": day_meta["parent_session_id"],
                        "child_session_id": session_id,
                        "day_label": day_meta["day_label"],
                        "day_index": day_meta["day_index"],
                        "run_id": day_meta.get("run_id"),
                    }
            except Exception as exc:
                rokid_day_merge_error = str(exc)
        if _env_bool("EM2MEM_SINGLE_ACTIVE_SESSION", True):
            from online_pipeline.active_session import clear_active_session

            active_session_cleared = clear_active_session(PROJECT_ROOT, session_id=session_id, reason="stream_end")
        return 200, {
            "status": "stream_ended",
            "session_id": session_id,
            "stream_id": state.get("stream_id"),
            "final_chunk_index": state.get("final_chunk_index"),
            "close_open_event": bool(request.close_open_event),
            "task_created": task_path is not None,
            "task_id": task_path.stem if task_path else state.get("stream_end_task_id"),
            "task_path": str(task_path) if task_path else state.get("stream_end_task_path"),
            "received_chunk_count": summary.get("received_chunk_count", 0),
            "processed_chunk_count": summary.get("processed_chunk_count", 0),
            "failed_chunk_count": len(summary.get("failed_chunks", []) or []),
            "open_event_closed": task_path is not None and bool(request.close_open_event),
            "frame_open_event_closed": bool((frame_mst_result or {}).get("closed_event_count")),
            "frame_mst": frame_mst_result,
            "frame_mst_error": frame_mst_error,
            "can_ask": True,
            "active_session_cleared": active_session_cleared,
            "active_rokid_session_cleared": active_rokid_session_cleared,
            "rokid_day_merge_task": rokid_day_merge_task,
            "rokid_day_merge_error": rokid_day_merge_error,
        }
    except Exception as exc:
        return 500, {"status": "error", "message": str(exc)}


@app.post("/stream/{session_id}/end")
async def stream_end(session_id: str, request: StreamEndRequest) -> JSONResponse:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    status_code, payload = _execute_stream_end(session_id, session_dir, request)
    return JSONResponse(status_code=status_code, content=payload)


@app.get("/stream/{session_id}/preview")
async def stream_preview(session_id: str) -> JSONResponse:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_pipeline.frame_stream import public_frame_stream_status_block
        from online_pipeline.rokid_ingest import public_rokid_status_block

        stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
        if not isinstance(stream_state, dict):
            stream_state = {}
        input_mode = stream_state.get("input_mode")
        frame_stream = public_frame_stream_status_block(session_dir, input_mode=input_mode)
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "session_id": session_id,
                "input_mode": input_mode,
                "stream_status": stream_state.get("status"),
                "frame_stream": frame_stream,
                "rokid": public_rokid_status_block(session_dir, input_mode=input_mode),
                "can_ask": bool(frame_stream.get("mcur_ready")),
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.get("/stream/{session_id}/status")
async def stream_status(session_id: str) -> JSONResponse:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_short_term.stream_chunk_manager import StreamChunkManager
        from online_pipeline.stream_status import build_stream_status

        StreamChunkManager(session_dir).reconcile_stream_upload_chunks()
        return JSONResponse(status_code=200, content=build_stream_status(PROJECT_ROOT, session_dir))
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/rokid/{session_id}/frame")
async def rokid_upload_frame(
    session_id: str,
    frame: Optional[UploadFile] = File(default=None),
    frame_index: int = Form(...),
    client_ts_ms: Optional[int] = Form(default=None),
    relative_ts_ms: Optional[int] = Form(default=None),
    source_ts_ms: Optional[int] = Form(default=None),
    device_ts_ms: Optional[int] = Form(default=None),
    timestamp_source: Optional[str] = Form(default=None),
    width: Optional[int] = Form(default=None),
    height: Optional[int] = Form(default=None),
    format_hint: Optional[str] = Form(default=None, alias="format"),
    source: str = Form(default="rokid_sdk_video"),
) -> JSONResponse:
    return await stream_upload_frame(
        session_id=session_id,
        frame=frame,
        frame_index=frame_index,
        client_ts_ms=client_ts_ms,
        relative_ts_ms=relative_ts_ms,
        source_ts_ms=source_ts_ms,
        device_ts_ms=device_ts_ms,
        timestamp_source=timestamp_source,
        width=width,
        height=height,
        format_hint=format_hint,
        source=source,
    )


@app.post("/rokid/{session_id}/audio_chunk")
async def rokid_upload_audio_chunk(
    session_id: str,
    audio: Optional[UploadFile] = File(default=None),
    audio_index: int = Form(...),
    client_ts_ms: Optional[int] = Form(default=None),
    relative_ts_ms: Optional[int] = Form(default=None),
    source_ts_ms: Optional[int] = Form(default=None),
    device_ts_ms: Optional[int] = Form(default=None),
    timestamp_source: Optional[str] = Form(default=None),
    duration_ms: Optional[int] = Form(default=None),
    sample_rate: Optional[int] = Form(default=None),
    channels: Optional[int] = Form(default=None),
    sample_width: Optional[int] = Form(default=None),
    encoding: Optional[str] = Form(default=None),
    audio_format: Optional[str] = Form(default=None, alias="format"),
    source: str = Form(default="rokid_sdk_audio"),
) -> JSONResponse:
    return await stream_upload_audio_chunk(
        session_id=session_id,
        audio=audio,
        audio_index=audio_index,
        client_ts_ms=client_ts_ms,
        relative_ts_ms=relative_ts_ms,
        source_ts_ms=source_ts_ms,
        device_ts_ms=device_ts_ms,
        timestamp_source=timestamp_source,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        encoding=encoding,
        audio_format=audio_format,
        source=source,
    )


@app.post("/rokid/{session_id}/audio_question")
async def rokid_audio_question(
    session_id: str,
    audio: Optional[UploadFile] = File(default=None),
    duration_ms: Optional[int] = Form(default=None),
    sample_rate: Optional[int] = Form(default=None),
    channels: Optional[int] = Form(default=None),
    sample_width: Optional[int] = Form(default=None),
    encoding: Optional[str] = Form(default=None),
    audio_format: Optional[str] = Form(default=None, alias="format"),
    mode: Optional[str] = Form(default="async"),
    retrieval_mode: str = Form(default="auto"),
    memory_mode: str = Form(default="auto"),
    top_k: int = Form(default=5),
    use_current: Optional[bool] = Form(default=True),
    use_image_evidence: Any = Form(default="auto"),
    max_image_evidence: Optional[int] = Form(default=6),
    use_interaction_cache: bool = Form(default=True),
    debug_router: bool = Form(default=True),
    long_term_retrieval_scheme: Optional[str] = Form(default=None),
    retrieval_scheme: Optional[str] = Form(default=None),
    client_source: str = Form(default="glasses"),
    input_method: str = Form(default="voice"),
) -> JSONResponse:
    return await stream_audio_question(
        session_id=session_id,
        audio=audio,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        encoding=encoding,
        audio_format=audio_format,
        mode=mode,
        retrieval_mode=retrieval_mode,
        memory_mode=memory_mode,
        top_k=top_k,
        use_current=use_current,
        use_image_evidence=use_image_evidence,
        max_image_evidence=max_image_evidence,
        use_interaction_cache=use_interaction_cache,
        debug_router=debug_router,
        long_term_retrieval_scheme=long_term_retrieval_scheme,
        retrieval_scheme=retrieval_scheme,
        client_source=client_source,
        input_method=input_method,
    )

@app.post("/rokid/{session_id}/audio_question/stream", response_model=None)
async def rokid_audio_question_stream(
    session_id: str,
    audio: Optional[UploadFile] = File(default=None),
    duration_ms: Optional[int] = Form(default=None),
    sample_rate: Optional[int] = Form(default=None),
    channels: Optional[int] = Form(default=None),
    sample_width: Optional[int] = Form(default=None),
    encoding: Optional[str] = Form(default=None),
    audio_format: Optional[str] = Form(default=None, alias="format"),
    retrieval_mode: str = Form(default="auto"),
    memory_mode: str = Form(default="auto"),
    top_k: int = Form(default=5),
    use_current: Optional[bool] = Form(default=True),
    use_image_evidence: Any = Form(default="auto"),
    max_image_evidence: Optional[int] = Form(default=6),
    use_interaction_cache: bool = Form(default=True),
    debug_router: bool = Form(default=True),
    client_source: str = Form(default="glasses"),
    input_method: str = Form(default="voice"),
) -> StreamingResponse:
    return await stream_audio_question_stream(
        session_id=session_id,
        audio=audio,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        encoding=encoding,
        audio_format=audio_format,
        retrieval_mode=retrieval_mode,
        memory_mode=memory_mode,
        top_k=top_k,
        use_current=use_current,
        use_image_evidence=use_image_evidence,
        max_image_evidence=max_image_evidence,
        use_interaction_cache=use_interaction_cache,
        debug_router=debug_router,
        client_source=client_source,
        input_method=input_method,
    )

@app.get("/rokid/{session_id}/status")
async def rokid_status(session_id: str) -> JSONResponse:
    guard = _require_rokid_session(session_id)
    if guard is not None:
        return guard
    return await stream_status(session_id)


@app.get("/stream/{session_id}/missing_chunks")
async def stream_missing_chunks(session_id: str) -> JSONResponse:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_short_term.stream_chunk_manager import StreamChunkManager

        manager = StreamChunkManager(session_dir)
        manager.reconcile_stream_upload_chunks()
        summary = manager.summary()
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "session_id": session_id,
                "missing_chunks": summary.get("missing_chunks", []),
                "retry_required_chunks": summary.get("retry_required_chunks", []),
                "conflict_chunks": summary.get("conflict_chunks", []),
                "waiting_chunks": summary.get("waiting_chunks", []),
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.get("/stream/{session_id}/metrics")
async def stream_metrics(session_id: str) -> JSONResponse:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_pipeline.stream_timeline import build_stream_metrics

        return JSONResponse(status_code=200, content=build_stream_metrics(PROJECT_ROOT, session_dir))
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.get("/stream/{session_id}/timeline")
async def stream_timeline(
    session_id: str,
    limit: int = 100,
    event_type: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> JSONResponse:
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        from online_pipeline.stream_timeline import read_timeline_events

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "session_id": session_id,
                "events": read_timeline_events(session_dir, limit=limit, event_type=event_type, chunk_index=chunk_index),
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/stream/{session_id}/retry_chunk")
async def stream_retry_chunk(session_id: str, request_body: Optional[dict[str, Any]] = Body(default=None)) -> JSONResponse:
    if request_body is None:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "missing request body; expected JSON with required integer field 'chunk_index'",
                "required_body": {"chunk_index": 0, "force": False},
                "hint": "call GET /stream/{session_id}/missing_chunks first and retry an index from retry_required_chunks or missing_chunks",
            },
        )
    if "chunk_index" not in request_body:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "missing required field: chunk_index",
                "required_body": {"chunk_index": 0, "force": False},
                "hint": "call GET /stream/{session_id}/missing_chunks first and retry an index from retry_required_chunks or missing_chunks",
            },
        )
    try:
        request = StreamRetryChunkRequest(
            chunk_index=int(request_body.get("chunk_index")),
            force=bool(request_body.get("force", False)),
        )
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"invalid retry_chunk body: {exc}",
                "required_body": {"chunk_index": 0, "force": False},
            },
        )
    if not _valid_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    inactive = _inactive_session_response(session_id)
    if inactive is not None:
        return inactive
    try:
        from online_preprocess.task_queue import enqueue_stream_asr_task, enqueue_stream_chunk_task, enqueue_stream_upload_task
        from online_pipeline.backpressure import compute_backpressure
        from online_short_term.stream_chunk_manager import StreamChunkManager

        manager = StreamChunkManager(session_dir)
        reconcile_result = manager.reconcile_stream_upload_chunks()
        if reconcile_result.get("recovered_count"):
            print(
                f"[stream_retry] reconciled orphan uploads session_id={session_id} "
                f"recovered_count={reconcile_result.get('recovered_count')}",
                flush=True,
            )
        def _retry_mutation(state: dict[str, Any]) -> dict[str, Any]:
            upload_chunks = [dict(item) for item in state.get("upload_chunks", state.get("received_chunks", [])) or [] if isinstance(item, dict)]
            upload = next(
                (
                    item
                    for item in upload_chunks
                    if int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == int(request.chunk_index)
                ),
                None,
            )
            if upload is None:
                missing = set(int(x) for x in state.get("missing_chunks", []) or [] if str(x).lstrip("-").isdigit())
                missing.add(int(request.chunk_index))
                state["missing_chunks"] = sorted(missing)
                return {
                    "error_status": 404,
                    "status": "error",
                    "message": "chunk file is not registered; frontend should upload it again",
                    "frontend_should_reupload_chunk": True,
                }
            conflicts = state.get("conflict_chunks", []) or []
            if any(isinstance(item, dict) and int(item.get("upload_chunk_index", -1)) == int(request.chunk_index) for item in conflicts):
                return {"error_status": 409, "status": "conflict", "conflict_chunks": conflicts}
            upload_path = session_dir / str(upload.get("path") or "")
            if not upload_path.exists():
                missing = set(int(x) for x in state.get("missing_chunks", []) or [] if str(x).lstrip("-").isdigit())
                missing.add(int(request.chunk_index))
                state["missing_chunks"] = sorted(missing)
                return {
                    "error_status": 404,
                    "status": "error",
                    "message": "chunk file is missing; frontend should upload it again",
                    "frontend_should_reupload_chunk": True,
                }

            created_tasks: list[str] = []
            existing_tasks: list[str] = []
            now = utc_now_iso()
            processing_chunks = [dict(item) for item in upload.get("processing_chunks", []) or [] if isinstance(item, dict)]
            if processing_chunks:
                updated_processing = []
                for proc in state.get("processing_chunks", []) or []:
                    if not isinstance(proc, dict):
                        continue
                    proc = dict(proc)
                    if int(proc.get("source_upload_chunk_index", proc.get("upload_chunk_index", -1))) != int(request.chunk_index):
                        updated_processing.append(proc)
                        continue
                    proc_status = str(proc.get("status") or "")
                    proc_index = int(proc.get("proc_index", proc.get("chunk_index", 0)))
                    if proc_status == "processed" and not request.force:
                        updated_processing.append(proc)
                        continue
                    active = _find_active_stream_task(
                        session_id=session_id,
                        task_type="stream_chunk",
                        match_fields={"proc_index": proc_index},
                    )
                    if active is not None:
                        state_key, task_path, _payload = active
                        proc["task_id"] = task_path.stem
                        proc["task_path"] = str(task_path)
                        if state_key.endswith("_queued"):
                            proc["status"] = "queued"
                        elif state_key.endswith("_in_progress"):
                            proc["status"] = "processing"
                        elif state_key.endswith("_done"):
                            proc["status"] = "processed"
                        existing_tasks.append(str(task_path))
                        updated_processing.append(proc)
                        continue
                    if proc_status == "failed" or request.force:
                        proc["status"] = "queued"
                        proc["error"] = None
                        proc["forced_retry"] = bool(request.force)
                        proc["retry_queued_at"] = now
                        task_path = enqueue_stream_chunk_task(
                            project_root=PROJECT_ROOT,
                            session_id=session_id,
                            stream_id=str(state.get("stream_id") or ""),
                            chunk_id=str(proc.get("chunk_id") or proc.get("processing_chunk_id") or ""),
                            chunk_index=proc_index,
                            proc_index=proc_index,
                            upload_chunk_index=int(request.chunk_index),
                            source_upload_chunk_id=str(upload.get("upload_chunk_id") or upload.get("chunk_id") or ""),
                            chunk_path=str(proc.get("path") or ""),
                            start_time=float(proc.get("start_time") or 0.0),
                            end_time=float(proc.get("end_time") or 0.0),
                            duration=float(proc.get("duration") or 0.0),
                            checksum=str(proc.get("checksum") or upload.get("checksum") or ""),
                            force=True,
                        )
                        proc["task_id"] = task_path.stem
                        proc["task_path"] = str(task_path)
                        created_tasks.append(str(task_path))
                    updated_processing.append(proc)
                state["processing_chunks"] = updated_processing
                upload["processing_chunks"] = [
                    dict(item)
                    for item in updated_processing
                    if isinstance(item, dict)
                    and int(item.get("source_upload_chunk_index", item.get("upload_chunk_index", -1))) == int(request.chunk_index)
                ]
                if request.force or str(upload.get("asr_status") or "") == "failed":
                    active_asr = _find_active_stream_task(
                        session_id=session_id,
                        task_type="stream_asr",
                        match_fields={"upload_chunk_index": int(request.chunk_index)},
                    )
                    if active_asr is not None:
                        _state_key, task_path, _payload = active_asr
                        upload["asr_task_id"] = task_path.stem
                        upload["asr_task_path"] = str(task_path)
                        existing_tasks.append(str(task_path))
                    else:
                        task_path = enqueue_stream_asr_task(
                            project_root=PROJECT_ROOT,
                            session_id=session_id,
                            stream_id=str(state.get("stream_id") or ""),
                            upload_chunk_id=str(upload.get("upload_chunk_id") or upload.get("chunk_id") or ""),
                            upload_chunk_index=int(request.chunk_index),
                            upload_chunk_path=str(upload.get("path") or ""),
                            processing_chunks=processing_chunks,
                            global_start_time=float(upload.get("stream_start_time", processing_chunks[0].get("start_time", 0.0)) or 0.0),
                            global_end_time=float(upload.get("stream_end_time", processing_chunks[-1].get("end_time", 0.0)) or 0.0),
                            asr_backend=os.getenv("EM2MEM_STREAM_ASR_BACKEND", "whisperx"),
                            reason="retry_chunk",
                            force=True,
                        )
                        upload["asr_status"] = "queued"
                        upload["asr_task_id"] = task_path.stem
                        upload["asr_task_path"] = str(task_path)
                        upload["asr_queued_at"] = now
                        created_tasks.append(str(task_path))
            else:
                if str(upload.get("status") or "") == "processed" and not request.force:
                    return {"status": "already_processed", "created_tasks": [], "existing_tasks": []}
                active_upload = _find_active_stream_task(
                    session_id=session_id,
                    task_type="stream_upload_chunk",
                    match_fields={"upload_chunk_index": int(request.chunk_index)},
                )
                if active_upload is not None:
                    state_key, task_path, _payload = active_upload
                    upload["task_id"] = task_path.stem
                    upload["task_path"] = str(task_path)
                    upload["status"] = "queued" if state_key.endswith("_queued") else "processing" if state_key.endswith("_in_progress") else "processed"
                    existing_tasks.append(str(task_path))
                else:
                    upload["status"] = "queued"
                    upload["error"] = None
                    upload["forced_retry"] = bool(request.force)
                    upload["retry_queued_at"] = now
                    task_path = enqueue_stream_upload_task(
                        project_root=PROJECT_ROOT,
                        session_id=session_id,
                        stream_id=str(state.get("stream_id") or ""),
                        upload_chunk_id=str(upload.get("upload_chunk_id") or upload.get("chunk_id") or ""),
                        upload_chunk_index=int(request.chunk_index),
                        upload_chunk_path=str(upload.get("path") or ""),
                        checksum=str(upload.get("checksum") or ""),
                        force=True,
                    )
                    upload["task_id"] = task_path.stem
                    upload["task_path"] = str(task_path)
                    created_tasks.append(str(task_path))

            new_uploads = []
            for item in upload_chunks:
                if int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == int(request.chunk_index):
                    new_uploads.append(upload)
                else:
                    new_uploads.append(item)
            state["upload_chunks"] = new_uploads
            state["received_chunks"] = new_uploads
            return {"status": "ok", "created_tasks": created_tasks, "existing_tasks": existing_tasks}

        updated_state, retry_result = manager.update_stream_state_locked(_retry_mutation)
        if retry_result.get("error_status"):
            return JSONResponse(
                status_code=int(retry_result.get("error_status") or 500),
                content={
                    "status": retry_result.get("status", "error"),
                    "session_id": session_id,
                    "chunk_index": request.chunk_index,
                    "message": retry_result.get("message"),
                    "frontend_should_reupload_chunk": bool(retry_result.get("frontend_should_reupload_chunk")),
                    "conflict_chunks": retry_result.get("conflict_chunks"),
                },
            )
        if retry_result.get("status") == "already_processed":
            return JSONResponse(
                status_code=200,
                content={"status": "already_processed", "session_id": session_id, "chunk_index": request.chunk_index, "task_created": False},
            )
        created_tasks = [str(path) for path in retry_result.get("created_tasks", []) or []]
        existing_tasks = [str(path) for path in retry_result.get("existing_tasks", []) or []]
        backpressure = compute_backpressure(project_root=PROJECT_ROOT, stream_latency=updated_state.get("latency"))
        response_status = "retry_queued" if created_tasks else "retry_already_queued" if existing_tasks else "nothing_to_retry"
        can_upload_next = str(updated_state.get("status") or "") == "running" and backpressure.get("recommended_action") != "pause_upload"
        return JSONResponse(
            status_code=200,
            content={
                "status": response_status,
                "session_id": session_id,
                "chunk_index": request.chunk_index,
                "forced_retry": bool(request.force),
                "task_created": bool(created_tasks),
                "task_paths": created_tasks,
                "existing_task_paths": existing_tasks,
                "backpressure": backpressure,
                "can_upload_next_chunk": can_upload_next,
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/runtime/clear_old_tasks")
async def clear_old_tasks_api(
    keep_session_id: str,
    x_em2mem_admin_token: Optional[str] = Header(default=None, alias="X-Em2Mem-Admin-Token"),
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    if not _valid_session_id(keep_session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid keep_session_id"})
    if not _admin_token_ok(x_em2mem_admin_token, authorization):
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "message": "runtime admin token required",
                "token_header": "X-Em2Mem-Admin-Token",
            },
        )
    try:
        from online_pipeline.active_session import clear_old_session_tasks, write_active_session

        write_active_session(PROJECT_ROOT, keep_session_id, reason="manual_clear_old_tasks")
        result = clear_old_session_tasks(PROJECT_ROOT, keep_session_id=keep_session_id, reason="manual_clear_old_tasks")
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "keep_session_id": keep_session_id,
                **result,
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/upload_video")
async def upload_video(
    file: Optional[UploadFile] = File(default=None),
) -> JSONResponse:
    if file is None:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "No file uploaded. Expected form field 'file'.",
            },
        )

    session_id = uuid4().hex[:12]
    session_dir = ONLINE_SESSIONS_DIR / session_id
    video_path = session_dir / "input.mp4"
    metadata_path = session_dir / "metadata.json"
    size_bytes = 0

    try:
        session_dir.mkdir(parents=True, exist_ok=False)

        with video_path.open("wb") as output_file:
            while True:
                chunk = await file.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                size_bytes += len(chunk)
                output_file.write(chunk)

        if size_bytes == 0:
            if video_path.exists():
                video_path.unlink()
            session_dir.rmdir()
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Uploaded file is empty."},
            )

        metadata = {
            "session_id": session_id,
            "original_filename": file.filename,
            "saved_video_path": str(video_path),
            "size_bytes": size_bytes,
            "upload_time": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_status(
            session_dir=session_dir,
            session_id=session_id,
            status="uploaded",
            stage="uploaded",
            progress=0,
            error=None,
        )

        preprocess_queued = _env_bool("EM2MEM_AUTO_PREPROCESS", True)
        task_path = None
        if preprocess_queued:
            task_path = enqueue_preprocess_task(
                project_root=PROJECT_ROOT,
                session_id=session_id,
                force=_env_bool("EM2MEM_FORCE_PREPROCESS", False),
            )
            write_status(
                session_dir=session_dir,
                session_id=session_id,
                status="processing",
                stage="preprocess_queued",
                progress=5,
                error=None,
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "uploaded",
                "session_id": session_id,
                "filename": file.filename,
                "video_path": str(video_path),
                "size_bytes": size_bytes,
                "preprocess_queued": preprocess_queued,
                "task_path": str(task_path) if task_path else None,
            },
        )
    except OSError as exc:
        if video_path.exists():
            video_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        if session_dir.exists():
            try:
                session_dir.rmdir()
            except OSError:
                pass
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to save uploaded file: {exc}",
            },
        )
    finally:
        await file.close()
