from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, utc_now_iso
from online_short_term.schemas import build_retrieval_text


SYSTEM_PROMPT = """You refine a provisional short-term micro-event for first-person video memory.

Rules:
1. Only describe what is visible in the attached keyframes or stated in transcript.
2. Do not infer hidden intentions or unseen actions.
3. Keep objects concrete and mark uncertainty by lowering confidence.
4. Return valid JSON only."""


USER_PROMPT_TEMPLATE = """Micro-event:
- event_id: {event_id}
- time range: {start_time} to {end_time} seconds
- boundary reason: {boundary_reason}
- transcript: {transcript}
- keyframes:
{keyframe_table}

Return JSON with exactly these fields:
refined_caption: concise grounded caption for this micro-event
visual_objects: list of objects with name, attributes, confidence
main_actions: list of objects with action, actor, objects, time_span, confidence
state_changes: list of objects with entity, attribute, before, after, time_span, confidence
entities: list of concrete entity names
confidence: number between 0 and 1"""


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in VLM response.")
    return json.loads(match.group(0))


def _parse_utc_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _generation_time(event: dict[str, Any]) -> float | None:
    for key in ("available_at", "end_time"):
        value = _safe_float(event.get(key))
        if value is not None:
            return value
    return None


def _stream_start_wall_time(session_dir: Path) -> datetime | None:
    state = read_json(Path(session_dir) / "stream" / "stream_state.json", default={})
    if not isinstance(state, dict):
        return None
    return _parse_utc_datetime(state.get("created_at") or state.get("started_at"))


def _refine_completed_time(
    event: dict[str, Any],
    session_dir: Path,
    completed_at: datetime,
) -> float | None:
    stream_start = _stream_start_wall_time(session_dir)
    if stream_start is not None:
        return max(0.0, (completed_at - stream_start).total_seconds())

    generated_at = _generation_time(event)
    event_created_at = _parse_utc_datetime(event.get("created_at"))
    if generated_at is None or event_created_at is None:
        return None
    return max(generated_at, generated_at + (completed_at - event_created_at).total_seconds())


def _apply_refine_timing(
    event: dict[str, Any],
    session_dir: Path,
    *,
    completed_at_iso: str,
) -> dict[str, Any]:
    item = dict(event)
    completed_at = _parse_utc_datetime(completed_at_iso)
    if completed_at is None:
        return item
    completed_time = _refine_completed_time(item, session_dir, completed_at)
    if completed_time is None:
        return item

    completed_time = round(float(completed_time), 3)
    item["refine_completed_at"] = completed_time
    generated_at = _generation_time(item)
    if generated_at is not None:
        item["refine_speed"] = round(max(0.0, completed_time - float(generated_at)), 3)
    refine = dict(item.get("refine") or {})
    refine["last_refine_completed_at"] = completed_at_iso
    item["refine"] = refine
    return item


def _bounded_refine_timeline(refine: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = refine.get("refine_timeline")
    if not isinstance(timeline, list):
        timeline = []
    timeline = [entry for entry in timeline if isinstance(entry, dict)]
    return timeline[-49:]


def _finish_refine_timeline(
    item: dict[str, Any],
    *,
    attempt: int,
    status: str,
    completed_at: str | None = None,
    failed_at: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    refine = dict(item.get("refine") or {})
    timeline = _bounded_refine_timeline(refine)
    for entry in reversed(timeline):
        if entry.get("attempt") == attempt and entry.get("status") == "running":
            entry["status"] = status
            entry["refine_completed_at"] = completed_at
            entry["refine_failed_at"] = failed_at
            entry["error"] = error
            break
    refine["refine_timeline"] = timeline
    item["refine"] = refine
    return item

ALLOWED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _normalize_reasoning_effort(value: Any) -> str:
    effort = str(value or "none").strip().lower()
    if effort not in ALLOWED_REASONING_EFFORTS:
        return "none"
    return effort


def _reasoning_effort_kwargs() -> dict[str, Any]:
    if os.getenv("WORLDMM_OPENAI_DISABLE_REASONING", "1").strip().lower() in {"1", "true", "yes", "on"}:
        return {"reasoning_effort": "none"}
    effort = os.getenv("WORLDMM_CHAT_REASONING_EFFORT") or os.getenv("WORLDMM_OPENAI_REASONING_EFFORT") or "none"
    return {"reasoning_effort": _normalize_reasoning_effort(effort)}


def _looks_like_unsupported_reasoning(exc: Exception) -> bool:
    text = repr(exc).lower()
    has_reasoning_field = "reasoning_effort" in text or "reasoning" in text
    has_unsupported_signal = (
        "unsupported" in text
        or "unrecognized" in text
        or "unknown" in text
        or "unexpected" in text
    )
    return has_reasoning_field and has_unsupported_signal


def _select_keyframes(event: dict[str, Any], max_images: int) -> list[dict[str, Any]]:
    frames = [frame for frame in event.get("keyframes", []) or [] if isinstance(frame, dict) and frame.get("path")]
    if max_images <= 0 or len(frames) <= max_images:
        return frames
    if max_images == 1:
        return [frames[0]]
    indexes = [round(i * (len(frames) - 1) / (max_images - 1)) for i in range(max_images)]
    selected = []
    seen = set()
    for idx in indexes:
        if idx not in seen:
            selected.append(frames[idx])
            seen.add(idx)
    return selected


class MicroEventRefiner:
    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        max_images: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.backend = (backend or os.getenv("WORLDMM_MST_REFINE_BACKEND", "openai")).strip().lower()
        self.model = model or os.getenv("WORLDMM_MST_REFINE_MODEL") or os.getenv("WORLDMM_VLM_MODEL", "gpt-4o-mini")
        self.max_images = int(max_images or os.getenv("WORLDMM_MST_REFINE_MAX_IMAGES", "4"))
        self.timeout = float(timeout or os.getenv("WORLDMM_MST_REFINE_TIMEOUT", "120"))
        self._client = None

    def refine_event(
        self,
        event: dict[str, Any],
        session_dir: Path,
        *,
        task_id: str | None = None,
        task_queued_at: str | None = None,
        task_worker_started_at: str | None = None,
        task_reason: str | None = None,
    ) -> dict[str, Any]:
        item = dict(event)
        refine = dict(item.get("refine") or {})
        previous_attempts = int(refine.get("refine_attempts", 0) or 0)
        attempt = previous_attempts + 1
        started_at_iso = utc_now_iso()
        refine["refine_attempts"] = attempt
        refine["last_refine_at"] = started_at_iso
        refine["last_refine_queued_at"] = task_queued_at
        refine["last_refine_worker_started_at"] = task_worker_started_at
        refine["last_refine_task_id"] = task_id
        refine["last_refine_task_reason"] = task_reason
        if previous_attempts > 0:
            refine["last_refine_retry_queued_at"] = task_queued_at
        refine["backend"] = self.backend
        timeline = _bounded_refine_timeline(refine)
        timeline.append(
            {
                "attempt": attempt,
                "is_retry": previous_attempts > 0,
                "task_id": task_id,
                "task_reason": task_reason,
                "queued_at": task_queued_at,
                "worker_started_at": task_worker_started_at,
                "refine_started_at": started_at_iso,
                "refine_completed_at": None,
                "refine_failed_at": None,
                "status": "running",
                "error": None,
            }
        )
        refine["refine_timeline"] = timeline
        item["status"] = "captioning"
        item["refine"] = refine
        try:
            payload = self._refine_mock(item) if self.backend == "mock" else self._refine_openai(item, session_dir)
            item = self._apply_payload(item, payload)
            item["status"] = "refined"
            item["caption_source"] = "refined"
            item["needs_refine"] = False
            item["refined_stale"] = False
            item["stale_reason"] = None
            item["version"] = int(item.get("version", 1) or 1) + 1
            completed_at_iso = utc_now_iso()
            item["updated_at"] = completed_at_iso
            refine = dict(item.get("refine") or {})
            refine["last_refine_error"] = None
            refine["backend"] = self.backend
            item["refine"] = refine
            item = _apply_refine_timing(item, session_dir, completed_at_iso=completed_at_iso)
            item = _finish_refine_timeline(item, attempt=attempt, status="refined", completed_at=completed_at_iso)
        except Exception as exc:
            failed_at_iso = utc_now_iso()
            item["status"] = "refine_failed"
            item["updated_at"] = failed_at_iso
            refine = dict(item.get("refine") or {})
            error_text = f"{type(exc).__name__}: {exc}"
            refine["last_refine_failed_at"] = failed_at_iso
            refine["last_refine_error"] = error_text
            refine["backend"] = self.backend
            item["refine"] = refine
            item = _finish_refine_timeline(
                item,
                attempt=attempt,
                status="refine_failed",
                failed_at=failed_at_iso,
                error=error_text,
            )
        item["retrieval_text"] = build_retrieval_text(item)
        return item

    def _refine_mock(self, event: dict[str, Any]) -> dict[str, Any]:
        transcript = str(event.get("transcript") or "").strip()
        time_range = f"{event.get('start_time')} to {event.get('end_time')} seconds"
        if transcript:
            caption = f"During {time_range}, the transcript says: {transcript[:160]}"
        else:
            caption = f"During {time_range}, keyframes show a provisional visual event with boundary reason {event.get('boundary_reason')}."
        return {
            "refined_caption": caption,
            "visual_objects": [],
            "main_actions": [],
            "state_changes": [],
            "entities": [],
            "confidence": 0.45,
        }

    def _refine_openai(self, event: dict[str, Any], session_dir: Path) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for WORLDMM_MST_REFINE_BACKEND=openai") from exc
        if self._client is None:
            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"),
                timeout=self.timeout,
            )
        frames = _select_keyframes(event, self.max_images)
        keyframe_table = "\n".join(f"- {frame.get('timestamp')}: {frame.get('path')}" for frame in frames) or "(no keyframes)"
        user_text = USER_PROMPT_TEMPLATE.format(
            event_id=event.get("event_id"),
            start_time=event.get("start_time"),
            end_time=event.get("end_time"),
            boundary_reason=event.get("boundary_reason"),
            transcript=event.get("transcript") or "(no transcript)",
            keyframe_table=keyframe_table,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for frame in frames:
            image_path = session_dir / str(frame.get("path", ""))
            if image_path.exists():
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(image_path)}})
        request_kwargs = _reasoning_effort_kwargs()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                **request_kwargs,
            )
        except Exception as exc:
            if ("reasoning_effort" not in request_kwargs and "reasoning" not in request_kwargs) or not _looks_like_unsupported_reasoning(exc):
                raise
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
        raw_text = response.choices[0].message.content or ""
        return _extract_json_object(raw_text)

    def _apply_payload(self, event: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        item = dict(event)
        caption = str(payload.get("refined_caption") or payload.get("event_caption_refined") or "").strip()
        if not caption:
            caption = str(item.get("event_caption_placeholder") or "A refined caption could not be generated.").strip()
        item["event_caption_refined"] = caption
        for key in ("visual_objects", "main_actions", "state_changes", "entities"):
            value = payload.get(key, [])
            item[key] = value if isinstance(value, list) else []
        try:
            item["confidence"] = float(payload.get("confidence", item.get("confidence", 0.55)))
        except Exception:
            item["confidence"] = item.get("confidence", 0.55)
        return item
