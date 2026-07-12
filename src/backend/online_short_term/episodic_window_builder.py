from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import utc_now_iso


SYSTEM_PROMPT = """You are building a 30-second episodic memory unit for a long-video memory system.

You are given a chronological list of refined micro-events within this 30-second window. Each micro-event has timestamps,
refined captions, transcripts, actions, objects, entities, and state changes.

Rules:
1. Only use information supported by the refined micro-events and transcript.
2. Do not infer hidden intentions or identities.
3. Use generic entity names when uncertain.
4. Preserve temporal order.
5. Keep source_micro_event_ids unchanged.
6. Return valid JSON only."""


USER_PROMPT_TEMPLATE = """30-second window:
- window_id: {window_id}
- time range: {start_time} to {end_time} seconds

Chronological refined micro-events:
{events_json}

Aggregate these refined micro-events into one coherent 30-second episodic memory. Do not simply concatenate the captions.

Return JSON with exactly these fields:
caption
fine_caption
scene
main_actions
state_changes
visual_objects
entities
transcript_summary
source_micro_event_ids
confidence"""


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in 30s episodic response.")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("30s episodic response JSON is not an object.")
    return obj


ALLOWED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _normalize_reasoning_effort(value: Any) -> str:
    effort = str(value or "none").strip().lower()
    if effort not in ALLOWED_REASONING_EFFORTS:
        return "none"
    return effort


def _reasoning_effort_kwargs() -> dict[str, Any]:
    if os.getenv("EM2MEM_OPENAI_DISABLE_REASONING", "1").strip().lower() in {"1", "true", "yes", "on"}:
        return {"reasoning_effort": "none"}
    effort = os.getenv("EM2MEM_CHAT_REASONING_EFFORT") or os.getenv("EM2MEM_OPENAI_REASONING_EFFORT") or "none"
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


def _caption_for_event(event: dict[str, Any]) -> str:
    for key in ("event_caption_refined", "event_caption_fast", "event_caption_placeholder", "retrieval_text"):
        text = _clean_text(event.get(key))
        if text:
            return text
    return f"A micro-event occurs from {event.get('start_time')}s to {event.get('end_time')}s."


def _dedupe_list(values: list[Any]) -> list[Any]:
    output = []
    seen = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _event_keyframe_paths(event: dict[str, Any]) -> list[str]:
    paths = []
    for frame in event.get("keyframes", []) or []:
        if isinstance(frame, dict) and frame.get("path"):
            paths.append(str(frame["path"]))
    for path in event.get("keyframe_paths", []) or []:
        if path:
            paths.append(str(path))
    return _dedupe_list(paths)


def _event_keyframe_captions(event: dict[str, Any]) -> list[dict[str, Any]]:
    captions = []
    for frame in event.get("keyframes", []) or []:
        if not isinstance(frame, dict):
            continue
        caption = _clean_text(frame.get("caption") or frame.get("keyframe_caption"))
        if caption:
            captions.append(
                {
                    "timestamp": frame.get("timestamp"),
                    "path": frame.get("path"),
                    "caption": caption,
                }
            )
    for item in event.get("keyframe_captions", []) or []:
        if isinstance(item, dict):
            captions.append(item)
        elif _clean_text(item):
            captions.append({"caption": _clean_text(item)})
    return _dedupe_list(captions)


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "status": event.get("status"),
        "refined_caption": event.get("event_caption_refined"),
        "transcript": event.get("transcript"),
        "main_actions": event.get("main_actions", []) or [],
        "state_changes": event.get("state_changes", []) or [],
        "visual_objects": event.get("visual_objects", []) or [],
        "entities": event.get("entities", []) or [],
        "keyframes": [
            {
                "timestamp": frame.get("timestamp"),
                "path": frame.get("path"),
                "role": frame.get("role"),
                "caption": frame.get("caption") or frame.get("keyframe_caption"),
            }
            for frame in event.get("keyframes", []) or []
            if isinstance(frame, dict)
        ],
    }


class MSTEpisodicWindowBuilder:
    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> None:
        self.backend = (backend or os.getenv("EM2MEM_MST_EPISODIC_BACKEND", "openai")).strip().lower()
        self.model = model or os.getenv("EM2MEM_MST_EPISODIC_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.4"
        self.timeout = float(timeout or os.getenv("EM2MEM_MST_EPISODIC_TIMEOUT", "120"))
        self.retries = int(retries if retries is not None else os.getenv("EM2MEM_MST_EPISODIC_RETRIES", "3"))
        self._client = None

    def build_episode(
        self,
        window: dict[str, Any],
        events: list[dict[str, Any]],
        *,
        session_id: str,
        session_dir: Path,
        version: int = 1,
    ) -> dict[str, Any]:
        ordered_events = sorted(events, key=lambda item: (float(item.get("start_time", 0.0)), str(item.get("event_id", ""))))
        if not ordered_events:
            raise ValueError(f"No micro-events for ready window {window.get('window_id')}")
        invalid = [event.get("event_id") for event in ordered_events if event.get("status") not in {"refined", "final"}]
        if invalid:
            raise ValueError(f"Window {window.get('window_id')} contains non-refined events: {invalid}")

        source_ids = [str(event.get("event_id")) for event in ordered_events if event.get("event_id")]
        start = float(window.get("start_time", ordered_events[0].get("start_time", 0.0)) or 0.0)
        end = float(window.get("end_time", ordered_events[-1].get("end_time", start)) or start)
        episode_id = f"mst_ep_{int(round(start)):06d}_{int(round(end)):06d}"
        segment_id = f"seg_{int(round(start)):06d}_{int(round(end)):06d}"

        fallback_used = False
        error_debug = None
        raw_response_preview = None
        backend_used = self.backend
        try:
            if self.backend == "mock":
                payload = self._aggregate_mock(window, ordered_events, source_ids)
            elif self.backend == "rule":
                payload = self._aggregate_rule(window, ordered_events, source_ids)
            elif self.backend == "openai":
                payload, raw_response_preview = self._aggregate_openai(window, ordered_events, source_ids)
            else:
                raise ValueError(f"Unsupported episodic backend: {self.backend}")
        except Exception as exc:
            fallback_used = True
            backend_used = "rule"
            error_debug = f"{type(exc).__name__}: {exc}"
            payload = self._aggregate_rule(window, ordered_events, source_ids)

        payload_source_ids = [str(x) for x in payload.get("source_micro_event_ids", []) or []]
        if set(payload_source_ids) != set(source_ids):
            payload["source_micro_event_ids"] = source_ids

        transcripts = _dedupe_list([_clean_text(event.get("transcript")) for event in ordered_events if _clean_text(event.get("transcript"))])
        transcript_segments = []
        seen_segments = set()
        for event in ordered_events:
            for segment in event.get("transcript_segments", []) or []:
                if not isinstance(segment, dict):
                    continue
                key = (
                    segment.get("start"),
                    segment.get("end"),
                    _clean_text(segment.get("text")),
                    segment.get("speaker"),
                )
                if key in seen_segments:
                    continue
                seen_segments.add(key)
                transcript_segments.append(segment)
        keyframe_paths = _dedupe_list([path for event in ordered_events for path in _event_keyframe_paths(event)])
        keyframe_captions = _dedupe_list([item for event in ordered_events for item in _event_keyframe_captions(event)])
        source_micro_events = [
            {
                "event_id": event.get("event_id"),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "caption": _caption_for_event(event),
                "status": event.get("status"),
                "version": event.get("version"),
            }
            for event in ordered_events
        ]

        now = utc_now_iso()
        return {
            "episode_id": episode_id,
            "session_id": session_id,
            "segment_id": segment_id,
            "start_time": start,
            "end_time": end,
            "duration": round(max(0.0, end - start), 3),
            "status": "complete",
            "version": int(version),
            "caption": _clean_text(payload.get("caption")) or "A refined 30-second episodic memory is available.",
            "fine_caption": _clean_text(payload.get("fine_caption")) or _clean_text(payload.get("caption")),
            "scene": _clean_text(payload.get("scene")),
            "transcript": " ".join(transcripts),
            "transcript_segments": transcript_segments,
            "transcript_summary": _clean_text(payload.get("transcript_summary")),
            "main_actions": payload.get("main_actions", []) if isinstance(payload.get("main_actions"), list) else [],
            "state_changes": payload.get("state_changes", []) if isinstance(payload.get("state_changes"), list) else [],
            "visual_objects": payload.get("visual_objects", []) if isinstance(payload.get("visual_objects"), list) else [],
            "entities": payload.get("entities", []) if isinstance(payload.get("entities"), list) else [],
            "keyframe_paths": keyframe_paths,
            "keyframe_captions": keyframe_captions,
            "source_micro_event_ids": source_ids,
            "source_micro_events": source_micro_events,
            "refined_event_count": sum(1 for event in ordered_events if event.get("status") in {"refined", "final"}),
            "provisional_event_count": sum(1 for event in ordered_events if event.get("status") not in {"refined", "final"}),
            "completeness_score": 1.0,
            "confidence": float(payload.get("confidence", 0.75) or 0.75),
            "source": {
                "type": "mst_micro_event_aggregation",
                "generator": "Stage5B2",
                "backend": backend_used,
            },
            "aggregation_debug": {
                "backend_requested": self.backend,
                "backend_used": backend_used,
                "model": self.model if backend_used == "openai" else None,
                "fallback_used": fallback_used,
                "error_debug": error_debug,
                "raw_response_preview": raw_response_preview,
            },
            "created_at": now,
            "updated_at": now,
        }

    def _aggregate_mock(self, window: dict[str, Any], events: list[dict[str, Any]], source_ids: list[str]) -> dict[str, Any]:
        start = window.get("start_time")
        end = window.get("end_time")
        captions = [_caption_for_event(event) for event in events]
        return {
            "caption": f"Mock 30-second memory from {start}s to {end}s covering {len(events)} refined micro-events.",
            "fine_caption": " ".join(captions),
            "scene": "",
            "main_actions": [],
            "state_changes": [],
            "visual_objects": [],
            "entities": [],
            "transcript_summary": "",
            "source_micro_event_ids": source_ids,
            "confidence": 0.5,
        }

    def _aggregate_rule(self, window: dict[str, Any], events: list[dict[str, Any]], source_ids: list[str]) -> dict[str, Any]:
        captions = []
        actions = []
        states = []
        objects = []
        entities = []
        transcripts = []
        scenes = []
        for event in events:
            captions.append(f"{float(event.get('start_time', 0.0)):.1f}-{float(event.get('end_time', 0.0)):.1f}s: {_caption_for_event(event)}")
            actions.extend(event.get("main_actions", []) or [])
            states.extend(event.get("state_changes", []) or [])
            objects.extend(event.get("visual_objects", []) or [])
            entities.extend(event.get("entities", []) or [])
            if _clean_text(event.get("transcript")):
                transcripts.append(_clean_text(event.get("transcript")))
            if _clean_text(event.get("scene")):
                scenes.append(_clean_text(event.get("scene")))
        fine_caption = " ".join(captions)
        first_caption = _caption_for_event(events[0])
        caption = first_caption if len(events) == 1 else f"{first_caption} The window contains {len(events)} refined micro-events in sequence."
        return {
            "caption": caption,
            "fine_caption": fine_caption,
            "scene": _dedupe_list(scenes)[0] if scenes else "",
            "main_actions": _dedupe_list(actions),
            "state_changes": _dedupe_list(states),
            "visual_objects": _dedupe_list(objects),
            "entities": _dedupe_list(entities),
            "transcript_summary": " ".join(_dedupe_list(transcripts)),
            "source_micro_event_ids": source_ids,
            "confidence": 0.68,
        }

    def _aggregate_openai(
        self,
        window: dict[str, Any],
        events: list[dict[str, Any]],
        source_ids: list[str],
    ) -> tuple[dict[str, Any], str]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for EM2MEM_MST_EPISODIC_BACKEND=openai") from exc
        if self._client is None:
            self._client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"),
                timeout=self.timeout,
            )
        events_json = json.dumps([_event_payload(event) for event in events], ensure_ascii=False, indent=2)
        user_text = USER_PROMPT_TEMPLATE.format(
            window_id=window.get("window_id"),
            start_time=window.get("start_time"),
            end_time=window.get("end_time"),
            events_json=events_json,
        )
        last_exc = None
        for _ in range(max(1, self.retries)):
            try:
                request_kwargs = _reasoning_effort_kwargs()
                try:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_text},
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
                            {"role": "user", "content": user_text},
                        ],
                    )
                raw_text = response.choices[0].message.content or ""
                payload = _extract_json_object(raw_text)
                payload["source_micro_event_ids"] = source_ids
                return payload, raw_text[:1000]
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"OpenAI 30s episodic aggregation failed after {self.retries} attempts: {last_exc}")
