from __future__ import annotations

from pathlib import Path
from typing import Any


def empty_evidence_payload(transcript: str = "") -> dict[str, Any]:
    return {
        "fine_caption": transcript or "",
        "scene": None,
        "keyframe_captions": [],
        "visual_objects": [],
        "main_actions": [],
        "state_changes": [],
        "conversation_focus": transcript if transcript else None,
        "speakers": [],
        "confidence": 0.0,
    }


def normalize_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_fallback_payload(segment: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    transcript = str(segment.get("transcript", "")).strip()
    keyframes = segment.get("keyframes", []) or []
    payload = empty_evidence_payload(transcript=transcript)
    payload["fine_caption"] = transcript or "No reliable caption could be generated from this segment."
    payload["keyframe_captions"] = [
        {
            "timestamp": frame.get("timestamp"),
            "path": frame.get("path"),
            "caption": None,
            "visible_entities": [],
            "visual_objects": [],
        }
        for frame in keyframes
    ]
    payload["confidence"] = 0.1 if transcript else 0.0
    if error:
        payload["error"] = error
    return payload


def normalize_caption_payload(payload: dict[str, Any], segment: dict[str, Any]) -> dict[str, Any]:
    fallback = build_fallback_payload(segment)
    normalized = {
        "fine_caption": normalize_string_or_none(payload.get("fine_caption")) or fallback["fine_caption"],
        "scene": normalize_string_or_none(payload.get("scene")),
        "keyframe_captions": normalize_list(payload.get("keyframe_captions")),
        "visual_objects": normalize_list(payload.get("visual_objects")),
        "main_actions": normalize_list(payload.get("main_actions")),
        "state_changes": normalize_list(payload.get("state_changes")),
        "conversation_focus": normalize_string_or_none(payload.get("conversation_focus")),
        "speakers": normalize_list(payload.get("speakers")),
        "confidence": normalize_float(payload.get("confidence"), default=fallback["confidence"]),
    }
    if "error" in payload:
        normalized["error"] = str(payload["error"])
    return normalized


def validate_keyframe_paths(session_dir: Path, keyframe_paths: list[str]) -> list[str]:
    return [path for path in keyframe_paths if (session_dir / path).exists()]
