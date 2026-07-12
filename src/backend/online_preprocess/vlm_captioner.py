from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from .evidence_schema import build_fallback_payload, normalize_caption_payload


SYSTEM_PROMPT = """You are converting a 30-second first-person video segment into a grounded evidence document for long-video memory retrieval.

Rules:
1. Only describe what is visible in the keyframes or stated in the transcript.
2. Do not infer hidden intentions.
3. Keep object names concrete.
4. If uncertain, mark confidence lower.
5. Return valid JSON only."""


USER_PROMPT_TEMPLATE = """Inputs:
- Segment id: {segment_id}
- Time range: {start} to {end} seconds
- Transcript: {transcript}
- Keyframes are attached as images. Their timestamps and paths are:
{keyframe_table}

Return JSON with exactly these fields:
fine_caption: string
scene: string or null
keyframe_captions: list of objects with timestamp, path, caption, visible_entities, visual_objects
visual_objects: list of objects with name, attributes, time_span, evidence_keyframes, confidence
main_actions: list of objects with action, actor, objects, time_span, evidence_keyframes, confidence
state_changes: list of objects with entity, attribute, before, after, time_span, confidence
conversation_focus: string or null
speakers: list of objects with speaker, text, time_span
confidence: number between 0 and 1"""


def select_keyframes_for_vlm(keyframes: list[dict[str, Any]], max_keyframes: int) -> list[dict[str, Any]]:
    if max_keyframes <= 0 or len(keyframes) <= max_keyframes:
        return keyframes
    if max_keyframes == 1:
        return [keyframes[0]]
    indexes = [
        round(i * (len(keyframes) - 1) / (max_keyframes - 1))
        for i in range(max_keyframes)
    ]
    selected = []
    seen = set()
    for idx in indexes:
        if idx not in seen:
            selected.append(keyframes[idx])
            seen.add(idx)
    return selected


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
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


ALLOWED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _normalize_reasoning_effort(value: Any) -> str:
    effort = str(value or "none").strip().lower()
    return effort if effort in ALLOWED_REASONING_EFFORTS else "none"


def _reasoning_effort_kwargs() -> dict[str, Any]:
    if os.getenv("EM2MEM_OPENAI_DISABLE_REASONING", "1").strip().lower() in {"1", "true", "yes", "on"}:
        return {"reasoning_effort": "none"}
    effort = os.getenv("EM2MEM_CHAT_REASONING_EFFORT") or os.getenv("EM2MEM_OPENAI_REASONING_EFFORT") or "none"
    return {"reasoning_effort": _normalize_reasoning_effort(effort)}


def _looks_like_unsupported_reasoning_effort(exc: Exception) -> bool:
    text = repr(exc).lower()
    return "reasoning_effort" in text and ("unsupported" in text or "unrecognized" in text or "unknown" in text or "unexpected" in text)


class VLMCaptioner:
    def caption_segment(
        self,
        segment: dict[str, Any],
        keyframe_paths: list[str],
        transcript: str,
    ) -> dict[str, Any]:
        raise NotImplementedError


class MockVLMCaptioner(VLMCaptioner):
    def caption_segment(
        self,
        segment: dict[str, Any],
        keyframe_paths: list[str],
        transcript: str,
    ) -> dict[str, Any]:
        payload = build_fallback_payload(segment)
        payload["fine_caption"] = transcript or f"Segment {segment.get('segment_id')} contains sampled keyframes but no transcript."
        payload["scene"] = None
        payload["confidence"] = 0.2
        return normalize_caption_payload(payload, segment)


class OpenAIVLMCaptioner(VLMCaptioner):
    def __init__(
        self,
        model: str,
        session_dir: Path,
        max_keyframes: int = 8,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAI VLM caption backend.") from exc

        self.model = model
        self.session_dir = session_dir
        self.max_keyframes = max_keyframes
        self.timeout = timeout
        self.temperature = temperature
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"),
            timeout=timeout,
        )

    def caption_segment(
        self,
        segment: dict[str, Any],
        keyframe_paths: list[str],
        transcript: str,
    ) -> dict[str, Any]:
        selected_keyframes = select_keyframes_for_vlm(segment.get("keyframes", []) or [], self.max_keyframes)
        keyframe_table = "\n".join(
            f"- {frame.get('timestamp')}: {frame.get('path')}"
            for frame in selected_keyframes
        )
        user_text = USER_PROMPT_TEMPLATE.format(
            segment_id=segment.get("segment_id"),
            start=segment.get("start"),
            end=segment.get("end"),
            transcript=transcript or "(no transcript)",
            keyframe_table=keyframe_table or "(no keyframes)",
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for frame in selected_keyframes:
            rel_path = str(frame.get("path", "")).strip()
            image_path = self.session_dir / rel_path
            if not rel_path or not image_path.exists():
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_path)},
                }
            )

        request_kwargs = _reasoning_effort_kwargs()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                **request_kwargs,
            )
        except Exception as exc:
            if "reasoning_effort" not in request_kwargs or not _looks_like_unsupported_reasoning_effort(exc):
                raise
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
            )
        raw_text = response.choices[0].message.content or ""
        parsed = _extract_json_object(raw_text)
        return normalize_caption_payload(parsed, segment)


def build_vlm_captioner(
    backend: str,
    session_dir: Path,
    model: str | None = None,
    max_keyframes: int = 8,
) -> VLMCaptioner:
    backend = backend.lower()
    if backend == "mock":
        return MockVLMCaptioner()
    if backend == "openai":
        return OpenAIVLMCaptioner(
            model=model or os.getenv("EM2MEM_VLM_MODEL", "gpt-4o-mini"),
            session_dir=session_dir,
            max_keyframes=max_keyframes,
            timeout=float(os.getenv("EM2MEM_VLM_TIMEOUT", "120")),
            temperature=float(os.getenv("EM2MEM_VLM_TEMPERATURE", "0")),
        )
    if backend == "local":
        raise NotImplementedError("Local VLM backend is reserved for a future implementation.")
    raise ValueError(f"Unsupported VLM caption backend: {backend}")
