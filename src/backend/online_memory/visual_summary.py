"""Keep image-grounded descriptions separate from narrative and speech text."""
from __future__ import annotations

import re
from typing import Any


# Adapted from the visual annotation principles in:
# https://github.com/zjunlp/LightMem-Ego/blob/main/research/em2mem/preprocess/
# multimodal_memory_cell/build_multimodal_event_record.py
VISUAL_SUMMARY_SYSTEM_PROMPT = """You summarize the visual evidence in a short video window.
Inspect the attached keyframes. Write an independent, concise visual_summary describing the visible
environment, people, objects, appearances, spatial relations and directly observable actions.
The caption and transcript are weak context only: do not copy or paraphrase them as visual evidence.
Do not include speech, plans, hidden intentions, identities or off-screen events unless the images
actually establish the corresponding visible fact. Preserve uncertainty and the clip's perspective;
do not force first-person wording. Use natural English and return valid JSON only:
{"visual_summary": "one to three concise sentences grounded in the attached images",
 "keyframe_captions": [{"timestamp": 0.0, "path": "exact attached path", "caption": "a brief description of this image"}]}.
For keyframe_captions describe only attached frames listed in frames_needing_descriptions.
Use their exact timestamps and paths. Describe each frame's visible state independently;
do not assign the whole event's actions to every frame or infer unseen transitions."""


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""


def _scene_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("dominant_scene", "scene", "location", "name", "label"):
            text = _scene_text(value.get(key))
            if text:
                return text
        return ""
    return _text(value)


def visual_summary_from_evidence(doc: dict[str, Any]) -> str:
    """Preserve independent summaries; legacy fallback uses visual metadata only.

    An explicitly failed image analysis remains empty. In particular, a caption
    or transcript must never silently stand in for an unsuccessful vision call.
    """
    status = doc.get("visual_summary_status")
    if status in {"failed", "unavailable"}:
        return ""
    summary = _text(doc.get("visual_summary"))
    narrative_values = {
        _text(doc.get(key))
        for key in ("text", "caption", "fine_caption", "caption_text", "text_caption", "transcript", "transcript_text")
    }
    if summary and (status in {"generated", "derived"} or summary not in narrative_values):
        return summary

    parts = []
    frame_captions = [_text(doc.get("keyframe_caption"))]
    for frame in doc.get("keyframe_captions", []) or []:
        frame_captions.append(_text(frame.get("caption") or frame.get("keyframe_caption")) if isinstance(frame, dict) else _text(frame))
    parts.extend(dict.fromkeys(text for text in frame_captions if text))
    scene = _scene_text(doc.get("scene")) or _scene_text(doc.get("scene_summary"))
    if scene:
        parts.append(f"Scene: {scene.rstrip('.')}.")
    objects = []
    for obj in doc.get("visual_objects", []) or doc.get("visual_object_threads", []) or []:
        name = _text(obj.get("name") or obj.get("label") or obj.get("object") or obj.get("canonical_label")) if isinstance(obj, dict) else _text(obj)
        if name and name not in objects:
            objects.append(name)
    if objects:
        parts.append(f"Visible objects: {', '.join(objects)}.")
    return " ".join(parts)


def aggregate_visual_summaries(children: list[dict[str, Any]]) -> str:
    summaries = [visual_summary_from_evidence(child) for child in children]
    return " ".join(dict.fromkeys(summary for summary in summaries if summary))


def build_child_visual_evidence(children: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Whitelist the visual observations available for parent summarization."""
    evidence = []
    for child in children:
        summary = visual_summary_from_evidence(child)
        if summary:
            evidence.append({
                "doc_id": child.get("doc_id"),
                "start": child.get("start", child.get("start_time")),
                "end": child.get("end", child.get("end_time")),
                "visual_summary": summary,
            })
    return evidence
