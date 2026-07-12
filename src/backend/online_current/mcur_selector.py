from __future__ import annotations

import re
from typing import Any

from online_current.schemas import as_float, env_int


VISUAL_QUERY_KEYWORDS = [
    "颜色", "外观", "画面", "手里", "拿着", "位置", "左边", "右边", "什么东西", "穿着",
    "color", "appearance", "frame", "scene", "view", "screen", "display", "visible", "shown",
    "showing", "holding", "wearing", "left", "right", "position", "look like", "on screen",
    "on the screen", "current frame", "current scene", "current view", "what do you see",
    "what can you see", "what is visible", "what is shown", "what is on screen",
    "what's on screen", "what is on the screen", "what's on the screen", "what am i seeing",
    "what am i looking at", "describe the scene", "describe current scene", "read the screen",
    "read the current screen", "text on screen", "what is this?", "what's this?",
    "what is that?", "what's that?",
]
SPEECH_QUERY_KEYWORDS = [
    "说了什么", "讲话", "提到", "声音",
    "what did", "what did he say", "what did she say", "what did they say", "what was said",
    "say", "said", "saying", "mentioned", "speech", "conversation", "audio", "voice", "transcript",
]


def _contains_any(text: str, keywords: list[str]) -> bool:
    lower = (text or "").lower()
    for keyword in keywords:
        needle = str(keyword or "").strip().lower()
        if not needle:
            continue
        if needle.isascii():
            prefix = r"(?<![a-z0-9])" if needle[0].isalnum() else ""
            suffix = r"(?![a-z0-9])" if needle[-1].isalnum() else ""
            if re.search(prefix + re.escape(needle) + suffix, lower):
                return True
        elif needle in lower:
            return True
    return False


class MCurFrameSelector:
    def select_frames_for_query(
        self,
        current_context: dict[str, Any],
        question: str,
        max_images: int | None = None,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        max_images = env_int("EM2MEM_MCUR_MAX_QUERY_IMAGES", 3) if max_images is None else int(max_images)
        max_frames = env_int("EM2MEM_MCUR_MAX_EVIDENCE_FRAMES", 5) if max_frames is None else int(max_frames)
        max_images = max(0, max_images)
        max_frames = max(1, max_frames)
        frames = list(current_context.get("frames", []) or [])
        frames.sort(key=lambda item: as_float(item.get("timestamp")))
        if not frames:
            return {
                "selected_frames": [],
                "selected_image_paths_for_mllm": [],
                "evidence_frames": [],
                "selection_reason": "no current frames available",
            }

        q = (question or "").lower()
        is_visual = _contains_any(q, VISUAL_QUERY_KEYWORDS)
        is_speech = _contains_any(q, SPEECH_QUERY_KEYWORDS)
        state = current_context.get("state") or {}
        core_seconds = as_float(state.get("core_seconds"), 10.0)
        current_time = as_float(current_context.get("current_time"), as_float(frames[-1].get("timestamp")))
        core_start = max(0.0, current_time - core_seconds)

        latest = frames[-1]
        core_frames = [frame for frame in frames if as_float(frame.get("timestamp")) >= core_start]
        high_diff = sorted(core_frames or frames, key=lambda item: -as_float(item.get("diff_score")))[: max_frames]
        recent = list(reversed(core_frames or frames))[: max_frames]

        ordered_candidates = [latest]
        if is_visual:
            ordered_candidates.extend(high_diff)
            ordered_candidates.extend(recent)
        elif is_speech:
            ordered_candidates.extend(recent[:2])
            ordered_candidates.extend(high_diff[:1])
        else:
            ordered_candidates.extend(high_diff[:2])
            ordered_candidates.extend(recent)

        selected = self._dedup_by_time(ordered_candidates, max_frames)
        evidence_frames = [
            {
                "path": frame.get("path"),
                "timestamp": round(as_float(frame.get("timestamp")), 3),
                "source_path": frame.get("source_path"),
                "diff_score": frame.get("diff_score"),
                "role": frame.get("role"),
                "source": "M_cur",
            }
            for frame in selected
            if frame.get("path")
        ]
        image_order = sorted(
            evidence_frames,
            key=lambda frame: (
                0 if frame.get("role") == "latest" else 1,
                -as_float(frame.get("diff_score")),
                -as_float(frame.get("timestamp")),
            ),
        )
        selected_image_paths = [str(frame["path"]) for frame in image_order[:max_images] if frame.get("path")]
        reason_parts = ["latest"]
        if is_visual:
            reason_parts.append("visual_attribute_query")
        if is_speech:
            reason_parts.append("speech_query")
        if high_diff:
            reason_parts.append("high_diff")
        return {
            "selected_frames": selected,
            "selected_image_paths_for_mllm": selected_image_paths,
            "evidence_frames": evidence_frames,
            "selection_reason": " + ".join(reason_parts),
        }

    def _dedup_by_time(self, frames: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_seconds: set[int] = set()
        for frame in frames:
            ts = as_float(frame.get("timestamp"))
            sec = int(ts)
            if sec in seen_seconds:
                continue
            if any(abs(ts - as_float(item.get("timestamp"))) < 1.5 for item in selected):
                continue
            selected.append(frame)
            seen_seconds.add(sec)
            if len(selected) >= limit:
                break
        selected.sort(key=lambda item: as_float(item.get("timestamp")))
        return selected
