from __future__ import annotations

import json
import os
from typing import Any


STATE_GROUNDING_INSTRUCTION = (
    "For state questions, distinguish directly confirmed presence, directly confirmed absence, "
    "and an unobservable state: not visible does not mean absent. Occluded hands or an object "
    "outside the frame support 'cannot confirm', not 'no longer holding it'. "
    "For NOW, use the latest supplied observation for the requested actor; earlier states are historical references. "
    "Give a complete state sentence instead of an isolated yes/no, and ensure its polarity agrees with the explanation. "
    "Keep posture and activity separate: holding a phone at an ear does not establish a connected call."
)


def _answer_language_instruction() -> str:
    language = str(os.getenv("EM2MEM_ANSWER_LANGUAGE", "zh") or "zh").strip().lower()
    if language in {"zh", "cn", "chinese", "中文", "zh-cn", "zh_hans", "zh-hans", "simplified_chinese"}:
        return "请始终用简体中文回答，不要使用繁体中文。即使证据文本是英文，也要翻译和概括成简体中文；专有名词、文件名、模型名可以保留原文。"
    if language in {"en", "english"}:
        return "Answer in English."
    if language in {"auto", "same", "same_as_question"}:
        return "Answer in the same language as the question."
    return f"Answer in {language}."


def build_current_prompt(question: str, current_context: dict[str, Any], selection: dict[str, Any]) -> str:
    state = current_context.get("state") or {}
    open_event = current_context.get("open_event") or {}
    frames = []
    for frame in selection.get("evidence_frames", []) or []:
        frames.append(
            {
                "timestamp": frame.get("timestamp"),
                "path": frame.get("path"),
                "diff_score": frame.get("diff_score"),
                "role": frame.get("role"),
                "source": frame.get("source"),
            }
        )
    payload = {
        "current_time": state.get("current_time"),
        "window_start_time": state.get("window_start_time"),
        "window_end_time": state.get("window_end_time"),
        "open_event": {
            "start_time": open_event.get("start_time"),
            "end_time": open_event.get("end_time"),
            "duration": open_event.get("duration"),
            "diff_stats": open_event.get("diff_stats"),
            "status": open_event.get("status"),
        },
        "recent_transcript": current_context.get("transcript") or "",
        "selected_frames": frames,
        "selection_reason": selection.get("selection_reason"),
    }
    return (
        "You are answering a current-perception question about a first-person video stream.\n"
        "Use only the current rolling memory below and any attached current frames. "
        "The current memory is a lightweight 30-second rolling buffer and may not contain refined captions.\n"
        "For a question about NOW, ground the current state in the frame labelled latest at current_time. "
        "Verify the premise of the question: naming a bag, phone or action does not establish that it is visible or held. "
        "First check the latest frame for an actual grasp/contact with the requested object; do not substitute a nearby fixed interior object. "
        "If a requested object cannot be identified in the latest frame, state that uncertainty instead of assigning an object category or color. "
        "Older frames are temporal references; an object visible only earlier must not be reported as held now. "
        "Every attached image has its own timestamp label. If the latest image is unclear, say so rather than substituting an earlier state.\n"
        "Describe only what is needed to answer the question. Do not add uncertain vehicle subtypes or object details. "
        "Distinguish visible posture from inferred activity: a phone held at an ear does not prove a connected call.\n"
        f"{STATE_GROUNDING_INSTRUCTION}\n"
        f"{_answer_language_instruction()} Be concise and state uncertainty when the current frames/transcript are insufficient.\n\n"
        f"Question: {question}\n\n"
        f"Current memory package:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
    )


def build_local_current_answer(question: str, current_context: dict[str, Any], selection: dict[str, Any]) -> str:
    del question
    state = current_context.get("state") or {}
    transcript = str(current_context.get("transcript") or "").strip()
    frames = selection.get("evidence_frames", []) or []
    frame_times = ", ".join(str(frame.get("timestamp")) for frame in frames[:5] if frame.get("timestamp") is not None)
    if transcript:
        return f"当前窗口大约是 {state.get('window_start_time')}-{state.get('window_end_time')} 秒；最近转写内容：{transcript}。选中的帧时间：{frame_times}。"
    return f"当前窗口大约是 {state.get('window_start_time')}-{state.get('window_end_time')} 秒；选中的帧时间：{frame_times}。当前没有可用转写。"
