from __future__ import annotations

import json
import os
from typing import Any


def _answer_language_instruction() -> str:
    language = str(os.getenv("WORLDMM_ANSWER_LANGUAGE", "zh") or "zh").strip().lower()
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
