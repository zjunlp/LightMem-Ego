"""
Multiscale caption generation using LLM summarization.

Adapted from the newer Em2Mem multiscale summarization flow, with light
compatibility helpers for the current evidence JSON format used in this repo.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


_PROMPTS: Dict[str, List[str]] = {}

_PROMPTS["egocentric"] = [
    """As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

Specific Requirements
1. Structure the events clearly.
- Merge related events and keep chronological order.
- Group events by location, task, or theme when appropriate.
2. Retain key information.
- The primary character's ("I") actions and decisions should remain clear.
- Keep important discussions, decisions, and task execution details.
- Preserve the purpose and method of important actions.
3. Be concise and remove redundancies.
- Avoid atmosphere, emotion, or unsupported interpretation.
- Compress long discussions into decisions, plans, and execution details.
4. Adhere strictly to facts.
- Do not add assumptions or unsupported details.
- Keep the actual order of events.

Output Format
- Each paragraph should represent one major event in summary-detail-summary style.
- Output in English only.
- Do not report the word count.""",
    """As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

Specific Requirements
1. Structure the events clearly.
- Merge related events and keep chronological order.
- Group events by location, task, or theme when appropriate.
2. Retain key information.
- The primary character's ("I") actions and decisions should remain clear.
- Keep important discussions, decisions, and task execution details.
- Preserve the purpose and method of important actions.
3. Be concise and remove redundancies.
- Avoid atmosphere, emotion, or unsupported interpretation.
- Compress long discussions into decisions, plans, and execution details.
4. Adhere strictly to facts.
- Do not add assumptions or unsupported details.
- Keep the actual order of events.

Output Format
- Each paragraph should represent one major event in summary-detail-summary style.
- Strictly stay below 300 words in total.
- Output in English only.
- Do not report the word count.""",
    """As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

Specific Requirements
1. Structure the events clearly.
- Merge related events and keep chronological order.
- Group events by location, task, or theme when appropriate.
2. Retain key information.
- The primary character's ("I") actions and decisions should remain clear.
- Keep important discussions, decisions, and task execution details.
- Preserve the purpose and method of important actions.
3. Be concise and remove redundancies.
- Avoid atmosphere, emotion, or unsupported interpretation.
- Compress long discussions into decisions, plans, and execution details.
4. Adhere strictly to facts.
- Do not add assumptions or unsupported details.
- Keep the actual order of events.

Output Format
- Each paragraph should represent one major event in summary-detail-summary style.
- Strictly stay below 1500 words in total.
- Output in English only.
- Do not report the word count.""",
]

_PROMPTS["general"] = [
    """As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information from a video, ensuring that all key actions are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

Specific Requirements
1. Structure the events clearly.
- Merge related events and keep chronological order.
- Group events by location, task, or theme when appropriate.
2. Retain key information.
- Keep all important actions, decisions, and task execution details.
- Preserve the purpose and method of important actions.
3. Be concise and remove redundancies.
- Avoid atmosphere, emotion, or unsupported interpretation.
- Compress long discussions into decisions, plans, and execution details.
4. Adhere strictly to facts.
- Do not add assumptions or unsupported details.
- Keep the actual order of events.

Output Format
- Each paragraph should represent one major event in summary-detail-summary style.
- Output in English only.
- Do not report the word count.""",
    """As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information from a video, ensuring that all key actions are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

Specific Requirements
1. Structure the events clearly.
- Merge related events and keep chronological order.
- Group events by location, task, or theme when appropriate.
2. Retain key information.
- Keep all important actions, decisions, and task execution details.
- Preserve the purpose and method of important actions.
3. Be concise and remove redundancies.
- Avoid atmosphere, emotion, or unsupported interpretation.
- Compress long discussions into decisions, plans, and execution details.
4. Adhere strictly to facts.
- Do not add assumptions or unsupported details.
- Keep the actual order of events.

Output Format
- Each paragraph should represent one major event in summary-detail-summary style.
- Strictly stay below 300 words in total.
- Output in English only.
- Do not report the word count.""",
    """As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information from a video, ensuring that all key actions are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

Specific Requirements
1. Structure the events clearly.
- Merge related events and keep chronological order.
- Group events by location, task, or theme when appropriate.
2. Retain key information.
- Keep all important actions, decisions, and task execution details.
- Preserve the purpose and method of important actions.
3. Be concise and remove redundancies.
- Avoid atmosphere, emotion, or unsupported interpretation.
- Compress long discussions into decisions, plans, and execution details.
4. Adhere strictly to facts.
- Do not add assumptions or unsupported details.
- Keep the actual order of events.

Output Format
- Each paragraph should represent one major event in summary-detail-summary style.
- Strictly stay below 1500 words in total.
- Output in English only.
- Do not report the word count.""",
]


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _ensure_list(values: Any) -> List[Any]:
    if isinstance(values, list):
        return values
    if values is None:
        return []
    return [values]


def _unique_keep_order(values: Sequence[Any], *, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _normalize_day(date: Any, default: str = "DAY1") -> str:
    text = str(date or "").strip()
    return text or default


def _day_sort_key(date: str) -> Tuple[int, str]:
    match = re.search(r"DAY(\d+)", str(date).upper())
    if match:
        return int(match.group(1)), str(date)
    return 10**9, str(date)


def _time_to_seconds(time_val: Any) -> int:
    digits = re.sub(r"\D", "", str(time_val or ""))
    if len(digits) < 6:
        return 0
    if len(digits) == 6:
        hh = int(digits[0:2])
        mm = int(digits[2:4])
        ss = int(digits[4:6])
    else:
        digits = digits[-8:]
        hh = int(digits[0:2])
        mm = int(digits[2:4])
        ss = int(digits[4:6])
    return hh * 3600 + mm * 60 + ss


def _seconds_to_time(seconds: int) -> int:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return hours * 1000000 + minutes * 10000 + secs * 100


def _zfill_time(value: Any) -> str:
    text = re.sub(r"\D", "", str(value or ""))
    return text.zfill(8) if text else ""


def _get_item_text(item: Dict[str, Any]) -> str:
    for key in ("text", "fine_caption", "summary_text", "caption_text"):
        text = str(item.get(key, "")).strip()
        if text:
            return text
    return ""


def _get_item_doc_id(item: Dict[str, Any], level_name: str = "") -> str:
    if item.get("doc_id"):
        return str(item["doc_id"]).strip()
    date = _normalize_day(item.get("date"))
    start_time = _zfill_time(item.get("start_time"))
    end_time = _zfill_time(item.get("end_time"))
    suffix = f"_{level_name}" if level_name else ""
    if date and start_time and end_time:
        return f"{date}_{start_time}_{end_time}{suffix}"
    return f"{level_name or 'node'}_{start_time}_{end_time}"


def _infer_level_name(name: str) -> str:
    text = str(name).strip()
    for candidate in ("30sec", "30s", "3min", "10min", "1h"):
        if re.search(rf"(^|_){re.escape(candidate)}($|_)", text):
            return candidate
    return text


def _sort_items(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            _day_sort_key(_normalize_day(item.get("date"))),
            _time_to_seconds(item.get("start_time")),
            _time_to_seconds(item.get("end_time")),
            _get_item_doc_id(item),
        ),
    )


def _extract_values(value: Any, candidate_keys: Sequence[str]) -> List[str]:
    results: List[str] = []
    for item in _ensure_list(value):
        if isinstance(item, dict):
            for key in candidate_keys:
                raw = item.get(key)
                if raw:
                    results.append(str(raw).strip())
                    break
        else:
            results.append(str(item).strip())
    return [x for x in results if x]


def _collect_thread_values(
    batch: Sequence[Dict[str, Any]],
    primary_field: str,
    fallback_field: str,
    candidate_keys: Sequence[str],
    limit: int = 12,
) -> List[str]:
    values: List[str] = []
    for item in batch:
        raw = item.get(primary_field)
        if raw:
            values.extend(_extract_values(raw, candidate_keys))
            continue
        raw = item.get(fallback_field)
        if raw:
            values.extend(_extract_values(raw, candidate_keys))
    return _unique_keep_order(values, limit=limit)


def _collect_speaker_stats(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    for item in batch:
        speaker_stats = item.get("speaker_stats")
        if speaker_stats:
            for stat in _ensure_list(speaker_stats):
                if isinstance(stat, dict):
                    speaker = str(stat.get("speaker", "")).strip()
                    if speaker:
                        counter[speaker] += int(stat.get("support_units", 1) or 1)
            continue
        for speaker in _extract_values(item.get("speakers", []), ("speaker", "name", "value")):
            counter[speaker] += 1
    return [
        {"speaker": speaker, "support_units": count}
        for speaker, count in counter.most_common()
    ]


def _extract_item_critical_lines(item: Dict[str, Any]) -> List[str]:
    if item.get("critical_speech_lines"):
        return _unique_keep_order(_ensure_list(item.get("critical_speech_lines")))

    raw_entries = item.get("raw_entries")
    if isinstance(raw_entries, list):
        transcript_lines = [
            str(entry.get("text", "")).strip()
            for entry in raw_entries
            if isinstance(entry, dict) and entry.get("type") == "transcript" and str(entry.get("text", "")).strip()
        ]
        lines = _unique_keep_order(transcript_lines)
        if lines:
            return lines

    transcript_text = str(item.get("transcript_text", "")).strip()
    if not transcript_text:
        return []
    pieces = re.split(r"(?<=\.)\s+|(?<=\")\s+|(?<=!)\s+|(?<=\?)\s+", transcript_text)
    return _unique_keep_order(pieces)


def _collect_critical_speech_lines(batch: Sequence[Dict[str, Any]], limit: int) -> List[str]:
    values: List[str] = []
    for item in batch:
        values.extend(_extract_item_critical_lines(item))
    return _unique_keep_order(values, limit=limit)


def _aggregate_scene_summary(batch: Sequence[Dict[str, Any]], limit: int = 5) -> Dict[str, Any]:
    counter: Counter[str] = Counter()
    for item in batch:
        scene_summary = item.get("scene_summary")
        if isinstance(scene_summary, dict):
            dist = scene_summary.get("scene_distribution")
            if isinstance(dist, dict) and dist:
                for scene, score in dist.items():
                    scene_text = str(scene).strip()
                    if not scene_text:
                        continue
                    try:
                        counter[scene_text] += float(score)
                    except (TypeError, ValueError):
                        counter[scene_text] += 1.0
                continue
            dominant_scene = str(scene_summary.get("dominant_scene", "")).strip()
            if dominant_scene:
                counter[dominant_scene] += 1.0
                continue

        scene = str(item.get("scene", "")).strip()
        if scene:
            counter[scene] += 1.0

    if not counter:
        return {"dominant_scene": "", "scene_distribution": {}}

    top_items = counter.most_common(limit)
    total = sum(score for _, score in top_items) or 1.0
    return {
        "dominant_scene": top_items[0][0],
        "scene_distribution": {
            scene: round(score / total, 4)
            for scene, score in top_items
        },
    }


def _collect_visual_objects(batch: Sequence[Dict[str, Any]], limit: int = 10) -> List[str]:
    values: List[str] = []
    for item in batch:
        raw = item.get("visual_object_threads")
        if raw:
            values.extend(_extract_values(raw, ("canonical_label", "object", "label", "value")))
            continue
        raw = item.get("visual_objects")
        if raw:
            values.extend(_extract_values(raw, ("object", "label", "value")))
    return _unique_keep_order(values, limit=limit)


def _build_visual_summary(scene_summary: Dict[str, Any], visual_objects: Sequence[str]) -> str:
    dominant_scene = str(scene_summary.get("dominant_scene", "")).strip()
    top_objects = [str(x).strip() for x in visual_objects if str(x).strip()][:6]
    if dominant_scene and top_objects:
        return (
            f"The visual context is mainly in {dominant_scene}, "
            f"with recurring visible objects such as {', '.join(top_objects)}."
        )
    if dominant_scene:
        return f"The visual context is mainly in {dominant_scene}."
    if top_objects:
        return f"Recurring visible objects include {', '.join(top_objects)}."
    return ""


def _collect_source_doc_ids(batch: Sequence[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for item in batch:
        source_ids = item.get("source_doc_ids")
        if source_ids:
            ids.extend(_extract_values(source_ids, ("doc_id", "value", "id")))
        else:
            ids.append(_get_item_doc_id(item))
    return _unique_keep_order(ids)


def _collect_child_ids(batch: Sequence[Dict[str, Any]]) -> List[str]:
    return [_get_item_doc_id(item) for item in batch]


def _summarize_batch(
    batch: Sequence[Dict[str, Any]],
    window_start: int,
    window_end: int,
    system_message: str,
    llm: Any,
    level_name: str,
    batch_index: int,
) -> Optional[Dict[str, Any]]:
    sorted_batch = _sort_items(batch)
    context = "\n".join(text for text in (_get_item_text(item) for item in sorted_batch) if text)
    if not context:
        return None

    try:
        response = llm.generate(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"All descriptions:\n{context}"},
            ]
        )
    except Exception as exc:
        print(f"Error processing batch {batch_index}: {exc}")
        return None

    if not response:
        return None

    date = _normalize_day(sorted_batch[0].get("date"))
    start_time = str(_seconds_to_time(window_start)).zfill(8)
    end_time = str(_seconds_to_time(window_end)).zfill(8)
    scene_summary = _aggregate_scene_summary(sorted_batch)
    visual_object_threads = _collect_visual_objects(sorted_batch)

    return {
        "doc_id": f"{date}_{start_time}_{end_time}_{level_name}",
        "level": level_name,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "text": str(response).strip(),
        "video_path": str(sorted_batch[0].get("video_path", "")).strip(),
        "action_threads": _collect_thread_values(sorted_batch, "action_threads", "main_actions", ("canonical_label", "action", "label", "value")),
        "object_threads": _collect_thread_values(sorted_batch, "object_threads", "salient_objects", ("canonical_label", "object", "label", "value")),
        "topic_threads": _collect_thread_values(sorted_batch, "topic_threads", "conversation_focus", ("canonical_label", "topic", "label", "value")),
        "speaker_stats": _collect_speaker_stats(sorted_batch),
        "critical_speech_lines": _collect_critical_speech_lines(
            sorted_batch,
            limit=10 if level_name == "3min" else 8,
        ),
        "scene_summary": scene_summary,
        "visual_summary": _build_visual_summary(scene_summary, visual_object_threads),
        "visual_object_threads": visual_object_threads,
        "child_ids": _collect_child_ids(sorted_batch),
        "source_doc_ids": _collect_source_doc_ids(sorted_batch),
    }


def _bucket_by_window(
    input_data: Sequence[Dict[str, Any]],
    window_seconds: int,
    default_date: str = "DAY1",
) -> List[Tuple[int, str, int, int, List[Dict[str, Any]]]]:
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)

    for item in input_data:
        date = _normalize_day(item.get("date"), default=default_date)
        start_sec = _time_to_seconds(item.get("start_time"))
        window_start = (start_sec // window_seconds) * window_seconds
        buckets[(date, window_start)].append(item)

    ordered_keys = sorted(buckets.keys(), key=lambda x: (_day_sort_key(x[0]), x[1]))
    return [
        (idx, date, window_start, window_start + window_seconds, _sort_items(buckets[(date, window_start)]))
        for idx, (date, window_start) in enumerate(ordered_keys)
    ]


def _summarize_level(
    input_data: Sequence[Dict[str, Any]],
    window_seconds: int,
    system_message: str,
    llm: Any,
    level_name: str,
    default_date: str = "DAY1",
) -> List[Dict[str, Any]]:
    window_batches = _bucket_by_window(input_data, window_seconds, default_date=default_date)
    if not window_batches:
        return []

    results_by_index: Dict[int, Dict[str, Any]] = {}
    max_workers = min(8, len(window_batches)) if window_batches else 1

    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(
                _summarize_batch,
                batch,
                window_start,
                window_end,
                system_message,
                llm,
                level_name,
                batch_index,
            ): batch_index
            for batch_index, _date, window_start, window_end, batch in window_batches
        }

        progress_bar = tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc=f"Summarizing {level_name}",
            unit="window",
            leave=False,
        )
        for future in progress_bar:
            batch_index = future_to_index[future]
            result = future.result()
            if result:
                results_by_index[batch_index] = result
        progress_bar.close()

    return [results_by_index[idx] for idx in sorted(results_by_index)]


def gen_multiscale(
    input_json: str,
    save_dir: str,
    llm: Any,
    windows: Sequence[int] = (180, 600, 3600),
    granularity_names: Sequence[str] = ("3min", "10min", "1h"),
    perspective: str = "egocentric",
    default_date: str = "DAY1",
) -> None:
    """
    Generate multiscale event summaries from base captions/evidence docs.

    The first level reads input_json. Each later level reads the previous
    level's output.
    """

    if len(windows) != len(granularity_names):
        raise ValueError("windows and granularity_names must have the same length")

    prompts = _PROMPTS.get(perspective, _PROMPTS["general"])
    os.makedirs(save_dir, exist_ok=True)

    prev_level_data = _load_json(input_json)
    if not isinstance(prev_level_data, list):
        raise ValueError("input_json must be a JSON list")

    prev_level_data = _sort_items(prev_level_data)

    for level, (window_sec, name) in enumerate(zip(windows, granularity_names)):
        output_path = os.path.join(save_dir, f"{name}.json")

        if os.path.exists(output_path):
            try:
                cached = _load_json(output_path)
                if isinstance(cached, list):
                    prev_level_data = _sort_items(cached)
                    print(f"  {name} already exists, skipping: {output_path}")
                    continue
            except json.JSONDecodeError:
                print(f"  {name} is corrupt, regenerating...")

        sys_msg = prompts[min(level, len(prompts) - 1)]
        print(f"  Generating {name} (window={window_sec}s) from {len(prev_level_data)} entries...")
        results = _summarize_level(
            prev_level_data,
            window_sec,
            sys_msg,
            llm,
            level_name=_infer_level_name(name),
            default_date=default_date,
        )
        _save_json(output_path, results)
        print(f"    -> {len(results)} entries written to {output_path}")
        prev_level_data = results
