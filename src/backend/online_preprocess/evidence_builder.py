from __future__ import annotations

from pathlib import Path
from typing import Any

from .evidence_schema import build_fallback_payload, normalize_caption_payload
from .io_utils import ensure_dir, read_json, relative_to_session, utc_now_iso, write_json, write_status
from .vlm_captioner import VLMCaptioner, build_vlm_captioner


def _append_log(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] {message}\n")


def _load_inputs(session_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    session_30sec_path = session_dir / "preprocess" / "session_30sec.json"
    segments_path = session_dir / "preprocess" / "segments_30s.json"
    transcript_path = session_dir / "preprocess" / "transcript.json"

    missing = [p for p in [session_30sec_path, segments_path, transcript_path, session_dir / "input.mp4"] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required evidence inputs: " + ", ".join(str(p) for p in missing))

    session_30sec = read_json(session_30sec_path, default=[])
    segments = read_json(segments_path, default=[])
    transcript = read_json(transcript_path, default=[])
    if not isinstance(session_30sec, list) or not isinstance(segments, list) or not isinstance(transcript, list):
        raise ValueError("Evidence inputs must be JSON lists.")
    return session_30sec, segments, transcript


def _merge_segment(base: dict[str, Any], detailed: dict[str, Any] | None) -> dict[str, Any]:
    if not detailed:
        merged = dict(base)
        if not merged.get("segment_id") and merged.get("doc_id"):
            merged["segment_id"] = merged["doc_id"]
        return merged
    merged = dict(base)
    for key in ["segment_id", "keyframes", "transcript_segments", "transcript", "clip_path", "source_video_path"]:
        if detailed.get(key) not in (None, "", []):
            merged[key] = detailed[key]
    if not merged.get("segment_id") and merged.get("doc_id"):
        merged["segment_id"] = merged["doc_id"]
    return merged


def _build_speakers(transcript_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    speakers = []
    for item in transcript_segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        speakers.append(
            {
                "speaker": item.get("speaker"),
                "text": text,
                "time_span": [item.get("start"), item.get("end")],
            }
        )
    return speakers


def _build_evidence_doc(
    session_id: str,
    segment: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    segment_id = str(segment.get("segment_id") or segment.get("doc_id") or "unknown_segment")
    keyframe_paths = list(segment.get("keyframe_paths") or [frame.get("path") for frame in segment.get("keyframes", []) if frame.get("path")])
    transcript_segments = list(segment.get("transcript_segments", []) or [])

    speakers = payload.get("speakers") or _build_speakers(transcript_segments)
    return {
        "doc_id": f"session_{session_id}_{segment_id}",
        "session_id": session_id,
        "segment_id": segment_id,
        "start_time": float(segment.get("start", segment.get("start_time", 0.0)) or 0.0),
        "end_time": float(segment.get("end", segment.get("end_time", 0.0)) or 0.0),
        "duration": float(segment.get("duration", 0.0) or 0.0),
        "source_video_path": segment.get("source_video_path") or segment.get("video_path") or "input.mp4",
        "clip_path": segment.get("clip_path"),
        "transcript": str(segment.get("transcript", "")).strip(),
        "transcript_segments": transcript_segments,
        "fine_caption": payload["fine_caption"],
        "scene": payload["scene"],
        "keyframe_captions": payload["keyframe_captions"],
        "visual_objects": payload["visual_objects"],
        "main_actions": payload["main_actions"],
        "state_changes": payload["state_changes"],
        "conversation_focus": payload["conversation_focus"],
        "speakers": speakers,
        "source_doc_ids": [],
        "keyframe_paths": keyframe_paths,
        "confidence": payload["confidence"],
        "status": "final" if "error" not in payload else "fallback",
        **({"error": payload["error"]} if "error" in payload else {}),
    }


def _build_captioned_entry(segment: dict[str, Any], evidence_doc: dict[str, Any]) -> dict[str, Any]:
    captioned = dict(segment)
    captioned.update(
        {
            "start_time": evidence_doc["start_time"],
            "end_time": evidence_doc["end_time"],
            "video_path": evidence_doc["source_video_path"],
            "caption": evidence_doc["fine_caption"],
            "fine_caption": evidence_doc["fine_caption"],
            "scene": evidence_doc["scene"],
            "keyframe_paths": evidence_doc["keyframe_paths"],
            "keyframe_caption": [
                {
                    "timestamp": item.get("timestamp"),
                    "path": item.get("path"),
                    "caption": item.get("caption"),
                }
                for item in evidence_doc["keyframe_captions"]
            ],
            "visual_objects": evidence_doc["visual_objects"],
            "main_actions": evidence_doc["main_actions"],
            "conversation_focus": evidence_doc["conversation_focus"],
            "evidence_doc_id": evidence_doc["doc_id"],
        }
    )
    return captioned


def build_session_evidence(
    session_id: str,
    sessions_root: Path,
    backend: str,
    model: str | None = None,
    max_keyframes: int = 8,
    force: bool = False,
    limit_segments: int | None = None,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    session_dir = sessions_root / session_id
    evidence_dir = session_dir / "evidence"
    captions_dir = session_dir / "captions"
    logs_dir = session_dir / "logs"
    evidence_path = evidence_dir / "session_evidence.json"
    captioned_path = captions_dir / "session_30sec_captioned.json"
    log_path = logs_dir / "evidence_builder.log"

    ensure_dir(evidence_dir)
    ensure_dir(captions_dir)
    ensure_dir(logs_dir)

    if evidence_path.exists() and captioned_path.exists() and not force and not dry_run:
        return evidence_path, captioned_path

    session_30sec, segments, _ = _load_inputs(session_dir)
    segment_by_id = {str(item.get("segment_id")): item for item in segments}
    merged_segments = [
        _merge_segment(base, segment_by_id.get(str(base.get("segment_id"))))
        for base in session_30sec
    ]
    if limit_segments is not None:
        merged_segments = merged_segments[:limit_segments]

    if dry_run:
        _append_log(log_path, f"dry-run ok: session={session_id} segments={len(merged_segments)}")
        return evidence_path, captioned_path

    write_status(session_dir, session_id, status="processing", stage="evidence_building", progress=70, error=None)
    captioner: VLMCaptioner = build_vlm_captioner(
        backend=backend,
        session_dir=session_dir,
        model=model,
        max_keyframes=max_keyframes,
    )

    evidence_docs = []
    captioned_entries = []
    total = max(len(merged_segments), 1)
    for idx, segment in enumerate(merged_segments):
        progress = 70 + int(15 * idx / total)
        write_status(session_dir, session_id, status="processing", stage="evidence_building", progress=progress, error=None)
        transcript = str(segment.get("transcript", "")).strip()
        keyframe_paths = list(segment.get("keyframe_paths") or [frame.get("path") for frame in segment.get("keyframes", []) if frame.get("path")])

        try:
            payload = captioner.caption_segment(segment=segment, keyframe_paths=keyframe_paths, transcript=transcript)
            payload = normalize_caption_payload(payload, segment)
            _append_log(log_path, f"segment ok: {segment.get('segment_id')}")
        except Exception as exc:
            payload = build_fallback_payload(segment, error=str(exc))
            _append_log(log_path, f"segment fallback: {segment.get('segment_id')} error={exc}")

        evidence_doc = _build_evidence_doc(session_id=session_id, segment=segment, payload=payload)
        evidence_docs.append(evidence_doc)
        captioned_entries.append(_build_captioned_entry(segment, evidence_doc))

    write_json(evidence_path, evidence_docs)
    write_json(captioned_path, captioned_entries)
    write_status(
        session_dir,
        session_id,
        status="processing",
        stage="evidence_done",
        progress=90,
        error=None,
        outputs={
            "session_evidence": relative_to_session(evidence_path, session_dir),
            "session_30sec_captioned": relative_to_session(captioned_path, session_dir),
        },
    )
    return evidence_path, captioned_path
