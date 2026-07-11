from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io_utils import OnlinePreprocessError, ensure_dir, relative_to_session


UNIFORM_NUM = 8
CANDIDATE_FPS = 1
MAX_KEYFRAMES = 16
CHANGE_TOPK = 8
MIN_TIME_GAP = 1.5
LOW_CHANGE_THRESHOLD = 0.15
MERGE_TOLERANCE = 0.75


def _uniform_timestamps(duration: float, count: int = UNIFORM_NUM) -> list[float]:
    if duration <= 0:
        return [0.0]
    end = max(duration - 1e-3, 0.0)
    return [round(float(ts), 3) for ts in np.linspace(0.0, end, num=count)]


def _candidate_timestamps(duration: float, fps: int = CANDIDATE_FPS) -> list[float]:
    if duration <= 0:
        return [0.0]
    times = [round(float(ts), 3) for ts in np.arange(0.0, duration, 1.0 / fps)]
    end = round(max(duration - 1e-3, 0.0), 3)
    if not times:
        times = [0.0]
    if abs(times[-1] - end) > 0.5:
        times.append(end)
    return sorted(set(times))


def _open_video(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OnlinePreprocessError(f"Failed to open video for keyframe sampling: {video_path}")
    return cap


def _read_frame(cap: cv2.VideoCapture, timestamp: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(timestamp, 0.0) * 1000.0)
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame
    return None


def _compute_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _compute_phash(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:8, :8]
    median = np.median(low[1:, :])
    return (low > median).astype(np.uint8).flatten()


def _diff_score(prev_frame: np.ndarray, cur_frame: np.ndarray) -> float:
    hist_prev = _compute_histogram(prev_frame)
    hist_cur = _compute_histogram(cur_frame)
    hist_diff = float(cv2.compareHist(hist_prev, hist_cur, cv2.HISTCMP_BHATTACHARYYA))

    phash_prev = _compute_phash(prev_frame)
    phash_cur = _compute_phash(cur_frame)
    phash_diff = float(np.count_nonzero(phash_prev != phash_cur) / phash_prev.size)

    return round(0.6 * hist_diff + 0.4 * phash_diff, 4)


def _select_change_timestamps(
    candidate_frames: list[tuple[float, np.ndarray]],
    topk: int = CHANGE_TOPK,
) -> list[dict[str, Any]]:
    if len(candidate_frames) < 2:
        return []

    scored: list[dict[str, Any]] = []
    for prev, cur in zip(candidate_frames, candidate_frames[1:]):
        score = _diff_score(prev[1], cur[1])
        scored.append({"timestamp": cur[0], "diff_score": score})

    if not scored or max(item["diff_score"] for item in scored) < LOW_CHANGE_THRESHOLD:
        return []

    chosen: list[dict[str, Any]] = []
    for item in sorted(scored, key=lambda x: x["diff_score"], reverse=True):
        if len(chosen) >= topk:
            break
        if any(abs(item["timestamp"] - existing["timestamp"]) < MIN_TIME_GAP for existing in chosen):
            continue
        chosen.append(item)
    return sorted(chosen, key=lambda x: x["timestamp"])


def _merge_samples(
    uniform_timestamps: list[float],
    change_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [
        {
            "timestamp": timestamp,
            "sampling_type": "uniform",
            "diff_score": None,
        }
        for timestamp in uniform_timestamps
    ]

    for change in change_samples:
        timestamp = float(change["timestamp"])
        diff_score = float(change["diff_score"])
        matched = None
        for item in merged:
            if abs(float(item["timestamp"]) - timestamp) <= MERGE_TOLERANCE:
                matched = item
                break
        if matched is not None:
            matched["sampling_type"] = "uniform+change"
            matched["diff_score"] = diff_score
        else:
            merged.append(
                {
                    "timestamp": timestamp,
                    "sampling_type": "change",
                    "diff_score": diff_score,
                }
            )

    merged.sort(key=lambda x: float(x["timestamp"]))
    return merged[:MAX_KEYFRAMES]


def _frame_filename(timestamp: float, used_names: set[str]) -> str:
    base = f"kf_{int(round(timestamp)):06d}"
    candidate = f"{base}.jpg"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    suffix = 1
    while True:
        candidate = f"{base}_{suffix:02d}.jpg"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def sample_keyframes_for_segments(
    session_dir: Path,
    segments: list[dict[str, Any]],
    keyframes_root: Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    ensure_dir(keyframes_root)

    for segment in segments:
        clip_path = session_dir / segment["clip_path"]
        if not clip_path.exists():
            raise OnlinePreprocessError(f"Segment clip missing for keyframe sampling: {clip_path}")

        duration = float(segment["duration"])
        uniform_times = _uniform_timestamps(duration)
        candidate_times = _candidate_timestamps(duration)

        cap = _open_video(clip_path)
        try:
            candidate_frames = []
            for timestamp in candidate_times:
                frame = _read_frame(cap, timestamp)
                if frame is not None:
                    candidate_frames.append((timestamp, frame))
        finally:
            cap.release()

        if not candidate_frames:
            raise OnlinePreprocessError(f"No frames could be read from clip: {clip_path}")

        change_samples = _select_change_timestamps(candidate_frames)
        samples = _merge_samples(uniform_times, change_samples)

        cap = _open_video(clip_path)
        try:
            segment_dir = keyframes_root / segment["segment_id"]
            ensure_dir(segment_dir)
            keyframes: list[dict[str, Any]] = []
            used_names: set[str] = set()
            for sample in samples:
                timestamp = float(sample["timestamp"])
                frame = _read_frame(cap, timestamp)
                if frame is None:
                    continue
                file_name = _frame_filename(timestamp, used_names)
                output_path = segment_dir / file_name
                if force or not output_path.exists():
                    ok = cv2.imwrite(str(output_path), frame)
                    if not ok:
                        raise OnlinePreprocessError(f"Failed to write keyframe image: {output_path}")
                keyframes.append(
                    {
                        "timestamp": round(timestamp, 3),
                        "path": relative_to_session(output_path, session_dir),
                        "sampling_type": sample["sampling_type"],
                        "diff_score": sample["diff_score"],
                    }
                )
        finally:
            cap.release()

        segment["keyframes"] = keyframes

    return segments
