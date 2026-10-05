#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


@dataclass(slots=True)
class TranscriptionResult:
    video_path: Path
    output_path: Path
    language: str | None
    duration: float | None
    segment_count: int


def discover_video_files(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video file: {input_path}")
        return [input_path]

    video_files = sorted(path for path in input_path.glob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES)
    if not video_files:
        raise FileNotFoundError(f"No supported video files found under: {input_path}")
    return video_files


def resolve_output_path(video_path: Path, input_path: Path, output_path: Path) -> Path:
    if input_path.is_file():
        if output_path.suffix.lower() == ".srt":
            return output_path
        return output_path / f"{video_path.stem}.srt"

    if output_path.suffix:
        raise ValueError("When `--input-path` is a directory, `--output-path` must be a directory.")

    relative_path = video_path.relative_to(input_path).with_suffix(".srt")
    return output_path / relative_path


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def clean_segment_text(text: str) -> str:
    return " ".join(text.strip().split())


def derive_num_workers(batch_size: int) -> int:
    return max(1, min(batch_size, os.cpu_count() or batch_size))


def build_model(model_name: str, batch_size: int, device: str = "auto") -> WhisperModel:
    return WhisperModel(
        model_name,
        device=device,
        num_workers=derive_num_workers(batch_size),
    )


def transcribe_video(model: WhisperModel, video_path: Path, output_path: Path) -> TranscriptionResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments, info = model.transcribe(
        str(video_path),
        vad_filter=True,
        condition_on_previous_text=True,
        word_timestamps=False,
    )

    segment_count = 0
    try:
        with output_path.open("w", encoding="utf-8") as f:
            for segment in segments:
                text = clean_segment_text(segment.text)
                if not text:
                    continue

                segment_count += 1
                f.write(f"{segment_count}\n")
                f.write(
                    f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}\n"
                )
                f.write(f"{text}\n\n")
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise

    return TranscriptionResult(
        video_path=video_path,
        output_path=output_path,
        language=getattr(info, "language", None),
        duration=getattr(info, "duration", None),
        segment_count=segment_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe Video-MME videos into `.srt` files.")
    parser.add_argument("--input-path", type=Path, default=Path("data/Video-MME/data"), help="Path to a video file or a directory of videos.")
    parser.add_argument("--output-path", type=Path, default=Path("data/Video-MME/transcript"), help="Output `.srt` file for a single input video, or transcript directory for video directories.")
    parser.add_argument("--model", type=str, default="distil-large-v3.5", help="faster-whisper model name or local model path.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of video files to transcribe concurrently.")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    input_path = args.input_path
    output_path = args.output_path

    video_files = discover_video_files(input_path)
    num_workers = derive_num_workers(args.batch_size)
    logger.info("Transcription settings: model=%s concurrent_files=%d num_workers=%d", args.model, args.batch_size, num_workers)

    model = build_model(args.model, batch_size=args.batch_size)

    processed = 0
    skipped = 0
    failed = 0

    progress_bar = tqdm(total=len(video_files), desc="Transcribing videos", unit="video")
    pending_jobs: list[tuple[Path, Path]] = []

    for video_path in video_files:
        transcript_path = resolve_output_path(video_path, input_path, output_path)
        progress_bar.set_postfix(file=video_path.name, processed=processed, skipped=skipped, failed=failed)

        if transcript_path.exists():
            skipped += 1
            progress_bar.update(1)
            progress_bar.set_postfix(file=video_path.name, processed=processed, skipped=skipped, failed=failed)
            continue

        pending_jobs.append((video_path, transcript_path))

    max_workers = max(1, min(args.batch_size, len(pending_jobs))) if pending_jobs else 1

    # Share one WhisperModel across concurrent calls; CTranslate2 workers handle parallel generation.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(transcribe_video, model, video_path, transcript_path): (video_path, transcript_path)
            for video_path, transcript_path in pending_jobs
        }

        for future in as_completed(future_to_job):
            video_path, _ = future_to_job[future]
            try:
                future.result()
            except Exception as exc:
                failed += 1
                logger.exception("Failed to transcribe %s: %s", video_path, exc)
            else:
                processed += 1

            progress_bar.update(1)
            progress_bar.set_postfix(file=video_path.name, processed=processed, skipped=skipped, failed=failed)

    progress_bar.close()

    logger.info("Finished transcription: processed=%d skipped=%d failed=%d output=%s", processed, skipped, failed, output_path)


if __name__ == "__main__":
    main()
