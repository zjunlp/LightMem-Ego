#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import pysrt
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from em2mem.llm import LLMModel

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
SYSTEM_PROMPT = """You are an expert video captioner.

You will receive a short video segment represented by ordered frames and optional transcript lines with timestamps.
Write a caption describing both the visual content and the audible content of the segment.

Guidelines:
- Describe visible actions, people, objects, and environment.
- Include relevant speech, sounds, or audio events.
- Keep the caption factual and neutral.
- Do not mention frames, timestamps, or that the input came from frames.
- Avoid speculation about emotions or intentions unless clearly visible or stated in speech.

Output only the final caption text."""


@dataclass(slots=True)
class SubtitleEntry:
    start_seconds: float
    end_seconds: float
    start_timestamp: str
    end_timestamp: str
    text: str


@dataclass(slots=True)
class FrameSample:
    timestamp_seconds: float
    image: Image.Image


@dataclass(slots=True)
class Segment:
    start_seconds: float
    end_seconds: float
    transcripts: list[SubtitleEntry]


@dataclass(slots=True)
class VideoReaderContext:
    reader: VideoReader
    average_fps: float
    total_frames: int
    lock: Lock


class CaptionGenerationError(RuntimeError):
    """Raised when a segment caption cannot be generated."""


def discover_transcript_files(transcript_path: Path) -> list[Path]:
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript path not found: {transcript_path}")

    if transcript_path.is_file():
        if transcript_path.suffix.lower() != ".srt":
            raise ValueError(f"Unsupported transcript file: {transcript_path}")
        return [transcript_path]

    transcript_files = sorted(path for path in transcript_path.rglob("*.srt") if path.is_file())
    if not transcript_files:
        raise FileNotFoundError(f"No transcript files found under: {transcript_path}")
    return transcript_files


def build_video_index(video_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    if not video_root.exists():
        raise FileNotFoundError(f"Video path not found: {video_root}")

    if video_root.is_file():
        if video_root.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video file: {video_root}")
        return {}, {video_root.stem: video_root}

    relative_index: dict[str, Path] = {}
    stem_index: dict[str, Path] = {}
    for video_file in sorted(path for path in video_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES):
        relative_key = video_file.relative_to(video_root).with_suffix("").as_posix()
        relative_index[relative_key] = video_file
        stem_index.setdefault(video_file.stem, video_file)

    if not stem_index:
        raise FileNotFoundError(f"No supported video files found under: {video_root}")
    return relative_index, stem_index


def resolve_video_path(
    transcript_file: Path,
    transcript_root: Path,
    video_path: Path,
    relative_index: dict[str, Path],
    stem_index: dict[str, Path],
) -> Path:
    if video_path.is_file():
        if video_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video file: {video_path}")
        return video_path

    if transcript_root.is_file():
        stem_match = stem_index.get(transcript_file.stem)
        if stem_match is None:
            raise FileNotFoundError(f"No matching video found for transcript: {transcript_file}")
        return stem_match

    relative_key = transcript_file.relative_to(transcript_root).with_suffix("").as_posix()
    if relative_key in relative_index:
        return relative_index[relative_key]

    stem_match = stem_index.get(transcript_file.stem)
    if stem_match is None:
        raise FileNotFoundError(f"No matching video found for transcript: {transcript_file}")
    return stem_match


def resolve_output_path(video_file: Path, video_path: Path, transcript_file: Path, transcript_path: Path, output_path: Path, unit_time: int | None = None) -> Path:
    if transcript_path.is_file():
        if output_path.suffix.lower() == ".json":
            return output_path
        if unit_time is not None:
            return output_path / video_file.stem / f"{unit_time}sec.json"
        return output_path / f"{video_file.stem}.json"

    if output_path.suffix:
        raise ValueError("When `--transcript-path` is a directory, `--output-path` must be a directory.")

    if unit_time is not None:
        return output_path / video_file.stem / f"{unit_time}sec.json"

    relative_path = transcript_file.relative_to(transcript_path).with_suffix(".json")
    if video_path.is_dir():
        relative_video_path = video_file.relative_to(video_path).with_suffix(".json")
        if relative_video_path.as_posix() == relative_path.as_posix():
            return output_path / relative_video_path
    return output_path / relative_path


def subrip_time_to_seconds(time_obj: pysrt.SubRipTime) -> float:
    return (
        time_obj.hours * 3600
        + time_obj.minutes * 60
        + time_obj.seconds
        + time_obj.milliseconds / 1000.0
    )


def format_clock(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1_000)
    if millis:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_caption_time(seconds: float, *, round_up: bool) -> str:
    if round_up:
        whole_seconds = max(0, int(math.ceil(seconds - 1e-9)))
    else:
        whole_seconds = max(0, int(math.floor(seconds + 1e-9)))

    hours, rem = divmod(whole_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}{minutes:02d}{secs:02d}00"


def parse_transcript(transcript_file: Path) -> list[SubtitleEntry]:
    subtitles = list(pysrt.open(str(transcript_file)))
    entries: list[SubtitleEntry] = []
    for subtitle in subtitles:
        text = " ".join(line.strip() for line in subtitle.text.splitlines() if line.strip())
        if not text:
            continue

        entries.append(
            SubtitleEntry(
                start_seconds=subrip_time_to_seconds(subtitle.start),
                end_seconds=subrip_time_to_seconds(subtitle.end),
                start_timestamp=str(subtitle.start),
                end_timestamp=str(subtitle.end),
                text=text,
            )
        )
    return entries


def get_video_duration(video_reader: VideoReader) -> tuple[float, float]:
    total_frames = len(video_reader)
    if total_frames <= 0:
        raise ValueError("Video file is empty.")

    average_fps = float(video_reader.get_avg_fps() or 0.0)
    if average_fps <= 0:
        average_fps = 1.0

    duration = total_frames / average_fps
    if duration <= 0:
        duration = 1.0 / average_fps
    return duration, average_fps


def compute_sample_seconds(start_seconds: float, end_seconds: float) -> list[float]:
    sample_seconds = [float(second) for second in range(int(math.floor(start_seconds)), int(math.ceil(end_seconds))) if start_seconds <= second < end_seconds]
    if not sample_seconds:
        sample_seconds = [start_seconds]
    return sample_seconds


def frame_to_image(frame: Any) -> Image.Image:
    image = Image.fromarray(frame)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def sample_segment_frames(video_reader_ctx: VideoReaderContext, start_seconds: float, end_seconds: float) -> list[FrameSample]:
    sample_seconds = compute_sample_seconds(start_seconds, end_seconds)
    frame_indices = [
        min(max(int(round(sample_second * video_reader_ctx.average_fps)), 0), video_reader_ctx.total_frames - 1)
        for sample_second in sample_seconds
    ]

    if not frame_indices:
        sample_seconds = [start_seconds]
        frame_indices = [0]

    with video_reader_ctx.lock:
        frame_batch = video_reader_ctx.reader.get_batch(frame_indices).asnumpy()

    return [
        FrameSample(timestamp_seconds=sample_second, image=frame_to_image(frame))
        for sample_second, frame in zip(sample_seconds, frame_batch, strict=True)
    ]


def build_segments(duration: float, transcript_entries: list[SubtitleEntry], unit_time: int) -> list[Segment]:
    num_segments = max(1, int(math.ceil(duration / unit_time)))
    segments: list[Segment] = []

    for segment_idx in range(num_segments):
        start_seconds = float(segment_idx * unit_time)
        end_seconds = min(float((segment_idx + 1) * unit_time), duration)
        overlapping_transcripts = [
            entry
            for entry in transcript_entries
            if entry.end_seconds > start_seconds and entry.start_seconds < end_seconds
        ]
        segments.append(
            Segment(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                transcripts=overlapping_transcripts,
            )
        )

    return segments


def build_segment_prompt(segment: Segment, frames: list[FrameSample]) -> list[dict[str, Any]]:
    transcript_lines = [
        f"- [{entry.start_timestamp} --> {entry.end_timestamp}] {entry.text}"
        for entry in segment.transcripts
    ]
    transcript_block = "\n".join(transcript_lines) if transcript_lines else "- No transcript lines overlap this segment."

    intro_text = "\n".join(
        [
            f"Segment window: {format_clock(segment.start_seconds)} to {format_clock(segment.end_seconds)}",
            "Transcript lines:",
            transcript_block,
            "The following frames are ordered chronologically within the segment.",
        ]
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": intro_text}]
    for frame in frames:
        content.append({"type": "text", "text": f"Frame timestamp: {format_clock(frame.timestamp_seconds)}"})
        content.append({"type": "image", "image": frame.image})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def generate_caption(segment: Segment, video_reader_ctx: VideoReaderContext, model: LLMModel) -> str:
    frames = sample_segment_frames(video_reader_ctx, segment.start_seconds, segment.end_seconds)
    try:
        response = model.generate(build_segment_prompt(segment, frames))
        return response.strip() if response else ""
    except Exception as exc:
        raise CaptionGenerationError(
            f"Error generating caption for segment {format_clock(segment.start_seconds)} to {format_clock(segment.end_seconds)}: {exc}"
        ) from exc
    finally:
        release_frames(frames)


def build_caption_entry(segment: Segment, video_path: Path, caption_text: str) -> dict[str, str]:
    return {
        "start_time": format_caption_time(segment.start_seconds, round_up=False),
        "end_time": format_caption_time(segment.end_seconds, round_up=True),
        "text": caption_text,
        "date": "DAY1",
        "video_path": video_path.as_posix(),
    }


def release_frames(frames: list[FrameSample]) -> None:
    for frame in frames:
        try:
            frame.image.close()
        except Exception:
            pass


def process_video(
    video_file: Path,
    transcript_file: Path,
    output_file: Path,
    model: LLMModel,
    unit_time: int,
) -> int:
    transcript_entries = parse_transcript(transcript_file)
    video_reader = VideoReader(str(video_file), ctx=cpu(0))
    duration, average_fps = get_video_duration(video_reader)
    video_reader_ctx = VideoReaderContext(
        reader=video_reader,
        average_fps=average_fps,
        total_frames=len(video_reader),
        lock=Lock(),
    )
    segments = build_segments(duration, transcript_entries, unit_time)

    results: list[tuple[int, str]] = []
    generation_errors: list[str] = []
    max_workers = min(64, os.cpu_count() or 1, len(segments)) if segments else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(generate_caption, segment, video_reader_ctx, model): idx
            for idx, segment in enumerate(segments)
        }
        progress = tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc=f"  Captioning {video_file.name}",
            unit="segment",
            leave=False,
        )
        for future in progress:
            idx = future_to_idx[future]
            try:
                caption_text = future.result()
            except Exception as exc:
                tqdm.write(f"      {exc}")
                generation_errors.append(str(exc))
                continue
            results.append((idx, caption_text))

    if generation_errors:
        raise CaptionGenerationError(f"Caption generation failed for {len(generation_errors)} segment(s); output file was not written.")

    ordered_results = sorted(results, key=lambda item: item[0])
    caption_entries = [
        build_caption_entry(segments[idx], video_file, caption_text)
        for idx, caption_text in ordered_results
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(caption_entries, f, indent=2, ensure_ascii=False)

    return len(caption_entries)


def validate_paths(video_path: Path, transcript_path: Path, output_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video path not found: {video_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript path not found: {transcript_path}")

    if video_path.is_file() and transcript_path.is_dir():
        raise ValueError("A single `--video-path` file requires a single `--transcript-path` file.")
    if video_path.is_dir() and transcript_path.is_file() and output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fine captions from videos and transcripts.")
    parser.add_argument("--video-path", type=Path, default=Path("data/Video-MME/data"), help="Path to a video file or a directory of videos.")
    parser.add_argument("--transcript-path", type=Path, default=Path("data/Video-MME/transcript"), help="Path to an `.srt` file or a directory of transcripts.")
    parser.add_argument("--output-path", type=Path, default=Path("data/Video-MME/caption"), help="Path to an existing caption file or directory.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="LLM model name.")
    parser.add_argument("--unit-time", type=int, default=10, help="Segment length in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caption files.")
    args = parser.parse_args()

    if args.unit_time < 1:
        parser.error("--unit-time must be at least 1.")

    validate_paths(args.video_path, args.transcript_path, args.output_path)

    transcript_files = discover_transcript_files(args.transcript_path)
    relative_index, stem_index = build_video_index(args.video_path)
    model = LLMModel(model_name=args.model)

    processed = 0
    skipped = 0
    failed = 0

    progress_bar = tqdm(transcript_files, desc="Generating fine captions", unit="video")
    for transcript_file in progress_bar:
        progress_bar.set_postfix(processed=processed, skipped=skipped, failed=failed, file=transcript_file.name)
        try:
            video_file = resolve_video_path(
                transcript_file=transcript_file,
                transcript_root=args.transcript_path,
                video_path=args.video_path,
                relative_index=relative_index,
                stem_index=stem_index,
            )
            output_file = resolve_output_path(
                video_file=video_file,
                video_path=args.video_path,
                transcript_file=transcript_file,
                transcript_path=args.transcript_path,
                output_path=args.output_path,
                unit_time=args.unit_time,
            )

            if output_file.exists() and not args.overwrite:
                skipped += 1
                continue

            segment_count = process_video(
                video_file=video_file,
                transcript_file=transcript_file,
                output_file=output_file,
                model=model,
                unit_time=args.unit_time,
            )
            processed += 1
            tqdm.write(f"Generated {segment_count} captions for {video_file.name} -> {output_file}")
        except Exception as exc:
            failed += 1
            tqdm.write(f"Failed to process {transcript_file}: {exc}")

    progress_bar.close()
    print(
        "Finished fine caption generation: "
        f"processed={processed} skipped={skipped} failed={failed} output={args.output_path}"
    )


if __name__ == "__main__":
    main()