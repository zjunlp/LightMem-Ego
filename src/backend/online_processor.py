from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from online_preprocess import (
    align_transcript_to_segments,
    extract_audio_wav,
    probe_video,
    sample_keyframes_for_segments,
    segment_video_into_clips,
    transcribe_audio_with_whisperx,
    write_empty_transcript_outputs,
    write_worldmm_session_files,
)
from online_preprocess.asr_whisperx import WhisperXRuntime
from online_preprocess.io_utils import OnlinePreprocessError, ensure_dir, write_json, write_status


def process_session(
    session_id: str,
    sessions_root: Path,
    whisperx_model: str,
    device: str,
    compute_type: str,
    language: str | None,
    model_dir: str | None,
    align_model_dir: str | None,
    skip_asr: bool,
    force: bool,
    asr_runtime: WhisperXRuntime | None = None,
) -> Path:
    session_dir = sessions_root / session_id
    input_video = session_dir / "input.mp4"
    preprocess_dir = session_dir / "preprocess"
    video_meta_path = preprocess_dir / "video_meta.json"
    audio_path = preprocess_dir / "audio.wav"
    transcript_srt_path = preprocess_dir / "transcript.srt"
    transcript_json_path = preprocess_dir / "transcript.json"
    clips_dir = preprocess_dir / "clips_30s"
    keyframes_dir = preprocess_dir / "keyframes"
    segments_json_path = preprocess_dir / "segments_30s.json"

    if not input_video.exists():
        raise OnlinePreprocessError(f"Session input video does not exist: {input_video}")

    if force and preprocess_dir.exists():
        shutil.rmtree(preprocess_dir)
    ensure_dir(preprocess_dir)

    current_stage = "uploaded"
    write_status(session_dir, session_id, status="processing", stage=current_stage, progress=0, error=None)

    try:
        current_stage = "probing"
        write_status(session_dir, session_id, status="processing", stage=current_stage, progress=5, error=None)
        video_meta = probe_video(
            input_video=input_video,
            output_json=video_meta_path,
            session_dir=session_dir,
            force=force,
        )

        current_stage = "extract_audio"
        write_status(session_dir, session_id, status="processing", stage=current_stage, progress=15, error=None)
        extract_audio_wav(
            input_video,
            audio_path,
            bool(video_meta["has_audio"]),
            force,
        )

        def run_asr() -> list[dict]:
            if skip_asr or not video_meta["has_audio"]:
                return write_empty_transcript_outputs(
                    output_srt=transcript_srt_path,
                    output_json=transcript_json_path,
                )
            return transcribe_audio_with_whisperx(
                audio_path=audio_path,
                output_srt=transcript_srt_path,
                output_json=transcript_json_path,
                model_name=whisperx_model,
                device=device,
                compute_type=compute_type,
                language=language,
                model_dir=model_dir,
                align_model_dir=align_model_dir,
                force=force,
                runtime=asr_runtime,
            )

        def run_visual_preprocess() -> list[dict]:
            visual_segments = segment_video_into_clips(
                input_video,
                clips_dir,
                session_dir,
                float(video_meta["duration"]),
                force,
            )
            return sample_keyframes_for_segments(
                session_dir=session_dir,
                segments=visual_segments,
                keyframes_root=keyframes_dir,
                force=force,
            )

        current_stage = "asr_visual"
        write_status(session_dir, session_id, status="processing", stage=current_stage, progress=35, error=None)
        with ThreadPoolExecutor(max_workers=2) as executor:
            asr_future = executor.submit(run_asr)
            visual_future = executor.submit(run_visual_preprocess)

            transcript_entries = asr_future.result()
            segments = visual_future.result()

        segments = align_transcript_to_segments(segments, transcript_entries)

        current_stage = "align_outputs"
        write_status(session_dir, session_id, status="processing", stage=current_stage, progress=85, error=None)

        write_json(segments_json_path, segments)
        write_worldmm_session_files(segments=segments, preprocess_dir=preprocess_dir)
        write_status(
            session_dir,
            session_id,
            status="done",
            stage="preprocess_done",
            progress=100,
            error=None,
        )
        return preprocess_dir
    except Exception as exc:
        write_status(
            session_dir,
            session_id,
            status="failed",
            stage=current_stage,
            progress=100,
            error=str(exc),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess an uploaded MP4 session for WorldMM online input.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default="online_sessions")
    parser.add_argument("--whisperx-model", default="large-v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--language", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--align-model-dir", default=None)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dir = process_session(
        session_id=args.session_id,
        sessions_root=Path(args.sessions_root),
        whisperx_model=args.whisperx_model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        model_dir=args.model_dir,
        align_model_dir=args.align_model_dir,
        skip_asr=args.skip_asr,
        force=args.force,
        asr_runtime=None,
    )
    print(f"Preprocess complete: {preprocess_dir}")


if __name__ == "__main__":
    main()
