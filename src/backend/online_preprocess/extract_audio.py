from __future__ import annotations

import wave
from pathlib import Path

from .io_utils import ensure_dir, ffmpeg_bin, run_command


def _write_empty_wav(path: Path, sample_rate: int = 16000) -> None:
    ensure_dir(path.parent)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"")


def extract_audio_wav(
    input_video: Path,
    output_wav: Path,
    has_audio: bool,
    force: bool = False,
    sample_rate: int = 16000,
) -> Path:
    if output_wav.exists() and not force:
        return output_wav

    ensure_dir(output_wav.parent)

    if not has_audio:
        _write_empty_wav(output_wav, sample_rate=sample_rate)
        return output_wav

    command = [
        ffmpeg_bin(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_wav),
    ]
    run_command(command, description="extract audio with ffmpeg")
    return output_wav
