from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path
from typing import Any

from .io_utils import OnlinePreprocessError, ensure_dir, read_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHISPERX_MODEL_DIR = PROJECT_ROOT / "models" / "whisperx"
DEFAULT_WHISPERX_ALIGN_MODEL_DIR = DEFAULT_WHISPERX_MODEL_DIR / "alignment"


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1_000
    millis = total_ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        lines.extend(
            [
                str(idx),
                f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}",
                text,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_empty_transcript_outputs(output_srt: Path, output_json: Path) -> list[dict[str, Any]]:
    ensure_dir(output_srt.parent)
    output_srt.write_text("", encoding="utf-8")
    write_json(output_json, [])
    return []


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _default_model_dir() -> Path:
    return Path(os.getenv("EM2MEM_WHISPERX_MODEL_DIR", str(DEFAULT_WHISPERX_MODEL_DIR)))


def _default_align_model_dir() -> Path:
    return Path(os.getenv("EM2MEM_WHISPERX_ALIGN_MODEL_DIR", str(DEFAULT_WHISPERX_ALIGN_MODEL_DIR)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_asr_model_name_or_path(model_name: str, model_dir: str | None) -> str:
    root = Path(model_dir) if model_dir else _default_model_dir()
    systran_name = f"faster-whisper-{model_name}"
    ct2_dir = root / systran_name
    if ct2_dir.exists() and (ct2_dir / "model.bin").exists():
        return str(ct2_dir)

    candidates = [
        root / model_name,
        root / f"models--Systran--{systran_name}" / "snapshots",
    ]

    for candidate in candidates:
        if candidate.exists():
            if (candidate / "model.bin").exists():
                return str(candidate)
            # WhisperX expects faster-whisper-style directories with model.bin.
            # If the local directory looks like a Hugging Face Whisper checkpoint,
            # fall back to the plain model name so WhisperX can resolve its own cache.
            return model_name

    # Support Hugging Face snapshot-style caches copied under models/whisperx.
    snapshot_root = candidates[2]
    if snapshot_root.exists():
        snapshots = sorted([p for p in snapshot_root.iterdir() if p.is_dir()])
        if snapshots:
            return str(snapshots[-1])

    return model_name


def _resolve_align_model_name_or_path(language: str, align_model_dir: str | None) -> str | None:
    root = Path(align_model_dir) if align_model_dir else _default_align_model_dir()
    language_key = language.lower()
    candidate_names = {
        "en": ["wav2vec2-large-960h-lv60-self", "wav2vec2-base-960h"],
        "zh": ["wav2vec2-large-xlsr-53-chinese-zh-cn"],
        "cn": ["wav2vec2-large-xlsr-53-chinese-zh-cn"],
    }.get(language_key, [])

    for name in candidate_names:
        candidate = root / name
        if candidate.exists():
            return str(candidate)
    return None


def _transcribe_with_language(model: Any, audio: Any, batch_size: int, language: str | None) -> dict[str, Any]:
    if language:
        try:
            return model.transcribe(audio, batch_size=batch_size, language=language)
        except TypeError:
            pass
    return model.transcribe(audio, batch_size=batch_size)


def _normalize_word(word: dict[str, Any]) -> dict[str, Any]:
    return {
        "word": str(word.get("word") or word.get("text") or "").strip(),
        "start": float(word["start"]) if word.get("start") is not None else None,
        "end": float(word["end"]) if word.get("end") is not None else None,
        "score": float(word["score"]) if word.get("score") is not None else None,
    }


def _normalize_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in raw_segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        words = [
            normalized_word
            for normalized_word in (_normalize_word(word) for word in segment.get("words", []) or [])
            if normalized_word["word"]
        ]
        normalized.append(
            {
                "start": float(segment.get("start") or 0.0),
                "end": float(segment.get("end") or 0.0),
                "text": text,
                "speaker": None,
                "confidence": None,
                "words": words,
            }
        )
    return normalized


class WhisperXRuntime:
    """Keep WhisperX ASR and alignment models warm inside a long-lived process."""

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "auto",
        compute_type: str = "auto",
        model_dir: str | None = None,
        align_model_dir: str | None = None,
        preload_align_languages: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        resolved_device = _resolve_device(device)
        self.device = resolved_device.split(":", 1)[0] if resolved_device.startswith("cuda:") else resolved_device
        self.device_index = int(resolved_device.split(":", 1)[1]) if resolved_device.startswith("cuda:") and ":" in resolved_device else 0
        self.compute_type = _resolve_compute_type(compute_type, self.device)
        self.model_dir = str(Path(model_dir) if model_dir else _default_model_dir())
        self.align_model_dir = str(Path(align_model_dir) if align_model_dir else _default_align_model_dir())
        self.preload_align_languages = preload_align_languages or []
        self.whisperx: Any | None = None
        self.asr_model: Any | None = None
        self.align_models: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.RLock()

    def load(self) -> None:
        with self._lock:
            self.load_asr_only()
            for language in self.preload_align_languages:
                self.get_align_model(language)

    def get_align_model(self, language: str | None) -> tuple[Any, Any] | None:
        if not language:
            return None
        with self._lock:
            self.load_asr_only()
            assert self.whisperx is not None

            key = language.lower()
            if key not in self.align_models:
                align_kwargs: dict[str, Any] = {
                    "language_code": key,
                    "device": self.device,
                    "model_dir": self.align_model_dir,
                }
                local_align_model = _resolve_align_model_name_or_path(key, self.align_model_dir)
                if local_align_model:
                    align_kwargs["model_name"] = local_align_model
                elif not _env_bool("EM2MEM_WHISPERX_ALLOW_ALIGN_DOWNLOAD", False):
                    return None
                self.align_models[key] = self.whisperx.load_align_model(**align_kwargs)
            return self.align_models[key]

    def load_asr_only(self) -> None:
        with self._lock:
            if self.whisperx is None:
                try:
                    import whisperx
                except ImportError as exc:
                    raise OnlinePreprocessError(
                        "whisperx is not installed. Install it before running ASR."
                    ) from exc
                self.whisperx = whisperx

            if self.asr_model is None:
                model_name_or_path = _resolve_asr_model_name_or_path(self.model_name, self.model_dir)
                self.asr_model = self.whisperx.load_model(
                    model_name_or_path,
                    device=self.device,
                    device_index=self.device_index,
                    compute_type=self.compute_type,
                    download_root=self.model_dir,
                )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self.load_asr_only()
            assert self.whisperx is not None
            assert self.asr_model is not None

            audio = self.whisperx.load_audio(str(audio_path))
            transcription = _transcribe_with_language(self.asr_model, audio, batch_size=batch_size, language=language)
            raw_segments = transcription.get("segments", []) or []

            aligned_segments = raw_segments
            align_language = transcription.get("language") or language
            if raw_segments and align_language:
                try:
                    align_bundle = self.get_align_model(align_language)
                    if align_bundle is not None:
                        align_model, metadata = align_bundle
                        aligned = self.whisperx.align(
                            raw_segments,
                            align_model,
                            metadata,
                            audio,
                            self.device,
                            return_char_alignments=False,
                        )
                        aligned_segments = aligned.get("segments", raw_segments) or raw_segments
                except Exception:
                    aligned_segments = raw_segments

            return _normalize_segments(aligned_segments)


def transcribe_audio_with_whisperx(
    audio_path: Path,
    output_srt: Path,
    output_json: Path,
    model_name: str = "large-v3",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    model_dir: str | None = None,
    align_model_dir: str | None = None,
    batch_size: int = 16,
    force: bool = False,
    runtime: WhisperXRuntime | None = None,
) -> list[dict[str, Any]]:
    if output_json.exists() and output_srt.exists() and not force:
        cached = read_json(output_json, default=[])
        if isinstance(cached, list):
            return cached

    if not audio_path.exists():
        raise OnlinePreprocessError(f"Audio file does not exist: {audio_path}")

    if runtime is None:
        runtime = WhisperXRuntime(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            model_dir=model_dir,
            align_model_dir=align_model_dir,
        )

    normalized = runtime.transcribe(audio_path=audio_path, language=language, batch_size=batch_size)
    _write_srt(output_srt, normalized)
    write_json(output_json, normalized)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WhisperX ASR and export SRT/JSON.")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--output-srt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-name", default="large-v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--language", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--align-model-dir", default=None)
    args = parser.parse_args()

    transcribe_audio_with_whisperx(
        audio_path=Path(args.audio_path),
        output_srt=Path(args.output_srt),
        output_json=Path(args.output_json),
        model_name=args.model_name,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        model_dir=args.model_dir,
        align_model_dir=args.align_model_dir,
    )


if __name__ == "__main__":
    main()
