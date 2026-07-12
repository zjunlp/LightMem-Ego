from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Em2MemOnlineLayout:
    session_dir: Path
    session_id: str

    @property
    def root(self) -> Path:
        return self.session_dir / "em2mem"

    @property
    def caption_root(self) -> Path:
        return self.root / "caption_root"

    @property
    def sidecar_root(self) -> Path:
        return self.root / "sidecar_root"

    @property
    def semantic_root(self) -> Path:
        return self.root / "semantic_root"

    @property
    def visual_root(self) -> Path:
        return self.root / "visual_root"

    @property
    def embeddings_root(self) -> Path:
        return self.root / "embeddings"

    @property
    def indexes_root(self) -> Path:
        return self.root / "indexes"

    @property
    def logs_root(self) -> Path:
        return self.session_dir / "logs"

    @property
    def memory_config_path(self) -> Path:
        return self.root / "memory_config.json"

    @property
    def subject(self) -> str:
        return self.session_id

    @property
    def caption_30sec_path(self) -> Path:
        return self.caption_root / f"{self.subject}_30sec.json"

    @property
    def caption_3min_path(self) -> Path:
        return self.caption_root / f"{self.subject}_3min.json"

    @property
    def caption_10min_path(self) -> Path:
        return self.caption_root / f"{self.subject}_10min.json"

    @property
    def caption_1h_path(self) -> Path:
        return self.caption_root / f"{self.subject}_1h.json"

    @property
    def visual_evidence_path(self) -> Path:
        return self.visual_root / "session_visual_evidence.json"

    @property
    def visual_embedding_path(self) -> Path:
        return self.embeddings_root / "visual_embeddings.pkl"


def ensure_em2mem_layout(layout: Em2MemOnlineLayout) -> None:
    for path in [
        layout.root,
        layout.caption_root,
        layout.sidecar_root,
        layout.sidecar_root / "30s",
        layout.sidecar_root / "3min",
        layout.sidecar_root / "10min",
        layout.sidecar_root / "1h",
        layout.semantic_root,
        layout.visual_root,
        layout.embeddings_root,
        layout.indexes_root,
        layout.logs_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def seconds_to_hhmmssff(seconds: float, fps_for_code: int = 100) -> str:
    total_frames = max(0, int(round(float(seconds) * fps_for_code)))
    total_seconds, frames = divmod(total_frames, fps_for_code)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}{minutes:02d}{secs:02d}{frames:02d}"


def hhmmssff_to_seconds(value: str | float | int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip().zfill(8)
    try:
        hours = int(text[0:2])
        minutes = int(text[2:4])
        seconds = int(text[4:6])
        frames = int(text[6:8])
    except ValueError:
        return 0.0
    return hours * 3600 + minutes * 60 + seconds + frames / 100.0


def rel_to_session(path: Path, session_dir: Path) -> str:
    return path.relative_to(session_dir).as_posix()
