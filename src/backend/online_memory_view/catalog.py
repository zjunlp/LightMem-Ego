from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .repository import LongTermMemoryRepository
from .schemas import (
    MEMORY_CATALOG_CURSOR_VERSION,
    MEMORY_CATALOG_PAGE_SIZE,
    MEMORY_SOURCE,
    SCHEMA_VERSION,
)


class MemoryCatalogError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details: dict[str, Any] = {}


@dataclass(frozen=True)
class MemoryCatalogResult:
    payload: dict[str, Any]


@dataclass(frozen=True)
class _CatalogEntry:
    payload: dict[str, Any]
    timestamp: float | None

    @property
    def sort_key(self) -> tuple[int, float, str]:
        return _sort_key(self.timestamp, str(self.payload["session_id"]))


@dataclass(frozen=True)
class _Cursor:
    updated_at: str | None
    session_id: str
    timestamp: float | None

    @property
    def sort_key(self) -> tuple[int, float, str]:
        return _sort_key(self.timestamp, self.session_id)


class LongTermMemoryCatalogService:
    def __init__(self, sessions_root: Path) -> None:
        self.sessions_root = Path(sessions_root)

    def load(self, cursor: str | None = None) -> MemoryCatalogResult:
        decoded_cursor = _decode_cursor(cursor) if cursor is not None else None
        entries = self._scan()
        entries.sort(key=lambda entry: entry.sort_key)
        total_ready_sessions = len(entries)
        if decoded_cursor is not None:
            entries = [entry for entry in entries if entry.sort_key > decoded_cursor.sort_key]

        candidates = entries[: MEMORY_CATALOG_PAGE_SIZE + 1]
        page = candidates[:MEMORY_CATALOG_PAGE_SIZE]
        has_more = len(candidates) > MEMORY_CATALOG_PAGE_SIZE
        next_cursor = _encode_cursor(page[-1]) if has_more and page else None
        return MemoryCatalogResult(
            payload={
                "schema_version": SCHEMA_VERSION,
                "status": "ok",
                "memory_source": MEMORY_SOURCE,
                "page_size": MEMORY_CATALOG_PAGE_SIZE,
                "total_ready_sessions": total_ready_sessions,
                "has_more": has_more,
                "next_cursor": next_cursor,
                "items": [entry.payload for entry in page],
            }
        )

    def _scan(self) -> list[_CatalogEntry]:
        if not self.sessions_root.exists() or not self.sessions_root.is_dir():
            raise MemoryCatalogError(500, "memory_catalog_failed", "长期记忆目录读取失败")
        try:
            session_dirs = [path for path in self.sessions_root.iterdir() if path.is_dir()]
        except OSError as exc:
            raise MemoryCatalogError(500, "memory_catalog_failed", "长期记忆目录读取失败") from exc

        entries: list[_CatalogEntry] = []
        for session_dir in session_dirs:
            session_id = session_dir.name
            if not _valid_session_id(session_id):
                continue
            try:
                upload_path = session_dir / 'video_upload.json'
                if upload_path.exists():
                    upload = json.loads(upload_path.read_text(encoding='utf-8'))
                    if upload.get('status') != 'ready' or upload.get('query_ready') is not True:
                        continue
                metadata = LongTermMemoryRepository(self.sessions_root, session_id).catalog_metadata()
                updated_at, timestamp = _normalize_config_time(metadata.get("updated_at"))
                metadata["updated_at"] = updated_at
                entries.append(_CatalogEntry(payload=metadata, timestamp=timestamp))
            except Exception:
                # A single malformed, changing, lagging, or unreadable session must
                # not make the global catalog unavailable.
                continue
        return entries


def _valid_session_id(session_id: str) -> bool:
    return bool(session_id) and all(ch.isalnum() or ch in {"-", "_"} for ch in session_id)


def _sort_key(timestamp: float | None, session_id: str) -> tuple[int, float, str]:
    if timestamp is None:
        return (1, 0.0, session_id)
    return (0, -timestamp, session_id)


def _normalize_config_time(value: Any) -> tuple[str | None, float | None]:
    if value is None or not str(value).strip():
        return None, None
    try:
        parsed = _parse_datetime(str(value))
    except ValueError:
        return None, None
    return _canonical_datetime(parsed), parsed.timestamp()


def _parse_datetime(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("empty datetime")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _encode_cursor(entry: _CatalogEntry) -> str:
    raw = json.dumps(
        {
            "v": MEMORY_CATALOG_CURSOR_VERSION,
            "updated_at": entry.payload.get("updated_at"),
            "session_id": entry.payload["session_id"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> _Cursor:
    try:
        if not value or len(value) > 4096 or "=" in value:
            raise ValueError("invalid cursor encoding")
        padding = "=" * (-len(value) % 4)
        decoded = base64.b64decode((value + padding).encode("ascii"), altchars=b"-_", validate=True)
        payload = json.loads(decoded.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"v", "updated_at", "session_id"}:
            raise ValueError("invalid cursor payload")
        version = payload.get("v")
        if isinstance(version, bool) or version != MEMORY_CATALOG_CURSOR_VERSION:
            raise ValueError("invalid cursor version")
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not _valid_session_id(session_id):
            raise ValueError("invalid cursor session")
        updated_at = payload.get("updated_at")
        if updated_at is None:
            timestamp = None
        elif isinstance(updated_at, str):
            parsed = _parse_datetime(updated_at)
            updated_at = _canonical_datetime(parsed)
            timestamp = parsed.timestamp()
        else:
            raise ValueError("invalid cursor datetime")
        return _Cursor(updated_at=updated_at, session_id=session_id, timestamp=timestamp)
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError, binascii.Error) as exc:
        raise MemoryCatalogError(400, "invalid_cursor", "分页游标无效") from exc
