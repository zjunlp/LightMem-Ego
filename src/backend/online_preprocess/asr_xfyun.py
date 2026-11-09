from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass
from email.utils import formatdate
from pathlib import Path
from typing import Any

import requests

from .io_utils import OnlinePreprocessError, ensure_dir, read_json, utc_now_iso, write_json


UPLOAD_HOST = "upload-ost-api.xfyun.cn"
OST_HOST = "ost-api.xfyun.cn"
SMALL_UPLOAD_URI = "/file/upload"
MPUPLOAD_INIT_URI = "/file/mpupload/init"
MPUPLOAD_UPLOAD_URI = "/file/mpupload/upload"
MPUPLOAD_COMPLETE_URI = "/file/mpupload/complete"
TASK_CREATE_URI = "/v2/ost/pro_create"
TASK_QUERY_URI = "/v2/ost/query"


_NO_SPEECH_ERROR_CODES = {"20304"}


class XfyunASRError(OnlinePreprocessError):
    """Raised when the Xfyun ASR WebAPI cannot produce a transcript."""


class XfyunNoSpeechError(XfyunASRError):
    """Raised when Xfyun reports a completed task with no transcribable speech."""


@dataclass(frozen=True)
class XfyunASRConfig:
    app_id: str
    api_key: str
    api_secret: str
    language: str = "zh_cn"
    accent: str = "mandarin"
    domain: str = "pro_ost_ed"
    audio_format: str = "audio/L16;rate=16000"
    poll_interval_seconds: float = 2.0
    timeout_seconds: float = 120.0
    request_timeout_seconds: float = 30.0
    small_upload_limit_bytes: int = 30 * 1024 * 1024
    upload_chunk_bytes: int = 10 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "XfyunASRConfig":
        app_id = _require_env("EM2MEM_XFYUN_APP_ID")
        api_key = _require_env("EM2MEM_XFYUN_API_KEY")
        api_secret = _require_env("EM2MEM_XFYUN_API_SECRET")
        return cls(
            app_id=app_id,
            api_key=api_key,
            api_secret=api_secret,
            language=os.getenv("EM2MEM_XFYUN_LANGUAGE", "zh_cn").strip() or "zh_cn",
            accent=os.getenv("EM2MEM_XFYUN_ACCENT", "mandarin").strip() or "mandarin",
            domain=os.getenv("EM2MEM_XFYUN_DOMAIN", "pro_ost_ed").strip() or "pro_ost_ed",
            audio_format=os.getenv("EM2MEM_XFYUN_AUDIO_FORMAT", "audio/L16;rate=16000").strip()
            or "audio/L16;rate=16000",
            poll_interval_seconds=_env_float("EM2MEM_XFYUN_POLL_INTERVAL_SECONDS", 2.0),
            timeout_seconds=_env_float("EM2MEM_XFYUN_TIMEOUT_SECONDS", 120.0),
            request_timeout_seconds=_env_float("EM2MEM_XFYUN_REQUEST_TIMEOUT_SECONDS", 30.0),
            small_upload_limit_bytes=_env_int("EM2MEM_XFYUN_SMALL_UPLOAD_LIMIT_BYTES", 30 * 1024 * 1024),
            upload_chunk_bytes=_env_int("EM2MEM_XFYUN_UPLOAD_CHUNK_BYTES", 10 * 1024 * 1024),
        )


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise XfyunASRError(f"Missing required Xfyun ASR environment variable: {name}")
    return value.strip()


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _is_no_speech_error_code(code: Any) -> bool:
    return str(code) in _NO_SPEECH_ERROR_CODES


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


def _body_bytes(body: bytes | str | None) -> bytes:
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    return body.encode("utf-8")


def _sha256_digest(body: bytes | str | None) -> str:
    return "SHA-256=" + base64.b64encode(hashlib.sha256(_body_bytes(body)).digest()).decode("utf-8")


def _json_body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _url(host: str, uri: str) -> str:
    return f"https://{host}{uri}"


def _request_id(prefix: str = "em2mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"[:64]


def _audio_encoding(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    if suffix == ".mp3":
        return "lame"
    return "raw"


def _content_type(audio_path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(audio_path))
    if guessed:
        return guessed
    if audio_path.suffix.lower() == ".wav":
        return "audio/wav"
    if audio_path.suffix.lower() == ".mp3":
        return "audio/mpeg"
    return "application/octet-stream"


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ms_to_seconds(value: Any) -> float:
    numeric = _to_float(value)
    if numeric is None:
        return 0.0
    return max(0.0, numeric / 1000.0)


def _first_word_candidate(ws_item: dict[str, Any]) -> dict[str, Any] | None:
    candidates = ws_item.get("cw")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    return first if isinstance(first, dict) else None


def _normalize_lattice(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            text = result.strip()
            return [{"start": 0.0, "end": 0.0, "text": text, "speaker": None, "confidence": None, "words": []}] if text else []

    if not isinstance(result, dict):
        return []

    lattice = result.get("lattice")
    if not isinstance(lattice, list):
        lattice = result.get("lattice2")
    if not isinstance(lattice, list):
        return []

    segments: list[dict[str, Any]] = []
    for item in lattice:
        if not isinstance(item, dict):
            continue
        st = item.get("json_1best", {}).get("st", {}) if isinstance(item.get("json_1best"), dict) else {}
        if not isinstance(st, dict):
            st = {}

        begin_ms = _to_float(st.get("bg"))
        if begin_ms is None:
            begin_ms = _to_float(item.get("begin")) or 0.0
        end_ms = _to_float(st.get("ed"))
        if end_ms is None:
            end_ms = _to_float(item.get("end")) or begin_ms

        text_parts: list[str] = []
        words: list[dict[str, Any]] = []
        for rt_item in st.get("rt", []) or []:
            if not isinstance(rt_item, dict):
                continue
            for ws_item in rt_item.get("ws", []) or []:
                if not isinstance(ws_item, dict):
                    continue
                cw = _first_word_candidate(ws_item)
                if cw is None:
                    continue
                word_text = str(cw.get("w") or "")
                word_property = str(cw.get("wp") or "")
                if not word_text or word_property == "g":
                    continue
                text_parts.append(word_text)

                word_start_frame = _to_float(ws_item.get("wb"))
                word_end_frame = _to_float(ws_item.get("we"))
                word_start = None
                word_end = None
                if word_start_frame is not None:
                    word_start = round((begin_ms + word_start_frame * 10.0) / 1000.0, 3)
                if word_end_frame is not None:
                    word_end = round((begin_ms + word_end_frame * 10.0) / 1000.0, 3)
                score = _to_float(cw.get("wc"))
                words.append(
                    {
                        "word": word_text,
                        "start": word_start,
                        "end": word_end,
                        "score": score,
                    }
                )

        text = "".join(text_parts).strip()
        if not text:
            continue
        start = round(max(0.0, begin_ms / 1000.0), 3)
        end = round(max(start, (end_ms or begin_ms) / 1000.0), 3)
        confidence = _to_float(st.get("sc"))
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": item.get("spk") or st.get("rl"),
                "confidence": confidence,
                "words": words,
            }
        )

    return sorted(segments, key=lambda segment: (float(segment.get("start") or 0.0), float(segment.get("end") or 0.0)))


class XfyunASRClient:
    def __init__(self, config: XfyunASRConfig | None = None, session: requests.Session | None = None) -> None:
        self.config = config or XfyunASRConfig.from_env()
        self.session = session or requests.Session()
        self.last_task_id: str | None = None

    def _signed_headers(self, host: str, uri: str, body: bytes | str | None, content_type: str, date: str | None = None) -> dict[str, str]:
        request_date = date or formatdate(timeval=None, localtime=False, usegmt=True)
        digest = _sha256_digest(body)
        signature_origin = f"host: {host}\ndate: {request_date}\nPOST {uri} HTTP/1.1\ndigest: {digest}"
        signature = base64.b64encode(
            hmac.new(
                self.config.api_secret.encode("utf-8"),
                signature_origin.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        authorization = (
            f'api_key="{self.config.api_key}",algorithm="hmac-sha256",'
            f'headers="host date request-line digest",signature="{signature}"'
        )
        return {
            "Accept": "application/json",
            "Authorization": authorization,
            "Content-Type": content_type,
            "Date": request_date,
            "Digest": digest,
            "Host": host,
        }

    def _send_prepared(self, prepared: requests.PreparedRequest, action: str) -> dict[str, Any]:
        try:
            response = self.session.send(prepared, timeout=self.config.request_timeout_seconds)
        except requests.RequestException as exc:
            raise XfyunASRError(f"Xfyun ASR {action} request failed: {exc}") from exc

        text = response.text
        payload: dict[str, Any] | None = None
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.status_code != 200:
            if action == "query task" and isinstance(payload, dict) and _is_no_speech_error_code(payload.get("code")):
                raise XfyunNoSpeechError(
                    f"Xfyun ASR {action} reported no-speech condition: code={payload.get('code')} message={payload.get('message')}"
                )
            raise XfyunASRError(f"Xfyun ASR {action} returned HTTP {response.status_code}: {text[:300]}")
        if not isinstance(payload, dict):
            raise XfyunASRError(f"Xfyun ASR {action} returned non-JSON response: {text[:300]}")
        code = payload.get("code")
        if code not in (0, "0", None):
            if action == "query task" and _is_no_speech_error_code(code):
                raise XfyunNoSpeechError(
                    f"Xfyun ASR {action} reported no-speech condition: code={code} message={payload.get('message')}"
                )
            raise XfyunASRError(f"Xfyun ASR {action} failed: code={code} message={payload.get('message')}")
        return payload

    def _prepare_json(self, host: str, uri: str, payload: dict[str, Any]) -> requests.PreparedRequest:
        body = _json_body(payload)
        headers = self._signed_headers(host, uri, body, "application/json")
        request = requests.Request("POST", _url(host, uri), data=body, headers=headers)
        return self.session.prepare_request(request)

    def _post_json(self, host: str, uri: str, payload: dict[str, Any], action: str) -> dict[str, Any]:
        return self._send_prepared(self._prepare_json(host, uri, payload), action)

    def _prepare_multipart(
        self,
        host: str,
        uri: str,
        fields: dict[str, Any],
        filename: str,
        content: bytes,
        content_type: str,
    ) -> requests.PreparedRequest:
        request = requests.Request(
            "POST",
            _url(host, uri),
            data={key: str(value) for key, value in fields.items()},
            files={"data": (filename, content, content_type)},
        )
        prepared = self.session.prepare_request(request)
        body = _body_bytes(prepared.body)
        signed_headers = self._signed_headers(
            host,
            uri,
            body,
            str(prepared.headers.get("Content-Type") or "multipart/form-data"),
        )
        for key, value in signed_headers.items():
            prepared.headers[key] = value
        return prepared

    def _post_multipart(
        self,
        host: str,
        uri: str,
        fields: dict[str, Any],
        filename: str,
        content: bytes,
        content_type: str,
        action: str,
    ) -> dict[str, Any]:
        prepared = self._prepare_multipart(host, uri, fields, filename, content, content_type)
        return self._send_prepared(prepared, action)

    def _small_upload(self, audio_path: Path, request_id: str) -> str:
        payload = self._post_multipart(
            UPLOAD_HOST,
            SMALL_UPLOAD_URI,
            {"app_id": self.config.app_id, "request_id": request_id},
            audio_path.name,
            audio_path.read_bytes(),
            _content_type(audio_path),
            "small upload",
        )
        file_url = payload.get("data", {}).get("url") if isinstance(payload.get("data"), dict) else None
        if not file_url:
            raise XfyunASRError("Xfyun ASR small upload did not return data.url")
        return str(file_url)

    def _multipart_upload(self, audio_path: Path, request_id: str) -> str:
        init_payload = self._post_json(
            UPLOAD_HOST,
            MPUPLOAD_INIT_URI,
            {"app_id": self.config.app_id, "request_id": request_id},
            "multipart upload init",
        )
        upload_id = init_payload.get("data", {}).get("upload_id") if isinstance(init_payload.get("data"), dict) else None
        if not upload_id:
            raise XfyunASRError("Xfyun ASR multipart upload init did not return data.upload_id")

        chunk_size = max(1, int(self.config.upload_chunk_bytes))
        with audio_path.open("rb") as audio_file:
            slice_id = 0
            while True:
                chunk = audio_file.read(chunk_size)
                if not chunk:
                    break
                self._post_multipart(
                    UPLOAD_HOST,
                    MPUPLOAD_UPLOAD_URI,
                    {
                        "app_id": self.config.app_id,
                        "request_id": request_id,
                        "upload_id": upload_id,
                        "slice_id": slice_id,
                    },
                    audio_path.name,
                    chunk,
                    "application/octet-stream",
                    f"multipart upload slice {slice_id}",
                )
                slice_id += 1

        complete_payload = self._post_json(
            UPLOAD_HOST,
            MPUPLOAD_COMPLETE_URI,
            {"app_id": self.config.app_id, "request_id": request_id, "upload_id": upload_id},
            "multipart upload complete",
        )
        file_url = complete_payload.get("data", {}).get("url") if isinstance(complete_payload.get("data"), dict) else None
        if not file_url:
            raise XfyunASRError("Xfyun ASR multipart upload complete did not return data.url")
        return str(file_url)

    def upload_file(self, audio_path: Path) -> str:
        request_id = _request_id("em2mem_upload")
        file_size = audio_path.stat().st_size
        if file_size < self.config.small_upload_limit_bytes:
            return self._small_upload(audio_path, request_id)
        return self._multipart_upload(audio_path, request_id)

    def create_task(self, audio_url: str, audio_path: Path) -> str:
        payload = {
            "common": {"app_id": self.config.app_id},
            "business": {
                "request_id": _request_id("em2mem_task"),
                "language": self.config.language,
                "domain": self.config.domain,
                "accent": self.config.accent,
            },
            "data": {
                "audio_url": audio_url,
                "audio_src": "http",
                "audio_size": audio_path.stat().st_size,
                "format": self.config.audio_format,
                "encoding": _audio_encoding(audio_path),
            },
        }
        response = self._post_json(OST_HOST, TASK_CREATE_URI, payload, "create task")
        task_id = response.get("data", {}).get("task_id") if isinstance(response.get("data"), dict) else None
        if not task_id:
            raise XfyunASRError("Xfyun ASR create task did not return data.task_id")
        return str(task_id)

    def query_task(self, task_id: str) -> dict[str, Any]:
        payload = {"common": {"app_id": self.config.app_id}, "business": {"task_id": task_id}}
        return self._post_json(OST_HOST, TASK_QUERY_URI, payload, "query task")

    def wait_for_result(self, task_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + max(1.0, self.config.timeout_seconds)
        while True:
            response = self.query_task(task_id)
            data = response.get("data")
            if not isinstance(data, dict):
                raise XfyunASRError("Xfyun ASR query response missing data")
            task_status = str(data.get("task_status") or "")
            if task_status in {"3", "4"}:
                result = data.get("result")
                if result is None:
                    raise XfyunASRError("Xfyun ASR completed without result")
                return result
            if task_status and task_status not in {"1", "2"}:
                raise XfyunASRError(f"Xfyun ASR task entered unexpected status: {task_status}")
            if time.monotonic() >= deadline:
                raise XfyunASRError(f"Xfyun ASR task timed out after {self.config.timeout_seconds:.1f}s")
            time.sleep(max(0.1, self.config.poll_interval_seconds))

    def transcribe(self, audio_path: Path) -> list[dict[str, Any]]:
        self.last_empty_transcript_reason = None
        audio_url = self.upload_file(audio_path)
        task_id = self.create_task(audio_url, audio_path)
        self.last_task_id = task_id
        try:
            result = self.wait_for_result(task_id)
        except XfyunNoSpeechError:
            self.last_empty_transcript_reason = "xfyun_no_speech_20304"
            return []
        return _normalize_lattice(result)


def _metadata_path(output_json: Path) -> Path:
    return output_json.with_suffix(output_json.suffix + ".meta.json")


def _is_xfyun_cache(output_json: Path) -> bool:
    metadata = read_json(_metadata_path(output_json), default={})
    return isinstance(metadata, dict) and str(metadata.get("backend") or "").lower() == "xfyun"


def transcribe_audio_with_xfyun(
    audio_path: Path,
    output_srt: Path,
    output_json: Path,
    language: str | None = None,
    force: bool = False,
    client: XfyunASRClient | None = None,
) -> list[dict[str, Any]]:
    if output_json.exists() and output_srt.exists() and not force:
        cached = read_json(output_json, default=[])
        if isinstance(cached, list) and _is_xfyun_cache(output_json):
            return cached

    if not audio_path.exists():
        raise OnlinePreprocessError(f"Audio file does not exist: {audio_path}")

    active_client = client or XfyunASRClient()
    empty_transcript_reason: str | None = None
    try:
        normalized = active_client.transcribe(audio_path)
        empty_transcript_reason = getattr(active_client, "last_empty_transcript_reason", None)
    except XfyunNoSpeechError:
        normalized = []
        empty_transcript_reason = "xfyun_no_speech_20304"
    if not normalized and empty_transcript_reason is None:
        empty_transcript_reason = "xfyun_empty_transcript"
    _write_srt(output_srt, normalized)
    write_json(output_json, normalized)
    metadata = {
        "backend": "xfyun",
        "provider": "iflytek",
        "xfyun_task_id": active_client.last_task_id,
        "audio_path": str(audio_path),
        "segment_count": len(normalized),
        "created_at": utc_now_iso(),
    }
    if empty_transcript_reason:
        metadata["empty_transcript_reason"] = empty_transcript_reason
    write_json(_metadata_path(output_json), metadata)
    return normalized
