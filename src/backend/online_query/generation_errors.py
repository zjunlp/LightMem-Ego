"""Preserve provider diagnostics and distinguish transport from image failures."""
from __future__ import annotations

import os
import re
from typing import Any


class GenerationCallError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def safe_error_text(value: Any) -> str:
    text = str(value)
    for name in ("OPENAI_API_KEY", "EM2MEM_EXTERNAL_OPENAI_API_KEY", "HEYROUTE_API_KEY"):
        key = os.getenv(name)
        if key:
            text = text.replace(key, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", text)
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED]", text)[:1800]


def exception_chain(error: BaseException) -> list[dict[str, Any]]:
    result, seen = [], set()
    while error is not None and id(error) not in seen and len(result) < 8:
        seen.add(id(error))
        item = {"type": type(error).__name__, "message": safe_error_text(error)}
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            item["status_code"] = status
        result.append(item)
        error = error.__cause__ or error.__context__
    return result


def failure_details(error: BaseException) -> dict[str, Any]:
    chain = exception_chain(error)
    names = " ".join(item["type"].lower() for item in chain)
    statuses = {item.get("status_code") for item in chain}
    kind = "generation_error"
    if "timeout" in names:
        kind = "upstream_timeout"
    elif 401 in statuses or "authenticationerror" in names:
        kind = "upstream_authentication"
    elif 403 in statuses or "permissiondenied" in names:
        kind = "upstream_permission"
    elif 429 in statuses or "ratelimiterror" in names:
        kind = "rate_limited"
    elif any(word in names for word in ("connectionerror", "connecterror", "readerror", "writeerror", "protocolerror", "sslerror", "ssleoferror", "connecttimeout")):
        kind = "upstream_connection"
    elif any(isinstance(code, int) and code >= 500 for code in statuses):
        kind = "upstream_service"
    def scrub(value):
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return safe_error_text(value) if isinstance(value, str) else value
    return {"error_type": kind, "error_chain": chain,
            "diagnostics": scrub(getattr(error, "diagnostics", {}))}


def allow_text_fallback(error: BaseException) -> bool:
    # Removing images cannot repair authentication, transport or service errors.
    # Other errors may be image/schema compatibility issues, so retain that path.
    return failure_details(error)["error_type"] == "generation_error"


def mark_answer_failure(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") not in {None, "ok"}:
        return result
    if result.get("generation_status") not in {"local_evidence_summary", "failed"}:
        return result
    raw = result.get("raw") or {}
    debug = raw.get("llm_debug") or raw.get("answer_debug") or {}
    details = debug.get("fallback_failure") or debug.get("primary_failure") or {}
    kind = details.get("error_type", "generation_error")
    messages = {
        "upstream_timeout": "回答服务等待超时，本次未生成有效答案。请稍后重试。",
        "upstream_connection": "连接回答服务时中断，本次未生成有效答案。请稍后重试。",
        "upstream_authentication": "回答服务的凭证无效，请联系管理员检查配置。",
        "upstream_permission": "回答服务拒绝了本次请求，请联系管理员检查权限。",
        "rate_limited": "回答服务暂时繁忙或限流，请稍后重试。",
        "upstream_service": "回答服务暂时异常，本次未生成有效答案。请稍后重试。",
    }
    if result.get("answer"):
        result.setdefault("local_evidence_summary", result["answer"])
    result.update(status="failed", answer="", error_type=kind,
                  message=messages.get(kind, "回答生成失败。已保留检索到的证据，可稍后重试。"),
                  sent_image_count=0, attached_image_count=0)
    return result
