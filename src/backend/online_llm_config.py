from __future__ import annotations

import os
from typing import Any


def chat_response_text(response: Any) -> str:
    """Read SDK objects and JSON/string responses from compatible gateways."""
    import json
    if isinstance(response, str):
        try:
            decoded = json.loads(response)
        except (ValueError, TypeError):
            return response
        if isinstance(decoded, dict) and ("choices" in decoded or "output_text" in decoded):
            return chat_response_text(decoded)
        return response
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = first.get("message", {}) if isinstance(first, dict) else first.message
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    output = response.get("output_text") if isinstance(response, dict) else getattr(response, "output_text", None)
    if isinstance(output, str):
        return output
    raise ValueError("Model response contains no readable message text")


LOCAL_QWEN_PROFILE = "local-qwen35"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def local_llm_enabled() -> bool:
    value = os.getenv("EM2MEM_LOCAL_LLM_ENABLED", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def local_chat_request_kwargs(enabled: bool | None = None) -> dict[str, Any]:
    """Return OpenAI SDK kwargs needed by the local Qwen chat template."""
    if enabled is None:
        enabled = local_llm_enabled()
    if not enabled:
        return {}
    non_thinking = os.getenv("EM2MEM_LOCAL_LLM_NON_THINKING", "1")
    if non_thinking.strip().lower() not in {"1", "true", "yes", "on"}:
        return {}
    return {
        "max_tokens": _env_int("EM2MEM_LOCAL_LLM_MAX_TOKENS", 4096),
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        }
    }


def merge_local_chat_request_kwargs(kwargs: dict[str, Any], enabled: bool | None = None) -> dict[str, Any]:
    merged = dict(kwargs)
    local_kwargs = local_chat_request_kwargs(enabled=enabled)
    if not local_kwargs:
        return merged

    if "max_tokens" not in merged and "max_completion_tokens" not in merged:
        merged["max_tokens"] = local_kwargs["max_tokens"]
    local_extra = dict(local_kwargs.get("extra_body") or {})
    existing_extra = dict(merged.get("extra_body") or {})
    local_template = dict(local_extra.get("chat_template_kwargs") or {})
    existing_template = dict(existing_extra.get("chat_template_kwargs") or {})
    existing_template.update(local_template)
    existing_extra.update(local_extra)
    existing_extra["chat_template_kwargs"] = existing_template
    merged["extra_body"] = existing_extra
    return merged


def local_cache_namespace() -> str:
    if not local_llm_enabled():
        return ""
    value = os.getenv("EM2MEM_LOCAL_LLM_CACHE_NAMESPACE", "local_qwen35")
    return value.strip().replace("-", "_")


def external_openai_api_key() -> str | None:
    return os.getenv("EM2MEM_EXTERNAL_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def external_openai_base_url() -> str | None:
    return os.getenv("EM2MEM_EXTERNAL_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")


def external_openai_model() -> str:
    return os.getenv("EM2MEM_EXTERNAL_OPENAI_MODEL") or os.getenv("EM2MEM_MST_REFINE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.4"


def local_openai_api_key() -> str:
    return os.getenv("EM2MEM_LOCAL_LLM_API_KEY", "local-qwen35")


def local_openai_base_url() -> str:
    return os.getenv(
        "EM2MEM_LOCAL_LLM_BASE_URL",
        f"http://{os.getenv('EM2MEM_LOCAL_LLM_HOST', '127.0.0.1')}:{os.getenv('EM2MEM_LOCAL_LLM_PORT', '18100')}/v1",
    )


def answer_openai_client_kwargs() -> dict[str, Any]:
    """Route answer clients with the same local/remote choice as their profile."""
    local = local_llm_enabled()
    return {
        "api_key": local_openai_api_key() if local else external_openai_api_key(),
        "base_url": local_openai_base_url() if local else external_openai_base_url(),
        "local_mode": local,
    }
