from __future__ import annotations

import os


DEFAULT_LONG_TERM_RETRIEVAL_SCHEME = "em2memory"
SUPPORTED_LONG_TERM_RETRIEVAL_SCHEMES = ("em2memory", "worldmm_legacy")

_SCHEME_ALIASES = {
    "em2memory": "em2memory",
    "em2memory_new": "em2memory",
    "em2_memory": "em2memory",
    "dense": "em2memory",
    "dense_rag": "em2memory",
    "event_dense": "em2memory",
    "worldmemory_dense": "em2memory",
    "legacy": "worldmm_legacy",
    "worldmm_legacy": "worldmm_legacy",
    "worldmm_original": "worldmm_legacy",
    "original": "worldmm_legacy",
    "hipporag": "worldmm_legacy",
    "hippo_rag": "worldmm_legacy",
}


def normalize_long_term_retrieval_scheme(value: str | None = None) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"", "auto", "default"}:
        env_value = (
            os.getenv("WORLDMM_LONG_TERM_RETRIEVAL_SCHEME")
            or os.getenv("WORLDMM_RETRIEVAL_SCHEME")
            or ""
        )
        env_raw = env_value.strip().lower().replace("-", "_")
        if env_raw and env_raw not in {"auto", "default"}:
            return normalize_long_term_retrieval_scheme(env_value)
        return DEFAULT_LONG_TERM_RETRIEVAL_SCHEME
    if raw in _SCHEME_ALIASES:
        return _SCHEME_ALIASES[raw]
    supported = ", ".join(SUPPORTED_LONG_TERM_RETRIEVAL_SCHEMES)
    aliases = ", ".join(sorted(_SCHEME_ALIASES))
    raise ValueError(
        f"unsupported long-term retrieval scheme: {value!r}; "
        f"supported schemes: {supported}; aliases: {aliases}"
    )


def retrieval_scheme_cache_key(session_id: str, long_term_retrieval_scheme: str | None = None) -> str:
    scheme = normalize_long_term_retrieval_scheme(long_term_retrieval_scheme)
    return f"{session_id}::ltr={scheme}"
