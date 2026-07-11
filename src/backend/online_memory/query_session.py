from __future__ import annotations

from pathlib import Path
from typing import Any

from online_query.query_engine import query_session as _cached_query_session
from online_retrieval_scheme import normalize_long_term_retrieval_scheme


def query_session(
    session_id: str,
    question: str,
    sessions_root: Path = Path("online_sessions"),
    top_k: int = 5,
    retriever_model: str | None = None,
    respond_model: str | None = None,
    output_json: Path | None = None,
    no_cache: bool = False,
    cache: Any = None,
    use_image_evidence: Any = "auto",
    max_image_frames: int = 4,
    retrieval_mode: str = "auto",
    max_image_evidence: int | None = 3,
    text_top_k: int | None = None,
    visual_top_k: int | None = None,
    final_evidence_k: int | None = None,
    long_term_retrieval_scheme: str | None = None,
    retrieval_scheme: str | None = None,
    stream_handler: Any = None,
) -> dict[str, Any]:
    """Compatibility wrapper for older imports.

    The old implementation spawned eval/query_rag.py for every question.
    The online path now keeps a LoadedQueryEngine in memory and reuses it
    across queries in the same worker/API process.
    """

    del retriever_model, respond_model
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(
        long_term_retrieval_scheme or retrieval_scheme
    )
    return _cached_query_session(
        session_id=session_id,
        question=question,
        sessions_root=sessions_root,
        top_k=top_k,
        no_cache=no_cache,
        cache=cache,
        output_json=output_json,
        retrieval_mode=retrieval_mode,
        use_image_evidence=use_image_evidence,
        max_image_frames=max_image_frames,
        max_image_evidence=max_image_evidence,
        text_top_k=text_top_k,
        visual_top_k=visual_top_k,
        final_evidence_k=final_evidence_k,
        long_term_retrieval_scheme=long_term_retrieval_scheme,
        stream_handler=stream_handler,
    )
