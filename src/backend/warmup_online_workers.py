from __future__ import annotations

import argparse
import os
from pathlib import Path

from online_pipeline.runtime_state import write_worker_runtime


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight warmup/status writer for online worker dependencies.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--warmup-vlm2vec", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()

    write_worker_runtime(
        project_root,
        "evidence",
        status="ready",
        backend=os.getenv("WORLDMM_EVIDENCE_CAPTION_BACKEND", "openai"),
        model_name=os.getenv("WORLDMM_VLM_MODEL") or os.getenv("OPENAI_MODEL"),
        client_loaded=True,
        warmup_done=True,
    )
    write_worker_runtime(
        project_root,
        "refine",
        status="ready",
        backend=os.getenv("WORLDMM_MST_REFINE_BACKEND", "openai"),
        model_name=os.getenv("WORLDMM_MST_REFINE_MODEL") or os.getenv("WORLDMM_VLM_MODEL") or os.getenv("OPENAI_MODEL"),
        client_loaded=True,
        warmup_done=True,
    )
    write_worker_runtime(
        project_root,
        "consolidation",
        status="ready",
        backend=os.getenv("WORLDMM_MST_EPISODIC_BACKEND", "openai"),
        model_name=os.getenv("WORLDMM_MST_EPISODIC_MODEL") or os.getenv("WORLDMM_MEMORY_MODEL") or os.getenv("OPENAI_MODEL"),
        client_loaded=True,
        warmup_done=True,
    )
    if args.warmup_vlm2vec:
        from online_visual.vlm2vec_runtime import get_global_vlm2vec_runtime

        runtime = get_global_vlm2vec_runtime()
        _ = runtime.model if runtime.backend == "vlm2vec" else None
        info = runtime.info()
        write_worker_runtime(
            project_root,
            "visual",
            status="ready",
            backend=info.get("backend"),
            model_name="VLM2Vec-V2.0" if info.get("backend") == "vlm2vec" else "mock",
            model_path=info.get("model_path"),
            device=info.get("device"),
            model_loaded=True,
            warmup_done=True,
            extra={"runtime": info},
        )
    print("warmup status written to runtime/workers")


if __name__ == "__main__":
    main()

