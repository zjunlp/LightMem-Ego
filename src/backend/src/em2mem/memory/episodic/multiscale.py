"""
Multiscale episodic memory generation.

This version removes the old Chroma/RagAgent/gen_event dependency chain and
uses the local em2mem-native gen_multiscale implementation instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

load_dotenv()

def _resolve_runtime_dependencies():
    try:
        from ...llm import LLMModel
        from .gen_multiscale import gen_multiscale
        return LLMModel, gen_multiscale
    except ImportError:
        current_dir = Path(__file__).resolve().parent
        src_root = Path(__file__).resolve().parents[3]
        for path in (str(current_dir), str(src_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from em2mem.llm import LLMModel
        from gen_multiscale import gen_multiscale
        return LLMModel, gen_multiscale


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _materialize_base_30sec(input_json: str, output_json: str) -> str:
    if os.path.abspath(input_json) == os.path.abspath(output_json):
        return input_json

    data = _load_json(input_json)
    if not isinstance(data, list):
        raise ValueError("json_path must point to a JSON list")
    _save_json(output_json, data)
    return output_json


def generate_multiscale_memory(
    db_name: str = "A1_JAKE_EVIDENCE",
    person_name: str = "A1_JAKE",
    json_path: str = "data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_evidence.json",
    diary_dir: str = ".cache/events_diary",
    save_path: str = "data/EgoLife/EgoLifeCap",
    model_name: str = os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    windows: Optional[List[int]] = None,
    granularity_names: Optional[List[str]] = None,
    perspective: str = "egocentric",
) -> None:
    """
    Generate 3min/10min/1h multiscale summaries from a 30sec evidence JSON.

    Args:
        db_name: Output folder name and filename prefix.
        person_name: Kept for backward compatibility with existing scripts.
        json_path: Path to the base 30sec/evidence JSON.
        diary_dir: Kept for backward compatibility; no longer used.
        save_path: Parent directory that will contain <db_name>/ outputs.
        model_name: LLM model name for summarization.
        windows: Window sizes in seconds for each multiscale level.
        granularity_names: Output filenames without .json.
        perspective: "egocentric" or "general".
    """

    _ = person_name, diary_dir

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Input json not found: {json_path}")

    windows = windows or [180, 600, 3600]
    granularity_names = granularity_names or [
        f"{db_name}_3min",
        f"{db_name}_10min",
        f"{db_name}_1h",
    ]

    if len(windows) != len(granularity_names):
        raise ValueError("windows and granularity_names must have the same length")

    output_dir = os.path.join(save_path, db_name)
    os.makedirs(output_dir, exist_ok=True)

    base_30sec_json = os.path.join(output_dir, f"{db_name}_30sec.json")
    base_input_json = _materialize_base_30sec(json_path, base_30sec_json)

    LLMModel, gen_multiscale = _resolve_runtime_dependencies()
    llm = LLMModel(model_name=model_name)
    gen_multiscale(
        input_json=base_input_json,
        save_dir=output_dir,
        llm=llm,
        windows=windows,
        granularity_names=granularity_names,
        perspective=perspective,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiscale episodic memory.")
    parser.add_argument("--db_name", default="A1_JAKE_EVIDENCE")
    parser.add_argument("--person_name", default="A1_JAKE")
    parser.add_argument(
        "--json_path",
        default="data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_evidence.json",
    )
    parser.add_argument(
        "--diary_dir",
        default=".cache/events_diary",
        help="Unused. Kept only for backward compatibility with existing scripts.",
    )
    parser.add_argument("--save_path", default="data/EgoLife/EgoLifeCap")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--windows", default="180,600,3600")
    parser.add_argument(
        "--granularity_names",
        default="",
        help="Comma-separated output filenames without .json. Empty uses <db_name>_3min,<db_name>_10min,<db_name>_1h.",
    )
    parser.add_argument(
        "--perspective",
        default="egocentric",
        choices=["egocentric", "general"],
        help="Prompt style for summarization.",
    )
    args = parser.parse_args()

    granularity_names = (
        [name.strip() for name in args.granularity_names.split(",") if name.strip()]
        if args.granularity_names.strip()
        else None
    )

    generate_multiscale_memory(
        db_name=args.db_name,
        person_name=args.person_name,
        json_path=args.json_path,
        diary_dir=args.diary_dir,
        save_path=args.save_path,
        model_name=args.model,
        windows=[int(window.strip()) for window in args.windows.split(",") if window.strip()],
        granularity_names=granularity_names,
        perspective=args.perspective,
    )
