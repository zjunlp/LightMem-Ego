from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_preprocess.io_utils import read_json


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Print online query worker/cache runtime state.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    runtime_path = Path(args.project_root).resolve() / "online_tasks" / "query_runtime.json"
    payload = read_json(runtime_path, default=None)
    if payload is None:
        payload = {
            "status": "not_running_or_no_runtime_file",
            "message": f"{runtime_path} not found",
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
