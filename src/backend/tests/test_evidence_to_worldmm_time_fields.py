import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _install_lightweight_stubs(monkeypatch) -> None:
    online_preprocess = types.ModuleType("online_preprocess")
    online_preprocess.__path__ = []
    io_utils = types.ModuleType("online_preprocess.io_utils")
    io_utils.read_json = lambda path, default=None: json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default
    io_utils.write_json = lambda path, data: Path(path).write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "online_preprocess", online_preprocess)
    monkeypatch.setitem(sys.modules, "online_preprocess.io_utils", io_utils)

    online_memory = types.ModuleType("online_memory")
    online_memory.__path__ = [str(ROOT / "online_memory")]
    monkeypatch.setitem(sys.modules, "online_memory", online_memory)


def _load_evidence_to_worldmm(monkeypatch):
    _install_lightweight_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location(
        "online_memory.evidence_to_worldmm",
        ROOT / "online_memory" / "evidence_to_worldmm.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, "online_memory.evidence_to_worldmm", module)
    spec.loader.exec_module(module)
    return module


def test_caption_item_preserves_parent_display_time_and_local_seconds(monkeypatch) -> None:
    evidence_to_worldmm = _load_evidence_to_worldmm(monkeypatch)

    item = evidence_to_worldmm.evidence_doc_to_caption_item(
        "parent1",
        {
            "doc_id": "parent1__DAY1__ev1",
            "segment_id": "seg_000000_000030",
            "date": "DAY1",
            "day_label": "DAY1",
            "start_time": "20453000",
            "end_time": "20460000",
            "local_start_time": 0.0,
            "local_end_time": 30.0,
            "display_date": "2026年7月8日",
            "display_start_time": "20:45:30",
            "display_end_time": "20:46:00",
            "display_time_range": "20:45:30-20:46:00",
            "display_datetime_start": "2026-07-08 20:45:30",
            "display_datetime_end": "2026-07-08 20:46:00",
            "timezone": "Asia/Shanghai",
            "fine_caption": "[2026年7月8日 20:45:30-20:46:00] laptop on desk",
        },
        0,
    )

    assert item["start_time"] == "20453000"
    assert item["end_time"] == "20460000"
    assert item["start"] == 0.0
    assert item["end"] == 30.0
    assert item["duration"] == 30.0
    assert item["local_start_time"] == 0.0
    assert item["local_end_time"] == 30.0
    assert item["display_time_range"] == "20:45:30-20:46:00"
    assert item["display_datetime_start"] == "2026-07-08 20:45:30"
    assert item["timezone"] == "Asia/Shanghai"


def test_caption_item_still_converts_relative_seconds(monkeypatch) -> None:
    evidence_to_worldmm = _load_evidence_to_worldmm(monkeypatch)

    item = evidence_to_worldmm.evidence_doc_to_caption_item(
        "child1",
        {
            "doc_id": "ev1",
            "start_time": 60.0,
            "end_time": 90.0,
            "fine_caption": "office view",
        },
        0,
    )

    assert item["start_time"] == "00010000"
    assert item["end_time"] == "00013000"
    assert item["start"] == 60.0
    assert item["end"] == 90.0
