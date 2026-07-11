from __future__ import annotations

import json
from pathlib import Path
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.modules.setdefault("cv2", types.SimpleNamespace())

from online_short_term.mst_retriever import MSTRetriever


class FakeStore:
    def __init__(self, events: list[dict]) -> None:
        self.events = events

    def get_state(self) -> dict:
        return {"mst_version": 1}

    def load_events(self) -> list[dict]:
        return list(self.events)

    def load_archive_events(self) -> list[dict]:
        return list(self.events)


def _event(idx: int, start: float, caption: str, *, refined: bool = True) -> dict:
    end = start + 4.0
    return {
        "event_id": f"event_{idx:03d}",
        "session_id": "summary_test",
        "start_time": start,
        "end_time": end,
        "status": "refined" if refined else "provisional",
        "event_caption_refined": caption if refined else None,
        "event_caption_placeholder": caption if not refined else "",
        "retrieval_text": caption,
        "diff_score": 0.5,
    }


def test_summary_query_uses_timeline_coverage() -> None:
    events = [
        _event(0, 5.0, "The wearer starts at a laptop desk."),
        _event(1, 36.0, "The wearer walks toward a door."),
        _event(2, 68.0, "The wearer uses a water dispenser and fills a clear bottle."),
        _event(3, 96.0, "The wearer leaves the water dispenser area."),
    ]
    for idx in range(4, 24):
        events.append(_event(idx, 120.0 + idx * 25.0, "The wearer works at a laptop in an office."))

    results = MSTRetriever(FakeStore(events)).search("Summarize everything so far.", top_k=5)
    captions = "\n".join(json.dumps(item, ensure_ascii=False) for item in results)

    assert len(results) > 5
    assert "water dispenser" in captions
    assert any(float(item["start_time"]) >= 500.0 for item in results)
    assert results == sorted(results, key=lambda item: float(item["start_time"]))


if __name__ == "__main__":
    test_summary_query_uses_timeline_coverage()
    print("ok")
