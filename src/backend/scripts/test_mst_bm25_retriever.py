from __future__ import annotations

import sys
import types
from pathlib import Path

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


def _event(event_id: str, start: float, caption: str, *, diff: float = 0.2) -> dict:
    end = start + 5.0
    return {
        "event_id": event_id,
        "session_id": "bm25_test",
        "start_time": start,
        "end_time": end,
        "status": "refined",
        "event_caption_refined": caption,
        "event_caption_placeholder": "",
        "retrieval_text": caption,
        "diff_score": diff,
    }


def test_bm25_entity_match_beats_recency_and_cache() -> None:
    events = [
        _event("phone_1", 10.0, "Hands hold a smartphone in front of an open laptop on a desk."),
        _event("phone_2", 18.0, "One hand still holds the phone while moving toward a black bag."),
    ]
    for index in range(8):
        start = 220.0 + index * 8.0
        events.append(
            _event(
                f"laptop_{index}",
                start,
                "Hands type on a laptop keyboard while a webpage remains open on the desk.",
                diff=0.45,
            )
        )

    cache_context = {
        "referenced_time_ranges": [
            {"start": 236.0, "end": 249.0, "score": 0.75},
            {"start": 252.0, "end": 260.0, "score": 0.75},
        ]
    }
    question = (
        "Using the previous context around 252.0-260.0s, answer: "
        "I once used my phone. But I forget where did I place it."
    )

    results = MSTRetriever(FakeStore(events)).search(question, top_k=5, cache_context=cache_context)

    top_ids = [item["event_id"] for item in results[:3]]
    assert "phone_1" in top_ids
    assert "phone_2" in top_ids
    assert results[0]["bm25_score"] > 0.0
    assert all(item["event_id"].startswith("phone") for item in results[:2])


if __name__ == "__main__":
    test_bm25_entity_match_beats_recency_and_cache()
    print("ok")
