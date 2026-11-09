"""Compatibility between the upstream timestamp fixes and upgraded retrieval."""
import pytest

from online_query.interaction_cache import _normalize_frame_timestamp


@pytest.mark.parametrize("value,path,expected", [
    (1500.125, "stream/keyframes/kf_001500125.jpg", 1500.125),
    (1500125, "stream/keyframes/kf_001500125.jpg", 1500.125),
    (1500.125, "evidence/keyframes/kf_001500125.jpg", 1500.125),
    (False, "stream/keyframes/kf_000001000.jpg", None),
    (float("nan"), "stream/keyframes/kf_000001000.jpg", None),
])
def test_cache_accepts_upstream_legacy_units_and_local_finite_seconds(value, path, expected):
    assert _normalize_frame_timestamp(value, path) == expected


def test_episodic_retrieval_harness_uses_upstream_weekday_timestamp_parser():
    from retrieval_source_helpers import EPISODIC, caption
    record = caption("weekday", "00:00:30", "00:01:00", "A person holds a cup.")
    record["date"] = "DAY2 周二"
    memory = EPISODIC.EpisodicMemoryRAG(None, None, None)
    memory.load_captions_from_data({"30sec": [record]})
    assert memory.captions["30sec"][0].timestamp_int == (200003000, 200010000)
