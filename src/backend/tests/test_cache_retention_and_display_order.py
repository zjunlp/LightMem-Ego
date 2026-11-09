"""Cache references must follow displayed images and expire without file writes."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from online_query import interaction_cache as cache_module
from online_query.coreference_resolver import CoreferenceResolver
from online_query.interaction_cache import InteractionCache


def result(times, answer):
    return {
        "status": "ok", "answer": answer,
        "raw": {"image_paths_used": [f"frame_{stamp:06d}.jpg" for stamp in times]},
        "evidence_frames": [{"path": f"frame_{stamp:06d}.jpg", "timestamp": stamp} for stamp in times],
        "timestamps": [{"start": min(times), "end": max(times)}],
    }


@pytest.mark.parametrize("answer", [
    "![first](frame_000040.jpg) then ![second](frame_000070.jpg)",
    "先看40秒，再看70秒。",
    "40、70秒都有对应的画面。",
])
def test_display_order_follows_answer_instead_of_send_order(tmp_path, answer):
    cache = InteractionCache("s", tmp_path)
    cache.update_from_query_result("找两张画面", "找两张画面", result([70, 40], answer))
    context = cache.latest_context()["latest_interaction"]
    assert [f["timestamp"] for f in context["displayed_evidence_frames"]] == [40, 70]
    resolver = CoreferenceResolver()
    assert resolver.resolve("上一答第一张图里有什么？", cache)["referenced_time_ranges"][0]["start"] == 40
    assert resolver.resolve("上一答最后一张图里有什么？", cache)["referenced_time_ranges"][0]["start"] == 70


def test_latest_capture_time_is_distinct_from_last_displayed_image(tmp_path):
    cache = InteractionCache("s", tmp_path)
    cache.update_from_query_result("找两张画面", "", result([40, 70], "![first](frame_000070.jpg) ![last](frame_000040.jpg)"))
    resolver = CoreferenceResolver()
    assert resolver.resolve("上一答最后一张图", cache)["referenced_time_ranges"][0]["start"] == 40
    assert resolver.resolve("上一答最晚那张图片", cache)["referenced_time_ranges"][0]["start"] == 70
    assert resolver.resolve("the latest frame in the previous answer", cache)["referenced_time_ranges"][0]["start"] == 70
    assert resolver.resolve("上一答最早那张图", cache)["referenced_time_ranges"][0]["start"] == 40


def test_non_displayed_sent_frame_cannot_become_the_latest_reference(tmp_path):
    cache = InteractionCache("s", tmp_path)
    cache.update_from_query_result("找画面", "", result([30, 29], "![引用图](frame_000029.jpg)"))
    actual = CoreferenceResolver().resolve("上一答实际展示的最晚图片", cache)
    assert actual["referenced_time_ranges"][0]["start"] == 29


def test_ttl_expires_turns_and_hot_anchors_without_a_file_change(tmp_path, monkeypatch):
    cache = InteractionCache("s", tmp_path, ttl_seconds=60)
    cache.update_from_query_result("什么时候拿手机", "", result([26], "26秒拿手机"))
    assert cache.latest_context()["latest_interaction"]
    assert cache.latest_context()["hot_time_ranges"]
    original_mtime = cache.cache_path.stat().st_mtime_ns
    clock = cache_module.time.time()
    monkeypatch.setattr(cache_module.time, "time", lambda: clock + 61)
    expired = cache.latest_context()
    assert expired["latest_interaction"] == {}
    assert expired["hot_time_ranges"] == []
    assert expired["hot_entities"] == []
    assert expired["hot_memories"] == []
    assert cache.cache_path.stat().st_mtime_ns == original_mtime
    assert not CoreferenceResolver().resolve("那个时候他戴墨镜吗", cache)["is_followup"]
    assert not InteractionCache("s", tmp_path, ttl_seconds=60).recent_interactions()


def test_expired_disk_turn_is_not_resurrected_when_appending(tmp_path):
    cache = InteractionCache("s", tmp_path, ttl_seconds=60)
    cache.update_from_query_result("找画面", "", result([26], "26秒拿手机"))
    payload = json.loads(cache.cache_path.read_text())
    old = datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()
    for turn in payload["recent_interactions"]:
        turn["created_at"] = old
    for key in ("hot_time_ranges", "hot_entities", "hot_memories"):
        for item in payload[key]:
            item["last_accessed_at"] = old
    cache.cache_path.write_text(json.dumps(payload))
    cache.update_from_query_result("找新图", "", result([80], "80秒拿手机"))
    saved = json.loads(cache.cache_path.read_text())
    assert [turn["answer"] for turn in saved["recent_interactions"]] == ["80秒拿手机"]
    assert all(item["start"] != 26 for item in saved["hot_time_ranges"])


def test_explicit_zero_ttl_disables_expiration(tmp_path, monkeypatch):
    cache = InteractionCache("s", tmp_path, ttl_seconds=0)
    cache.update_from_query_result("找画面", "", result([26], "26秒拿手机"))
    original = cache_module.time.time()
    monkeypatch.setattr(cache_module.time, "time", lambda: original + 86400)
    assert cache.latest_context()["latest_interaction"]
