from online_evidence_coverage import (
    observation_detail_features,
    order_frame_candidates_by_complementarity,
)


def _frame(timestamp, description, *, local=1.0):
    return {
        "path": f"frame_{timestamp}.jpg",
        "timestamp": timestamp,
        "_local_relevance": local,
        "_action_support_level": 1,
        "_coverage_observations": [{"description": description, "path": f"frame_{timestamp}.jpg"}],
    }


def test_event_frame_order_prefers_complementary_action_details_over_duplicate_stills():
    frames = [
        _frame(27, "右前方男子把手机贴在耳边，用双手拿着。"),
        _frame(29, "右前方男子把手机贴在耳边，改为单手拿着。"),
        _frame(30, "右前方男子把手机贴在耳边，仍然单手拿着。"),
    ]

    ordered = order_frame_candidates_by_complementarity("他是怎么拿着手机的？", frames)

    assert [frame["timestamp"] for frame in ordered] == [27, 29, 30]
    assert ordered[1]["_complementarity_rank"] < ordered[2]["_complementarity_rank"]
    assert "detail:" in " ".join(observation_detail_features("他是怎么拿着手机的？", {
        "description": "右前方男子把手机贴在耳边，改为单手拿着。"
    }))


def test_event_frame_order_keeps_distinct_before_after_states():
    frames = [
        _frame(600, "她戴着墨镜，遮阳帽还在头上。"),
        _frame(607, "她把墨镜摘下，遮阳帽仍在头上。"),
        _frame(610, "她仍未戴墨镜，遮阳帽还在头上。"),
    ]

    ordered = order_frame_candidates_by_complementarity("之前到后来她的穿戴有什么变化？", frames)

    assert [frame["timestamp"] for frame in ordered[:2]] == [600, 607]
    assert len(observation_detail_features("之前到后来她的穿戴有什么变化？", {
        "before": "戴着墨镜",
        "after": "摘下墨镜",
        "entity": "墨镜",
        "description": "她把墨镜摘下。",
    })) > 0


def test_free_form_detail_does_not_promote_a_distant_unasked_object():
    features = observation_detail_features(
        "较早同时戴口罩和墨镜，到后来脸部状态有什么变化？",
        {"description": "A masked person wearing sunglasses and a seat belt sits in a car and holds a black smartphone near the torso."},
    )

    assert "detail:smartphone" not in features


def test_change_marker_beats_a_later_continuation_when_details_are_close():
    question = "前面刚能认出手机的时候，他是怎么拿着的？"
    changed = observation_detail_features(question, {"description": "Phone is raised closer to the torso and angled more horizontally."})
    repeated = observation_detail_features(question, {"description": "Phone remains held above lap; posture remains similar."})

    assert "detail:change_marker" in changed
    assert "detail:continuation_marker" in repeated
