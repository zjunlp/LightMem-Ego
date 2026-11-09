from online_query.query_router import QueryRouter
from online_query.memory_plan import RetrievalPlanner


def test_router_auto_keeps_image_evidence_for_summary_and_temporal_queries() -> None:
    router = QueryRouter()
    context = {
        "visual_ready": False,
        "short_term_ready": True,
        "current_ready": True,
        "current_stale": False,
        "long_term_ready": True,
    }

    summary = router.route("what happened on DAY1", request_options={"use_image_evidence": "auto"}, session_context=context)
    temporal = router.route("when did I move the phone?", request_options={"use_image_evidence": "auto"}, session_context=context)

    assert summary["use_image_evidence"] is True
    assert summary["max_image_evidence"] > 0
    assert temporal["use_image_evidence"] is True
    assert temporal["max_image_evidence"] > 0


def test_router_allows_explicit_image_evidence_off() -> None:
    router = QueryRouter()

    decision = router.route("what happened on DAY1", request_options={"use_image_evidence": False})

    assert decision["use_image_evidence"] is False


def test_retrieval_planner_auto_keeps_anchor_image_evidence_without_visual_index() -> None:
    planner = RetrievalPlanner()
    plan = planner.plan(
        memory_decision={
            "query_type": "long_term_summary",
            "memory_route": {"use_long_term": True},
        },
        request_options={"use_image_evidence": "auto"},
        runtime_state={"long_term_ready": True, "visual_embedding_ready": False},
    )

    assert plan["use_image_evidence"] is True
    assert plan["max_image_evidence"] > 0
    assert plan["retrieval_plan"]["M_lt"]["mode"] == "text_only"


def test_retrieval_planner_allows_explicit_image_evidence_off() -> None:
    planner = RetrievalPlanner()
    plan = planner.plan(
        memory_decision={
            "query_type": "general_qa",
            "memory_route": {"use_long_term": True},
        },
        request_options={"use_image_evidence": False},
        runtime_state={"long_term_ready": True, "visual_embedding_ready": False},
    )

    assert plan["use_image_evidence"] is False
    assert plan["max_image_evidence"] == 0
