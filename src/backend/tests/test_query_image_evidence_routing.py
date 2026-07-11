import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_query_routing_classes():
    package = types.ModuleType("online_query")
    package.__path__ = [str(ROOT / "online_query")]
    sys.modules["online_query"] = package
    for module_name in ("router_schema", "memory_plan", "query_router"):
        full_name = f"online_query.{module_name}"
        spec = importlib.util.spec_from_file_location(full_name, ROOT / "online_query" / f"{module_name}.py")
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return sys.modules["online_query.query_router"].QueryRouter, sys.modules["online_query.memory_plan"].RetrievalPlanner


def test_router_auto_keeps_image_evidence_for_summary_and_temporal_queries() -> None:
    QueryRouter, _RetrievalPlanner = _load_query_routing_classes()
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
    QueryRouter, _RetrievalPlanner = _load_query_routing_classes()
    router = QueryRouter()

    decision = router.route("what happened on DAY1", request_options={"use_image_evidence": False})

    assert decision["use_image_evidence"] is False


def test_retrieval_planner_auto_keeps_anchor_image_evidence_without_visual_index() -> None:
    _QueryRouter, RetrievalPlanner = _load_query_routing_classes()
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
    _QueryRouter, RetrievalPlanner = _load_query_routing_classes()
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
