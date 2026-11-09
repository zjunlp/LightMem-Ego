"""Offline contracts for query integration; no answer/embedding API calls."""
from types import SimpleNamespace

import pytest

from online_current.mcur_query import build_current_prompt
from online_query import query_engine as engine


@pytest.mark.parametrize("question", [
    "之前有没有口罩和墨镜都戴着的时候？那时跟现在有什么不同？",
    "Compare what he held earlier with what he is holding now.",
    "截至现在，他有没有拿过手机？",
])
def test_history_and_mixed_requests_do_not_bypass_router(question, monkeypatch):
    monkeypatch.setenv("EM2MEM_CURRENT_FAST_PATH_ENABLED", "1")
    assert not engine._wants_current_fast_path(question)


def test_current_switch_uses_fast_path_and_explicit_overrides_survive(monkeypatch):
    monkeypatch.setenv("EM2MEM_CURRENT_FAST_PATH_ENABLED", "1")
    assert engine._wants_current_fast_path("再回到现在，他还拿着手机吗？")
    assert engine._wants_current_fast_path("Compare earlier with now", memory_mode="current")
    assert not engine._wants_current_fast_path("现在拿什么", use_current=False)


@pytest.fixture
def fallback_session(tmp_path, monkeypatch):
    session = tmp_path / "s"
    for second in (591, 606, 610):
        path = session / f"current/frames/frame_{second:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    state = {"current_time": 610, "window_start_time": 580, "window_end_time": 610,
             "mcur_ready": True, "current_visual_ready": True, "is_stale": False}
    frames = [{"timestamp": t, "path": f"current/frames/frame_{t:06d}.jpg",
               "diff_score": .2, "role": "latest" if t == 610 else "reference"}
              for t in (606, 610)]
    context = {"state": state, "mcur_ready": True, "is_stale": False,
               "current_time": 610, "window_start_time": 580, "window_end_time": 610, "frames": frames,
               "open_event": {"start_time": 601, "end_time": 610, "status": "open", "keyframes": frames}}
    st = [{"event_id": "closed", "start_time": 581, "end_time": 601, "status": "refined",
           "score": .9, "event_caption_refined": "男子先前调整过眼镜。",
           "keyframes": [{"timestamp": 591, "path": "current/frames/frame_000591.jpg"}]}]
    class Current:
        def __init__(self, *_):
            self.transcript_path = session / "current/transcript_partial.jsonl"
            self.frames_path = session / "current/current_frames.jsonl"
            self.frames_path.touch()
        def get_state(self): return dict(state)
        def is_ready(self): return True
        def is_stale(self, *_): return False
        def get_current_context(self): return context
    class Short:
        def __init__(self, *_): pass
        def is_ready(self): return True
        def get_state(self): return {"active_time_span": [581, 601]}
    class Retriever:
        def __init__(self, *_): pass
        def search(self, *_args, **_kwargs): return st
    monkeypatch.setattr(engine, "MCurStore", Current)
    monkeypatch.setattr(engine, "MSTStore", Short)
    monkeypatch.setattr(engine, "MSTRetriever", Retriever)
    monkeypatch.setattr(engine, "_frame_open_event_synthetic_evidence", lambda *_: None)
    monkeypatch.setenv("EM2MEM_MST_ANSWER_WITH_LLM", "0")
    monkeypatch.setenv("EM2MEM_QUERY_MAX_TOTAL_IMAGES", "6")
    return SimpleNamespace(root=tmp_path, context=context)


def _fallback(data, **overrides):
    args = dict(session_id="s", question="他最近有没有调整眼镜？", sessions_root=data.root,
                top_k=5, latency={}, retrieval_mode="auto", use_image_evidence=True,
                max_image_evidence=6, text_top_k=5, visual_top_k=5, final_evidence_k=4,
                use_interaction_cache=False, cache_mode="off")
    args.update(overrides)
    return engine._query_short_term_only(**args)


def test_no_lt_fallback_keeps_open_current_tail_alongside_closed_st(fallback_session):
    result = _fallback(fallback_session)
    times = [f["timestamp"] for f in result["evidence_frames"]]
    assert 610 in times
    assert any(e["source_memory"] == "M_cur" for e in result["raw"]["selected_evidence"])
    assert result["current_used"]
    assert not result["short_term_only"]
    assert result["recent_recall_sufficiency"]["requires_current_tail"]


@pytest.mark.parametrize("options", [{"use_current": False}, {"use_image_evidence": False},
                                    {"max_image_evidence": 0}])
def test_fallback_respects_caller_source_and_image_optouts(fallback_session, options):
    result = _fallback(fallback_session, **options)
    if options.get("use_current") is False:
        assert not result.get("current_used")
        assert all(f.get("timestamp") != 610 for f in result["evidence_frames"])
    else:
        assert result["sent_image_count"] == 0


def test_current_prompt_distinguishes_absence_from_occlusion():
    prompt = build_current_prompt("他还拿着包吗？", {"state": {"current_time": 116}}, {})
    assert "not visible does not mean absent" in prompt.lower()
    assert "complete state" in prompt.lower()


def test_all_generation_paths_share_state_grounding_contract():
    loaded = object.__new__(engine.LoadedQueryEngine)
    prompts = [
        loaded._build_visual_answer_prompt("他现在拿着手机吗", "memory_router", [], [], []),
        engine._build_short_term_llm_prompt("他现在拿着手机吗", [], {}),
    ]
    for prompt in prompts:
        assert "not visible does not mean absent" in prompt.lower()
        assert "complete state" in prompt.lower()


def test_short_term_prompt_uses_final_events_including_current_transcript():
    prompt = engine._build_short_term_llm_prompt(
        "最近说了什么", [{"event_id": "excluded", "event_caption_refined": "EXCLUDED_RAW_EVENT"}], {},
        selected_evidence=[{"evidence_id": "tail", "source_memory": "M_cur", "status": "current",
                            "start_time": 601, "end_time": 610, "transcript": "CURRENT_TAIL_TRANSCRIPT"}],
    )
    assert "CURRENT_TAIL_TRANSCRIPT" in prompt
    assert "EXCLUDED_RAW_EVENT" not in prompt
    assert "unrefined" in prompt


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("image_failure", [False, True])
def test_sync_stream_and_text_fallback_keep_grounding_prompt(tmp_path, monkeypatch, streaming, image_failure):
    import sys
    from retrieval_source_helpers import UTIL
    monkeypatch.setitem(sys.modules, "em2mem.memory.utils", UTIL)
    monkeypatch.setenv("EM2MEM_QUERY_ANSWER_RETRIES", "1")
    calls = []
    class Model:
        def generate(self, prompt):
            calls.append(prompt)
            if image_failure and isinstance(prompt, list):
                raise RuntimeError("mock image failure")
            return "手部被遮挡，目前无法确认是否仍拿着包。"
        def stream_generate(self, prompt, on_chunk):
            answer = self.generate(prompt)
            on_chunk(answer)
            return answer
    loaded = object.__new__(engine.LoadedQueryEngine)
    loaded.session_dir = tmp_path
    loaded.em2mem_memory = SimpleNamespace(respond_llm_model=Model())
    events = [{"evidence_id": "tail", "source_memory": "M_cur", "status": "current", "caption": "unrefined current",
               "start_time": 601, "end_time": 610, "keyframe_paths": ["current/frame_000610.jpg"]}]
    chunks = []
    result = loaded._generate_answer_from_evidence(
        "现在还拿着包吗", "memory_router", [], [], [], True, 1,
        selected_image_paths=["current/frame_000610.jpg"], selected_evidence=events,
        stream_handler=chunks.append if streaming else None,
    )
    assert result.llm_debug["generation_status"] == ("text_only_success" if image_failure else "multimodal_success")
    assert len(result.image_paths_used) == (0 if image_failure else 1)
    assert len(calls) == (2 if image_failure else 1)
    for call in calls:
        prompt = call if isinstance(call, str) else call[-1]["content"][0]["text"]
        assert "not visible does not mean absent" in prompt
    if streaming:
        assert chunks


def test_current_only_fallback_keeps_stream_handler(fallback_session, monkeypatch):
    chunks = []
    class Model:
        def generate(self, prompt): return "眼镜在头顶。"
        def stream_generate(self, prompt, on_chunk):
            on_chunk("眼镜在头顶。")
            return "眼镜在头顶。"
    monkeypatch.setattr(engine, "_get_short_term_answer_model", lambda: Model())
    _fallback(fallback_session, use_short_term=False, stream_handler=chunks.append)
    assert any(chunk.get("type") == "delta" for chunk in chunks)


@pytest.mark.parametrize("mixed,confidence,ambiguous,suppressed", [
    (False, .9, False, True), (True, .9, False, False),
    (False, .4, False, False), (False, .9, True, False),
])
def test_recent_tail_respects_reliable_past_event_reference(mixed, confidence, ambiguous, suppressed):
    route = {"query_constraints": {"requires_current": mixed, "requires_history": True}}
    evaluation = {"requires_current_tail": True, "current_tail_coverage": {"tail_range": [601,610]}}
    cache = {"temporal_relation": "same_time", "confidence": confidence,
             "reference_ambiguous": ambiguous, "referenced_time_ranges": [{"start": 581, "end": 601}]}
    engine._apply_current_tail_requirement(route, evaluation, cache)
    assert bool(route.get("requires_current_tail")) is not suppressed
    assert evaluation["requires_current_tail"] is not suppressed


def test_recent_reference_does_not_inject_unrelated_current_tail(fallback_session, monkeypatch):
    cache = {"is_followup": True, "cache_hit": True, "confidence": .9,
             "temporal_relation": "same_time", "referenced_time_ranges": [{"start": 581, "end": 601}],
             "resolved_question": "刚才那段他有没有调整眼镜？"}
    monkeypatch.setattr(engine, "CoreferenceResolver", lambda: SimpleNamespace(resolve=lambda *a, **kw: dict(cache)))
    result = _fallback(fallback_session, question="刚才那段他有没有调整眼镜？",
                       use_interaction_cache=True, cache_mode="read_only")
    assert not result["recent_recall_sufficiency"]["requires_current_tail"]
    assert all(f["timestamp"] <= 601 for f in result["evidence_frames"])
