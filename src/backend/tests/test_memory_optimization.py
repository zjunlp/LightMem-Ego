"""Behavioral regressions found in the real Dongfeng evidence path."""
from __future__ import annotations

import json
import pytest

from online_evidence_content import prompt_evidence, action_window
from online_query.evidence_packer import EvidencePacker
from online_query.memory_fusion import MemoryFusion
from online_query.memory_plan import RetrievalPlanner
from online_query.memory_router import MemoryRouter
from online_query.query_engine import _apply_image_fallback_to_route
from online_short_term.mst_retriever import _bm25_terms


def event(name, score, retrieval, times, **extra):
    return {"evidence_id": name, "source_memory": "M_lt", "start_time": min(times), "end_time": max(times), "final_score": score, "retrieval_score": retrieval, "caption": "An occupant is seated.", "keyframe_paths": [f"{name}_{t}.jpg" for t in times], "metadata": {"keyframes": [{"path": f"{name}_{t}.jpg", "timestamp": t} for t in times]}, **extra}


def pack(events, k, query="show the scene"):
    return EvidencePacker().pack(query, {"final_evidence_k": len(events), "use_image_evidence": True, "max_image_evidence": k}, {"final_evidence": events})


def test_image_priority_uses_fusion_and_shares_budget_across_events():
    events = [event("earlier", .72, .9, [30, 40, 50, 60]), event("later", .63, 1., [60, 70, 80, 90])]
    assert pack(events, 1)["selected_image_paths_for_mllm"] == ["earlier_30.jpg"]
    paths = pack(events, 3)["selected_image_paths_for_mllm"]
    assert paths[0].startswith("earlier_")
    assert any(path.startswith("later_") for path in paths)


def test_image_selection_reaches_late_action_instead_of_event_prefix():
    ev = event("scene", .6, .5, [0, 5, 10, 15, 20, 25, 30], main_actions=[{"action": "picks up a red cup", "time_span": [24., 28.]}])
    assert pack([ev], 1, "拿起红色杯子的画面")["selected_image_paths_for_mllm"] == ["scene_25.jpg"]


def test_no_localized_observation_uses_time_coverage():
    ev = event("scene", .6, .5, list(range(0, 31, 2)))
    assert pack([ev], 3)["selected_image_paths_for_mllm"] == ["scene_0.jpg", "scene_16.jpg", "scene_30.jpg"]


def test_shared_images_do_not_delete_short_term_text_or_speech():
    memories = {"text_results": [{"memory_id": "lt", "score": .8, "start_time": 0., "end_time": 30., "caption": "A person holds a bag", "keyframe_paths": ["same.jpg"]}], "short_term_results": [{"event_id": "st", "score": .7, "start_time": 20., "end_time": 30., "event_caption_refined": "The gray bag is placed beside the seat", "transcript": "Put it here", "keyframes": [{"path": "same.jpg", "timestamp": 25.}]}]}
    fused = MemoryFusion().fuse(memories, {"memory_priority": {"M_lt": 1., "M_st": .8}}, {"final_evidence_k": 4, "retrieval_plan": {"M_cache": {"enabled": False}}}, "general_qa")
    assert {ev["source_memory"] for ev in fused["final_evidence"]} == {"M_lt", "M_st"}
    short = next(ev for ev in fused["final_evidence"] if ev["source_memory"] == "M_st")
    assert short["transcript"] == "Put it here"


def test_diff_is_not_a_semantic_or_visual_similarity():
    ev = MemoryFusion()._from_short_term({"event_id": "st", "score": .2, "diff_score": .9}, {"M_st": 1.})
    assert ev["diff_score"] == .9
    assert ev["semantic_score"] == ev["visual_score"] == 0.


def test_prompt_preserves_whole_caption_and_visual_summary_before_extra_facts():
    ev = event("scene", .6, .5, [0, 30], caption="A person is sitting. " * 80 + "The person puts on sunglasses.", visual_summary="The seated person wears dark sunglasses.", main_actions=[{"action": "puts on sunglasses", "time_span": [24, 28]}], supporting_facts=[{"triple": ["person", "wears", "sunglasses"], "support_event_ids": ["scene"]}])
    view, debug = prompt_evidence("什么时候戴墨镜", [ev], max_tokens=450)
    payload = json.dumps(view, ensure_ascii=False)
    assert "puts on sunglasses" in payload
    assert view[0]["caption"] == ev["caption"]
    assert view[0]["visual_summary"] == ev["visual_summary"]
    assert debug["budget_exceeded"] and debug["omitted_units"] > 0
    assert debug["budget_shortfall_tokens"] == debug["estimated_tokens"] - debug["token_budget"]


def test_explicit_event_budget_is_not_silently_raised_to_ten():
    plan = RetrievalPlanner().plan({"query_type": "entity_tracking", "memory_route": {"use_long_term": True}}, {"final_evidence_k": 4, "text_top_k": 2}, {"long_term_ready": True})
    assert plan["final_evidence_k"] == 4
    assert plan["candidate_budgets"]["M_lt"] == plan["retrieval_plan"]["M_lt"]["candidate_budget"] == 2


def test_image_fallback_preserves_user_switch_and_total_budget():
    route = {"use_image_evidence": True, "use_image_evidence_source": "user_override", "max_image_evidence": 9}
    _apply_image_fallback_to_route(route, {"used_image_fallback": True})
    assert route["max_image_evidence"] == 9
    route = {"use_image_evidence": False, "use_image_evidence_source": "user_override", "max_image_evidence": 0}
    _apply_image_fallback_to_route(route, {"used_image_fallback": True})
    assert not route["use_image_evidence"] and route["max_image_evidence"] == 0
    route = {"use_image_evidence": True, "use_image_evidence_source": "auto", "max_image_evidence": 0}
    _apply_image_fallback_to_route(route, {"used_image_fallback": True})
    assert route["max_image_evidence"] == 0


def test_summary_respects_explicit_sources_and_auto_includes_unaggregated_tail():
    runtime = {"short_term_ready": True, "long_term_ready": True, "mst_time_span": [0, 116], "mlt_time_span": [0, 90]}
    auto = MemoryRouter().route("总结整个视频", {}, runtime)
    assert auto["memory_route"]["use_short_term"]
    assert auto["short_term_tail_range"] == [90, 116]
    explicit = MemoryRouter().route("总结整个视频", {"use_short_term": True, "use_long_term": False}, runtime)
    assert explicit["memory_route"]["use_short_term"] and not explicit["memory_route"]["use_long_term"]


def test_single_character_and_cross_language_retrieval_terms():
    assert "backpack" in _bm25_terms("刚才拿起了什么颜色的包", expand_aliases=True)
    assert "sunglasses" in _bm25_terms("什么时候戴墨镜", expand_aliases=True)


def test_constant_scores_do_not_create_evidence_and_history_support_is_as_of():
    from retrieval_source_helpers import EPISODIC, SEMANTIC, UNIFIED, FakeEmbedding
    assert UNIFIED.EM2Memory._normalize_dict(None, {"a": 0., "b": 0.}) == {"a": 0., "b": 0.}
    assert EPISODIC.EpisodicMemoryRAG._normalize_scores(None, [0.]) == [0.]
    semantic = SEMANTIC.SemanticMemory(FakeEmbedding())
    semantic.load_triples_from_data({"facts": [{"fact_id": "f", "triple": ["person", "holds", "cup"], "semantic_summary": "A later consolidated summary", "support_count": 2, "confidence": .99, "support_records": [{"doc_id": "early", "date": "DAY1", "start_time": "00000000", "end_time": "00003000", "confidence": .6}, {"doc_id": "late", "date": "DAY1", "start_time": "00003000", "end_time": "00010000", "confidence": .9}]}]})
    early = semantic.timestamp_to_triples[100003000][0]
    late = semantic.timestamp_to_triples[100010000][0]
    assert early.support_count == 1 and early.confidence == .6
    assert early.source_doc_ids == ["early"]
    assert late.support_count == 2 and set(late.source_doc_ids) == {"early", "late"}


def test_action_time_schema_supports_refiner_output():
    assert action_window({"time_span": [22., 28.]}) == (22., 28.)


def test_brief_local_change_can_pass_the_periodic_sampler(tmp_path, monkeypatch):
    from PIL import Image, ImageDraw
    from online_pipeline.frame_stream import FrameStreamStore, sha256_path
    store = FrameStreamStore(tmp_path / "session")
    store.initialize(stream_id="local_change", target_fps=1.)
    def register(index, timestamp, changed=False):
        path = tmp_path / f"upload{index}.png"
        image = Image.new("RGB", (128, 72), "black")
        if changed:
            ImageDraw.Draw(image).rectangle((32, 18, 63, 35), fill="white")
        image.save(path)
        return store.register_frame(tmp_path=path, frame_index=index, checksum=sha256_path(path), suffix=".png", size_bytes=path.stat().st_size, client_ts_ms=None, relative_ts_ms=timestamp, source_ts_ms=None, timestamp_source="test", width=128, height=72, source="test")["frame"]
    assert register(0, 0)["memory_accepted"]
    assert not register(1, 100, True)["memory_accepted"]  # promotion rate cap
    change = register(2, 300, True)
    assert change["memory_accepted"] and change["memory_acceptance_reason"] == "localized_visual_change"
    assert not register(3, 600, True)["memory_accepted"]  # identical image
    monkeypatch.setenv("EM2MEM_FRAME_STREAM_CHANGE_PROMOTION", "0")
    monkeypatch.setenv("EM2MEM_FRAME_STREAM_CHANGE_THRESHOLD", "0")
    assert not register(4, 900, True)["memory_accepted"]


def test_same_image_and_close_time_do_not_hide_a_different_observation(tmp_path):
    from PIL import Image
    for name, color in [("a.png", "black"), ("b.png", "white")]:
        Image.new("RGB", (64, 36), color).save(tmp_path / name)
    frames = [{"path": name, "timestamp": time, "segment_id": "event"} for name, time in [("a.png", 0.), ("b.png", .5)]]
    selected, _ = EvidencePacker(tmp_path)._dedup_frames(frames, None)
    assert len(selected) == 2


@pytest.mark.parametrize("trace_location", ["top", "debug"])
def test_followup_uses_answer_cited_sent_frames_and_distinguishes_later(tmp_path, trace_location):
    from online_query.interaction_cache import InteractionCache
    from online_query.coreference_resolver import CoreferenceResolver
    cache = InteractionCache("test", tmp_path)
    result = {"answer": "26秒 frame_000026.jpg 手机贴耳，35秒 frame_000035.jpg 仍保持。", "timestamps": [{"start": 0., "end": 90.}], "raw": {"image_paths_used": ["frame_000026.jpg", "frame_000035.jpg", "frame_000084.jpg"]}, "evidence_frames": [{"path": f"frame_{time:06d}.jpg", "timestamp": float(time)} for time in [26, 35, 43, 84]]}
    if trace_location == "debug":
        result["raw"] = {"llm_debug": {"actual_image_paths": result["raw"]["image_paths_used"]}}
    cache.update_from_query_result(question="什么时候手机贴耳", resolved_question="什么时候手机贴耳", result=result)
    current = CoreferenceResolver().resolve("那个时候他戴墨镜了吗", cache)
    assert current["temporal_relation"] == "same_time"
    assert current["referenced_time_ranges"][0]["start"] == 26.
    assert current["referenced_time_ranges"][0]["end"] == 35.
    assert {frame["timestamp"] for frame in current["referenced_evidence_frames"]} == {26., 35.}
    later = CoreferenceResolver().resolve("后来呢", cache)
    assert later["temporal_relation"] == "after"
    assert later["referenced_time_ranges"][0]["start"] == 35.
    assert later["referenced_evidence_frames"] == []


def test_same_time_followup_keeps_its_anchor_before_a_better_matching_alternative():
    memories = {"text_results": [{"memory_id": "then", "score": .2, "start_time": 20., "end_time": 30., "caption": "The phone is at the ear", "keyframe_paths": ["phone.jpg"]}, {"memory_id": "later", "score": 1., "start_time": 80., "end_time": 90., "caption": "The sunglasses are worn", "keyframe_paths": ["glasses.jpg"]}]}
    fused = MemoryFusion().fuse(memories, {"memory_priority": {"M_lt": 1.}}, {"final_evidence_k": 4, "retrieval_plan": {"M_cache": {"enabled": False}}}, "followup", {"temporal_relation": "same_time", "referenced_time_ranges": [{"start": 26., "end": 35.}]})
    assert [event["evidence_id"] for event in fused["final_evidence"]] == ["then", "later"]


def test_explicit_user_window_limits_frames_inside_an_overlapping_event():
    ev = event("scene", .7, .4, [20, 26, 30, 40])
    result = EvidencePacker().pack("26到35秒", {"use_image_evidence": True, "max_image_evidence": 3, "query_constraints":{"time_range":[26.,35.]}}, {"final_evidence": [ev]}, {"temporal_relation": "same_time", "referenced_time_ranges": [{"start": 26., "end": 35.}]})
    assert set(result["selected_image_paths_for_mllm"]) == {"scene_26.jpg", "scene_30.jpg"}


def test_growing_dense_corpus_only_encodes_new_texts(tmp_path):
    from retrieval_source_helpers import EPISODIC, FakeEmbedding, NoModels, EARLY, LATE, LATE_CUTOFF
    def load(model, rows):
        memory = EPISODIC.EpisodicMemoryRAG(model, NoModels(), NoModels(), granularities=['30sec'])
        memory.dense_cache_dir = str(tmp_path)
        memory.load_captions_from_data({'30sec': rows})
        memory.index(LATE_CUTOFF)
        return memory
    first = FakeEmbedding()
    load(first, [EARLY])
    assert first.calls == ['alpha task']
    second = FakeEmbedding()
    memory = load(second, [EARLY, LATE])
    assert second.calls == ['beta task']
    assert memory.dense_cache_stats['30sec'] == {'reused_texts': 1, 'encoded_texts': 1}


def test_recent_details_cannot_evict_selected_historical_comparison_anchors():
    long_events = [{"memory_id": name, "score": .1, "start_time": start, "end_time": start+30, "caption": caption} for name,start,caption in [("phone",180.,"holds a phone"),("coat",360.,"puts on a black coat")]]
    short_events = [{"event_id": f"recent{i}", "score": .9, "start_time": 500.+i*10, "end_time": 510.+i*10, "event_caption_refined": "wears a black coat"} for i in range(8)]
    result = MemoryFusion().fuse({"text_results":long_events,"short_term_results":short_events},{"memory_priority":{"M_lt":.85,"M_st":.7}},{"final_evidence_k":6,"retrieval_plan":{"M_cache":{"enabled":False}}},"temporal_reasoning")
    assert {"phone","coat"}.issubset({e["evidence_id"] for e in result["final_evidence"]})
    assert len(result["final_evidence"])==6


def test_summary_keeps_ranked_anchors_and_unaggregated_tail_within_budget():
    long_events=[{"memory_id":f"lt{i}","score":.1+i*.01,"start_time":i*30.,"end_time":(i+1)*30.,"caption":"earlier clothes"} for i in range(12)]
    short=[{"event_id":"tail","score":.9,"start_time":360.,"end_time":388.,"event_caption_refined":"latest clothes"}]
    result=MemoryFusion().fuse({"text_results":long_events,"short_term_results":short},{"memory_priority":{"M_lt":1.,"M_st":.85}},{"final_evidence_k":5,"retrieval_plan":{"M_lt":{"candidate_budget":12},"M_cache":{"enabled":False}}},"long_term_summary")
    selected=result["final_evidence"]
    assert len(selected)==5
    # Timeline alternatives enter retrieval's pool; fusion must not replace
    # the strongest anchors with a second keyword-coverage selection.
    assert {item["evidence_id"] for item in selected[:-1]} == {"lt8", "lt9", "lt10", "lt11"}
    assert selected[-1]["evidence_id"]=="tail"


def test_summary_image_budget_covers_beginning_and_latest_period():
    events=[event(str(i),.5+i*.01,.9,[i*30,(i+1)*30]) for i in range(6)]
    packed=EvidencePacker().pack("总结整个视频的穿戴变化",{"query_type":"long_term_summary","final_evidence_k":6,"use_image_evidence":True,"max_image_evidence":3},{"final_evidence":events})
    assert packed["selected_image_paths_for_mllm"]==["0_0.jpg","2_90.jpg","5_180.jpg"]


def test_temporal_reasoning_retains_current_context_along_with_historical_anchor():
    result=MemoryFusion().fuse({"current_results":[{"current_context":{"state":{"window_start_time":60.,"window_end_time":90.,"current_time":90.},"transcript":"The car has just stopped"}}],"text_results":[{"memory_id":f"past{i}","score":.8,"start_time":i*30.,"end_time":(i+1)*30.,"caption":"previous road condition"} for i in range(3)]},{"memory_priority":{"M_cur":.1,"M_lt":.85}},{"final_evidence_k":2,"retrieval_plan":{"M_cache":{"enabled":False}}},"temporal_reasoning")
    assert {item["source_memory"] for item in result["final_evidence"]}=={"M_cur","M_lt"}


def test_short_summary_budget_retains_earliest_and_latest_state():
    from online_short_term.mst_retriever import MSTRetriever
    events=[{"event_id":f"e{i}","start_time":i*10.,"end_time":(i+1)*10.,"event_caption_refined":"some visible action","confidence":.9 if i%3==0 else .1} for i in range(18)]
    selected=MSTRetriever(None)._timeline_summary(events,top_k=3)
    assert len(selected)==3
    assert selected[0]["start_time"]==0. and selected[-1]["end_time"]==180.
    selected_all=MSTRetriever(None)._timeline_summary(events,top_k=8)
    assert "e15" in {event["event_id"] for event in selected_all}  # representative action
    assert "e17" in {event["event_id"] for event in selected_all}  # latest state


def test_hybrid_and_text_views_of_same_event_merge_unique_details():
    text={"memory_id":"doc","canonical_segment_id":"seg","start_time":0.,"end_time":30.,"score":.7,"caption":"holds a phone","keyframe_paths":["a.jpg"]}
    hybrid={"canonical_segment_id":"seg","text":{**text,"caption":"red phone case"},"fused_score":.6}
    result=MemoryFusion().fuse({"text_results":[text],"fused_results":[hybrid]},{"memory_priority":{"M_lt":1.}},{"final_evidence_k":4,"retrieval_plan":{"M_lt":{"candidate_budget":4},"M_cache":{"enabled":False}}},"general_qa")
    assert len(result["final_evidence"])==1
    assert "holds a phone" in result["final_evidence"][0]["caption"] and "red phone case" in result["final_evidence"][0]["caption"]


def test_summary_rule_priority_is_not_reported_as_raw_text_similarity():
    event=MemoryFusion()._from_short_term({"event_id":"timeline","start_time":0.,"end_time":10.,"score":1.,"raw_score":None,"score_policy":"summary_timeline_coverage_priority","summary_timeline_fallback":True},{"M_st":1.})
    assert event["raw_retrieval_score"] is None
    assert event["score_policy"]=="summary_timeline_coverage_priority"


def test_action_overview_keeps_a_middle_action_and_latest_frame():
    events=[event("start",.6,.5,[0,5,10]),event("action",.5,.4,[100,105,110]),event("latest",.4,.3,[110,113,116])]
    packed=EvidencePacker().pack("从开始依次做了哪些事",{"query_type":"long_term_summary","final_evidence_k":3,"use_image_evidence":True,"max_image_evidence":3},{"final_evidence":events})
    assert packed["selected_image_paths_for_mllm"]==["start_0.jpg","action_105.jpg","latest_116.jpg"]
