"""Evidence-loss regressions from the Dongfeng prefix demonstrations."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json
import threading
import time

import pytest

from online_evidence_content import prompt_evidence, relevance, supplemental_observation_matches
from online_evidence_coverage import select_coverage, coverage_features, evidence_time
from online_query.coreference_resolver import CoreferenceResolver
from online_query.evidence_packer import EvidencePacker
from online_query.evidence_validation import validate_evidence
from online_query.interaction_cache import InteractionCache
from online_query.memory_fusion import MemoryFusion
from refine_mst_micro_events import is_auto_refine_eligible


def event(name, start, observations, score=.5):
    return {"evidence_id": name, "source_memory": "M_lt", "start_time": start, "end_time": start+30,
            "final_score": score, "visual_observations": [
                {"timestamp": stamp, "path": f"{name}_{stamp}.jpg", "description": text}
                for stamp, text in observations], "keyframe_paths": [f"{name}_{stamp}.jpg" for stamp, _ in observations]}


def test_followup_preserves_four_distinct_occurrences_in_both_metadata_and_question():
    spans = [{"start": a, "end": b} for a,b in [(60,67),(195,207),(233,240),(429,436)]]
    cache = SimpleNamespace(latest_context=lambda: {"latest_interaction": {"time_ranges": spans}})
    result = CoreferenceResolver().resolve("那个时候她穿什么颜色？", cache)
    assert result["referenced_time_ranges"] == spans
    assert "429.0-436.0s" in result["resolved_question"]
    assert "multiple observed periods" in result["occurrence_instruction"]


def test_followup_fusion_protects_a_late_low_scoring_occurrence():
    items = [event(str(i), t, [(t, "woman in a white shirt")], score) for i,t,score in [(0,60,.9),(1,65,.8),(2,195,.7),(3,233,.6),(4,429,.1)]]
    selected = MemoryFusion()._preserve_followup_context(items, [{"start":t,"end":t+7} for t in [60,195,233,429]], 4)
    assert {item["start_time"] for item in selected} == {60,195,233,429}


def test_same_actor_and_color_does_not_promote_unknown_image_above_bag_evidence():
    question = "画面右侧穿黑色上衣的人拿起什么颜色的包？"
    bag = event("bag",90,[(96,"The front-right man in a black shirt lifts a gray backpack"),(100,"The front-right man opens the gray bag")])
    phone = event("phone",20,[(25,"The front-right man in a black shirt holds a phone at his ear")], .9)
    result = EvidencePacker().pack(question,{"use_image_evidence":True,"max_image_evidence":6},{"final_evidence":[bag,phone]})
    assert result["selected_image_paths_for_mllm"][:2] == ["bag_96.jpg","bag_100.jpg"]
    assert result['image_selection_summary']['inspection_fallback_count']==1
    assert relevance(question,bag["visual_observations"][0]) > relevance(question,phone["visual_observations"][0])


def test_clothing_category_does_not_make_a_shirt_satisfy_a_mask_condition():
    q="找他戴白色口罩，同时拿手机的画面"
    phone=event("phone",0,[(10,"The man in a white shirt holds a phone")])
    both=event("both",30,[(35,"The man wears a white mask and holds a phone")])
    constraints={"requirements":{"simultaneous":True}}
    assert [doc for doc,_ in supplemental_observation_matches(q,[phone,both],constraints)]==["both"]


def test_a_phone_in_another_actors_clause_remains_unverified_below_direct_evidence():
    scene=event("scene",0,[(2,"The front-right man reaches toward the controls. The rear man in a black shirt holds a phone."),
                            (26,"The front-right man holds a phone at his ear.")])
    result=EvidencePacker().pack("前排右侧穿黑色上衣的人什么时候把手机贴到耳边",{"use_image_evidence":True,"max_image_evidence":6},{"final_evidence":[scene]})
    assert result["selected_image_paths_for_mllm"]==["scene_26.jpg","scene_2.jpg"]
    assert result['image_selection_summary']['inspection_fallback_count']==1


def test_overview_preserves_a_brief_internal_state_not_just_event_endpoints():
    items = [event("start",0,[(0,"A woman wears a tan visor and white shirt")]),
             event("change",150,[(150,"The woman in a white shirt holds sunglasses"),(183,"Dark sunglasses cover her eyes"),(190,"She has removed the sunglasses")]),
             event("end",660,[(688,"The woman wears a black jacket and sunglasses on top of her head")])]
    items[1]["end_time"] = 195  # The brief observation belongs inside its event.
    result = EvidencePacker().pack("概括她的穿戴变化",{"query_type":"long_term_summary","final_evidence_k":3,"use_image_evidence":True,"max_image_evidence":4},{"final_evidence":items})
    images = result["selected_image_paths_for_mllm"]
    assert "change_183.jpg" in images
    assert "start_0.jpg" in images and "end_688.jpg" in images


def test_state_coverage_beats_repeated_high_scoring_head_turns():
    items = [event("start",0,[(0,"The woman wears a white shirt and visor")]),
             event("change",150,[(183,"The woman removes the visor and wears sunglasses over her eyes")], .1),
             *[event(f"repeat{i}",300+i*30,[(300+i*30,"The woman in a white shirt turns her head")],.9) for i in range(3)],
             event("end",660,[(688,"The woman wears a black jacket")])]
    selected = select_coverage("概括她的穿戴变化",items,3)
    assert [item["evidence_id"] for item in selected] == ["start","change","end"]


def test_later_rich_state_does_not_replace_the_first_observed_wearing():
    items=[event("start",0,[(0,"The woman wears a visor")]),
           event("first",150,[(183,"Dark sunglasses cover her eyes")]),
           event("later",240,[(251,"She wears sunglasses"),(255,"Sunglasses are on top of her head"),(272,"She has removed sunglasses")]),
           event("end",660,[(688,"She wears a black jacket")])]
    assert "first" in {item["evidence_id"] for item in select_coverage("概括穿戴变化",items,3)}


def test_selector_day_labels_preserve_real_time_order_instead_of_input_order():
    assert evidence_time({"start_time":"DAY1 00:00:00"})==0
    assert evidence_time({"start_time":"DAY1 00:10:00"})==600
    assert evidence_time({"start_time":"DAY2 00:00:01"})==86401
    candidates=[{"id":"middle","start_time":"DAY1 00:05:00"},
                {"id":"end","start_time":"DAY1 00:11:00"},
                {"id":"start","start_time":"DAY1 00:00:00"}]
    assert [item["id"] for item in select_coverage("概括视频",candidates,2)]==["start","end"]


def test_one_object_disappearing_does_not_remove_another_visible_object():
    item=event("scene",30,[(44,"The spray bottle is no longer visible, and she bends forward with her face shaded by the visor.")])
    assert "state:hat:removed" not in coverage_features("总结穿戴变化",item)


def test_road_widening_is_not_evidence_of_sparse_traffic():
    item=event("road",0,[(9,"The road opens toward a junction, with several vehicles near the sides.")])
    assert "state:traffic:sparse" not in coverage_features("概括前方车辆变化",item)
    near=event("queue",30,[(30,"A white hatchback is close ahead in the lane; the road scene remains stopped or slow-moving.")])
    assert {"state:traffic:near","state:traffic:stopped_or_slow"} <= coverage_features("概括前方车辆变化",near)


def test_early_state_details_do_not_consume_the_only_middle_image_slot():
    items=[event("start",0,[(0,"A small car is visible in the distance"),(9,"The road opens toward a junction"),(30,"A hatchback is close ahead")]),
           event("queue",30,[(39,"The road scene remains stopped or slow-moving"),(90,"Vehicles are dense and closely spaced")]),
           event("middle",180,[(206,"Vehicles are distant and the road is clear")]),
           event("end",330,[(359,"Traffic remains stopped")])]
    packed=EvidencePacker().pack("概括前方车辆变化，区分开头中间和末段",{"query_type":"long_term_summary","final_evidence_k":4,"use_image_evidence":True,"max_image_evidence":4},{"final_evidence":items})
    assert {"start_0.jpg","middle_206.jpg","end_359.jpg"} <= set(packed["selected_image_paths_for_mllm"])


def test_summary_keeps_background_from_a_discarded_child_without_its_images():
    items=[event("start",0,[(0,"The woman wears a visor")]),event("other",600,[(607,"Sunglasses are on top of her head")])]
    items[1]["parent_context"]=[{"doc_id":"last_parent","start_time":540,"end_time":660,"caption":"The sunglasses move from the eyes to the top of her head."}]
    result=MemoryFusion().fuse({"text_results":items},{"question":"概括她的穿戴变化","query_constraints":{"summary":True}},{"final_evidence_k":1},"long_term_summary")
    assert len(result["final_evidence"])==1
    first=result["final_evidence"][0]
    assert first["parent_context"][0]["doc_id"]=="last_parent"
    assert "other_607.jpg" not in first["keyframe_paths"]


def test_summary_keeps_a_compact_shared_parent_under_the_same_budget():
    items = [event("child",600,[(t,"The woman in a black jacket turns toward the window") for t in range(600,625)])]
    items[0]["parent_context"] = [{"doc_id":"parent","start_time":540,"end_time":660,"caption": "The woman looks around the car. "*70 + "She moves the sunglasses from her eyes to the top of her head.","scope":"coarse background"}]
    packed,stats = prompt_evidence("概括她的穿戴变化",items,max_tokens=1200)
    assert stats["estimated_tokens"] <= 1200
    assert stats["retained_parent_summaries"] == 1
    assert "sunglasses" in packed[0]["parent_context"][0]["caption"]
    assert packed[0]["parent_context"][0]["scope"] == "coarse background"


def test_validator_checks_actual_compressed_evidence_and_all_followup_ranges():
    full = [event("change",150,[(183,"Dark sunglasses cover her eyes")])]
    check = validate_evidence("概括穿戴变化",[{"caption":"woman sits"}],[],{"summary":True,"referenced_time_ranges":[{"start":429,"end":436}]},source_events=full)
    assert "summary_state_changes_lost_during_prompt_compression" in check["issues"]
    assert check["uncovered_reference_ranges"] == [{"start":429,"end":436}]


def test_two_stale_cache_instances_append_without_losing_either_turn(tmp_path):
    first,second = InteractionCache("s",tmp_path),InteractionCache("s",tmp_path)
    def update(cache,answer):
        return cache.update_from_query_result("发生什么","发生什么",{"status":"ok","answer":answer})
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda args:update(*args),[(first,"first"),(second,"second")]))
    turns = InteractionCache("s",tmp_path).recent_interactions()
    assert {row["answer"] for row in turns} == {"first","second"}
    assert len({row["turn_id"] for row in turns}) == 2


def test_transient_refine_retry_has_bounded_backoff_and_attempts(monkeypatch):
    monkeypatch.delenv("EM2MEM_MST_REFINE_RETRY_BACKOFF_SECONDS",raising=False)
    monkeypatch.setenv("EM2MEM_MST_REFINE_MAX_ATTEMPTS","3")
    now=datetime.now(timezone.utc)
    item={"status":"refine_failed","refine":{"refine_attempts":1,"last_refine_error":"APIConnectionError: Connection error", "last_refine_failed_at":now.isoformat()}}
    assert not is_auto_refine_eligible(item,now=now+timedelta(seconds=10))
    assert is_auto_refine_eligible(item,now=now+timedelta(seconds=16))
    item["refine"]["refine_attempts"]=3
    assert not is_auto_refine_eligible(item,now=now+timedelta(hours=1))
    item["refine"].update(refine_attempts=1,last_refine_error="AuthenticationError 401")
    assert not is_auto_refine_eligible(item,now=now+timedelta(hours=1))


def test_idle_worker_recovers_a_transient_refinement_without_new_input(tmp_path,monkeypatch):
    import argparse
    import online_mst_refine_worker as worker
    from online_short_term.mst_store import MSTStore
    from online_preprocess.task_queue import get_queue_dirs
    monkeypatch.setenv("EM2MEM_SINGLE_ACTIVE_SESSION","0")
    monkeypatch.setenv("EM2MEM_AUTO_MST_CONSOLIDATION","0")
    monkeypatch.setenv("EM2MEM_MST_REFINE_MAX_ATTEMPTS","3")
    monkeypatch.delenv("EM2MEM_MST_REFINE_RETRY_BACKOFF_SECONDS",raising=False)
    store=MSTStore(tmp_path/"online_sessions/s")
    store.save_archive_events([{"event_id":"failed","start_time":0,"end_time":10,"status":"refine_failed","refine":{"refine_attempts":1,"last_refine_error":"APIConnectionError: Connection error","last_refine_failed_at":(datetime.now(timezone.utc)-timedelta(seconds=20)).isoformat()}}])
    worker.run_worker(argparse.Namespace(project_root=str(tmp_path),sessions_root=str(tmp_path/"online_sessions"),backend="mock",model=None,max_concurrency=1,scan_sessions=True,scan_limit_sessions=2,limit_events=2,force_refine=False,verbose=False,once=True,poll_interval=.05))
    assert store.load_archive_events()[0]["status"]=="refined"
    assert len(list(get_queue_dirs(tmp_path)["mst_refine_done"].glob("*.json")))==1


def test_summary_candidate_coverage_respects_the_indexed_prefix():
    from retrieval_source_helpers import UNIFIED
    memory=object.__new__(UNIFIED.EM2Memory)
    entries=[]
    for i in range(15):
        caption="A woman in a white shirt turns her head"
        if i==10: caption="The woman removes sunglasses from her eyes and places them on top of her head"
        entries.append(SimpleNamespace(doc_id=str(i),timestamp_int=(i*3000,(i+1)*3000),text=caption,visual_summary=caption,metadata={},start_time=i*30,end_time=(i+1)*30))
    by_id={e.doc_id:e for e in entries}
    memory.episodic_memory=SimpleNamespace(captions={"30sec":entries},_get_thread_prefix_size=lambda g:12,_matches_requested_time=lambda e:True,get_caption_by_doc_id=lambda doc,g:by_id[doc],get_parent_caption=lambda *a:None,get_triplets_by_doc_id=lambda *a:[])
    memory.visual_memory=SimpleNamespace(get_clip_by_doc_id=lambda *a:None)
    memory._compute_event_role_scores=lambda **kw:{doc:{"trigger":score,"antecedent":score,"broader":score} for doc,score in kw["episodic_norm"].items()}
    for attr,value in {"selector_max_candidates":8,"selector_global_top_n":6,"selector_trigger_top_n":2,"selector_antecedent_top_n":2,"selector_broader_top_n":2}.items():setattr(memory,attr,value)
    candidates=memory._build_event_selector_candidates("总结她的穿戴变化",{str(i):1-i*.05 for i in range(6)},{},{})
    assert "10" in {row["doc_id"] for row in candidates}
    assert all(int(row["doc_id"])<12 for row in candidates)


def test_current_lane_answers_while_history_preload_is_blocked(tmp_path,monkeypatch):
    import argparse
    import signal
    import online_query_worker as worker
    import online_query.query_engine as engine
    from online_preprocess.task_queue import enqueue_query_task, get_queue_dirs
    from online_current.mcur_store import MCurStore
    from PIL import Image
    monkeypatch.setenv("EM2MEM_SINGLE_ACTIVE_SESSION","0")
    monkeypatch.setenv("EM2MEM_QUERY_CURRENT_FAST_PATH","1")
    monkeypatch.setenv("EM2MEM_INTERACTION_CACHE_ENABLED","0")
    monkeypatch.setattr(engine,"PROJECT_ROOT",tmp_path)
    class Model:
        def generate(self,prompt):return "当前画面是一只杯子。"
    monkeypatch.setattr(engine,"_get_short_term_answer_model",lambda:Model())
    session=tmp_path/"online_sessions/s"
    frame=session/"stream/frames/frame_000026.jpg"
    frame.parent.mkdir(parents=True);Image.new("RGB",(16,16),"white").save(frame)
    MCurStore(session).update_from_frame_stream(frame_index=26,frame_path="stream/frames/frame_000026.jpg",relative_ts_ms=26000)
    started,release=threading.Event(),threading.Event()
    def preload():
        started.set();assert release.wait(10);return None
    monkeypatch.setattr(worker,"_preload_text_embedding",preload)
    handlers={}
    monkeypatch.setattr(worker.signal,"signal",lambda signum,handler:handlers.setdefault(signum,handler))
    args=argparse.Namespace(project_root=str(tmp_path),sessions_root="online_sessions",cache_max_sessions=1,ttl_seconds=3600,once=False,current_lane=True,poll_interval=.05,preload_vlm2vec=False,preload_sessions="",preload_recent_memory_ready=0)
    thread=threading.Thread(target=worker.run_worker,args=(args,),daemon=True);thread.start()
    try:
        assert started.wait(5)
        task=enqueue_query_task(tmp_path,"s","现在画面里有什么？",retrieval_mode="current",use_interaction_cache=False)
        done=get_queue_dirs(tmp_path)["query_done"]/task.name
        deadline=time.monotonic()+5
        while not done.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert done.exists(), "Current question was blocked by the unrelated preload"
        assert not release.is_set()
        assert json.loads(done.read_text())["result"]["status"]=="ok"
    finally:
        handlers[signal.SIGTERM](signal.SIGTERM,None)
        release.set();thread.join(5)
    assert not thread.is_alive()
