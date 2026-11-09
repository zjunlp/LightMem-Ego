"""Historical image selection remains usable after its ST source expires."""
from copy import deepcopy

from online_evidence_content import image_observation_records, prompt_evidence
from online_memory.evidence_to_em2mem import evidence_doc_to_caption_item
from online_memory.frame_captions import inherit_frame_captions, short_term_frame_captions
from online_query.evidence_packer import EvidencePacker, image_evidence_label
from online_query.evidence_schema import evidence_frames
from online_query.memory_fusion import MemoryFusion
from online_short_term.episodic_window_builder import MSTEpisodicWindowBuilder
from online_short_term.mst_store import MSTStore
from online_short_term.mst_to_episodic import _evidence_doc_from_episode
from test_memory_editor_connection import memory_session


def micro(name, times, descriptions):
    return {"event_id": name, "version": 2, "start_time": min(times), "end_time": max(times),
            "status": "refined", "event_caption_refined": "A person works in a kitchen.",
            "keyframes": [{"timestamp": t, "path": f"stream/frames/frame_{t:06d}.jpg", "caption": ""}
                          for t in times],
            "visual_observations": [{"timestamp": t, "path": f"stream/frames/frame_{t:06d}.jpg", "description": description}
                                    for t, description in zip(times, descriptions)]}


def test_lt_keeps_capture_labels_through_publication_and_after_st_eviction(tmp_path, monkeypatch):
    event = micro("kitchen", [0, 15, 30], ["A person stands beside a cabinet.",
        "The person holds a blue mug with both hands.", "The person walks toward the door."])
    builder = MSTEpisodicWindowBuilder(backend="rule")
    monkeypatch.setattr(builder, "_build_visual_summary", lambda *args: ("A person in a kitchen.", "generated", {}))
    episode = builder.build_episode({"start_time": 0, "end_time": 30}, [event], session_id="kitchen", session_dir=tmp_path)
    record = evidence_doc_to_caption_item("kitchen", _evidence_doc_from_episode(episode), 0)
    assert len(record["keyframe_captions"]) == 3
    mug = next(row for row in record["keyframe_captions"] if row["timestamp"] == 15)
    assert mug["source_micro_event_id"] == "kitchen" and mug["source_micro_event_version"] == 2
    assert record["visual_summary"] == "A person in a kitchen."
    store = MSTStore(tmp_path)
    store.recent_window_seconds = 90
    store.append_events([event])
    store.append_events([micro("later", [200, 210], ["A person is seated."] * 2)])
    assert "kitchen" not in {row["event_id"] for row in store.load_events()}
    # No ST source is given to either fusion or image packing.
    selected = MemoryFusion().fuse({"text_results": [{**record, "memory_id": record["doc_id"], "score": .7}]},
        {}, {"final_evidence_k": 1}, "general_qa")["final_evidence"]
    packed = EvidencePacker().pack("How was the blue mug held?", {"use_image_evidence": True,
        "max_image_evidence": 1, "final_evidence_k": 1}, {"final_evidence": selected})
    assert packed["selected_image_paths_for_mllm"] == [mug["path"]]
    frame = next(row for row in packed["selected_evidence_frames"] if row["path"] == mug["path"])
    assert frame["frame_relevance_score"] > 0
    assert frame["frame_score_method"] == "stored_description_and_action_time"
    label = image_evidence_label(mug["path"], selected)
    assert label["capture_time_seconds"] == 15 and "blue mug" in label["frame_description"]
    assert label["source_micro_event_id"] == "kitchen"
    prompt, _ = prompt_evidence("How was the mug held?", selected, image_paths=[mug["path"]])
    assert prompt[0]["caption"] and not prompt[0].get("keyframe_captions")


def test_source_observation_must_match_a_saved_capture():
    event = micro("reading", [5, 10], ["A person holds a book.", "A person closes the book."])
    event["visual_observations"] += [
        {"path": "missing.jpg", "timestamp": 9, "description": "Unattached observation."},
        {"path": event["keyframes"][0]["path"], "timestamp": 999, "description": "Wrong capture."}]
    rows = short_term_frame_captions(event)
    assert {row["timestamp"] for row in rows} == {5, 10}
    assert all(row["caption"] for row in rows)
    assert "Wrong capture" not in str(rows) and "Unattached observation" not in str(rows)


def test_backfill_preserves_synopsis_and_only_matching_source_frames():
    source = micro("parcel", [10, 15, 20], ["A parcel is closed.", "A person opens the parcel.", "A parcel is open."])
    record = {"caption": "The person opens a parcel.", "visual_summary": "A cardboard parcel on a table.",
              "start_time": 10, "end_time": 19, "keyframe_paths": [row["path"] for row in source["keyframes"]],
              "source_micro_event_ids": ["parcel"], "keyframe_captions": []}
    original = deepcopy((record, source))
    repaired = inherit_frame_captions(record, [source, micro("unrelated", [15], ["Unrelated source."])])
    assert {row["timestamp"] for row in repaired["keyframe_captions"]} == {10, 15}
    assert repaired["caption"] == record["caption"] and repaired["visual_summary"] == record["visual_summary"]
    assert (record, source) == original
    assert inherit_frame_captions(repaired, [source]) == repaired


def test_empty_native_caption_does_not_hide_stored_image_description():
    event = {"evidence_id": "plant", "source_memory": "M_lt", "keyframes": [{"path": "plant.jpg", "timestamp": 2, "caption": ""}],
             "metadata": {"keyframe_captions": [{"path": "plant.jpg", "timestamp": 2, "caption": "A person waters a plant."}]}}
    assert evidence_frames(event)[0]["caption"] == "A person waters a plant."
    assert image_observation_records(event)[0]["description"] == "A person waters a plant."


def test_event_synopsis_cannot_become_a_single_frame_observation():
    event = {"evidence_id": "walk", "source_memory": "M_lt", "caption": "The person enters and later leaves.",
             "keyframe_captions": [{"path": "door.jpg", "timestamp": 1, "caption": ""}]}
    assert image_observation_records(event) == []
    label = image_evidence_label("door.jpg", [event], [{"path": "door.jpg", "timestamp": 1, "caption": event["caption"]}])
    assert "frame_description" not in label


def test_duplicate_lt_views_do_not_spend_unique_event_budget():
    text = [{"memory_id": f"e{i}", "canonical_segment_id": f"e{i}", "start_time": i*30, "end_time": i*30+30,
             "caption": "A person handles a parcel.", "score": .1 + i*.1} for i in range(6)]
    fused = [{"canonical_segment_id": row["memory_id"], "text": row, "fused_score": row["score"]*.6,
              "start_time": row["start_time"], "end_time": row["end_time"], "visual_items": []} for row in text]
    result = MemoryFusion().fuse({"text_results": text, "fused_results": fused}, {},
        {"final_evidence_k": 8, "retrieval_plan": {"M_lt": {"candidate_budget": 8}}}, "general_qa")
    assert {row["evidence_id"] for row in result["final_evidence"]} == {f"e{i}" for i in range(6)}
    assert result["fusion_summary"]["input_counts"]["M_lt"] == 6
    assert result["fusion_summary"]["long_term_duplicate_views_removed"] == 6


def test_frame_repair_publishes_labels_without_rewriting_memory_text(memory_session):
    import json
    from online_memory.visual_summary_repair import active_records, publish_repair
    root, sid, layout = memory_session
    _, records = active_records(root / sid)
    source = micro("saved", [5], ["A red bag is held with both hands."])
    path = source["keyframes"][0]["path"]
    records[0]["keyframe_paths"] = [path]
    layout.caption_30sec_path.write_text(json.dumps(records))
    original = layout.caption_30sec_path.read_bytes()
    parents = {p.name: p.read_bytes() for p in layout.caption_root.glob("*.json") if "30sec" not in p.name}
    result = publish_repair(root / sid, {}, frame_caption_sources=[source], runtime_validator=lambda *_: None)
    config, updated = active_records(root / sid)
    assert result["version"] == 2 and layout.caption_30sec_path.read_bytes() == original
    assert updated[0]["text"] == records[0]["text"]
    assert updated[0].get("visual_summary") == records[0].get("visual_summary")
    assert updated[0]["keyframe_captions"][0]["caption"] == source["visual_observations"][0]["description"]
    from online_memory_incremental.snapshot_manager import pin_query_snapshot
    pinned = pin_query_snapshot(root / sid, config)
    target = root / sid / pinned["caption_root"]
    assert {name: (target / name).read_bytes() for name in parents} == parents
    assert publish_repair(root / sid, {}, frame_caption_sources=[source], runtime_validator=lambda *_: None)["status"] == "unchanged"
