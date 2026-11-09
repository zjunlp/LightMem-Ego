"""Visual evidence must survive generation and storage independently of captions."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from online_memory.evidence_to_em2mem import (
    build_multiscale_caption_items,
    evidence_doc_to_caption_item,
)
from online_short_term.episodic_window_builder import MSTEpisodicWindowBuilder
from online_short_term.mst_to_episodic import _caption_doc_from_episode, _evidence_doc_from_episode


TEXT = "The man says he will call a doctor tomorrow."
VISUAL = "A seated man holds a dark smartphone inside a car."
WINDOW = {"window_id": "win_000000_000030", "start_time": 0.0, "end_time": 30.0}


def _events(frame_path=None):
    return [{
        "event_id": "phone", "start_time": 0.0, "end_time": 30.0,
        "status": "refined", "event_caption_refined": TEXT,
        "scene": "car/interior", "transcript": "I will call a doctor tomorrow.",
        "visual_objects": [{"name": "smartphone"}],
        "keyframes": [{"path": frame_path, "timestamp": 28.0}] if frame_path else [],
    }]


def _text_payload():
    return {
        "caption": TEXT, "fine_caption": TEXT, "scene": "car/interior",
        "main_actions": [], "state_changes": [], "visual_objects": [{"name": "smartphone"}],
        "entities": [], "transcript_summary": "A planned phone call.",
        "source_micro_event_ids": ["phone"], "confidence": 0.9,
    }


def _builder_with_api_boundary(visual_output):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        content = kwargs["messages"][-1]["content"]
        has_images = isinstance(content, list) and any(x.get("type") == "image_url" for x in content)
        if has_images:
            if isinstance(visual_output, Exception):
                raise visual_output
            data = visual_output
        else:
            data = _text_payload()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])

    builder = MSTEpisodicWindowBuilder(backend="openai", model="gpt-test", retries=1)
    builder._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return builder, requests


def _write_frame(tmp_path):
    from PIL import Image

    path = tmp_path / "stream/frames/frame_000028.jpg"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "blue").save(path)
    return str(path.relative_to(tmp_path))


def test_episode_visual_summary_uses_images_and_survives_both_adapters(tmp_path):
    path = _write_frame(tmp_path)
    builder, requests = _builder_with_api_boundary({"visual_summary": VISUAL})
    episode = builder.build_episode(WINDOW, _events(path), session_id="test", session_dir=tmp_path)

    assert episode["fine_caption"] == TEXT
    assert episode.get("visual_summary") == VISUAL
    assert episode["visual_summary_status"] == "generated"
    image_requests = [r for r in requests if isinstance(r["messages"][-1]["content"], list)]
    assert len(image_requests) == 1
    content = image_requests[0]["messages"][-1]["content"]
    image = next(part for part in content if part["type"] == "image_url")
    import base64
    assert base64.b64decode(image["image_url"]["url"].split(",", 1)[1]) == (tmp_path / path).read_bytes()
    assert "28.0" in json.dumps(content)

    for convert in (_caption_doc_from_episode, _evidence_doc_from_episode):
        doc = convert(episode)
        assert doc["visual_summary"] == VISUAL
        assert evidence_doc_to_caption_item("test", doc, 0)["visual_summary"] == VISUAL


def test_same_visual_call_labels_only_matching_attached_unlabelled_frames(tmp_path):
    path = _write_frame(tmp_path)
    builder, requests = _builder_with_api_boundary({"visual_summary": VISUAL, "keyframe_captions": [
        {"timestamp": 28, "path": path, "caption": "A person holds a dark phone low in both hands."},
        {"timestamp": 29, "path": path, "caption": "Wrong timestamp."},
        {"timestamp": 28, "path": "not-attached.jpg", "caption": "Wrong image."}]})
    episode = builder.build_episode(WINDOW, _events(path), session_id="test", session_dir=tmp_path)
    assert len(episode["keyframe_captions"]) == 1
    caption = episode["keyframe_captions"][0]
    assert caption["path"] == path and caption["timestamp"] == 28
    assert caption["description_source"] == "long_term_visual_keyframes"
    assert caption["source_episode_id"] == episode["episode_id"]
    assert "Wrong" not in str(episode["keyframe_captions"])
    assert len(requests) == 2  # One text aggregation plus the existing visual call.
    assert not episode.get("visual_observations")


def test_visual_call_reuses_existing_short_term_labels(tmp_path):
    path = _write_frame(tmp_path)
    builder, requests = _builder_with_api_boundary({"visual_summary": VISUAL, "keyframe_captions": [
        {"timestamp": 28, "path": path, "caption": "An unnecessary replacement."}]})
    events = _events(path)
    events[0]["visual_observations"] = [{"timestamp": 28, "path": path, "description": "A phone rests in the person's hand."}]
    episode = builder.build_episode(WINDOW, events, session_id="test", session_dir=tmp_path)
    assert episode["keyframe_captions"][0]["caption"] == events[0]["visual_observations"][0]["description"]
    image_request = next(row for row in requests if isinstance(row["messages"][-1]["content"], list))
    assert json.loads(image_request["messages"][-1]["content"][0]["text"])["frames_needing_descriptions"] == []


@pytest.mark.parametrize("visual_output", [RuntimeError("vision unavailable"), {}, {"visual_summary": ["invalid"]}])
def test_visual_failure_preserves_text_and_never_uses_it_as_visual_summary(tmp_path, visual_output):
    path = _write_frame(tmp_path)
    builder, _ = _builder_with_api_boundary(visual_output)
    episode = builder.build_episode(WINDOW, _events(path), session_id="test", session_dir=tmp_path)

    assert episode["fine_caption"] == TEXT
    assert episode.get("visual_summary") == ""
    assert episode["visual_summary_status"] == "failed"
    assert episode["visual_summary_debug"]["error"]
    item = evidence_doc_to_caption_item("test", _evidence_doc_from_episode(episode), 0)
    assert item["visual_summary"] == ""


def test_missing_images_do_not_generate_a_visual_summary_from_speech(tmp_path):
    builder, requests = _builder_with_api_boundary({"visual_summary": VISUAL})
    episode = builder.build_episode(WINDOW, _events("missing.jpg"), session_id="test", session_dir=tmp_path)
    assert episode["fine_caption"] == TEXT
    assert episode.get("visual_summary") == ""
    assert episode["visual_summary_status"] == "unavailable"
    assert len(requests) == 1


@pytest.mark.parametrize("legacy_summary", [None, TEXT])
def test_adapter_derives_missing_or_aliased_visual_summary_only_from_visual_metadata(legacy_summary):
    doc = {
        "start_time": 0.0, "end_time": 30.0, "fine_caption": TEXT,
        "scene": {"location": "car/interior"}, "visual_objects": [{"name": "smartphone"}],
        "visual_summary": legacy_summary,
    }
    item = evidence_doc_to_caption_item("test", doc, 0)
    assert item["visual_summary"] == "Scene: car/interior. Visible objects: smartphone."
    assert "doctor" not in item["visual_summary"]


def _children():
    return [evidence_doc_to_caption_item("test", {
        "doc_id": name, "start_time": start, "end_time": start + 30,
        "fine_caption": TEXT, "visual_summary": visual,
    }, i) for i, (name, start, visual) in enumerate([
        ("cup", 0.0, "A blue cup is on the table."),
        ("book", 30.0, "A red book is beside the cup."),
    ])]


def test_rule_multiscale_combines_child_visual_summaries_instead_of_text():
    item = build_multiscale_caption_items(_children(), 180, "3min")[0]
    assert item["visual_summary"] == "A blue cup is on the table. A red book is beside the cup."
    assert "doctor" not in item["visual_summary"]


@pytest.mark.parametrize("response_visual", ["A blue cup and a red book rest on a table.", None])
def test_incremental_rewrite_updates_visual_summary_and_receives_child_visual_evidence(tmp_path, monkeypatch, response_visual):
    import em2mem.llm
    from online_memory_incremental.incremental_store import IncrementalMemoryAppender

    requests = []

    class APIBoundary:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompt, **kwargs):
            requests.append(json.loads(prompt[-1]["content"]))
            return json.dumps({"text": "The speaker discusses a future appointment.", "visual_summary": response_visual})

    monkeypatch.setenv("EM2MEM_INCREMENTAL_DIRTY_MULTISCALE_BACKEND", "llm")
    monkeypatch.setattr(em2mem.llm, "LLMModel", APIBoundary)
    children = _children()
    item = build_multiscale_caption_items(children, 180, "3min")[0]
    appender = IncrementalMemoryAppender(session_id="test", sessions_root=tmp_path)
    appender._maybe_rewrite_dirty_windows_with_llm({item["doc_id"]: item}, children, "3min")

    assert item["text"] == "The speaker discusses a future appointment."
    expected = response_visual or "A blue cup is on the table. A red book is beside the cup."
    assert item["visual_summary"] == expected
    assert requests[0]["child_30s_memories"][0]["visual_summary"] == "A blue cup is on the table."
    assert "visual_summary" in requests[0]["return_fields"]


def test_legacy_evidence_normalization_and_storage_keep_independent_visual_summary():
    from online_preprocess.evidence_schema import normalize_caption_payload
    from online_preprocess.evidence_builder import _build_evidence_doc, _build_captioned_entry

    segment = {"segment_id": "test", "start": 0.0, "end": 30.0, "transcript": "Spoken context."}
    payload = normalize_caption_payload({"fine_caption": TEXT, "visual_summary": VISUAL}, segment)
    doc = _build_evidence_doc("test", segment, payload)
    captioned = _build_captioned_entry(segment, doc)
    assert doc.get("visual_summary") == VISUAL
    assert captioned.get("visual_summary") == VISUAL


def test_failed_child_vision_cannot_be_reintroduced_as_parent_visual_facts(tmp_path, monkeypatch):
    import em2mem.llm
    from online_memory_incremental.incremental_store import IncrementalMemoryAppender

    class APIBoundary:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompt, **kwargs):
            return json.dumps({"text": TEXT, "visual_summary": "An unsupported smartphone scene."})

    monkeypatch.setenv("EM2MEM_INCREMENTAL_DIRTY_MULTISCALE_BACKEND", "llm")
    monkeypatch.setattr(em2mem.llm, "LLMModel", APIBoundary)
    child = evidence_doc_to_caption_item("test", {
        "start_time": 0.0, "end_time": 30.0, "fine_caption": TEXT,
        "visual_objects": [{"name": "smartphone"}],
        "visual_summary": "", "visual_summary_status": "failed",
    }, 0)
    item = build_multiscale_caption_items([child], 180, "3min")[0]
    appender = IncrementalMemoryAppender(session_id="test", sessions_root=tmp_path)
    appender._maybe_rewrite_dirty_windows_with_llm({item["doc_id"]: item}, [child], "3min")
    assert item["visual_summary"] == ""
    assert item["visual_summary_status"] == "unavailable"


def test_mixed_success_multiscale_excludes_failed_child_visual_metadata(tmp_path, monkeypatch):
    import em2mem.llm
    from online_memory_incremental.incremental_store import IncrementalMemoryAppender

    requests = []

    class APIBoundary:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompt, **kwargs):
            requests.append(json.loads(prompt[-1]["content"]))
            return json.dumps({"text": TEXT, "visual_summary": "A blue cup is on a desk."})

    monkeypatch.setenv("EM2MEM_INCREMENTAL_DIRTY_MULTISCALE_BACKEND", "llm")
    monkeypatch.setattr(em2mem.llm, "LLMModel", APIBoundary)
    children = _children()
    children[1].update({
        "visual_summary": "", "visual_summary_status": "failed",
        "scene_summary": {"dominant_scene": "hospital bed"},
        "visual_objects": ["hospital bed"],
    })
    item = build_multiscale_caption_items(children, 180, "3min")[0]
    appender = IncrementalMemoryAppender(session_id="test", sessions_root=tmp_path)
    appender._maybe_rewrite_dirty_windows_with_llm({item["doc_id"]: item}, children, "3min")
    assert "hospital bed" not in json.dumps(requests[0])
    assert requests[0]["child_visual_evidence"][0]["visual_summary"] == "A blue cup is on the table."
    assert item["visual_summary"] == "A blue cup is on a desk."


@pytest.mark.parametrize("vision_fails", [False, True])
def test_full_multiscale_generation_keeps_child_visual_summaries_separate_from_narrative(vision_fails):
    from em2mem.memory.episodic.gen_multiscale import _summarize_batch

    requests = []

    class APIBoundary:
        def generate(self, prompt, **kwargs):
            requests.append(prompt)
            if len(requests) == 1:
                return "A speaker discusses an appointment."
            if vision_fails:
                raise RuntimeError("summary service unavailable")
            return json.dumps({"visual_summary": "A blue cup and a red book are on a table."})

    parent = _summarize_batch(_children(), 0, 180, "Summarize events.", APIBoundary(), "3min", 0)
    assert parent["text"] == "A speaker discusses an appointment."
    expected = "A blue cup is on the table. A red book is beside the cup." if vision_fails else "A blue cup and a red book are on a table."
    assert parent["visual_summary"] == expected
    assert len(requests) == 2
    visual_request = json.dumps(requests[1])
    assert "blue cup" in visual_request and "red book" in visual_request
    assert "doctor" not in visual_request


def test_legacy_vlm_without_actual_images_never_accepts_visual_facts(tmp_path):
    from online_preprocess.vlm_captioner import OpenAIVLMCaptioner
    from online_preprocess.evidence_builder import _build_evidence_doc

    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        data = {"fine_caption": TEXT, "visual_summary": "A hospital bed is visible."}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])

    captioner = OpenAIVLMCaptioner.__new__(OpenAIVLMCaptioner)
    captioner.model, captioner.session_dir, captioner.max_keyframes, captioner.temperature = "gpt-test", tmp_path, 8, 0.0
    captioner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    segment = {"segment_id": "missing", "start": 0.0, "end": 30.0,
               "keyframes": [{"path": "missing.jpg", "timestamp": 5.0}]}
    payload = captioner.caption_segment(segment, ["missing.jpg"], "I will call a doctor tomorrow.")
    assert payload["visual_summary"] == ""
    assert payload["visual_summary_status"] == "unavailable"
    assert _build_evidence_doc("test", segment, payload)["visual_summary_status"] == "unavailable"
    assert requests == []


def test_legacy_vlm_uses_available_paths_even_without_keyframe_records(tmp_path):
    from online_preprocess.vlm_captioner import OpenAIVLMCaptioner

    path = _write_frame(tmp_path)
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        data = {"fine_caption": TEXT, "visual_summary": VISUAL}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])

    captioner = OpenAIVLMCaptioner.__new__(OpenAIVLMCaptioner)
    captioner.model, captioner.session_dir, captioner.max_keyframes, captioner.temperature = "gpt-test", tmp_path, 8, 0.0
    captioner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    payload = captioner.caption_segment({"segment_id": "paths", "start": 0, "end": 30}, [path], "Spoken context.")
    assert sum(part.get("type") == "image_url" for part in requests[0]["messages"][-1]["content"]) == 1
    assert payload["visual_summary"] == VISUAL
    assert payload["visual_summary_status"] == "generated"
