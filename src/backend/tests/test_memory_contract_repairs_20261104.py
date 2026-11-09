"""Cross-layer memory contracts: real stores/packing, no provider calls."""
import json
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image

from online_current.mcur_store import MCurStore
from online_evidence_content import supplemental_requirement_support
from online_memory_incremental.snapshot_manager import SnapshotManager, pin_query_snapshot
from online_preprocess.io_utils import write_json_atomic
from online_query.evidence_packer import EvidencePacker, image_evidence_label
from online_query.interaction_cache import InteractionCache
from online_query.memory_fusion import MemoryFusion
from online_query.query_cache import SessionEngineCache
from online_query.query_engine import LoadedQueryEngine, _load_memory_config
from online_query_constraints import parse_query_constraints
from online_short_term.micro_event_refiner import MicroEventRefiner


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Memory contract tests must not call a provider")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def prompt_evidence(prompt):
    return json.loads(prompt.rsplit("Evidence: ", 1)[1].split("\n", 1)[0])


def image(session, stamp):
    path = f"stream/frames/frame_{stamp:06d}.jpg"
    target = session / path
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), (stamp % 255, 0, 0)).save(target)
    return path


def test_cached_current_capture_survives_alias_eviction_and_keeps_label(tmp_path):
    session = tmp_path / "s"
    store = MCurStore(session)
    for stamp in (26, 70):
        path = image(session, stamp)
        store.update_from_chunk({"end_time": stamp}, [{"path": path, "timestamp": stamp}], [])
        if stamp == 26:
            current = {**store.load_frames()[0], "source": "M_cur"}
            cache = InteractionCache("s", session)
            cached = cache._trim_evidence_frames([current])[0]
    assert not (session / cached["path"]).exists()
    assert cached["source_path"] == current["source_path"]
    constraints = {"time_range": [26, 26]}
    fused = MemoryFusion().fuse({}, {"memory_route": {"use_interaction_cache": True},
                                   "query_constraints": constraints}, {"final_evidence_k": 2},
                              "followup", {"is_followup": True, "referenced_evidence_frames": [cached]})
    packed = EvidencePacker(session).pack("当时在做什么？", {
        "query_constraints": constraints, "use_image_evidence": True,
        "max_image_evidence": 2, "final_evidence_k": 2}, fused)
    assert packed["selected_image_paths_for_mllm"] == [cached["source_path"]]
    label = image_evidence_label(cached["source_path"], packed["selected_evidence"], packed["selected_evidence_frames"])
    assert label["capture_time_seconds"] == 26
    assert label["cache_reused"] is True and label["original_source"] == "M_cur"


def test_original_capture_fallback_never_substitutes_another_time(tmp_path):
    original = image(tmp_path, 70)
    renamed = tmp_path / "stream/frames/kf_000070.jpg"
    (tmp_path / original).rename(renamed)
    original = str(renamed.relative_to(tmp_path))
    frames, _ = EvidencePacker(tmp_path)._dedup_frames([
        {"path": "current/frames/cur_kf_000026000.jpg", "timestamp": 26, "source_path": original}], 2)
    assert frames == []


@pytest.mark.parametrize("description", [
    "The driver wears a mask and sunglasses.",
    "The driver wears a mask, and the passenger wears sunglasses.",
    "The driver wears a mask and holds sunglasses in his hand.",
    "司机戴着口罩，手里拿着墨镜。",
    "前排男子戴着口罩，后排男子戴着墨镜。",
])
def test_conjunction_text_is_an_inspection_candidate_not_a_fact(description):
    question = "前排男子有没有同时戴着口罩和墨镜？" if description.startswith("前排") else "司机有没有同时戴着口罩和墨镜？"
    observation = {"timestamp": 25, "description": description}
    event = {"doc_id": "e", "start_time": 20, "end_time": 30, "visual_observations": [observation]}
    support = supplemental_requirement_support(question, event, parse_query_constraints(question, 100))
    assert support["confirmed"] is False
    assert support["observation_matched"] is True
    assert support["support_level"] == "text_candidate"
    assert support["observation"] == observation


@pytest.mark.parametrize("raw_kind", ["text", "short_term", "visual"])
def test_authoritative_empty_fusion_never_reintroduces_filtered_memory(raw_kind):
    engine = object.__new__(LoadedQueryEngine)
    late = {"caption": "OUTSIDE_SCOPE", "keyframe_caption": "OUTSIDE_SCOPE", "start_time": 50, "end_time": 60}
    prompt = engine._build_visual_answer_prompt("第0到10秒发生了什么？", "hybrid",
        [late] if raw_kind == "text" else [], [late] if raw_kind == "visual" else [], [],
        short_term_results=[{**late, "event_caption_refined": "OUTSIDE_SCOPE"}] if raw_kind == "short_term" else [],
        selected_evidence=[], query_constraints=parse_query_constraints("第0到10秒发生了什么？", 100))
    assert prompt_evidence(prompt) == []
    assert "OUTSIDE_SCOPE" not in prompt


def test_legacy_visual_fallback_preserves_zero_time():
    engine = object.__new__(LoadedQueryEngine)
    prompt = engine._build_visual_answer_prompt("Describe the event", "visual_only", [],
        [{"start_time": 0.0, "end_time": 10.0, "timestamp": 7, "keyframe_caption": "A visible action."}], [])
    assert prompt_evidence(prompt)[0]["time"] == [0, 10]


def test_published_snapshot_is_pinned_while_writer_stages_another_generation(tmp_path):
    session = tmp_path / "s"
    for folder, filename, payload in [("caption_root", "s_30sec.json", [{"text": "PUBLISHED"}]),
                                      ("sidecar_root", "triples.json", ["OLD_RELATION"]),
                                      ("semantic_root", "facts.json", ["OLD_FACT"])]:
        write_json_atomic(session / "em2mem" / folder / filename, payload)
    snapshot = SnapshotManager(session).build_fast_snapshot(1, components={"episodic": 1, "graph": 0, "semantic": 0},
        long_term_partial_ready=True, long_term_full_ready=False, semantic_lagging=True, graph_lagging=True)
    config = {"status": "memory_ready", "latest_ready_memory_version": 1,
              "latest_snapshot_path": str(snapshot.relative_to(session)), "query_rag_args": {
                  "episodic_caption_root": "em2mem/caption_root", "episodic_sidecar_root": "em2mem/sidecar_root",
                  "semantic_root": "em2mem/semantic_root"}}
    write_json_atomic(session / "em2mem/memory_config.json", config)
    write_json_atomic(session / "em2mem/caption_root/s_30sec.json", [{"text": "UNPUBLISHED"}])
    _, pinned = _load_memory_config(session)
    selected = session / pinned["query_rag_args"]["episodic_caption_root"] / "s_30sec.json"
    assert json.loads(selected.read_text())[0]["text"] == "PUBLISHED"
    assert pinned["lag"]["semantic_lagging"] is True
    assert pinned["latest_semantic_ready_version"] == 0
    assert pinned["query_rag_args"]["semantic_root"].startswith(str(snapshot.relative_to(session)))
    assert json.loads((session / "em2mem/memory_config.json").read_text()) == config


def test_pinning_accepts_edit_layout_but_never_falls_back_from_missing_snapshot(tmp_path):
    (tmp_path / "edited/em2mem/caption_root").mkdir(parents=True)
    edited = pin_query_snapshot(tmp_path, {"latest_snapshot_path": "edited", "query_rag_args": {}})
    assert edited["query_rag_args"]["episodic_caption_root"] == "edited/em2mem/caption_root"
    (tmp_path / "em2mem/caption_root").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        pin_query_snapshot(tmp_path, {"latest_snapshot_path": "missing"})


def test_same_visual_version_publications_are_immutable_and_remain_independently_queryable(tmp_path):
    from online_visual_embedding_builder import _write_visual_checkpoint
    session = tmp_path / "s"
    visual = session / "em2mem/embeddings"
    visual.mkdir(parents=True)
    (visual / "visual_embeddings.pkl").write_bytes(b"FIRST_GENERATION")
    first = _write_visual_checkpoint(session_dir=session, visual_root=visual, visual_version=1,
                                    index_backend="numpy", item_count=1)
    (visual / "visual_embeddings.pkl").write_bytes(b"SECOND_GENERATION")
    second = _write_visual_checkpoint(session_dir=session, visual_root=visual, visual_version=1,
                                     index_backend="numpy", item_count=2)
    assert first != second
    assert (session / first / "visual_embeddings.pkl").read_bytes() == b"FIRST_GENERATION"
    (session / "snapshot/caption_root").mkdir(parents=True)
    pinned = pin_query_snapshot(session, {"latest_snapshot_path": "snapshot", "visual_embedding_ready": True,
        "visual_checkpoint_path": first, "visual_embedding_path": "em2mem/embeddings/visual_embeddings.pkl"})
    assert (session / pinned["visual_embedding_path"]).read_bytes() == b"FIRST_GENERATION"
    assert pinned["visual_embedding_ready"] is True


@pytest.mark.parametrize("payload", [{}, {"refined_caption": " "}, {"refined_caption": []}, {"refined_caption": {"text": "x"}}])
def test_empty_refinement_remains_retryable_and_cannot_promote_placeholder(tmp_path, monkeypatch, payload):
    refiner = MicroEventRefiner(backend="openai")
    monkeypatch.setattr(refiner, "_refine_openai", lambda *args: payload)
    event = {"event_id": "e", "start_time": 0, "end_time": 10, "version": 1,
             "event_caption_placeholder": "Placeholder only", "needs_refine": True}
    result = refiner.refine_event(event, tmp_path)
    assert result["status"] == "refine_failed" and result["needs_refine"] is True
    assert result["version"] == 1 and not result.get("event_caption_refined")
    assert result["refine"]["last_refine_error"]


@pytest.mark.parametrize("text", ["25～26秒", "25到26秒", "25–26 seconds", "0.4166666667至0.4333333333分钟"])
def test_interval_citations_retain_all_actual_captures_and_no_unsent_frames(tmp_path, text):
    cache = InteractionCache("s", tmp_path)
    frames = [{"path": f"frame_{stamp}.jpg", "timestamp": stamp} for stamp in (25, 26, 40)]
    result = {"answer": text + "把手机贴耳。", "evidence_frames": frames,
              "raw": {"image_paths_used": [frames[0]["path"], frames[1]["path"]]}}
    assert [f["timestamp"] for f in cache._answer_cited_frames(result)] == [25, 26]


class Engine:
    def __init__(self, sid):
        self.session_id = sid
        self.last_accessed_at = time.time()
        self.closed = False
    def touch(self): self.last_accessed_at = time.time()
    def needs_reload(self): return False
    def close(self): self.closed = True


@pytest.mark.parametrize("action", ["evict", "install", "invalidate", "clear"])
def test_active_query_engine_survives_retirement_until_last_lease(tmp_path, monkeypatch, action):
    cache = SessionEngineCache(max_sessions=1)
    monkeypatch.setattr(cache, "_close_engine", lambda engine: engine.close())
    engine = Engine("a")
    first, _, _ = cache.get_or_load("a", lambda _: engine, acquire=True)
    entered, resume = threading.Event(), threading.Event()
    observed = []
    def query():
        with cache.borrow("a", lambda _: engine) as (second, _, _):
            entered.set()
            assert resume.wait(5)
            observed.append(second.closed)
    thread = threading.Thread(target=query)
    thread.start()
    assert entered.wait(5)
    if action == "evict": cache.get_or_load("b", lambda _: Engine("b"))
    elif action == "install": cache.install(Engine("a"))
    elif action == "invalidate": cache.invalidate("a")
    else: cache.clear()
    assert not engine.closed
    cache.release(first)
    assert not engine.closed
    resume.set()
    thread.join(5)
    assert observed == [False] and not thread.is_alive()
    assert engine.closed and not cache._retired and not cache._users


def test_query_failure_releases_retired_engine_lease(tmp_path, monkeypatch):
    from online_query import query_engine as module
    write_json_atomic(tmp_path / "s/em2mem/memory_config.json", {"status": "memory_ready", "latest_ready_memory_version": 1})
    cache = SessionEngineCache()
    monkeypatch.setattr(cache, "_close_engine", lambda engine: engine.close())
    class FailingEngine(Engine):
        @contextmanager
        def runtime_session_context(self, **kwargs): yield self
        def query(self, **kwargs):
            cache.invalidate("s")
            assert self.closed is False
            raise RuntimeError("generation failed")
    engine = FailingEngine("s")
    monkeypatch.setattr(module, "load_query_engine", lambda *args, **kwargs: engine)
    with pytest.raises(RuntimeError, match="generation failed"):
        module.query_session("s", "Describe the previous activity", sessions_root=tmp_path,
                             cache=cache, memory_mode="long_term", use_current=False, use_short_term=False)
    assert engine.closed and not cache._users and not cache._retired


def test_shared_engine_serializes_mutable_runtime_contexts(tmp_path):
    session = tmp_path / "long"
    session.mkdir()
    engine = LoadedQueryEngine(session_id="long", session_dir=session,
        memory_config_path=session / "em2mem/memory_config.json", memory_config={},
        em2mem_memory=None, query_args=None, visual_evidence_data=[], semantic_path=None, visual_ready=False)
    entered, attempted, second_entered, resume = (threading.Event() for _ in range(4))
    observed = []
    def first():
        with engine.runtime_session_context(realtime_session_id="a"):
            entered.set()
            assert resume.wait(5)
            observed.append(engine.runtime_session_id)
    def second():
        assert entered.wait(5)
        attempted.set()
        with engine.runtime_session_context(realtime_session_id="b"):
            second_entered.set()
            observed.append(engine.runtime_session_id)
    one, two = threading.Thread(target=first), threading.Thread(target=second)
    one.start(); two.start()
    assert attempted.wait(5) and not second_entered.wait(.05)
    resume.set()
    one.join(5); two.join(5)
    assert observed == ["a", "b"] and not one.is_alive() and not two.is_alive()
