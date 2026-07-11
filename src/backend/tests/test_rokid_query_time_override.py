import importlib.util
import json
import sys
import types
from pathlib import Path


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _install_import_stubs() -> None:
    online_query = sys.modules.get("online_query") or types.ModuleType("online_query")
    online_query.__path__ = [str(Path(__file__).resolve().parents[1] / "online_query")]
    sys.modules["online_query"] = online_query

    online_preprocess = sys.modules.get("online_preprocess") or types.ModuleType("online_preprocess")
    online_preprocess.__path__ = []
    io_utils = sys.modules.get("online_preprocess.io_utils") or types.ModuleType("online_preprocess.io_utils")
    io_utils.read_json = lambda path, default=None: json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default
    io_utils.utc_now_iso = lambda: "2026-07-08T00:00:00+00:00"
    io_utils.write_json = lambda path, data: _write_json(Path(path), data)
    io_utils.write_json_atomic = lambda path, data: _write_json(Path(path), data)
    sys.modules["online_preprocess"] = online_preprocess
    sys.modules["online_preprocess.io_utils"] = io_utils

    for name in (
        "online_current.mcur_query",
        "online_current.mcur_selector",
        "online_current.mcur_store",
        "online_query.coreference_resolver",
        "online_query.evidence_packer",
        "online_query.interaction_cache",
        "online_query.memory_fusion",
        "online_query.memory_plan",
        "online_query.memory_router",
        "online_query.query_router",
        "online_query.stream_query_context",
        "online_short_term.mst_retriever",
        "online_short_term.mst_store",
        "online_pipeline.stream_timeline",
        "online_visual.visual_index",
        "online_visual.visual_items",
        "online_visual.visual_schema",
        "online_visual.vlm2vec_runtime",
        "online_retrieval_scheme",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    sys.modules["online_current.mcur_query"].build_current_prompt = lambda *args, **kwargs: ""
    sys.modules["online_current.mcur_query"].build_local_current_answer = lambda *args, **kwargs: ""
    sys.modules["online_current.mcur_selector"].MCurFrameSelector = object
    sys.modules["online_current.mcur_store"].MCurStore = object
    sys.modules["online_query.coreference_resolver"].CoreferenceResolver = object
    sys.modules["online_query.evidence_packer"].EvidencePacker = object
    sys.modules["online_query.interaction_cache"].InteractionCache = object
    sys.modules["online_query.memory_fusion"].MemoryFusion = object
    sys.modules["online_query.memory_plan"].RetrievalPlanner = object
    sys.modules["online_query.memory_router"].MemoryRouter = object
    sys.modules["online_query.query_router"].QueryRouter = object
    sys.modules["online_query.stream_query_context"].load_stream_query_context = lambda *args, **kwargs: {}
    sys.modules["online_short_term.mst_retriever"].MSTRetriever = object
    sys.modules["online_short_term.mst_store"].MSTStore = object
    sys.modules["online_pipeline.stream_timeline"].append_timeline_event = lambda *args, **kwargs: None
    sys.modules["online_visual.visual_index"].VisualSearchIndex = object
    sys.modules["online_visual.visual_index"].load_visual_index = lambda *args, **kwargs: None
    sys.modules["online_visual.visual_items"].read_visual_items = lambda *args, **kwargs: []
    sys.modules["online_visual.visual_schema"].normalize_retrieval_mode = lambda value: value
    sys.modules["online_visual.vlm2vec_runtime"].get_global_vlm2vec_runtime = lambda *args, **kwargs: None
    sys.modules["online_visual.vlm2vec_runtime"].l2_normalize = lambda value: value
    sys.modules["online_retrieval_scheme"].normalize_long_term_retrieval_scheme = lambda value=None: value or "em2memory"


_install_import_stubs()

MODULE_PATH = Path(__file__).resolve().parents[1] / "online_query" / "query_engine.py"
SPEC = importlib.util.spec_from_file_location("query_engine", MODULE_PATH)
assert SPEC is not None
query_engine = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(query_engine)


def test_rokid_query_time_override_uses_latest_runtime_relative_time(tmp_path: Path) -> None:
    _write_json(tmp_path / "stream" / "rokid_state.json", {"latest_frame_relative_ts_ms": 90_000})
    day_context = {
        "is_rokid_day_child": True,
        "day_label": "DAY2",
        "start_datetime": "2026-07-08 10:03:12",
        "timezone": "Asia/Shanghai",
        "time_source": "client_device",
    }

    override = query_engine._rokid_query_time_override(day_context, tmp_path)

    assert override["until_date"] == "DAY2"
    assert override["until_time"] == "10044200"
    assert override["relative_seconds"] == 90.0
    assert override["display_datetime"] == "2026-07-08 10:04:42"
