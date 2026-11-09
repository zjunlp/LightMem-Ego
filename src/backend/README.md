<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../figs/lockup_compact_dark.png">
    <img src="../../figs/lockup_compact.png" width="420" alt="LightMem-Ego">
  </picture>
</div>

# LightMem-Ego · Backend

<p>
  <a href="../../README.md">← Back to LightMem-Ego</a> &nbsp;·&nbsp;
  <a href="DEPLOYMENT.md">Deployment Guide</a> &nbsp;·&nbsp;
  <a href="../../deploy/DOCKER.md">Docker</a> &nbsp;·&nbsp;
  <a href="docs/online_stream_api_contract.md">API Contract</a>
</p>

The FastAPI service and worker pipeline behind [LightMem-Ego](../../README.md). It accepts a full video upload, chunked or realtime frame/audio streams, and live RTMP/WHIP ingest; builds current, short-term, and long-term multimodal memory; and answers questions with timestamped evidence.

The installable package is named `em2mem-online-server`. The long-term memory tier (`M_lt`) is built by **EM²Mem** — our event-centric multimodal memory framework, accepted at **EMNLP 2026 Findings** ([arXiv:2609.00551](https://arxiv.org/abs/2609.00551)).

The [online memory upgrade notes](docs/online_memory_upgrade.md) describe retrieval,
evidence packing, cache/publication consistency, historical-video APIs, and the
offline regression suite.

## ✨ Capabilities

- **Multiple input modes** — full video upload, chunk fallback streams, realtime frame/audio HTTP input, and live media ingest (RTMP/WHIP via SRS).
- **Three-tier memory** — `M_cur` current rolling memory, `M_st` short-term micro-events, and `M_lt` incremental long-term memory built by EM²Mem.
- **ASR pipeline** — rolling audio windows and transcript backfill for stream chunks, with WhisperX or the Xfyun WebAPI as backends.
- **Evidence-grounded QA** — text, visual, current, short-term, and long-term retrieval, packed into a compact evidence view before answering.
- **Streaming answers** — server-sent-event endpoints for token-by-token replies.
- **Memory management** — inspection, editing, versioning, rollback, and a memory graph API.
- **Worker isolation** — preprocessing, ASR, memory, visual embedding, and query work each run as separate processes.

## 🚀 Quick Start

> [!IMPORTANT]
> Two things are required: an OpenAI-compatible LLM endpoint (base URL, API key, model names), and Xfyun ASR credentials if you want speech transcribed. A GPU and local model weights are optional.

### Docker (recommended)

The root Compose stack builds this backend, its workers, and the web frontend, and configures Xfyun ASR plus mock visual embeddings so it starts without model weights:

```bash
cp deploy/.env.example .env     # fill in your LLM endpoint, keys, and model names
docker compose up --build
```

> [!NOTE]
> The first build takes a few minutes. Visual embeddings default to `mock` so the stack comes up without model weights; enable the GPU profiles for full visual and text retrieval.

See [`deploy/DOCKER.md`](../../deploy/DOCKER.md) for GPU model services, SRS/RTMP live ingest, and data persistence.

### From source

**Requirements**

- Python 3.10 or newer.
- `ffmpeg` and `ffprobe` on `PATH`, or explicit `EM2MEM_FFMPEG_BIN` / `EM2MEM_FFPROBE_BIN`.
- An OpenAI-compatible API endpoint for captioning, refinement, memory construction, and answering.
- Optional: a CUDA GPU and local model weights for WhisperX, text embeddings, VLM2Vec, and local LLM inference. Weights are not included.

```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .            # add extras: -e ".[asr]", ".[gpu]", ".[dev]"
cp .env.example .env                  # configure model paths and API credentials
scripts/start_api.sh
```

Then start the worker set in another shell:

```bash
scripts/start_online_all_workers.sh
```

Verify the API is up:

```bash
curl http://127.0.0.1:8000/ping
```

For the full GPU server walkthrough — split `.venv` / `.venv_whisperx` environments, model downloads, and smoke tests — see [`DEPLOYMENT.md`](DEPLOYMENT.md).

## 🏗️ Architecture

```text
Input Sources
  |-- full video upload
  |-- chunk fallback stream
  |-- realtime frame/audio HTTP input
  |-- live media ingest (RTMP / WHIP)
        |
        v
Realtime Ingest Adapter
        |
        v
M_cur / M_st / rolling ASR
        |
        v
Refinement / Consolidation
        |
        v
Long-term Memory (M_lt)
        |
        v
Query Worker -> Answer + Evidence
```

| Worker | Role |
| :--- | :--- |
| `online_worker.py` | Preprocessing and ASR. |
| `online_stream_worker.py` | Chunk fallback stream processing. |
| `online_live_ingest_worker.py` | Live media ingest pull for RTMP/WHIP sources. |
| `online_asr_worker.py` | Standalone ASR tasks. |
| `online_evidence_worker.py` | Evidence construction. |
| `online_memory_worker.py` | Long-term memory build and update. |
| `online_mst_refine_worker.py` | Short-term micro-event refinement. |
| `online_mst_consolidation_worker.py` | Micro-event consolidation into `M_lt`. |
| `online_visual_worker.py` | Visual embedding and indexing. |
| `online_query_worker.py` | Asynchronous query execution. |

## 📊 Results

### End-to-end pipeline

Evaluated on a small, intentionally balanced set — 27 queries (nine per scenario) over five everyday-life videos, about 45.7 minutes of footage — driven from the phone and glasses-style clients. Retrieval, time to first token and end-to-end QA are reported as P50 / P90; retrieval is scored against the evidence that supports the answer.

**Retrieval accuracy**

| Scenario | R@1 | R@3 | R@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Object finding | 22.2 | 66.7 | 77.8 | 0.454 |
| Conversation recall | 44.4 | 55.6 | 55.6 | 0.481 |
| Life summarization | 88.9 | 100.0 | 100.0 | 0.944 |
| **Overall** | **51.9** | **74.1** | **77.8** | **0.627** |

**Latency** (P50 / P90)

| Stage | Phone P50 | Phone P90 | Glasses P50 | Glasses P90 |
| :--- | :---: | :---: | :---: | :---: |
| *Short-term memory QA* | | | | |
| Retrieval | 76 ms | 131 ms | 44 ms | 87 ms |
| Time to first token | 532 ms | 643 ms | 423 ms | 494 ms |
| Answer generation | 6.13 s | 10.11 s | 6.81 s | 9.14 s |
| **End-to-end** | **6.42 s** | **10.34 s** | **6.95 s** | **9.31 s** |
| *Long-term memory QA* | | | | |
| Retrieval | 2.99 s | 3.84 s | 3.06 s | 3.44 s |
| Time to first token | 4.64 s | 5.44 s | 4.74 s | 5.07 s |
| Answer generation | 5.78 s | 9.56 s | 4.37 s | 9.16 s |
| **End-to-end** | **10.57 s** | **13.93 s** | **8.61 s** | **13.60 s** |

Answer accuracy is 51.9% (LLM-judge) and 55.6% (human) overall — see the [main README](../../README.md#results) for the per-scenario breakdown.

### Long-term memory engine — EM²Mem

The long-term memory tier (`M_lt`) is built by EM²Mem. Average accuracy (%) across three long-video and egocentric benchmarks, as reported in the EM²Mem paper:

| Method | EgoLifeQA | Ego-R1 Bench | Video-MME (L) |
| :--- | :---: | :---: | :---: |
| GPT-5 | 48.6 | 46.3 | 74.3 |
| HippoRAG | 59.6 | 56.0 | 52.1 |
| M3-Agent | 53.5 | 52.0 | 55.3 |
| Ego-R1 | 53.0 | 52.0 | 42.7 |
| WorldMM | 65.6 | 65.3 | 76.6 |
| **EM²Mem** | **66.0** | **67.7** | **76.8** |

Against the strongest baseline (WorldMM, reproduced under the same evaluation setting), EM²Mem also averages **98.21 s** per query versus 459.00 s (**4.67× faster**) and uses **63.7% fewer** total tokens.

<details>
<summary><b>Full per-category results</b> — EgoLifeQA, Ego-R1 Bench, Video-MME (L)</summary>

**EgoLifeQA** — Ent. / EvR. / Hab. / Rel. / Task. EM²Mem row in bold; † marks WorldMM reproduced under the same evaluation setting.

| Method | Ent. | EvR. | Hab. | Rel. | Task | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B | 35.2 | 30.2 | 39.3 | 46.4 | 46.0 | 38.6 |
| Gemini 2.5 Pro | 43.2 | 40.5 | 41.0 | 55.2 | 52.4 | 46.4 |
| GPT-5 | 47.2 | 42.1 | 47.5 | 53.6 | 55.6 | 48.6 |
| VideoChat-Flash | 28.8 | 32.5 | 37.7 | 37.6 | 38.1 | 34.2 |
| Time-R1 | 39.2 | 50.8 | 65.6 | 48.8 | 47.6 | 48.8 |
| Video-RTS | 40.8 | 48.4 | 62.3 | 48.8 | 47.6 | 48.2 |
| LightRAG | 40.8 | 48.4 | 67.2 | 50.4 | 44.4 | 48.8 |
| HippoRAG | 48.8 | 60.3 | 70.5 | 60.8 | 66.7 | 59.6 |
| Video-RAG | 49.6 | 56.3 | 67.2 | 55.2 | 54.0 | 55.4 |
| EgoRAG | 40.0 | 56.3 | 62.3 | 54.4 | 52.4 | 52.0 |
| Ego-R1 | 51.2 | 53.2 | 63.9 | 50.4 | 50.8 | 53.0 |
| HippoMM | 45.6 | 53.2 | 70.5 | 55.2 | 58.7 | 54.6 |
| M3-Agent | 44.4 | 54.8 | 62.3 | 56.8 | 54.0 | 53.5 |
| WorldMM | 62.4 | 64.3 | 75.4 | 62.4 | 71.4 | 65.6 |
| WorldMM† | 57.6 | 65.1 | 68.9 | 68.8 | 60.3 | 64.0 |
| **EM²Mem** | **60.8** | 61.1 | 63.9 | **72.8** | **74.6** | **66.0** |

**Ego-R1 Bench**

| Method | Ent. | EvR. | Hab. | Rel. | Task | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B | 31.8 | 41.5 | 38.5 | 42.1 | 44.7 | 35.7 |
| Gemini 2.5 Pro | 43.9 | 56.1 | 53.9 | 47.4 | 47.4 | 46.7 |
| GPT-5 | 41.8 | 58.5 | 53.9 | 52.6 | 50.0 | 46.3 |
| VideoChat-Flash | 43.4 | 43.9 | 38.5 | 31.6 | 44.7 | 42.7 |
| Time-R1 | 49.2 | 48.8 | 46.2 | 42.1 | 44.7 | 48.0 |
| Video-RTS | 47.6 | 46.3 | 53.9 | 52.6 | 47.4 | 48.0 |
| LightRAG | 54.0 | 61.0 | 46.2 | 42.1 | 42.1 | 52.3 |
| HippoRAG | 54.5 | 65.9 | 69.2 | 52.6 | 50.0 | 56.0 |
| Video-RAG | 48.7 | 58.5 | 53.9 | 47.4 | 44.7 | 49.7 |
| EgoRAG | 46.6 | 56.1 | 46.2 | 47.4 | 55.3 | 49.0 |
| Ego-R1 | 50.8 | 63.4 | 38.5 | 36.8 | 57.9 | 52.0 |
| HippoMM | 51.9 | 56.1 | 46.2 | 52.6 | 57.9 | 53.0 |
| M3-Agent | 52.4 | 58.5 | 38.5 | 42.1 | 52.6 | 52.0 |
| WorldMM | 64.6 | 70.7 | 76.9 | 57.9 | 63.2 | 65.3 |
| **EM²Mem** | **74.6** | 53.7 | 69.2 | 47.4 | 57.9 | **67.7** |

**Video-MME (L)**

| Method | ARES | AREC | ATTR | CNT | ISYN | OCR | ORES | OREC | SPER | SRES | TPER | TRES | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B | 62.2 | 54.0 | 51.9 | 43.8 | 68.1 | 42.9 | 62.9 | 57.4 | 33.3 | 45.5 | 33.3 | 67.0 | 61.0 |
| Gemini 2.5 Pro | 56.9 | 47.6 | 66.7 | 41.7 | 71.8 | 57.1 | 53.3 | 40.7 | 0.0 | 72.7 | 66.7 | 48.4 | 55.7 |
| GPT-5 | 71.1 | 69.8 | 70.4 | 47.9 | 88.3 | 57.1 | 75.8 | 74.1 | 33.3 | 72.7 | 50.0 | 75.8 | 74.3 |
| VideoChat-Flash | 35.0 | 42.9 | 37.0 | 31.3 | 34.4 | 42.9 | 60.0 | 46.3 | 33.3 | 54.5 | 33.3 | 46.2 | 44.1 |
| Time-R1 | 20.6 | 28.6 | 25.9 | 35.4 | 31.9 | 35.7 | 53.3 | 48.2 | 33.3 | 36.4 | 50.0 | 44.0 | 37.6 |
| Video-RTS | 43.3 | 52.4 | 40.7 | 39.6 | 33.7 | 42.9 | 60.8 | 53.7 | 33.3 | 45.5 | 50.0 | 49.5 | 47.9 |
| LightRAG | 41.7 | 30.2 | 40.7 | 35.4 | 54.0 | 50.0 | 46.7 | 61.1 | 33.3 | 45.5 | 50.0 | 52.8 | 46.6 |
| HippoRAG | 45.6 | 47.6 | 40.7 | 37.5 | 52.2 | 42.9 | 52.9 | 64.8 | 66.7 | 54.5 | 50.0 | 70.3 | 52.1 |
| Video-RAG | 51.7 | 47.6 | 37.0 | 39.6 | 49.7 | 57.1 | 62.1 | 68.5 | 66.7 | 45.5 | 50.0 | 68.1 | 55.4 |
| EgoRAG | 31.1 | 55.6 | 33.3 | 22.9 | 41.1 | 28.6 | 44.6 | 48.2 | 33.3 | 54.5 | 66.7 | 48.4 | 41.1 |
| Ego-R1 | 37.2 | 52.4 | 40.7 | 35.4 | 38.0 | 35.7 | 42.1 | 51.9 | 66.7 | 63.6 | 50.0 | 52.8 | 42.7 |
| HippoMM | 41.1 | 42.9 | 55.6 | 35.4 | 38.7 | 35.7 | 37.9 | 53.7 | 33.3 | 54.5 | 50.0 | 47.3 | 41.6 |
| M3-Agent | 52.2 | 57.1 | 59.3 | 45.8 | 51.5 | 42.9 | 54.6 | 64.8 | 33.3 | 45.5 | 50.0 | 71.4 | 55.3 |
| WorldMM | 81.1 | 73.0 | 70.4 | 54.2 | 85.3 | 42.9 | 75.0 | 77.8 | 33.3 | 72.7 | 66.7 | 79.1 | 76.6 |
| WorldMM† | 73.3 | 68.3 | 77.8 | 60.4 | 80.2 | 50.0 | 72.4 | 77.8 | 33.3 | 90.9 | 66.7 | 71.1 | 73.1 |
| **EM²Mem** | 77.2 | **76.2** | **77.8** | **64.6** | 80.7 | **64.3** | **77.0** | **77.8** | **33.3** | 81.8 | 50.0 | **79.1** | **76.8** |

Reproduction scripts: [`experiments/egolife`](https://github.com/zjunlp/LightMem/tree/main/experiments/egolife#results).

</details>

## ⚙️ Configuration

All settings come from environment variables; the release ships placeholders in [`.env.example`](.env.example).

> [!NOTE]
> The two env files disagree on purpose: the source `.env.example` defaults to ASR backend `whisperx` (a local GPU model), while the Docker stack defaults to `xfyun` (a hosted WebAPI). Pick deliberately via `EM2MEM_STREAM_ASR_BACKEND` / `EM2MEM_AUDIO_ASR_BACKEND`.

<details>
<summary><b>API and pipeline</b></summary>

```bash
EM2MEM_API_HOST=127.0.0.1
EM2MEM_API_PORT=8000
EM2MEM_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
EM2MEM_PIPELINE_MODE=mst
EM2MEM_AUTO_PREPROCESS=1
EM2MEM_AUTO_MEMORY=1
EM2MEM_AUTO_MST_CONSOLIDATION=1
EM2MEM_AUTO_VISUAL_EMBEDDING=1
```

</details>

<details>
<summary><b>Models and API credentials</b></summary>

```bash
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=<optional-openai-compatible-base-url>
EM2MEM_MEMORY_MODEL=gpt-5.4
EM2MEM_QUERY_RESPOND_MODEL=gpt-5.4
EM2MEM_MST_REFINE_MODEL=gpt-5.4
EM2MEM_MST_EPISODIC_MODEL=gpt-5.4
```

</details>

<details>
<summary><b>ASR</b></summary>

`EM2MEM_STREAM_ASR_BACKEND` / `EM2MEM_AUDIO_ASR_BACKEND` select the backend — see the note under [Configuration](#-configuration).

```bash
EM2MEM_STREAM_ASR_BACKEND=whisperx      # or xfyun
EM2MEM_AUDIO_ASR_BACKEND=whisperx       # or xfyun
EM2MEM_AUDIO_ASR_WINDOW_MS=5000
EM2MEM_AUDIO_ASR_HOP_MS=5000
EM2MEM_WHISPERX_MODEL=medium
EM2MEM_WHISPERX_DEVICE=cuda
EM2MEM_WHISPERX_MODEL_DIR=/path/to/whisperx
# Xfyun WebAPI credentials (required when the backend is xfyun)
EM2MEM_XFYUN_APP_ID=
EM2MEM_XFYUN_API_KEY=
EM2MEM_XFYUN_API_SECRET=
```

</details>

<details>
<summary><b>Visual and text embeddings</b></summary>

```bash
EM2MEM_VISUAL_BACKEND=remote            # or vlm2vec / mock
EM2MEM_VLM2VEC_MODEL_PATH=/path/to/VLM2Vec-V2.0
EM2MEM_VLM2VEC_EMBED_URL=http://127.0.0.1:18091
EM2MEM_ALLOW_HF_DOWNLOAD=0
```

`EM2MEM_VISUAL_BACKEND=mock` is useful for structural tests without model weights:

```bash
EM2MEM_VISUAL_BACKEND=mock scripts/start_online_query_worker.sh
```

</details>

<details>
<summary><b>Streaming limits</b></summary>

```bash
EM2MEM_FRAME_STREAM_MAX_BYTES=8388608
EM2MEM_AUDIO_CHUNK_MAX_BYTES=8388608
EM2MEM_FRAME_STREAM_TARGET_FPS=1
EM2MEM_STREAM_PROCESSING_CHUNK_SECONDS=5
EM2MEM_LIVE_RTMP_ENABLED=0
EM2MEM_WEBRTC_WHIP_ENABLED=0
```

</details>

## 🔌 API

| Group | Endpoints |
| :--- | :--- |
| **Health** | `GET /ping`, `GET /runtime`, `GET /pipeline_runtime` |
| **Sessions** | `POST /stream/start`, `POST /rokid/stream/start`, `GET /stream/{session_id}/status`, `POST /stream/{session_id}/end` |
| **Capture** | `POST /stream/{session_id}/frame`, `POST /stream/{session_id}/audio_chunk`, `POST /stream/{session_id}/chunk` |
| **Live ingest** | `POST /stream/{session_id}/live/ingest/start`, `POST /stream/{session_id}/live/ingest/stop` |
| **Memory** | `GET /session/{session_id}/current`, `GET /session/{session_id}/short_term`, `PUT /session/{session_id}/memories/30sec`, `GET /session/{session_id}/memory-graph` |
| **Queries** | `POST /ask/{session_id}`, `POST /ask/{session_id}/stream`, `GET /query_task/{task_id}`, `GET /session/{session_id}/qa_history` |
| **Media** | `GET /session/{session_id}/file?path=...`, `GET /stream/{session_id}/preview`, `POST /upload_video` |

<details>
<summary><b>curl examples</b></summary>

Start a stream:

```bash
curl -X POST http://127.0.0.1:8000/stream/start \
  -H 'Content-Type: application/json' \
  -d '{"input_mode":"frame_audio","chunk_duration":5.0}'
```

Send one frame:

```bash
curl -X POST http://127.0.0.1:8000/stream/<session_id>/frame \
  -F frame=@/path/to/frame.jpg \
  -F client_ts_ms=1710000000000 \
  -F relative_ts_ms=0
```

Send one audio chunk:

```bash
curl -X POST http://127.0.0.1:8000/stream/<session_id>/audio_chunk \
  -F audio=@/path/to/audio.wav \
  -F audio_index=0 \
  -F client_ts_ms=1710000000000 \
  -F relative_ts_ms=0 \
  -F duration_ms=1000
```

Ask a question and poll the task:

```bash
curl -X POST http://127.0.0.1:8000/ask/<session_id> \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is happening now?","memory_mode":"auto"}'

curl http://127.0.0.1:8000/query_task/<task_id>
```

</details>

The full request/response contract lives in [`docs/online_stream_api_contract.md`](docs/online_stream_api_contract.md).

### Realtime input modes

- **HTTP frame/audio stream** — push frames and audio chunks to `/frame` and `/audio_chunk`.
- **Chunk fallback** — upload video chunks to `/stream/{session_id}/chunk`; the stream worker schedules processing and ASR tasks.
- **Live media ingest** — create RTMP/WHIP sources, then run `online_live_ingest_worker.py` to pull frames and audio into the same ingest adapter.
- **Rokid Glass adapter** — start `/stream/start` with `input_mode=rokid_frame_audio`, then upload JPEG/WebP frames and WAV/PCM audio chunks. The backend never calls the Rokid SDK; it reuses `ingest_frame`, `ingest_audio_chunk`, `M_cur`, `M_st`, rolling ASR, and the existing query path. Timestamps follow `relative_ts_ms = SystemClock.elapsedRealtime() - streamStartElapsedMs`.

## 🧵 Scripts

| Script | Purpose |
| :--- | :--- |
| `scripts/start_api.sh` | Start the FastAPI server. |
| `scripts/start_online_all_workers.sh` | Start the default worker set. |
| `scripts/start_online_worker.sh` | Preprocessing/ASR worker. |
| `scripts/start_online_stream_worker.sh` | Chunk fallback stream worker. |
| `scripts/start_online_query_worker.sh` | Query worker. |
| `scripts/start_online_memory_worker.sh` | Long-term memory worker. |
| `scripts/start_online_visual_worker.sh` | Visual embedding worker. |
| `scripts/start_online_live_ingest_worker.sh` | Live ingest worker. |
| `scripts/start_online_mst_refine_worker.sh` | Micro-event refinement worker. |
| `scripts/start_online_mst_consolidation_worker.sh` | Micro-event consolidation worker. |
| `scripts/stop_server_and_workers.sh` | Stop the API and workers. |
| `scripts/smoke_test_local_qwen35.py` | Smoke-test a local Qwen3.5 endpoint. |

Run multiple refinement workers with one command:

```bash
EM2MEM_MST_REFINE_WORKER_COUNT=4 scripts/start_online_all_workers.sh
```

## 🖥️ Local Models

The pipeline can run against remote OpenAI-compatible endpoints or local model servers. The local Qwen3.5 profile serves an OpenAI-compatible API from an isolated vLLM environment:

```bash
scripts/setup_local_qwen35_env.sh
scripts/download_local_qwen35_model.sh
scripts/select_llm_profile.sh local-qwen35
scripts/stop_server_and_workers.sh --keep-api --force
```

It serves Qwen3.5 under the configured model name (`Qwen3.5-9B` by default), keeps the external model for retrieval/refine workers, uses GPU 2 by default, and moves stream ASR to GPU 1. To return to the configured remote endpoint:

```bash
scripts/select_llm_profile.sh remote
scripts/stop_server_and_workers.sh --keep-api
```

## 🧪 Tests

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## 📁 Layout

- `api_server.py` — FastAPI entry point and public HTTP API.
- `online_current/` — `M_cur` current memory.
- `online_short_term/` — `M_st` micro-events and refinement.
- `online_streaming/` — partial transcripts and ASR backfill.
- `online_pipeline/` — realtime ingest, live sources, backpressure, runtime state.
- `online_preprocess/` — video segmentation, keyframe sampling, ASR, evidence creation.
- `online_memory/`, `online_memory_incremental/` — EM²Mem layout, incremental updates, HippoRAG cache handling.
- `online_query/` — query planning, routing, retrieval, evidence packing, answer generation.
- `online_visual/` — visual index and VLM2Vec runtime integration.
- `online_memory_edit/`, `online_memory_view/` — memory editing and inspection APIs.
- `src/em2mem/` — runtime memory, LLM, and embedding components used by the server.
- `src/HippoRAG/` — vendored runtime subset used by long-term retrieval.
- `scripts/`, `deploy/srs/srs.conf` — worker/server helpers and a minimal SRS config.

## 🔐 Security And Data

> [!WARNING]
> Never commit `.env`. This release ships no secrets, environment files, certificates, tokens, model weights, or server-specific paths — supply credentials through environment variables or a deployment secret manager.

Runtime sessions, task queues, logs, generated indexes, FAISS files, uploads, and media outputs are written to Git-ignored directories such as `online_sessions/`, `online_tasks/`, `runtime/`, and `logs/`, and are excluded from this release.

## ⚠️ Limitations

- External model dependencies and model weights are not bundled.
- Runtime data and generated memory indexes are not included.
- Reverse proxy, TLS, and authentication layers are deployment-specific and not included.
- WebRTC/SRS/RTMP deployments need separate infrastructure and network configuration.
- Local GPU package selection depends on your CUDA, PyTorch, FAISS, and WhisperX environment.

## 📄 Citation And License

If you use this backend in a paper or artifact, cite **LightMem-Ego**. The long-term memory tier (`M_lt`) is built by **EM²Mem** (accepted at **EMNLP 2026 Findings**), so cite it as well when you use that module.

```bibtex
@article{chen2026lightmemego,
  title={LightMem-Ego: Your AI Memory for Everyday Life},
  author={Chen, Yijun and Xiao, Boyi and Zhao, Yixian and Xia, Haoting and Xu, Buqiang and Fang, Jizhan and Li, Yanya and Zheng, Yaqi and Wang, Xuehai and Xue, Zirui and others},
  journal={arXiv preprint arXiv:2607.11487},
  year={2026}
}
```

```bibtex
@article{chen2026em2mem,
  title={EM$^{2}$Mem: Event-Centric Multimodal Memory for Large Language Models},
  author={Chen, Yijun and Zheng, Yaqi and Li, Yanya and Xiao, Boyi and Xu, Buqiang and Qiao, Shuofei and Fang, Jizhan and Deng, Xinle and Yao, Yunzhi and Wang, Xuehai and others},
  journal={arXiv preprint arXiv:2609.00551},
  year={2026}
}
```

Released under the repository [`LICENSE`](../../LICENSE).
