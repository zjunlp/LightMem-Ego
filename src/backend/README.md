# WorldMM Online Server

WorldMM Online Server is an online long-video and realtime multimodal memory QA server. It accepts uploaded videos, chunked stream fallback input, direct frame/audio realtime input, and live media ingest sources, then builds current, short-term, and long-term multimodal memories for query-time evidence retrieval and answer generation.

## Key Features

- Full video upload pipeline through `/upload_video`
- Chunk-based online stream fallback through `/stream/start` and `/stream/{session_id}/chunk`
- Realtime frame input through `/stream/{session_id}/frame`
- Realtime audio input through `/stream/{session_id}/audio_chunk`
- Unified realtime ingest adapter for frame/audio events
- Live media ingest worker for RTMP/WHIP-style deployments
- `M_cur` current rolling memory
- `M_st` short-term micro-event memory
- `M_lt` incremental long-term memory
- ASR transcript backfill for stream chunks and audio windows
- Query workers with text, visual, current, short-term, and long-term evidence retrieval

## Architecture

```text
Input Sources
  |-- full video upload
  |-- chunk fallback stream
  |-- realtime frame/audio HTTP input
  |-- live media ingest
        |
        v
Realtime Ingest Adapter
        |
        v
M_cur / M_st / ASR
        |
        v
Refinement / Consolidation
        |
        v
Long-term Memory (M_lt)
        |
        v
Query Worker
        |
        v
Answer + Evidence
```

## Repository Structure

- `api_server.py`: FastAPI entry point and public HTTP API.
- `DEPLOYMENT.md`: GPU server deployment guide, including `.venv` and `.venv_whisperx`.
- `online_worker.py`: preprocessing and ASR worker.
- `online_stream_worker.py`: chunk fallback stream worker.
- `online_live_ingest_worker.py`: live media ingest worker.
- `online_query_worker.py`: asynchronous query worker.
- `online_memory_worker.py`: long-term memory build and update worker.
- `online_visual_worker.py`: visual embedding worker.
- `online_current/`: `M_cur` current memory.
- `online_short_term/`: `M_st` micro-event memory and refinement.
- `online_streaming/`: partial transcript and ASR backfill.
- `online_pipeline/`: realtime ingest, live source, backpressure, runtime state.
- `online_preprocess/`: video segmentation, keyframe sampling, ASR, evidence creation.
- `online_memory/` and `online_memory_incremental/`: WorldMM layout, incremental updates, HippoRAG cache handling.
- `online_query/`: query planning, routing, retrieval, evidence packing, and answer generation.
- `online_visual/`: visual index and VLM2Vec runtime integration.
- `src/worldmm/`: runtime WorldMM memory, LLM, and embedding components used by the server.
- `src/HippoRAG/`: vendored runtime subset needed by long-term retrieval.
- `scripts/`: server, worker, RTMP/SRS, and realtime input helper scripts.
- `deploy/srs/srs.conf`: minimal SRS configuration for local live ingest experiments.

## Requirements

- Python 3.10 or newer.
- `ffmpeg` and `ffprobe` on `PATH`, or explicit `WORLDMM_FFMPEG_BIN` / `WORLDMM_FFPROBE_BIN`.
- Optional CUDA GPU for ASR, visual embeddings, and local model inference.
- OpenAI-compatible API access for LLM-backed captioning, refinement, memory construction, and answering.
- Optional local model weights for WhisperX, Qwen embedding models, VLM2Vec, and VLM captioning. Model weights are not included in this release.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

Edit `.env` with your model paths and API credentials. Then start the API:

```bash
scripts/start_api.sh
```

For a full GPU server deployment, including the split `.venv` / `.venv_whisperx`
environment setup, see `DEPLOYMENT.md`.

Start the default online worker set:

```bash
scripts/start_online_all_workers.sh
```

For a lighter local structure test, use mock visual embeddings:

```bash
WORLDMM_VISUAL_BACKEND=mock scripts/start_online_query_worker.sh
```

## Environment Variables

The release includes `.env.example` with placeholders only. Common variables:

```bash
WORLDMM_API_HOST=127.0.0.1
WORLDMM_API_PORT=8000
WORLDMM_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
WORLDMM_AUTO_PREPROCESS=1
WORLDMM_FRAME_STREAM_MAX_BYTES=8388608
WORLDMM_AUDIO_CHUNK_MAX_BYTES=8388608
WORLDMM_AUDIO_ASR_WINDOW_MS=5000
WORLDMM_AUDIO_ASR_HOP_MS=5000
WORLDMM_AUDIO_ASR_MIN_WINDOW_MS=4500
WORLDMM_AUDIO_ASR_FLUSH_MIN_MS=2000
WORLDMM_LIVE_RTMP_ENABLED=0
WORLDMM_WEBRTC_WHIP_ENABLED=0
WORLDMM_FFMPEG_BIN=ffmpeg
WORLDMM_FFPROBE_BIN=ffprobe
WORLDMM_WHISPERX_MODEL_DIR=/path/to/whisperx
WORLDMM_VLM2VEC_MODEL_PATH=/path/to/VLM2Vec-V2.0
WORLDMM_VISUAL_BACKEND=vlm2vec
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=<optional-openai-compatible-base-url>
```

## API Examples

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

Ask a question asynchronously:

```bash
curl -X POST http://127.0.0.1:8000/ask/<session_id> \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is happening now?","memory_mode":"auto"}'
```

Poll a query task:

```bash
curl http://127.0.0.1:8000/query_task/<task_id>
```

Start or stop live ingest:

```bash
curl -X POST http://127.0.0.1:8000/stream/<session_id>/live/ingest/start
curl -X POST http://127.0.0.1:8000/stream/<session_id>/live/ingest/stop
```

## Worker Commands

```bash
scripts/start_api.sh
scripts/start_online_worker.sh
scripts/start_online_stream_worker.sh
scripts/start_online_query_worker.sh
scripts/start_online_memory_worker.sh
scripts/start_online_visual_worker.sh
scripts/start_online_live_ingest_worker.sh
scripts/start_online_mst_refine_worker.sh
scripts/start_online_mst_consolidation_worker.sh
scripts/start_online_all_workers.sh
```

To start multiple refine workers with one command:

```bash
WORLDMM_MST_REFINE_WORKER_COUNT=4 scripts/start_online_all_workers.sh
```

## Realtime Input Modes

- HTTP frame/audio stream: push frames and audio chunks directly to `/frame` and `/audio_chunk`.
- Chunk fallback: upload video chunks to `/stream/{session_id}/chunk`; the stream worker materializes processing chunks and ASR tasks.
- Live media ingest: create RTMP/WHIP live sources, then run `online_live_ingest_worker.py` to pull frames and audio into the same realtime ingest adapter.
- Rokid Glass backend adapter: start `/stream/start` with `input_mode=rokid_frame_audio`, then have a phone-side Rokid Connector upload JPEG/WebP frames and WAV/PCM audio chunks to the returned `/frame` and `/audio_chunk` URLs. The backend does not call the Rokid SDK; it reuses `ingest_frame`, `ingest_audio_chunk`, `M_cur`, `M_st`, rolling ASR, and the existing query path.

Rokid Connector timestamp rule:

```text
relative_ts_ms = Android SystemClock.elapsedRealtime() - streamStartElapsedMs
```

For the first connector version, convert SDK `NV21` frames to JPEG before upload and prefer WAV 16 kHz mono audio chunks. See `docs/online_stream_api_contract.md` and `docs/stage_rokid_backend_adapter_plan.md`.

## Data And Generated Files

Runtime sessions, task queues, logs, generated indexes, FAISS files, pickles, model weights, uploads, and media outputs are intentionally excluded from this release. They are recreated under ignored runtime directories such as `online_sessions/`, `online_tasks/`, `runtime/`, and `logs/`.

## Security

No secrets, `.env` files, private certificates, tokens, model weights, or server-specific paths are included. Provide credentials through environment variables or deployment secret managers. Do not commit `.env`, runtime data, generated media, task queues, logs, or model artifacts.

## Limitations

- External model dependencies are not bundled.
- Runtime data and generated memory indexes are not included.
- Deployment-specific reverse proxy, TLS, and authentication layers are not included.
- WebRTC/SRS/RTMP production deployments require separate infrastructure and network configuration.
- Local GPU package selection depends on your CUDA, PyTorch, FAISS, and WhisperX environment.

## Citation And Acknowledgements

If you use this code in a paper or artifact, cite the associated WorldMM work when available and acknowledge the external model and retrieval components used in your deployment. This release does not claim any acceptance venue or benchmark result by itself.
