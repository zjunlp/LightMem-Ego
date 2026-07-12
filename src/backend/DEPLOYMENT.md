# Deployment Guide

This guide describes how to deploy the Em2Mem Online Server on a new GPU server with internet access. It assumes the API server, workers, model caches, and optional SRS live media service run on the same machine, so no relay or SSH tunnel is required.

## Target Layout

```text
lightmem_ego-online-server-release/
  .venv/             general API, query, memory, visual, and live ingest runtime
  .venv_whisperx/    ASR/preprocess runtime with WhisperX and CUDA packages
  .env               local secrets and deployment configuration
  online_sessions/   generated runtime sessions, ignored by git
  online_tasks/      generated task queues, ignored by git
  logs/              worker logs, ignored by git
  models/            optional local model cache, ignored by git
```

Use `.venv` for most services. Use `.venv_whisperx` for `online_worker.py`, because WhisperX, PyTorch, CUDA, and audio alignment packages often need a separate dependency set. `scripts/start_online_worker.sh` automatically activates `.venv_whisperx` when it exists.

## System Requirements

- Linux server with an NVIDIA GPU and working CUDA driver.
- Python 3.10 or newer. Python 3.11 is recommended for the general server environment.
- `ffmpeg` and `ffprobe` on `PATH`, or explicit `EM2MEM_FFMPEG_BIN` and `EM2MEM_FFPROBE_BIN`.
- Docker if you want to run the bundled SRS live media helper.
- Outbound internet access for Python packages, Hugging Face model downloads, and OpenAI-compatible APIs.
- OpenAI-compatible API key and base URL, or another configured LLM backend supported by the code.

Install basic system packages with your distribution package manager:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-dev build-essential ffmpeg git curl unzip
```

## Unpack The Release

```bash
cd /opt/em2mem
unzip /path/to/lightmem_ego-online-server-release.zip
cd lightmem_ego-online-server-release
```

Any path is fine. Avoid placing model weights or runtime data inside git-tracked directories unless they are ignored.

## Create `.venv`

The `.venv` environment runs the API, query worker, memory worker, visual worker, stream worker, and live ingest worker.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Install the GPU package set that matches your CUDA driver. A typical setup is:

```bash
python -m pip install torch torchvision --index-url <your-pytorch-cuda-wheel-index>
```

If you want all optional GPU packages in this environment:

```bash
python -m pip install -e ".[gpu]"
```

## Create `.venv_whisperx`

The `.venv_whisperx` environment runs `online_worker.py`, including video preprocessing, chunk ASR, realtime audio ASR, WhisperX, partial transcript storage, and transcript backfill.

```bash
python3.10 -m venv .venv_whisperx
source .venv_whisperx/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[asr,gpu]"
python -m pip install whisperx
```

If WhisperX or PyTorch needs a specific CUDA wheel, install that wheel first, then reinstall WhisperX. Keep `.venv_whisperx` separate from `.venv` when ASR dependencies conflict with query or visual dependencies.

## Configure `.env`

Create a local `.env` from the template:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```bash
EM2MEM_API_HOST=0.0.0.0
EM2MEM_API_PORT=8000
EM2MEM_CORS_ORIGINS=http://localhost:5173,https://your-frontend.example.com

OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=<your-openai-compatible-base-url>

EM2MEM_FFMPEG_BIN=ffmpeg
EM2MEM_FFPROBE_BIN=ffprobe

EM2MEM_ALLOW_HF_DOWNLOAD=1
TRANSFORMERS_OFFLINE=0
HF_HUB_OFFLINE=0
HF_DATASETS_OFFLINE=0
HF_HOME=/data/em2mem/hf

EM2MEM_WHISPERX_MODEL=medium
EM2MEM_WHISPERX_DEVICE=cuda
EM2MEM_WHISPERX_COMPUTE_TYPE=float16
EM2MEM_WHISPERX_MODEL_DIR=/data/em2mem/models/whisperx
EM2MEM_WHISPERX_ALIGN_MODEL_DIR=/data/em2mem/models/whisperx/alignment

EM2MEM_VLM2VEC_MODEL_PATH=/data/em2mem/models/VLM2Vec-V2.0
EM2MEM_VISUAL_BACKEND=vlm2vec
EM2MEM_VLM2VEC_DEVICE=cuda
EM2MEM_VLM2VEC_DTYPE=float16
```

Do not commit `.env`. Keep API keys, tokens, local model paths, and hostnames in `.env` or a deployment secret manager.

## Live Media Without Relay

When SRS and the workers run on the same server, leave the worker pull override empty:

```bash
EM2MEM_LIVE_PULL_BASE_URL=
```

Use direct local/internal pull URLs instead:

```bash
EM2MEM_LIVE_RTMP_ENABLED=1
EM2MEM_WEBRTC_WHIP_ENABLED=1
EM2MEM_LIVE_RTMP_SCHEME=rtmp
EM2MEM_LIVE_RTMP_DOMAIN=<server-public-hostname-or-ip>
EM2MEM_LIVE_RTMP_PUBLIC_PORT=1935
EM2MEM_LIVE_RTMP_INTERNAL_PULL_BASE=rtmp://127.0.0.1:1935/live

EM2MEM_WEBRTC_WHIP_SCHEME=http
EM2MEM_WEBRTC_WHIP_DOMAIN=<server-public-hostname-or-ip>
EM2MEM_WEBRTC_WHIP_PUBLIC_PORT=1985
EM2MEM_WEBRTC_WHIP_PATH=/rtc/v1/whip/
EM2MEM_WEBRTC_WHIP_APP=live
```

If API, SRS, and workers are split across machines later, set `EM2MEM_LIVE_PULL_BASE_URL` only for the worker machine. For the single-server deployment described here, it should stay empty.

## Rolling Audio ASR Defaults

Realtime live audio remains sliced into short ingest chunks, but ASR tasks are scheduled on stable 5 second windows:

```bash
EM2MEM_LIVE_INGEST_AUDIO_SEGMENT_MS=1500
EM2MEM_AUDIO_ASR_ENABLED=1
EM2MEM_AUDIO_ASR_BACKEND=whisperx
EM2MEM_AUDIO_ASR_WINDOW_MS=5000
EM2MEM_AUDIO_ASR_HOP_MS=5000
EM2MEM_AUDIO_ASR_MIN_WINDOW_MS=4500
EM2MEM_AUDIO_ASR_FLUSH_MIN_MS=2000
EM2MEM_AUDIO_ASR_MAX_WINDOW_MS=7000
EM2MEM_AUDIO_ASR_MAX_PENDING_WINDOWS=3
```

On live ingest stop, remaining buffered audio of at least `EM2MEM_AUDIO_ASR_FLUSH_MIN_MS` is queued as a final flush ASR window. Shorter tails are recorded as dropped tail duration in stream status.

## Start SRS

For a local SRS test deployment:

```bash
source .venv/bin/activate
scripts/start_srs_docker.sh
```

Open or proxy these ports as needed:

- `1935/tcp`: RTMP publish and pull.
- `1985/tcp`: SRS HTTP API and WHIP endpoint in the bundled helper.
- `8080/tcp`: SRS HTTP server, if enabled.
- `8000/tcp`: Em2Mem API.

Production deployments should add TLS, authentication, reverse proxy rules, and firewall policy outside this repository.

## Start The API And Workers

Use separate terminals, tmux panes, systemd units, or a process supervisor.

API:

```bash
source .venv/bin/activate
scripts/start_api.sh
```

All workers:

```bash
source .venv/bin/activate
scripts/start_online_all_workers.sh
```

Scale refine workers:

```bash
source .venv/bin/activate
EM2MEM_MST_REFINE_WORKER_COUNT=4 scripts/start_online_all_workers.sh
```

Manual worker split:

```bash
source .venv/bin/activate
scripts/start_online_stream_worker.sh
scripts/start_online_live_ingest_worker.sh
scripts/start_online_mst_refine_worker.sh
scripts/start_online_mst_consolidation_worker.sh
scripts/start_online_visual_worker.sh
scripts/start_online_memory_worker.sh
scripts/start_online_query_worker.sh
```

ASR/preprocess worker:

```bash
source .venv_whisperx/bin/activate
scripts/start_online_worker.sh
```

`scripts/start_online_all_workers.sh` also starts `scripts/start_online_worker.sh`; because that script auto-activates `.venv_whisperx`, the preprocess worker can use the WhisperX environment while other workers use `.venv`.

## Smoke Tests

Health and stream creation:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/stream/start \
  -H 'Content-Type: application/json' \
  -d '{"input_mode":"frame_audio","chunk_duration":5.0}'
```

Ask an asynchronous query:

```bash
curl -X POST http://127.0.0.1:8000/ask/<session_id> \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is happening now?","memory_mode":"auto"}'
curl http://127.0.0.1:8000/query_task/<task_id>
```

Audit live audio ASR windows without triggering ASR:

```bash
python scripts/audit_live_audio_asr_windows.py --session-id <session_id> --json
```

Monitor the pipeline:

```bash
python monitor_online_pipeline.py --watch
```

## Generated Data

The following are generated at runtime and intentionally excluded from the release zip:

- `online_sessions/`
- `online_tasks/`
- `runtime/`
- `logs/`
- model weights and Hugging Face caches
- generated media, FAISS indexes, pickle caches, checkpoints, and uploads

Back up these directories separately if you need to preserve sessions or memory indexes.
