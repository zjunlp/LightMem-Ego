<div align="center">
  <img src="./figs/lightmem_ego_crop.png" width="62%" alt="LightMem-Ego Logo">
</div>

<h1 align="center">LightMem-Ego: Your AI Memory for Everyday Life</h1>

<p align="center">
  <b>A streaming multimodal memory system for smart glasses, web capture, and everyday-life question answering.</b>
</p>

<p align="center">
  <a href="#citation">
    <img src="https://img.shields.io/badge/Paper-Coming%20Soon-red" alt="Paper">
  </a>
  <a href="https://github.com/zjunlp/LightMem-Ego">
    <img src="https://img.shields.io/github/stars/zjunlp/LightMem-Ego?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/zjunlp/LightMem-Ego/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-See%20LICENSE-green.svg" alt="License">
  </a>
  <img src="https://img.shields.io/github/last-commit/zjunlp/LightMem-Ego?color=blue" alt="Last Commit">
  <img src="https://img.shields.io/badge/PRs-Welcome-blue" alt="PRs Welcome">
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#demo">Demo</a> |
  <a href="#system-design">System Design</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#repository-layout">Repository Layout</a> |
  <a href="#related-works">Related Works</a> |
  <a href="#privacy-notice">Privacy</a>
</p>

---

<span id="overview"></span>

## Overview

**LightMem-Ego** is an end-to-end egocentric memory system for everyday-life assistance. It connects a Rokid AI Glass Android app, a browser frontend, and an online backend service so users can stream first-person camera/audio context, build structured memory from daily experience, and ask questions about current or past moments.

The system organizes continuous visual-audio experience into three memory scopes:

- **Current memory** for ongoing scene understanding.
- **Short-term memory** for recent events, actions, and conversations.
- **Long-term memory** for consolidated episodes, routines, preferences, and semantic facts.

LightMem-Ego is designed for practical scenarios such as object finding, conversation recall, life summarization, routine discovery, and hands-free wearable assistance.

<div align="center">
  <img src="./figs/system_design.png" width="95%" alt="LightMem-Ego System Design">
</div>

---

<span id="highlights"></span>

## Highlights

- **Streaming egocentric capture**: captures first-person visual frames and microphone audio from smart glasses or the browser.
- **Timeline-aligned multimodal memory**: aligns frames, audio chunks, transcripts, and metadata on a shared session timeline.
- **Hierarchical memory organization**: maintains current, short-term, and long-term memory for different temporal scopes.
- **Memory-grounded question answering**: retrieves timestamped multimodal evidence before generating answers.
- **Glasses + web deployment**: supports a Rokid AI Glass app for hands-free interaction and a browser frontend for desktop/mobile use.
- **Modular backend**: separates stream ingestion, session management, memory construction, retrieval, and QA workers.

---

<span id="demo"></span>

## Demo

Demo video: [YouTube](https://www.youtube.com/watch?v=BZuIxn00xlc) · [Bilibili](https://www.bilibili.com/video/BV1oANw62EA3/?vd_source=2537e8437f33dacc6255c196ac8292c3)

<p align="center">
  <a href="https://www.bilibili.com/video/BV1oANw62EA3/">
    <img src="./figs/thumbnail.png" width="80%" alt="LightMem-Ego Demo Video">
  </a>
</p>

<p align="center">
  <a href="https://www.bilibili.com/video/BV1oANw62EA3/">
    Watch the full demo video
  </a>
</p>

---

<span id="system-design"></span>

## System Design

LightMem-Ego is organized as three cooperating components:

1. **AI Glass App**
   Captures first-person camera frames and microphone audio, controls live sessions, submits voice questions, and displays memory-grounded answers on the glasses.

2. **Backend Service**
   Receives live streams, manages sessions, builds current/short-term/long-term memories, retrieves evidence, and returns answers.

3. **Web Frontend**
   Provides a browser interface for live capture, memory interaction, session review, and backend-powered QA.

```text
Web frontend         \
                     -> Backend API and workers -> Memory / retrieval / QA
Rokid AI Glass app  /
```

At runtime, either the web frontend or the glasses app can open a live session with the backend, send visual/audio data, and receive memory-grounded answers.

---

<span id="repository-layout"></span>

## Repository Layout

```text
src/
  frontend/       # Vite + React web frontend
  backend/        # FastAPI service, online workers, and memory-processing logic
  ai_glass_app/   # Rokid AI Glass Android app
```

Component documentation:

- [`src/frontend/README.md`](src/frontend/README.md)
- [`src/backend/README.md`](src/backend/README.md)
- [`src/ai_glass_app/README.md`](src/ai_glass_app/README.md)

---

<span id="components"></span>

## Components

### `src/frontend/`

The web frontend is a Vite + React app. It supports browser camera/microphone capture, session start/stop, live ingest controls, question submission, answer display, and evidence review.

The API base URL is configured with `VITE_API_BASE_URL` at build time, with a production fallback in `online_web/src/api/lightmem_egoApi.js`.

### `src/backend/`

The backend is a FastAPI-based online server. It exposes stream and query APIs, manages live sessions, runs workers for preprocessing/ASR/memory updates, and serves memory-grounded answers.

The backend uses the `src/em2mem/` runtime package for memory, LLM, and embedding components. Runtime sessions, logs, model weights, generated indexes, and private `.env` files are intentionally excluded.

### `src/ai_glass_app/`

The glasses app is an Android client for Rokid AI Glass. It starts and stops live capture, streams camera/audio data, records short voice questions, and renders answers on a glasses-friendly UI.

The backend endpoint is configured in:

```text
src/ai_glass_app/app/src/main/java/cn/zjukg/lightmem/glass/lightmem_ego/LightMemEgoConfig.kt
```

---

<span id="quick-start"></span>

## Quick Start

Each component has its own setup and runtime requirements. Start with the README for the component you want to run.

### Frontend

```bash
cd src/frontend/online_web
npm install
npm run dev
```

For production build:

```bash
npm run build
```

### Backend

```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
scripts/start_api.sh
```

Start the default online worker set:

```bash
scripts/start_online_all_workers.sh
```

### Glasses App

Windows:

```powershell
cd src\ai_glass_app
.\gradlew.bat assembleDebug
```

macOS or Linux:

```bash
cd src/ai_glass_app
./gradlew assembleDebug
```

---

<span id="scenarios"></span>

## Supported Scenarios

| Scenario | Example Query | Memory Scope |
| :--- | :--- | :--- |
| **Object Finding** | "Where did I leave my badge?" | Current / short-term memory |
| **Conversation Recall** | "What did the doctor tell me after checking the report?" | Short-term memory + transcript context |
| **Life Summarization** | "What did I do this afternoon?" | Short-term and long-term memory |
| **Routine Discovery** | "What do I usually do after arriving at the office?" | Long-term semantic memory |
| **Wearable Assistance** | "What am I looking at now?" | Current memory |

---

<span id="related-works"></span>

## Related Works
This repository belongs to ZJUNLP LightMem series, focusing on solving context bloat, excessive token consumption and low cache utilization for long-running LLM agents:
- [LightMem](https://github.com/zjunlp/LightMem) — A lightweight and efficient memory management framework designed for Large Language Models and AI Agents
- [LightMem2](https://github.com/zjunlp/LightMem2) — A modular framework for long-running agent memory and context management
- [LightMem-Ego](https://github.com/zjunlp/LightMem-Ego) — A lightweight streaming multimodal memory system for everyday-life assistance

<span id="privacy-notice"></span>

## Privacy Notice

LightMem-Ego may process camera frames, microphone audio, transcripts, generated answers, and memory data depending on deployment configuration. Before deploying with real users, review endpoint configuration, data retention policy, access control, and user consent flow.

This repository is intended for research and demonstration. Production deployments should implement privacy-preserving capture, sensitive-content filtering, encrypted storage, access control, retention/deletion policies, and user-controlled memory editing.

---

<span id="license"></span>

## License

See [`LICENSE`](LICENSE).

---

<span id="citation"></span>

## Citation

Paper and citation information will be added when available.

---

<span id="acknowledgements"></span>

## Acknowledgements

LightMem-Ego builds on the broader line of work on memory-augmented agents, egocentric multimodal understanding, and wearable AI assistants. We thank all contributors and collaborators who helped develop the system.
