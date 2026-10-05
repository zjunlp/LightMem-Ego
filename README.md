<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./figs/hero_lockup_dark.png">
    <img src="./figs/hero_lockup.png" width="100%" alt="LightMem-Ego: Your AI Memory for Everyday Life">
  </picture>
</div>

<p align="center">
  <b>An open-source, self-hostable multimodal memory system for smart glasses and phones.</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.11487"><img src="https://img.shields.io/badge/arXiv-2607.11487-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2607.11487"><img src="https://img.shields.io/badge/HuggingFace-Paper-yellow?logo=huggingface&logoColor=white" alt="Hugging Face Paper"></a>
  <a href="https://arxiv.org/abs/2609.00551"><img src="https://img.shields.io/badge/EM%C2%B2Mem-2609.00551-b31b1b?logo=arxiv&logoColor=white" alt="EM²Mem paper"></a>
  <img src="https://img.shields.io/badge/EMNLP%202026%20Findings-Accepted-blueviolet" alt="EMNLP 2026 Findings">
  <a href="https://github.com/zjunlp/LightMem-Ego/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://lightmem-ego.zjukg.cn/"><b>🌐 Try in Browser</b></a> &nbsp;·&nbsp;
  <a href="#quick-start"><b>🚀 Quick Start</b></a> &nbsp;·&nbsp;
  <a href="https://www.bilibili.com/video/BV1oANw62EA3/"><b>🎬 Watch the Demo</b></a> &nbsp;·&nbsp;
  <a href="https://github.com/zjunlp/LightMem-Ego/releases/download/v1.0.0/app-release.apk"><b>📱 Glasses APK</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React%2019-61DAFB?logo=react&logoColor=black" alt="React 19">
  <img src="https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white" alt="Vite">
  <img src="https://img.shields.io/badge/Android-3DDC84?logo=android&logoColor=white" alt="Android">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

<p align="center">
  <b>LightMem-Ego</b> is the end-to-end system; its long-term tier (<code>M_lt</code>) is powered by <a href="https://arxiv.org/abs/2609.00551"><b>EM²Mem</b></a> (EMNLP 2026 Findings), part of the ZJUNLP <a href="https://github.com/zjunlp/LightMem">LightMem</a> project family.
</p>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./figs/research_strip_dark.png">
    <img src="./figs/research_strip.png" width="88%" alt="EM²Mem versus the strongest baseline: 76.8 Video-MME (L), 67.7 Ego-R1 Bench, 66.0 EgoLifeQA, 4.67 times faster per query">
  </picture>
</div>

<details>
<summary><b>📑 Table of contents</b></summary>

- [📢 News](#news)
- [🎬 Demo](#demo)
- [🎯 Why LightMem-Ego](#why)
- [🚀 Quick Start](#quick-start)
- [💬 What You Can Ask](#scenarios)
- [🏗️ How It Works](#architecture)
- [📊 Results](#results)
- [🆚 How It Compares](#comparison)
- [📦 Repository Layout](#repository-layout)
- [🗺️ Roadmap](#roadmap)
- [📄 Citation](#citation)
- [🔗 Related Projects](#related-works)
- [⚖️ License](#license)
- [🔐 Privacy](#privacy)

</details>

---

<span id="news"></span>

## 📢 News

- **[2026-09]** ✨ Multi-session support, online memory editing, and streaming answers land in the backend and web UI.
- **[2026-08]** 🎉🎉🎉 [**EM²Mem: Event-Centric Multimodal Memory for Large Language Models**](https://arxiv.org/abs/2609.00551) — the long-term memory engine behind this backend — has been accepted to **EMNLP 2026 Findings**!
- **[2026-07-13]** 📄 [**LightMem-Ego: Your AI Memory for Everyday Life**](https://arxiv.org/abs/2607.11487) is released on arXiv.
- **[2026-07]** 📦 **v1.0.0 released** — [download the Rokid AI Glass APK](https://github.com/zjunlp/LightMem-Ego/releases/tag/v1.0.0) and reproduce the full stack with [Docker](https://github.com/zjunlp/LightMem-Ego/blob/main/deploy/DOCKER.md).
- **[2026-05]** 🎉 [**LightMem-Ego: Your AI Memory for Everyday Life**](https://github.com/zjunlp/LightMem-Ego) is open-sourced.

---

<span id="demo"></span>

## 🎬 Demo

Ask the glasses a question in the middle of your day, and get an answer grounded in what you actually saw and heard. Prefer typing? Join the same live session from the web page.

> [!TIP]
> **No hardware? Try it right now.** The [live web demo](https://lightmem-ego.zjukg.cn/) runs the full LightMem-Ego workflow in your browser — on a phone too, where it captures from the phone's own camera and microphone. No glasses, no local installation.

<p align="center">
  <a href="https://www.bilibili.com/video/BV1oANw62EA3/"><img src="./figs/demo_v2.gif" width="85%" alt="The glasses record; the user asks where a plastic bottle was placed; the answer comes back with the timestamped evidence"></a>
</p>

<div align="center">
  <a href="https://www.bilibili.com/video/BV1oANw62EA3/"><picture><source media="(prefers-color-scheme: dark)" srcset="./figs/demo_caption_v3_dark.png"><img src="./figs/demo_caption_v3.png" width="81%" alt="Ask on the glasses, get a memory-grounded answer on the HUD — watch the full demo."></picture></a>
</div>

<p align="center">
  <a href="https://www.youtube.com/watch?v=BZuIxn00xlc"><picture><source media="(prefers-color-scheme: dark)" srcset="./figs/watch_youtube_dark.png"><img src="./figs/watch_youtube.png" height="34" alt="Watch the full demo on YouTube"></picture></a>
  &nbsp;&nbsp;<a href="https://www.bilibili.com/video/BV1oANw62EA3/"><picture><source media="(prefers-color-scheme: dark)" srcset="./figs/watch_bilibili_dark.png"><img src="./figs/watch_bilibili.png" height="34" alt="Watch the full demo on Bilibili"></picture></a>
</p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="./src/ai_glass_app/assets/glass_1_01.png" width="250" alt="Asking a question on Rokid AI Glasses">
      <br><b>Hands-free on Rokid AI Glasses</b><br>Ask by voice or preset question
    </td>
    <td align="center" width="50%">
      <img src="./src/ai_glass_app/assets/demo_slide7_cropped_for_emnlp_01.png" width="250" alt="Memory-grounded answer over a real-world scene">
      <br><b>Answers grounded in memory</b><br>Timestamps + visual evidence
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="./src/ai_glass_app/assets/frontend_rokid.png" width="620" alt="Asking typed questions about the same live session from the web">
      <br><b>Same session on the web</b><br>Type questions when speaking isn't convenient
    </td>
  </tr>
</table>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./figs/feature_strip_v2_dark.png">
    <img src="./figs/feature_strip_v2.png" width="100%" alt="Runs on glasses and phones; self-hostable; timestamped evidence; three-tier memory">
  </picture>
</div>

<h5 align="center">⭐ If LightMem-Ego is useful to you, a star helps more people find it.</h5>

---

<span id="why"></span>

## 🎯 Why LightMem-Ego

- 🎥 **Always-on egocentric capture** — streams first-person camera frames and microphone audio from Rokid AI Glasses or a phone.
- 🧠 **Three-tier memory** — a rolling *current* memory, *short-term* micro-events, and consolidated *long-term* episodes, routines, and preferences.
- ⏱️ **One aligned timeline** — frames, audio chunks, ASR transcripts and metadata all share a single session timeline.
- 🔍 **Memory-grounded answers** — every answer ships with the timestamped visual and transcript evidence behind it.
- 👓 **Glasses and web, one session** — start capture on the glasses, keep asking from the web page in the same live session.
- 🐳 **Self-hostable** — `docker compose up --build` brings up the web UI and the full backend worker pipeline.

Text memory systems only know what you typed, live assistants only know the current scene, and video systems only let you search afterwards. LightMem-Ego covers all three — the system-by-system table is in [How It Compares](#comparison).

---

<span id="quick-start"></span>

## 🚀 Quick Start

> [!IMPORTANT]
> Every path except the hosted demo needs an OpenAI-compatible LLM endpoint (base URL, API key, model names) and [Xfyun](https://www.xfyun.cn/) ASR credentials for speech. The default stack also expects `Qwen3-Embedding-4B` weights under `docker-data/models/`.

| Path | Setup | What it adds | GPU |
| :--- | :--- | :--- | :---: |
| 🌐 **[Live demo](https://lightmem-ego.zjukg.cn/)** | none | the full hosted workflow | — |
| 🐳 **[Docker](#docker)** | Docker + your LLM endpoint | the standard self-hosted setup | no |
| 👁 **[+ visual retrieval](#visual-retrieval)** | `--profile models` + VLM2Vec weights | frame-level visual matching | yes |
| ⚡ **[+ local Qwen](#local-qwen)** | vLLM server on your GPU | lower first-token latency | yes |
| 👓 **[Rokid glasses](#rokid-glasses)** | APK + any backend above | hands-free capture, HUD answers | no |

> [!NOTE]
> The last two are **independent add-ons**, not requirements. The plain Docker setup is what we run day to day — add visual retrieval when caption and transcript evidence is not enough, and local Qwen when a remote API feels slow.

<span id="docker"></span>

### 🐳 Docker

```bash
git clone https://github.com/zjunlp/LightMem-Ego.git
cd LightMem-Ego
cp deploy/.env.example .env     # LLM endpoint, keys, model names, Xfyun credentials
docker compose up --build
```

Open **http://localhost:8080**. The web container proxies `/api` to the backend, so no CORS setup is needed. The first build takes a few minutes.

By default `EM2MEM_VISUAL_BACKEND=mock`, so retrieval runs on captions and transcripts. Current memory, short-term micro-events, long-term consolidation and evidence-grounded answers all behave as in the [demo](#demo) — only frame-level visual matching is off.

<span id="visual-retrieval"></span>

### 👁 Add visual retrieval (optional)

Put `VLM2Vec-V2.0` and `Qwen3-Embedding-4B` under `docker-data/models/`, then start the model services:

```bash
docker compose --profile models up --build
```

Point the backend at them in `.env`:

```bash
EM2MEM_VISUAL_BACKEND=remote
EM2MEM_TEXT_EMBED_BACKEND=remote
```

On a GPU host, add the override so the workers get the GPU as well:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml --profile models up --build
```

This needs the NVIDIA Container Toolkit and model directories matching the paths in `.env` — see [`deploy/DOCKER.md`](deploy/DOCKER.md) for details.

<span id="local-qwen"></span>

### ⚡ Local Qwen for lower latency (optional)

This is where a GPU helps most. Every memory write and every answer otherwise round-trips to a remote API; serving the LLM locally cuts first-token latency noticeably. The scripts build an isolated vLLM environment and switch the backend onto it:

```bash
cd src/backend
scripts/setup_local_qwen35_env.sh
scripts/download_local_qwen35_model.sh
scripts/select_llm_profile.sh local-qwen35
scripts/stop_server_and_workers.sh --keep-api --force
```

Details and the smoke test: [`src/backend/README.md`](src/backend/README.md).

<span id="rokid-glasses"></span>

### 👓 Rokid AI Glasses

Install the released APK:

```bash
adb install -r app-release.apk
```

Or build it (JDK + Android SDK):

```bash
cd src/ai_glass_app
./gradlew assembleDebug        # Windows: .\gradlew.bat assembleDebug
```

Set `API_BASE_URL` in [`LightMemEgoConfig.kt`](src/ai_glass_app/app/src/main/java/cn/zjukg/lightmem/glass/lightmem_ego/LightMemEgoConfig.kt) to your own backend — it points at our demo server by default. Details: [`src/ai_glass_app/README.md`](src/ai_glass_app/README.md).

### Building from source

<details>
<summary><b>Web frontend</b> (Node.js + npm)</summary>

```bash
cd src/frontend/online_web
npm install
npm run dev
```

Point it at your backend by creating `online_web/.env.local`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Details: [`src/frontend/README.md`](src/frontend/README.md)

</details>

<details>
<summary><b>Backend</b> (Python 3.10+, ffmpeg/ffprobe)</summary>

```bash
cd src/backend
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip && python -m pip install -e .
cp .env.example .env            # configure model paths and API credentials
scripts/start_api.sh
scripts/start_online_all_workers.sh
```

Details: [`src/backend/README.md`](src/backend/README.md) and [`DEPLOYMENT.md`](src/backend/DEPLOYMENT.md).

</details>

---

<span id="scenarios"></span>

## 💬 What You Can Ask

| Scenario | Example question | Memory used |
| :--- | :--- | :--- |
| **Object finding** | "Where did I leave my badge?" | Current + short-term |
| **Conversation recall** | "What did the doctor tell me after checking the report?" | Short-term + transcript |
| **Day summarization** | "What did I do this afternoon?" | Short-term + long-term |
| **Routine discovery** | "What do I usually do after arriving at the office?" | Long-term semantic |
| **Live assistance** | "What am I looking at right now?" | Current |

---

<span id="architecture"></span>

## 🏗️ How It Works

<div align="center">
  <img src="./figs/system_design.png" width="90%" alt="LightMem-Ego system design">
</div>

```text
Rokid AI Glasses ─┐
                  ├─► Stream API ─► M_cur ─► M_st ─► M_lt ─► Retrieval ─► Answer + Evidence
Browser (web) ────┘                 current  short   long
```

| Memory tier | Scope | Example |
| :--- | :--- | :--- |
| **`M_cur`** current memory | The ongoing scene, updated as frames arrive | "What am I looking at?" |
| **`M_st`** short-term memory | Recent micro-events, actions, and conversations | "What did she just tell me?" |
| **`M_lt`** long-term memory | Consolidated episodes, routines, preferences, semantic facts | "What do I usually do on Fridays?" |

The backend divides each session into short event anchors and stores multimodal evidence per anchor. The long-term tier (`M_lt`) is built by **EM²Mem**, our event-centric multimodal memory framework (EMNLP 2026 Findings, [arXiv:2609.00551](https://arxiv.org/abs/2609.00551)): events are the retrieval unit, and episodic and semantic graphs link them across a session. At query time the system retrieves aligned event-level evidence — captions, transcripts, frames, timestamps — instead of reconstructing context at inference.

<div align="center">
  <img src="./figs/em2mem_architecture.png" width="100%" alt="EM²Mem architecture: event-centric memory schema, event-linked graph construction, and lightweight retrieval">
</div>

*EM²Mem in one picture. A video is segmented into 30-second event anchors, and each anchor becomes a memory cell holding dense captions, transcripts, keyframes, and metadata. Episodic and semantic graphs link those cells, and retrieval reads grounded evidence from them instead of re-aligning raw fragments at query time.*

---

<span id="results"></span>

## 📊 Results

### End-to-end system — LightMem-Ego

> [!NOTE]
> These numbers come from a small, intentionally balanced set — **27 queries (nine per scenario) over five everyday-life videos, about 45.7 minutes of footage** — collected with the **phone and glasses client profiles used in the paper**. In the open-source release the phone client is the web frontend running in a **mobile browser**, capturing from the phone's own camera and microphone — a native app is on the [roadmap](#roadmap). It measures the current prototype rather than a public leaderboard. Reported in the [LightMem-Ego paper](https://arxiv.org/abs/2607.11487).

**Retrieval accuracy** — Recall@k over the retrieved memory entries, with MRR for the first relevant hit:

| Scenario | R@1 | R@3 | R@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Object finding | 22.2 | 66.7 | 77.8 | 0.454 |
| Conversation recall | 44.4 | 55.6 | 55.6 | 0.481 |
| Life summarization | 88.9 | 100.0 | 100.0 | 0.944 |
| **Overall** | **51.9** | **74.1** | **77.8** | **0.627** |

**Answer accuracy** — experience QA over daily scenarios:

| Scenario | LLM-Judge | Human |
| :--- | :---: | :---: |
| Object finding | 44.4 | 55.6 |
| Conversation recall | 33.3 | 33.3 |
| Life summarization | 77.8 | 77.8 |
| **Overall** | **51.9** | **55.6** |

**Latency** — P50 / P90 across two client profiles:

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

*Glasses columns are the glasses-style client profile. Short-term queries stay near-interactive; long-term ones trade latency for temporal coverage.*

### Long-term memory engine — EM²Mem

The long-term tier (`M_lt`) is built by EM²Mem. Average accuracy (%) across three long-video and egocentric benchmarks, as reported in the EM²Mem paper:

| Method | EgoLifeQA | Ego-R1 Bench | Video-MME (L) |
| :--- | :---: | :---: | :---: |
| GPT-5 | 48.6 | 46.3 | 74.3 |
| HippoRAG | 59.6 | 56.0 | 52.1 |
| M3-Agent | 53.5 | 52.0 | 55.3 |
| Ego-R1 | 53.0 | 52.0 | 42.7 |
| WorldMM | 65.6 | 65.3 | 76.6 |
| **EM²Mem** | **66.0** | **67.7** | **76.8** |

Against the strongest baseline (WorldMM, reproduced under the same evaluation setting):

| Metric | EM²Mem | WorldMM | Gain |
| :--- | :---: | :---: | :---: |
| Avg. latency per query | **98.21 s** | 459.00 s | 4.67× faster |
| Wall-clock evaluation time | **6,138 s** | 229,502 s | 37.4× faster |
| Total tokens | **15.27M** | 42.03M | 63.7% fewer |

EM²Mem moves multimodal alignment and graph organization into offline memory construction, so inference reads from pre-built event-indexed memory cells instead of re-aligning isolated fragments. Full per-category tables are in the [EM²Mem research README](research/em2mem/README.md#reported-results); the online backend details are in the [backend README](src/backend/README.md#results).

---

<span id="comparison"></span>

## 🆚 How It Compares

Representative commercial assistants, text-based memory systems, and egocentric multimodal assistants. This compares publicly described capabilities rather than measured performance.

| System | Platform & input | Real-time A/V stream | Current / short-term MM memory | Long-term episodic | Long-term semantic | Timestamped evidence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| ChatGPT Memory | Text chat | — | — | — | Partial | — |
| Mem0-style memory | Text and agent memory | — | Partial | — | ✓ | Partial |
| Memories.ai | Video archives and visual memory | Partial | Partial | ✓ | Partial | Partial |
| Gemini Live | Phone | ✓ | Partial | — | — | — |
| Ray-Ban Meta AI Glasses | Glasses | Partial | Partial | — | — | — |
| Vinci | Phone or wearable camera | ✓ | ✓ | Partial | Partial | Partial |
| VisualClaw | Streaming video with agent workspace | Partial | Partial | — | Partial | Partial |
| VisionClaw | Smart glasses | ✓ | Partial | — | — | Partial |
| Egocentric Co-Pilot | Smart glasses with web agents | ✓ | ✓ | Partial | Partial | Partial |
| EgoButler | AI-glasses egocentric video and audio | Partial | Partial | Partial | Partial | ✓ |
| **LightMem-Ego** | **Phone and glasses-style client** | **✓** | **✓** | **✓** | **✓** | **✓** |

**✓** implemented as an explicit first-class component · **Partial** limited, implicit, offline, session-level, or modality-restricted · **—** not explicitly supported or not publicly described. Adapted from the [LightMem-Ego paper](https://arxiv.org/abs/2607.11487). *Phone was the paper's evaluation client — the open-source clients are the browser frontend and the Rokid AI Glasses app.*

---

<span id="repository-layout"></span>

## 📦 Repository Layout

| Path | What's inside | Docs |
| :--- | :--- | :--- |
| [`src/ai_glass_app/`](src/ai_glass_app/) | Android app for Rokid AI Glasses (Kotlin, Jetpack Compose, CameraX) | [README](src/ai_glass_app/README.md) |
| [`src/frontend/`](src/frontend/) | Vite + React web UI for capture, sessions, QA, and evidence review | [README](src/frontend/README.md) |
| [`src/backend/`](src/backend/) | FastAPI service plus the online worker pipeline (ASR, memory, retrieval, QA) | [README](src/backend/README.md) |
| [`research/em2mem/`](research/em2mem/) | Offline EM²Mem research implementation, preprocessing, and evaluation code | [README](research/em2mem/README.md) |
| [`compose.yaml`](compose.yaml), [`deploy/`](deploy/) | Docker Compose stack and deployment notes | [DOCKER.md](deploy/DOCKER.md) |

---

<span id="roadmap"></span>

## 🗺️ Roadmap

- [ ] Ship a native phone app — today the phone client is the web frontend in a mobile browser.
- [ ] Release the end-to-end evaluation dataset and reproducibility scripts.
- [ ] Pluggable ASR, VLM, and embedding backends beyond the current defaults.
- [ ] Support wearable devices beyond Rokid AI Glass.
- [ ] On-device filtering and user-controlled memory editing for privacy-sensitive capture.
- [ ] One-click deployment template for a full cloud deployment.

---

<span id="citation"></span>

## 📄 Citation

If you find LightMem-Ego useful, please cite our paper:

```bibtex
@article{chen2026lightmemego,
  title={LightMem-Ego: Your AI Memory for Everyday Life},
  author={Chen, Yijun and Xiao, Boyi and Zhao, Yixian and Xia, Haoting and Xu, Buqiang and Fang, Jizhan and Li, Yanya and Zheng, Yaqi and Wang, Xuehai and Xue, Zirui and others},
  journal={arXiv preprint arXiv:2607.11487},
  year={2026}
}
```

The long-term memory tier (`M_lt`) of the backend is built by **EM²Mem**, which has been accepted to **EMNLP 2026 Findings**. Please cite it as well when you use that module:

```bibtex
@article{chen2026em2mem,
  title={EM$^{2}$Mem: Event-Centric Multimodal Memory for Large Language Models},
  author={Chen, Yijun and Zheng, Yaqi and Li, Yanya and Xiao, Boyi and Xu, Buqiang and Qiao, Shuofei and Fang, Jizhan and Deng, Xinle and Yao, Yunzhi and Wang, Xuehai and others},
  journal={arXiv preprint arXiv:2609.00551},
  year={2026}
}
```

---

<span id="related-works"></span>

## 🔗 Related Projects

This repository belongs to the ZJUNLP **LightMem** series, which targets context bloat, excessive token consumption, and low cache utilization in long-running LLM agents:

- [LightMem](https://github.com/zjunlp/LightMem) — a lightweight and efficient memory management framework for LLMs and AI agents
- [LightRSI](https://github.com/zjunlp/LightRSI) — a modular framework for recursive improvement in long-horizon LLM agents
- [EM²Mem](https://arxiv.org/abs/2609.00551) **(EMNLP 2026 Findings)** — event-centric multimodal memory for long-video QA, and the long-term memory engine behind this system ([code overview](https://github.com/zjunlp/LightMem/blob/main/EM2Mem.md))

---

<span id="acknowledgements"></span>

## 🙏 Acknowledgements

LightMem-Ego builds on the broader line of work on memory-augmented agents, egocentric multimodal understanding, and wearable AI assistants. We thank all contributors and collaborators who helped develop the system.

---

<span id="license"></span>

## ⚖️ License

Released under the [MIT License](LICENSE).

---

<span id="privacy"></span>

## 🔐 Privacy

LightMem-Ego processes camera frames, microphone audio, transcripts, and generated memories. It is released for research and demonstration; a production deployment needs HTTPS, access control, encryption at rest, a data retention/deletion policy, and explicit user consent. Runtime media and memory are written to local, Git-ignored directories.

---

<span id="star-history"></span>

## ⭐ Star History

<p align="center">
  <a href="https://github.com/zjunlp/LightMem-Ego"><img src="https://img.shields.io/github/stars/zjunlp/LightMem-Ego?style=social" alt="GitHub Stars"></a>
</p>

<div align="center">
  <a href="https://star-history.com/#zjunlp/LightMem-Ego&Date"><img src="https://api.star-history.com/svg?repos=zjunlp/LightMem-Ego&type=Date" width="70%" alt="Star history chart"></a>
</div>
