<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../figs/lockup_compact_dark.png">
    <img src="../../figs/lockup_compact.png" width="420" alt="LightMem-Ego">
  </picture>
</div>

# LightMem-Ego · Web Frontend

<p>
  <a href="../../README.md">← Back to LightMem-Ego</a> &nbsp;·&nbsp;
  <a href="https://lightmem-ego.zjukg.cn/">Live Demo</a> &nbsp;·&nbsp;
  <a href="README_DEPLOY.md">Deployment Notes</a>
</p>

A Vite + React web workspace for the [LightMem-Ego](../../README.md) memory system. Capture from your browser camera and microphone, replay saved videos, ask questions about past events, and inspect timestamped memories and visual evidence.

## ✨ Features

- **Browser A/V capture** — start and stop a live session using the camera and microphone, on desktop or in a phone browser, and stream frames and audio chunks to the backend.
- **Live session view** — watch the current session, its pipeline state, and stream status as memory is built.
- **Ask panel** — submit typed questions and render memory-grounded answers, including token-by-token streaming replies.
- **Evidence strip** — open the frames, transcripts, and timestamps that support each answer.
- **Memory browser and editor** — review sessions and catalogs, rename sessions, edit 30-second memory entries, and roll back changes.
- **Memory graph** — explore episodic and semantic links between memories; select an event, fact or entity to read its description, scene, critical speech, triples and first/last-seen time.
- **Independent memory workspace** — open `/memories/` directly, or use `?memory_session=<session-id>` to locate one session and return to its video.
- **Saved-video Q&A** — `/history-demo/` preserves playback position, conversation history, task receipts and edit drafts.
- **Video uploads** — upload a video up to 95 MiB, wait for full offline memory/index construction, then replay and ask. Refreshing resumes status checks for the original upload.
- **Shared appearance** — Chinese/English, light/dark themes and font-size preferences across pages.
- **Editing gate** — viewers can browse and ask; editing controls require administrator mode in the current browser tab.
- **Multiple input modes** — browser capture, Rokid glasses sessions, and demo uploads share one UI.

## 🚀 Quick Start

> [!IMPORTANT]
> This app needs a reachable LightMem-Ego [backend](../backend/README.md). On its own it can capture video, but it cannot build memory or answer questions. Want to skip setup? Use the [live demo](https://lightmem-ego.zjukg.cn/) instead.

**Requirements:** Node.js 20+ and npm. The backend needs FFmpeg for video ingestion.

```bash
cd src/frontend/online_web
npm ci
npm run dev
```

Open the URL Vite prints (default <http://localhost:5173>).

The development server proxies `/api` to `http://127.0.0.1:8000` by default. To select another local backend, run `LIGHTMEM_BACKEND_URL=http://127.0.0.1:8001 npm run dev`. This value configures the development proxy only; production uses Nginx or the Docker proxy.

To point the app at a local backend, create `online_web/.env.local`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The backend must allow your dev origin — `EM2MEM_CORS_ORIGINS` in the backend `.env` defaults to `http://localhost:5173,http://127.0.0.1:5173`.

## ⚙️ Configuration

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `/api` | Backend API base URL. The relative default works when a proxy (Nginx or the Docker frontend container) maps `/api` to the backend. |
| `VITE_DEMO_API_BASE_URL` | value of `VITE_API_BASE_URL` | Optional separate base URL for the demo upload/test endpoints. |

> [!NOTE]
> Production bundles capture both variables at **build time**, so rebuild before deploying changed values. Restart Vite after editing a development `.env.local`. The local file is Git-ignored.

## 📜 Scripts

| Command | What it does |
| :--- | :--- |
| `npm run dev` | Start the Vite dev server on `0.0.0.0`. |
| `npm run build` | Produce a production bundle in `online_web/dist/`. |
| `npm run preview` | Serve the production build locally. |
| `npm test` | Run the Vitest suite. |
| `npm run test:history` | Run saved-video controller, editing, localization and upload unit tests. |
| `npm run test:browser` | Build all pages and run offline Chromium UI/navigation checks; requires FFmpeg and `npx playwright install chromium`. |

`dist/` is intentionally not committed — it is reproducible from source and the lockfile.

## 📁 Layout

```text
src/frontend/
  online_web/          # Vite + React application
    src/api/           # Backend API wrapper (lightmem_egoApi.js, memoryImportApi.js)
    src/components/    # Live view, ask panel, answer card, evidence strip, memory editor, graph
    src/hooks/         # Session, stream, question, import and memory-edit state
    src/webrtc/        # WHIP/SRS browser streaming helpers
    src/styles/        # Application styles (app.css, workspace.css, theme.css)
    src/utils/         # Shared helpers
    src/i18n.jsx       # English/Chinese UI strings for every page
    src/theme.js       # Light/dark theme preference
    src/fontScale.js   # Small/default/large text-size preference
    public/theme-init.js  # Applies the stored theme before first paint
    memories/          # Independent memory-management HTML entry
    history-demo/      # Saved-video HTML entry
  history_demo/        # Replay/upload modules and portable offline tests
  deploy/              # Example Nginx configuration
  README_DEPLOY.md     # Production deployment notes
```

## 🔌 Backend APIs Used

| Group | Endpoints |
| :--- | :--- |
| **Sessions** | `POST /stream/start`, `POST /rokid/stream/start`, `GET /stream/{session_id}/status`, `GET /rokid/stream/active` |
| **Capture** | `POST /stream/{session_id}/frame`, `POST /stream/{session_id}/audio_chunk`, `POST /rokid/{session_id}/frame`, `POST /rokid/{session_id}/audio_chunk` |
| **Live ingest** | `POST /stream/{session_id}/live/ingest/start`, `POST /stream/{session_id}/live/ingest/stop` |
| **Questions** | `POST /ask/{session_id}`, `POST /ask/{session_id}/stream` (streaming), `POST /stream/{session_id}/audio_question` |
| **Answers** | `GET /query_task/{task_id}`, `GET /session/{session_id}/qa_history` |
| **Memory** | `GET /session/{session_id}/memories`, `PUT /session/{session_id}/memories/30sec`, `POST /session/{session_id}/memories/30sec/rollback`, `GET /session/{session_id}/memory-graph`, `GET /memories/sessions` |
| **Evidence** | `GET /session/{session_id}/file?path=...`, `GET /stream/{session_id}/preview` |
| **Saved video** | `GET /session/{session_id}/demo-media`, `GET /session/{session_id}/demo-video` (byte ranges) |
| **Video upload** | `POST /video-uploads`, `GET /video-uploads/{upload_id}`, `POST /video-uploads/{upload_id}/resume` |

The replay page requests `GET /query_task/{task_id}?compact=1` so repeated retrieval traces do not delay answer delivery. Complete answers, evidence captions and capture times remain available; full diagnostics are retained by the backend's default task response.

The updated backend branch supplies saved-video/upload APIs. Optional text-memory import, session renaming and individual-memory JSON export are shown only if the connected backend advertises their required routes in `/openapi.json`. Their absence does not block viewing, capture or Q&A.

Administrator mode uses a local presentation code (`lightmem-admin` by default, configurable through `window.__LIGHTMEM_ADMIN_CODE__`). This controls the editing UI; it is not server authentication. Backend `canEdit`, record permissions, version and idempotency checks remain in force.

The full contract is documented in [`../backend/docs/online_stream_api_contract.md`](../backend/docs/online_stream_api_contract.md).

## 🩺 Troubleshooting

- **Camera or microphone is blocked.** Browsers only allow `getUserMedia` on `localhost` or over HTTPS. Use `http://localhost:5173` locally, or serve the build over HTTPS. A phone has no `localhost` exemption, so testing the mobile-browser path needs HTTPS.
- **Requests fail with a CORS error.** Add the frontend origin to `EM2MEM_CORS_ORIGINS` in the backend `.env` and restart the API.
- **The UI loads but answers never arrive.** Confirm a query worker is running (`scripts/start_online_query_worker.sh`) and that the session status endpoint returns an active session.
- **`/api` requests hit the wrong host.** `VITE_API_BASE_URL` is baked in at build time — rebuild after changing it.

## 🚢 Deployment

Production build:

```bash
cd src/frontend/online_web
npm run build
```

Serve `online_web/dist/` with a reverse proxy that forwards `/api` to the backend, and terminate TLS so camera and microphone access keep working. See [`README_DEPLOY.md`](README_DEPLOY.md) and [`deploy/nginx-online-web.conf.example`](deploy/nginx-online-web.conf.example).

> [!TIP]
> The Docker stack at the repository root builds and serves this frontend on port 8080 alongside the backend — the fastest route to a working deployment. See [`deploy/DOCKER.md`](../../deploy/DOCKER.md).
