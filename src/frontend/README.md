# LightMem-Ego Frontend

This directory contains the web frontend for LightMem-Ego. The frontend is a Vite + React application that talks to the backend API, captures browser camera/microphone input, manages live sessions, submits questions, and displays memory-grounded answers with evidence.

## Layout

```text
src/frontend/
  online_web/          # Vite + React application
    src/api/           # Backend API wrapper
    src/components/    # UI panels and evidence display
    src/hooks/         # Session, stream, and question state
    src/styles/        # Application styles
    src/webrtc/        # WHIP/SRS browser streaming helpers
  deploy/              # Example Nginx configuration
  README_DEPLOY.md     # Production deployment notes
```

## Requirements

- Node.js and npm.
- A reachable LightMem-Ego backend API.
- HTTPS for production camera, microphone, and WebRTC access. Browsers allow these device APIs on `localhost`, but public deployments should use HTTPS.

## API Configuration

The frontend reads the backend API base URL from:

```text
VITE_API_BASE_URL
```

If the variable is not set, the production fallback is defined in:

```text
online_web/src/api/lightmem_egoApi.js
```

Default production API:

```text
https://lightmem-ego.zjukg.cn/api
```

For local development, create `online_web/.env.local`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The `.env.local` file is intentionally ignored by Git.

## Development

```bash
cd src/frontend/online_web
npm install
npm run dev
```

Open the URL printed by Vite. The development server is for testing the frontend; it still needs a running backend API for live capture and QA.

## Build

```bash
cd src/frontend/online_web
npm run build
```

The production build is generated in:

```text
online_web/dist/
```

`dist/` is intentionally not committed because it can be recreated from the source code and dependency lockfile.

## Main Backend APIs Used

The frontend uses these backend API groups:

- `POST /stream/start` to create a live session.
- `POST /stream/{session_id}/frame` to upload browser camera frames.
- `POST /stream/{session_id}/audio_chunk` to upload browser audio chunks.
- `POST /stream/{session_id}/live/ingest/start` and `/stop` to control live ingest when enabled.
- `GET /stream/{session_id}/status` to read stream/session status.
- `POST /ask/{session_id}` to submit a question.
- `GET /query_task/{task_id}` to poll asynchronous answer generation.
- `GET /session/{session_id}/file?path=...` to load evidence assets returned by the backend.

## Deployment

For Nginx deployment notes, see [`README_DEPLOY.md`](README_DEPLOY.md).
