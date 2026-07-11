# Frontend Update: Demo Mode Works With Normal Start API

This document explains the frontend changes needed after the backend update that lets demo mode work when the server is started with the normal API script.

## Summary

Previously, demo mode required the backend to be started with:

```bash
bash demo/start_demo_api.sh
```

Now demo routes are auto-registered when the normal API starts:

```bash
bash scripts/start_api.sh
```

If the `demo/` directory exists and `WORLDMM_ENABLE_DEMO_ROUTES` is not disabled, `api_server:app` now exposes:

```http
POST /demo/upload
POST /demo/{session_id}/start
POST /demo/{session_id}/tick
POST /demo/{session_id}/pause
POST /demo/{session_id}/stop
GET  /demo/{session_id}/status
GET  /demo/{session_id}/video
GET  /demo/{session_id}/manifest
```

The old `demo/start_demo_api.sh` still works, but frontend should not require it.

## Why The Previous Error Happened

The frontend showed:

```text
Streaming answer ended before a done event was received.
```

The most likely cause was a backend/frontend mismatch:

- Frontend selected demo mode.
- Backend was started with `scripts/start_api.sh`.
- Old `api_server:app` did not expose `/demo/*`.
- Demo upload/tick state was not correctly available.
- The streaming ask request ended before the expected SSE `done` event.

After this backend update, `scripts/start_api.sh` can serve both normal APIs and demo APIs.

## Frontend Changes Required

### 1. Do Not Depend On `demo/start_demo_api.sh`

Frontend should only assume one backend base URL.

For example:

```ts
const API_BASE_URL = "https://omnispark.zjukg.cn";
```

Demo mode should call paths under the same base URL:

```http
POST {API_BASE_URL}/demo/upload
POST {API_BASE_URL}/demo/{session_id}/tick
POST {API_BASE_URL}/ask/{session_id}/stream
```

### 2. Demo Upload Must Use `/demo/upload`

Request:

```http
POST /demo/upload
Content-Type: multipart/form-data
```

Required form field:

```text
video=<uploaded video file>
```

Important: the field name is `video`, not `file`.

Recommended fields:

```text
sample_fps=1
auto_prepare=true
enqueue_preprocess=false
```

Response example:

```json
{
  "session_id": "demo_xxx",
  "video_url": "/demo/demo_xxx/video",
  "start_url": "/demo/demo_xxx/start",
  "tick_url": "/demo/demo_xxx/tick",
  "ask_stream_url": "/ask/demo_xxx/stream",
  "prepared": true,
  "duration": 30.0,
  "frame_count": 31,
  "preprocess_queued": false
}
```

Frontend must store this `session_id`.

### 3. Use Returned `video_url` On Main Screen

After upload, return to the main interface and set the video element source to:

```text
video_url
```

If the backend returns a relative path, resolve it against the API base URL:

```ts
const videoSrc = new URL(video_url, API_BASE_URL).toString();
```

### 4. Main Start Button Must Start Demo Playback State

When the user clicks the main Start button in demo mode:

```http
POST /demo/{session_id}/start
Content-Type: application/json

{
  "current_time": 0,
  "playback_speed": 1
}
```

Then call `video.play()`.

### 5. Send Demo Tick While Video Plays

While the video is playing, send a tick every 500-1000 ms:

```http
POST /demo/{session_id}/tick
Content-Type: application/json

{
  "current_time": video.currentTime,
  "paused": false,
  "playback_speed": video.playbackRate || 1
}
```

This writes the current video window into backend `M_cur`.

### 6. Send One Tick Immediately Before Asking

Before asking a current-frame question, call `/demo/{session_id}/tick` once with the latest `video.currentTime`.

Then call:

```http
POST /ask/{session_id}/stream
Content-Type: application/json

{
  "question": "现在画面里有什么？",
  "response_mode": "stream",
  "memory_mode": "auto",
  "use_current": true,
  "use_short_term": false,
  "use_long_term": false,
  "use_image_evidence": true
}
```

### 7. SSE Handling Requirement

The streaming answer UI must wait for a `done` SSE event.

Expected events include:

```text
start
delta
evidence
done
error
ping
```

If the stream closes before `done`, show a recoverable error and log:

- HTTP status code
- Raw response body if available
- Last received SSE event
- `session_id`
- request payload

Do not permanently disable the send button after this error. Reset the ask UI to an idle/retry state.

### 8. Error Handling For Demo Mode

Handle these cases clearly:

- `404` on `/demo/upload`: backend did not load demo routes. Ask backend to restart normal API after the update.
- `409 not_ready` on `/ask/{session_id}/stream`: frontend likely did not call `/demo/{session_id}/tick` before asking.
- `500` on `/demo/upload`: video prepare failed. Show backend `message`.
- Stream ended before `done`: show retry, re-enable send button, and log Network details.

## Manual Test Checklist

1. Start backend with normal script:

```bash
bash scripts/start_api.sh
```

2. Confirm:

```http
POST /demo/upload
```

does not return 404.

3. Upload a short video using field name `video`.
4. Use returned `video_url` in the main video player.
5. Click Start and call `/demo/{session_id}/start`.
6. Confirm `/demo/{session_id}/tick` is called while video plays.
7. Before asking, call `/demo/{session_id}/tick` once.
8. Ask with `/ask/{session_id}/stream`.
9. Confirm frontend receives `done`.
10. Confirm send button is usable again after success or failure.

## Prompt For Frontend Agent

Use this prompt to ask a frontend coding agent to implement the update:

```text
You are modifying the WorldMM frontend demo mode.

Backend update: demo routes are now auto-registered on the normal API app. The backend should be started with bash scripts/start_api.sh, and demo mode should use the same API base URL as normal modes. Do not require a separate demo backend process.

Implement these frontend changes:

1. In demo mode, upload videos to POST /demo/upload using multipart/form-data with field name "video". Do not use the old /upload_video field name "file".
2. Store the returned session_id, video_url, start_url, tick_url, and ask_stream_url.
3. After upload, return to the main screen and set the video player source to video_url resolved against the API base URL.
4. When the main Start button is clicked in demo mode, call POST /demo/{session_id}/start with {"current_time":0,"playback_speed":1}, then play the video.
5. While the video plays, call POST /demo/{session_id}/tick every 500-1000 ms with current_time=video.currentTime, paused=false, and playback_speed=video.playbackRate || 1.
6. Immediately before sending any current-frame question, call /demo/{session_id}/tick once with the latest video.currentTime.
7. Ask with POST /ask/{session_id}/stream and payload:
   {
     "question": userQuestion,
     "response_mode": "stream",
     "memory_mode": "auto",
     "use_current": true,
     "use_short_term": false,
     "use_long_term": false,
     "use_image_evidence": true
   }
8. Fix SSE handling so the UI waits for the done event, handles error events, and re-enables the send button if the stream closes before done.
9. If the stream closes before done, log HTTP status, raw body if available, last SSE event, session_id, and request payload. Show a retryable error instead of leaving the UI stuck.
10. Add a manual test path: start normal backend with scripts/start_api.sh, upload a demo video, start playback, tick, ask "现在画面里有什么？", and confirm a done event is received.

Keep existing non-demo phone/Rokid flows working. Only change demo-mode routing and SSE recovery behavior.
```

