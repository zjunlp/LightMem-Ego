# Demo-Test Frontend Contract

This document describes the frontend changes for the cross-day demo-test mode.

## Services

Recommended production setup: use the normal API only.

`api_server.py` registers demo and demo-test routes by default when `WORLDMM_ENABLE_DEMO_ROUTES` is unset or true. Start the normal API:

```bash
bash scripts/start_api.sh
```

With this repository's current `.env`, the main API is:

```text
normal API with demo-test routes: http://127.0.0.1:18080
```

The public nginx already proxies `/api/*` to the normal API, so production frontend config should be:

```ts
API_BASE_URL = "https://omnispark.zjukg.cn/api"
DEMO_API_BASE_URL = "https://omnispark.zjukg.cn/api"
```

With this setup, demo-test upload goes to:

```http
POST https://omnispark.zjukg.cn/api/demo-test/upload
```

The separate demo API is still available for local development or isolation. To use it, run the normal API and demo API on different ports:

```bash
bash scripts/start_api.sh
bash demo/start_demo_api.sh
```

`demo/start_demo_api.sh` now defaults to:

```text
host = WORLDMM_DEMO_API_HOST or WORLDMM_API_HOST or 127.0.0.1
port = WORLDMM_DEMO_API_PORT or WORLDMM_API_PORT + 1
```

With this repository's current `.env`, this usually means:

```text
normal API: http://127.0.0.1:18080
demo API:   http://127.0.0.1:18081
```

For local frontend development with the separate demo API, configure both bases explicitly:

```ts
API_BASE_URL = "http://<backend-host>:18080"
DEMO_API_BASE_URL = "http://<backend-host>:18081"
```

If you still choose the separate demo API in production, do not make the browser connect to a raw backend port such as `:18081`. Put demo routes behind the same public origin as the frontend, then use:

```ts
API_BASE_URL = "https://omnispark.zjukg.cn/api"
DEMO_API_BASE_URL = "https://omnispark.zjukg.cn"
```

The reverse proxy must forward `/demo-test/*` to the demo API process. If the demo API also handles demo ask streaming, forward `/ask/*` to the demo API for demo mode, or use a dedicated prefix such as `/demo-api/*`.

Example nginx rules:

```nginx
client_max_body_size 4096m;

location /api/ {
    proxy_pass http://127.0.0.1:18080/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location /demo-test/ {
    proxy_pass http://127.0.0.1:18081/demo-test/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_request_buffering off;
    proxy_buffering off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}

location /ask/ {
    proxy_pass http://127.0.0.1:18081/ask/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
    proxy_cache off;
    gzip off;
}
```

For demo-test, call upload/playback/tick/video endpoints on `DEMO_API_BASE_URL`. In the recommended single-API setup, `DEMO_API_BASE_URL` equals `API_BASE_URL`.

## Upload

Hidden tools menu should add a `demo-test` mode.

Upload two videos in one request:

```http
POST {DEMO_API_BASE_URL}/demo-test/upload
Content-Type: multipart/form-data

day1_video=<file>
day2_video=<file>
day1_start=2026-06-30 18:56:02
day2_start=2026-07-01 21:26:32
sample_fps=1
auto_prepare=true
enqueue_offline=false
```

Important response fields:

```json
{
  "session_id": "demo_test_xxx",
  "clips": [
    {
      "clip_id": "day1",
      "display_date": "2026年6月30日",
      "start_time": "18:56:02",
      "video_url": "/demo-test/demo_test_xxx/video/day1"
    },
    {
      "clip_id": "day2",
      "display_date": "2026年7月1日",
      "start_time": "21:26:32",
      "video_url": "/demo-test/demo_test_xxx/video/day2"
    }
  ],
  "ask_stream_url": "/ask/demo_test_xxx/stream",
  "tick_url": "/demo-test/demo_test_xxx/tick"
}
```

Frontend should store:

```ts
sessionId
clips[]
activeClipId
demoApiBaseUrl
askStreamUrl
```

Resolve relative URLs with `DEMO_API_BASE_URL`.

## Playback

When user selects day1 or day2, set the video source to:

```ts
DEMO_API_BASE_URL + clip.video_url
```

When user clicks Start:

```http
POST {DEMO_API_BASE_URL}/demo-test/{session_id}/start
Content-Type: application/json

{
  "clip_id": "day1",
  "current_time": 0,
  "playback_speed": 1
}
```

While playing, send a tick every 500-1000ms:

```http
POST {DEMO_API_BASE_URL}/demo-test/{session_id}/tick
Content-Type: application/json

{
  "clip_id": "day1",
  "current_time": video.currentTime,
  "paused": video.paused,
  "playback_speed": video.playbackRate
}
```

Tick response includes:

```json
{
  "active_clip_id": "day1",
  "display_date": "2026年6月30日",
  "display_time": "18:56:18",
  "display_datetime": "2026-06-30 18:56:18",
  "local_current_time": 16.2,
  "current_time": 16.2
}
```

Show `display_date` and `display_time` in the lower-right time overlay.

For day2, the overlay should show:

```text
2026年7月1日 21:26:32 + video.currentTime
```

## Asking Current-Frame Questions

Before asking, send one final tick using the current video time.

Then call streaming ask:

```http
POST {DEMO_API_BASE_URL}/ask/{session_id}/stream
Content-Type: application/json

{
  "question": "现在画面里有什么？",
  "response_mode": "stream",
  "memory_mode": "auto",
  "use_current": true,
  "use_short_term": false,
  "use_long_term": false,
  "use_image_evidence": true,
  "max_image_evidence": 3
}
```

This is for questions about the currently playing frame.

## Cross-Day Long-Term Memory

For the best showcase, long-term memory should be built before the live demo.

Queue child offline processing:

```http
POST {DEMO_API_BASE_URL}/demo-test/{session_id}/enqueue_offline
Content-Type: application/json

{
  "force_preprocess": false,
  "enqueue_evidence": false,
  "force_evidence": false
}
```

After preprocess finishes for child sessions, queue evidence:

```http
POST {DEMO_API_BASE_URL}/demo-test/{session_id}/enqueue_offline
Content-Type: application/json

{
  "force_preprocess": false,
  "enqueue_evidence": true,
  "force_evidence": false
}
```

Poll:

```http
GET {DEMO_API_BASE_URL}/demo-test/{session_id}/status
```

When both child clips show `evidence_ready=true`, build parent memory:

```http
POST {DEMO_API_BASE_URL}/demo-test/{session_id}/build_memory
Content-Type: application/json

{
  "force": true,
  "allow_manifest_fallback": false,
  "skip_semantic": false
}
```

For local smoke tests only, use:

```json
{"force": true, "allow_manifest_fallback": true, "skip_semantic": true}
```

Fallback mode creates weak time-aware evidence from sampled frames and is not recommended for the final presentation.

## Frontend UX

Add these UI states:

- `demo-test upload`: choose day1 and day2 files.
- `preparing`: waiting for upload/auto_prepare.
- `day1 playback`: play day1 video with date/time overlay.
- `day2 playback`: play day2 video with date/time overlay.
- `memory building`: optional status while child preprocess/evidence and parent memory are prepared.

Use the same chat UI for questions. The only required ask change is sending a final tick before current-frame questions.

## Prompt For Frontend Codex

```text
You are modifying the frontend for WorldMM demo-test mode.

Add a hidden tools option named "demo-test". In this mode, the user uploads two videos, day1_video and day2_video, to POST {DEMO_API_BASE_URL}/demo-test/upload as multipart/form-data. Use defaults day1_start="2026-06-30 18:56:02", day2_start="2026-07-01 21:26:32", sample_fps=1, auto_prepare=true. Store the returned parent session_id and clips.

Keep existing API_BASE_URL for normal app calls. Add DEMO_API_BASE_URL as an explicit environment/config value. In the recommended production setup, set DEMO_API_BASE_URL equal to API_BASE_URL, for example https://omnispark.zjukg.cn/api, so returned relative demo-test URLs resolve to /api/demo-test/*. Do not hard-code or auto-force port 8001/18081 in browser code. The separate demo API path is optional only for local development or service isolation.

After upload, return to the main playback UI. Let the user choose day1/day2. Set the video element src to DEMO_API_BASE_URL + clip.video_url. On Start, POST /demo-test/{session_id}/start with {clip_id,current_time:0,playback_speed}. During playback send POST /demo-test/{session_id}/tick every 500-1000ms with {clip_id,current_time:video.currentTime,paused:video.paused,playback_speed:video.playbackRate}.

Render tick response display_date and display_time in the lower-right overlay. The overlay should update continuously while the video plays.

Before sending any current-frame question, send one final tick with the latest video.currentTime. Then call POST {DEMO_API_BASE_URL}/ask/{session_id}/stream with streaming enabled. For current-frame questions include use_current=true, use_short_term=false, use_long_term=false, use_image_evidence=true, max_image_evidence=3. Keep the existing streaming renderer.

Add optional controls or developer actions to call /demo-test/{session_id}/enqueue_offline, poll /demo-test/{session_id}/status, and call /demo-test/{session_id}/build_memory. Show child_status preprocess_ready/evidence_ready and memory_ready when available.

Do not break existing normal upload, normal stream, demo, or ask flows.
```
