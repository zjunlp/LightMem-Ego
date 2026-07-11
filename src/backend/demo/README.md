# Demo Video Playback Mode

This directory adds optional demo routes without changing the production API files.
For release builds, omit the whole `demo/` directory.

## Start

Run the demo wrapper alongside `scripts/start_api.sh`:

```bash
cd /zjunlp/chenyijun/worldmm-online-server-release
bash scripts/start_api.sh
bash demo/start_demo_api.sh
```

The wrapper imports the existing `api_server.app` and registers `/demo/*` routes.
All existing routes such as `/ask/{session_id}` and `/ask/{session_id}/stream` remain available.
By default, the normal API uses `WORLDMM_API_PORT` (usually 8000), while the demo API uses
`WORLDMM_DEMO_API_PORT` or `WORLDMM_API_PORT + 1` (usually 8001).

## What Changed

- Added a demo-only API wrapper in `demo/api.py`.
- Added `/demo/*` routes for uploading a local video, extracting sampled frames, controlling playback state, and syncing the current frame into the existing online `M_cur` memory.
- The normal ask APIs are unchanged. Demo questions still call `/ask/{session_id}` or `/ask/{session_id}/stream`.
- Demo uploads are also linked/copied to `online_sessions/{session_id}/input.mp4`, so the existing offline preprocess pipeline can be reused when needed.

## Frontend Flow

1. Hidden tools menu selects `demo`.
2. Upload a local video:

```http
POST /demo/upload
Content-Type: multipart/form-data

video=<file>
sample_fps=1
auto_prepare=true
enqueue_preprocess=false
```

Response includes:

```json
{
  "session_id": "demo_xxx",
  "video_url": "/demo/demo_xxx/video",
  "start_url": "/demo/demo_xxx/start",
  "tick_url": "/demo/demo_xxx/tick",
  "ask_stream_url": "/ask/demo_xxx/stream",
  "prepared": true,
  "duration": 123.4,
  "frame_count": 124,
  "preprocess_queued": false
}
```

3. Return to the main page and use `video_url` as the video source.
4. When the user clicks the main Start button:

```http
POST /demo/{session_id}/start
Content-Type: application/json

{"current_time": 0, "playback_speed": 1}
```

5. While the video plays, send ticks every 500-1000 ms:

```http
POST /demo/{session_id}/tick
Content-Type: application/json

{"current_time": 12.34, "paused": false, "playback_speed": 1}
```

Each tick writes the current playback window into the existing online `M_cur` files.
The normal query path then treats the demo video as if frames were arriving live.

6. Immediately before asking a current-frame question, send one more tick using the video element's current time.

7. Ask questions with the existing streaming API:

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

Use this for questions such as `现在画面里有什么？`, `这个人手里拿着什么？`, `画面左边是什么？`.

## Optional Offline Pipeline

The demo mode can also reuse the old offline pipeline, but this is optional because it is heavier and slower.

To enqueue preprocess during upload:

```http
POST /demo/upload
Content-Type: multipart/form-data

video=<file>
sample_fps=1
auto_prepare=true
enqueue_preprocess=true
force_preprocess=false
```

For an already uploaded demo session:

```http
POST /demo/{session_id}/sync_input
Content-Type: application/json

{"enqueue_preprocess": true, "force_preprocess": false}
```

This creates or refreshes `online_sessions/{session_id}/input.mp4` and, if requested, queues the existing preprocess task. Status can be polled with:

```http
GET /session/{session_id}/status
```

## Frontend Contract

- Demo mode should store the returned `session_id` and keep using it for the main page.
- The main video element should use `video_url`.
- The main Start button should call `/demo/{session_id}/start`, then call `video.play()`.
- While playing, call `/demo/{session_id}/tick` every 500-1000 ms.
- Before each current-frame ask, call `/demo/{session_id}/tick` once with the latest `video.currentTime`.
- For best UX, call `/ask/{session_id}/stream` and render SSE `delta` events as they arrive.
- If the user exits demo mode, stop sending ticks and call `/demo/{session_id}/pause` or `/demo/{session_id}/stop`.

## Notes

- Upload/prepare extracts demo frames offline with ffmpeg at `sample_fps`.
- Playback only reveals frames up to the frontend's reported `current_time`.
- This mode intentionally does not modify production files outside `demo/`.
- For long videos, use `sample_fps=0.5` or `1` to keep preparation fast.
- `auto_prepare=true` is recommended for short demo videos. For very long videos, upload with `auto_prepare=false`, show a preparing state, then call `/demo/{session_id}/prepare`.

## Demo-Test Cross-Day Mode

`/demo-test/*` is a showcase mode for two videos in one logical session. It is registered by the same `demo/api.py` wrapper and uses the same `demo/start_demo_api.sh` entrypoint.

Upload both videos:

```http
POST /demo-test/upload
Content-Type: multipart/form-data

day1_video=<file>
day2_video=<file>
day1_start=2026-06-30 18:56:02
day2_start=2026-07-01 21:26:32
sample_fps=1
auto_prepare=true
enqueue_offline=false
```

The response contains one parent `session_id` and two clip URLs:

```json
{
  "session_id": "demo_test_xxx",
  "clips": [
    {"clip_id": "day1", "video_url": "/demo-test/demo_test_xxx/video/day1"},
    {"clip_id": "day2", "video_url": "/demo-test/demo_test_xxx/video/day2"}
  ],
  "ask_stream_url": "/ask/demo_test_xxx/stream"
}
```

Start or tick a clip:

```http
POST /demo-test/{session_id}/start
{"clip_id":"day1","current_time":0,"playback_speed":1}

POST /demo-test/{session_id}/tick
{"clip_id":"day2","current_time":12.3,"paused":false,"playback_speed":1}
```

Each tick returns `display_date`, `display_time`, and `display_datetime`. The frontend should render those in the lower-right time display.

For current-frame questions, call one final tick with the video element's current time, then use the existing streaming ask endpoint with `use_current=true`.

For cross-day long-term memory:

1. Queue child offline processing when desired:

```http
POST /demo-test/{session_id}/enqueue_offline
{"force_preprocess":false,"enqueue_evidence":false}
```

After preprocess finishes, call it again with `enqueue_evidence=true`, or use existing worker automation if enabled.

2. Build the parent long-term memory after child evidence is ready:

```http
POST /demo-test/{session_id}/build_memory
{"force":true,"allow_manifest_fallback":false,"skip_semantic":false}
```

For a smoke test without VLM evidence, set `allow_manifest_fallback=true`; this creates time-aware fallback evidence from sampled frames, but answer quality will be weaker.
