# Multi-Session and Rokid API Contract

This update separates the public Rokid Glass API from the phone/web stream API and switches the server to multi-session mode by default.

## Runtime Mode

The default is now:

```bash
EM2MEM_SINGLE_ACTIVE_SESSION=0
```

In this mode:

- Every client must keep its own `session_id`.
- `/stream/active` should not be used by normal clients.
- Starting a new stream no longer cancels or aborts other users' queued tasks.
- Workers continue to process tasks by `session_id`.

To temporarily return to old single-user demo behavior:

```bash
EM2MEM_SINGLE_ACTIVE_SESSION=1
```

## Phone/Web API

Phone or browser frame/audio streaming should continue to use:

```http
POST /stream/start
```

Example:

```json
{
  "input_mode": "frame_audio_stream",
  "owner_id": "user_001",
  "device_id": "phone_001",
  "device_type": "phone",
  "metadata": {
    "client": "web_frontend"
  }
}
```

Use the returned URLs:

```json
{
  "session_id": "...",
  "frame_upload_url": "/stream/{session_id}/frame",
  "audio_upload_url": "/stream/{session_id}/audio_chunk",
  "live_ingest_start_url": null,
  "single_active_session": false
}
```

Ask APIs are unchanged:

```http
POST /ask/{session_id}/stream
GET  /stream/{session_id}/status
```

Rokid input modes are rejected on `/stream/start`. Use `/rokid/stream/start`.

## Rokid Glass API

Rokid should use the dedicated namespace:

```http
POST /rokid/stream/start
POST /rokid/{session_id}/frame
POST /rokid/{session_id}/audio_chunk
GET  /rokid/{session_id}/status
```

For SDK frame/audio upload:

```json
{
  "owner_id": "user_001",
  "device_id": "rokid_sn_001",
  "device_type": "rokid",
  "metadata": {
    "glass_sn": "rokid_sn_001",
    "phone_model": "android"
  }
}
```

The server forces:

```json
{
  "input_mode": "rokid_frame_audio"
}
```

The response contains Rokid-specific upload URLs:

```json
{
  "session_id": "...",
  "frame_upload_url": "/rokid/{session_id}/frame",
  "audio_upload_url": "/rokid/{session_id}/audio_chunk",
  "rokid": {
    "enabled": true,
    "input_mode": "rokid_frame_audio"
  }
}
```

Frame upload form fields:

```http
POST /rokid/{session_id}/frame
Content-Type: multipart/form-data

frame=<jpeg/png/webp>
frame_index=0
relative_ts_ms=0
format=jpeg
source=rokid_sdk_video
```

Audio upload form fields:

```http
POST /rokid/{session_id}/audio_chunk
Content-Type: multipart/form-data

audio=<wav/pcm/aac/m4a/mp3>
audio_index=0
relative_ts_ms=0
duration_ms=1000
format=wav
source=rokid_sdk_audio
```

For Rokid RTMP:

```json
{
  "input_mode": "rokid_live_rtmp",
  "owner_id": "user_001",
  "device_id": "rokid_sn_001"
}
```

Then use:

```http
POST /rokid/{session_id}/live/ingest/start
POST /rokid/{session_id}/live/ingest/stop
```

## Frontend Requirements

- Store `session_id` per browser tab/user/device.
- Never depend on `/stream/active` in multi-session mode.
- Use `/stream/*` URLs for phone/web streams.
- Use `/rokid/*` URLs for Rokid Glass streams.
- Always call ask/status APIs with the session owned by the current UI.
- Optional but recommended: send `owner_id`, `device_id`, and `device_type` on start.

## Worker Notes

No separate Rokid worker is required. The dedicated Rokid API writes the same session-local files and queues the same downstream tasks. Multiple users share the worker pool, so GPU-heavy workers may still queue under load; this change prevents one new session from cancelling the others.

