# Frontend Update: Multi-Session and Rokid Split

This document explains the frontend changes needed after the backend update that separates Rokid Glass APIs from phone/web stream APIs and enables multi-session mode.

## Summary

Backend behavior changed in two important ways:

- The server now runs in multi-session mode by default: `WORLDMM_SINGLE_ACTIVE_SESSION=0`.
- Rokid Glass has its own API namespace: `/rokid/*`.

The frontend must stop treating the backend as having one global active stream. Every browser tab, user, phone stream, Rokid stream, and demo session must keep and use its own `session_id`.

## What Frontend Must Change

### 1. Stop Relying On `/stream/active`

Do not use `/stream/active` to discover the current stream in normal user flows.

In multi-session mode it returns:

```json
{
  "status": "multi_session_mode",
  "active": false,
  "session_id": null,
  "message": "single active session is disabled; clients must keep and use their own session_id"
}
```

Frontend should instead store the `session_id` returned by start/upload APIs.

Recommended storage:

- Keep active `session_id` in page state.
- Persist it in localStorage/sessionStorage if refresh recovery is needed.
- If the app supports login or multiple devices, store sessions by `owner_id` and `device_id`.

### 2. Phone/Web Stream Still Uses `/stream/*`

For phone camera, browser frame/audio, or normal web live stream:

```http
POST /stream/start
```

Request example:

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

Use the URLs returned by the backend:

```json
{
  "session_id": "...",
  "frame_upload_url": "/stream/{session_id}/frame",
  "audio_upload_url": "/stream/{session_id}/audio_chunk",
  "single_active_session": false
}
```

Then ask and poll status with that same `session_id`:

```http
POST /ask/{session_id}/stream
GET  /stream/{session_id}/status
```

### 3. Rokid Glass Must Use `/rokid/*`

Rokid should not call `/stream/start`.

Start Rokid frame/audio mode:

```http
POST /rokid/stream/start
```

Request example:

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

The backend forces:

```json
{
  "input_mode": "rokid_frame_audio"
}
```

Use returned URLs:

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

Frame upload:

```http
POST /rokid/{session_id}/frame
Content-Type: multipart/form-data

frame=<jpeg/png/webp>
frame_index=0
relative_ts_ms=0
format=jpeg
source=rokid_sdk_video
```

Audio upload:

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

Rokid status:

```http
GET /rokid/{session_id}/status
```

Ask still uses the normal ask endpoint:

```http
POST /ask/{session_id}/stream
```

### 4. Rokid RTMP Mode

If Rokid uses RTMP push:

```http
POST /rokid/stream/start
```

Request:

```json
{
  "input_mode": "rokid_live_rtmp",
  "owner_id": "user_001",
  "device_id": "rokid_sn_001",
  "device_type": "rokid"
}
```

Then use:

```http
POST /rokid/{session_id}/live/ingest/start
POST /rokid/{session_id}/live/ingest/stop
GET  /rokid/{session_id}/status
```

### 5. Error Handling To Add

If frontend accidentally starts Rokid with `/stream/start`, backend returns `400`:

```json
{
  "status": "error",
  "message": "Rokid input modes are served by /rokid/stream/start. Use the dedicated Rokid API instead of /stream/start.",
  "rokid_start_url": "/rokid/stream/start"
}
```

If frontend sends a phone session to `/rokid/{session_id}/frame`, backend returns `409`:

```json
{
  "status": "wrong_input_mode",
  "message": "this session was not started through the Rokid API",
  "expected_input_modes": ["rokid_frame_audio", "rokid_live_rtmp"]
}
```

Handle these by showing a clear frontend error and resetting the selected device flow.

## Frontend State Model

Recommended frontend state shape:

```ts
type DeviceKind = "phone" | "rokid" | "demo";

type ActiveSession = {
  sessionId: string;
  deviceKind: DeviceKind;
  ownerId?: string;
  deviceId?: string;
  inputMode?: string;
  frameUploadUrl?: string | null;
  audioUploadUrl?: string | null;
  statusUrl?: string;
  askStreamUrl: string;
};
```

Do not store a single global active session unless the UI explicitly only supports one tab/device. If the UI can show multiple devices, store a map:

```ts
Record<string, ActiveSession>
```

where the key can be `deviceKind:deviceId` or a backend `session_id`.

## Migration Checklist

- Replace any normal-flow `/stream/active` usage with local `session_id` state.
- Keep phone/web start flow on `/stream/start`.
- Move Rokid start flow to `/rokid/stream/start`.
- Move Rokid frame upload to `/rokid/{session_id}/frame`.
- Move Rokid audio upload to `/rokid/{session_id}/audio_chunk`.
- Move Rokid status polling to `/rokid/{session_id}/status`.
- Preserve existing ask flow: `/ask/{session_id}/stream`.
- Include `owner_id`, `device_id`, and `device_type` when starting streams when available.
- Make sure each tab/device uses its own `session_id` for ask/status/upload.
- Remove logic that assumes starting a new stream cancels the previous one.

## Acceptance Tests

1. Start a phone stream and a Rokid stream at the same time.
2. Confirm phone uses `/stream/*` URLs and Rokid uses `/rokid/*` URLs.
3. Upload phone frames and Rokid frames concurrently.
4. Ask `/ask/{phone_session_id}/stream` and confirm it answers from phone session.
5. Ask `/ask/{rokid_session_id}/stream` and confirm it answers from Rokid session.
6. Confirm starting the Rokid stream does not make phone session return `inactive_session`.
7. Confirm `/stream/active` is not needed for the normal UI.

## Prompt For Frontend Agent

Use this prompt to ask a frontend coding agent to implement the update:

```text
You are modifying the frontend for WorldMM Online Server.

Backend has changed to multi-session mode and Rokid Glass APIs are now separated from phone/web stream APIs.

Implement these frontend changes:

1. Stop relying on GET /stream/active in normal user flows. Store and reuse the session_id returned by start/upload APIs.
2. For phone/web streaming, keep using POST /stream/start with input_mode="frame_audio_stream". Use returned frame_upload_url and audio_upload_url.
3. For Rokid Glass, use POST /rokid/stream/start instead of /stream/start. Do not send Rokid sessions through /stream/start.
4. For Rokid SDK frame/audio uploads, use:
   - POST /rokid/{session_id}/frame
   - POST /rokid/{session_id}/audio_chunk
   - GET /rokid/{session_id}/status
5. For Rokid RTMP mode, start with POST /rokid/stream/start and input_mode="rokid_live_rtmp", then call:
   - POST /rokid/{session_id}/live/ingest/start
   - POST /rokid/{session_id}/live/ingest/stop
6. Keep ask unchanged: POST /ask/{session_id}/stream.
7. Add optional owner_id, device_id, and device_type fields when starting phone or Rokid streams if the frontend has them.
8. Maintain session state per tab/device instead of one global active session. Do not assume starting one stream cancels another.
9. Add error handling for backend 400 Rokid-start misuse and 409 wrong_input_mode responses.
10. Add a small integration check or manual test path that can start phone and Rokid sessions simultaneously and ask each session independently.

Do not remove existing phone/web functionality. Keep backward-compatible UI where possible, but route Rokid through the new /rokid namespace.
```

