# Stage Rokid Backend Adapter Plan

## Scope

This stage adds a backend compatibility layer for Rokid Glass as an input source. It does not implement Android SDK code and does not call any Rokid glasses API from the backend.

The Rokid SDK should run on the glasses side or a phone-side Android connector. That connector captures frames and audio, normalizes them, and uploads to this backend.

## Backend Path

```text
Rokid Glass
-> Rokid SDK / Phone Connector
-> upload frame/audio
-> /stream/start(input_mode=rokid_frame_audio)
-> /stream/<session_id>/frame
-> /stream/<session_id>/audio_chunk
-> ingest_frame / ingest_audio_chunk
-> M_cur / M_st / rolling ASR / transcript backfill / M_lt / query
```

Internally, `rokid_frame_audio` reuses the existing realtime frame/audio session structure and does not create a Rokid-only memory pipeline, ASR worker, or query path.

## Connector Recommendations

- Video: upload JPEG or WebP frames. Recommended first version: JPEG, width around 640 px, quality around 75, 1 FPS.
- Audio: upload WAV 16 kHz mono chunks around 1000 ms. Raw `pcm_s16le` is supported when `sample_rate`, `channels`, and `sample_width=2` are provided, but WAV is simpler and preferred.
- Timestamps: use the same stream origin for video and audio:

```text
relative_ts_ms = Android SystemClock.elapsedRealtime() - streamStartElapsedMs
```

If the SDK provides `NV21`, convert `NV21 -> JPEG` inside the connector before upload. Backend raw `NV21` support is optional and currently disabled with a clear error message.

## Status State

Rokid sessions write a lightweight `stream/rokid_state.json`:

```json
{
  "input_mode": "rokid_frame_audio",
  "device_type": "rokid",
  "transport": "phone_sdk",
  "timestamp_mode": "connector_relative_ts_ms",
  "latest_frame_relative_ts_ms": 12000,
  "latest_audio_relative_ts_ms": 11500,
  "timestamp_warnings": 0
}
```

`GET /stream/<session_id>/status` exposes this as a `rokid` block and still includes the standard `frame_stream`, `audio_stream`, `frame_mst`, `memory`, and task blocks.

## Next Stage

The next stage is an Android Rokid Connector that:

1. Starts a session with `input_mode=rokid_frame_audio`.
2. Stores `streamStartElapsedMs = SystemClock.elapsedRealtime()`.
3. Uploads JPEG/WebP frames to `frame_upload_url`.
4. Uploads WAV or supported PCM chunks to `audio_upload_url`.
5. Reuses one timeline for both media streams.
6. Periodically polls `/stream/<session_id>/status` for health and warnings.
