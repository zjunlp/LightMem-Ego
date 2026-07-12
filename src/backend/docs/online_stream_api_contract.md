# Online Stream API Contract

This contract covers backend realtime stream inputs. All realtime frame/audio inputs converge on the same backend adapters:

- frame input -> `ingest_frame(...)` -> `M_cur` / `M_st`
- audio input -> `ingest_audio_chunk(...)` -> rolling ASR window queue -> transcript backfill
- query -> existing router / retriever / evidence / answer path

## Supported Input Modes

`POST /stream/start` accepts:

- `chunk`
- `frame_audio_stream`
- `frame_stream` alias for `frame_audio_stream`
- `live_pusher_rtmp`
- `web_webrtc_whip`
- `rokid_frame_audio`

`rokid_frame_audio` is a backend adapter for Rokid Glass connector uploads. The backend does not call the Rokid SDK. The SDK runs on the glasses or an Android phone connector, captures camera/audio, and uploads normalized frame/audio payloads to the existing HTTP endpoints.

## Rokid Start

Request:

```json
{
  "input_mode": "rokid_frame_audio",
  "metadata": {
    "source": "rokid_glass",
    "device_type": "rokid",
    "transport": "phone_sdk",
    "sdk": "rokid",
    "sdk_version": "...",
    "device_id": "...",
    "glass_sn": "...",
    "phone_model": "..."
  }
}
```

Response includes the normal stream fields plus:

```json
{
  "status": "stream_started",
  "input_mode": "rokid_frame_audio",
  "session_id": "...",
  "stream_id": "...",
  "frame_upload_url": "/stream/<sid>/frame",
  "audio_upload_url": "/stream/<sid>/audio_chunk",
  "rokid": {
    "enabled": true,
    "adapter": "phone_sdk_connector",
    "expected_video": {
      "preferred_format": "jpeg",
      "accepted_formats": ["jpg", "jpeg", "png", "webp"],
      "optional_raw_formats": ["nv21"],
      "recommended_fps": 1,
      "recommended_width": 640,
      "recommended_quality": 75
    },
    "expected_audio": {
      "preferred_format": "wav",
      "accepted_formats": ["wav", "pcm", "aac", "m4a", "mp3"],
      "recommended_chunk_ms": 1000,
      "asr_window_ms": 3000
    },
    "timestamp": {
      "required": "relative_ts_ms",
      "recommended_clock": "android_elapsed_realtime"
    }
  },
  "can_ask": false
}
```

The session structure is the same lightweight realtime frame/audio stream structure. `rokid_frame_audio` does not start live ingest, does not depend on SRS/WebRTC/RTMP, and does not create a Rokid-specific memory, ASR, or query path.

## Rokid Frame Upload

Endpoint:

```bash
POST /stream/<session_id>/frame
```

Multipart fields:

- `frame`: required file field
- `frame_index`: required integer; gaps are allowed
- `relative_ts_ms`: strongly required; if omitted, the backend falls back to server receive elapsed time and records a warning
- `client_ts_ms`: optional wall-clock client timestamp
- `device_ts_ms` or `source_ts_ms`: optional device timestamp
- `format`: `jpg`, `jpeg`, `png`, `webp`; `nv21` returns a clear unsupported raw-frame error
- `source`: defaults to `rokid_sdk_video` for Rokid sessions
- `width`, `height`: optional metadata

Example:

```bash
curl -X POST http://127.0.0.1:8000/stream/$SESSION_ID/frame \
  -F "frame=@/path/to/frame_001.jpg" \
  -F "frame_index=1" \
  -F "relative_ts_ms=1000" \
  -F "client_ts_ms=1710000000000" \
  -F "format=jpg" \
  -F "source=rokid_sdk_video"
```

First version recommendation: convert Rokid SDK `NV21` frames to JPEG/WebP in the Android connector before upload. Backend raw `NV21` upload is optional and currently returns:

```text
raw nv21 upload is not enabled; please convert to jpeg on Rokid Connector
```

## Rokid Audio Upload

Endpoint:

```bash
POST /stream/<session_id>/audio_chunk
```

Multipart fields:

- `audio`: required file field
- `audio_index`: required integer; gaps are allowed
- `relative_ts_ms`: strongly required; if omitted, the backend falls back to server receive elapsed time and records a warning
- `duration_ms`: recommended
- `format`: `wav`, `pcm`, `aac`, `m4a`, or `mp3`
- `sample_rate`: recommended, required for raw PCM
- `channels`: recommended, required for raw PCM
- `sample_width`: required for raw PCM; must be `2`
- `encoding`: raw PCM supports `pcm_s16le`
- `source`: defaults to `rokid_sdk_audio` for Rokid sessions

Example WAV upload:

```bash
curl -X POST http://127.0.0.1:8000/stream/$SESSION_ID/audio_chunk \
  -F "audio=@/path/to/audio_001.wav" \
  -F "audio_index=1" \
  -F "relative_ts_ms=1000" \
  -F "duration_ms=1000" \
  -F "format=wav" \
  -F "sample_rate=16000" \
  -F "channels=1" \
  -F "source=rokid_sdk_audio"
```

Example raw PCM upload:

```bash
curl -X POST http://127.0.0.1:8000/stream/$SESSION_ID/audio_chunk \
  -F "audio=@/path/to/audio_001.pcm" \
  -F "audio_index=1" \
  -F "relative_ts_ms=1000" \
  -F "duration_ms=1000" \
  -F "format=pcm" \
  -F "sample_rate=16000" \
  -F "channels=1" \
  -F "sample_width=2" \
  -F "encoding=pcm_s16le" \
  -F "source=rokid_sdk_audio"
```

The backend wraps supported raw `pcm_s16le` into WAV before registering the chunk. First version recommendation remains: send WAV 16 kHz mono from the connector.

Audio chunks from `frame_audio_stream`, Rokid direct upload, WebRTC/WHIP, and RTMP live ingest are accepted as short slices. ASR does not transcribe every short slice directly in the upload or ingest request path. The backend buffers accepted short chunks in `stream/audio_buffer_index.json` and queues rolling ASR windows through the existing `stream_asr` worker path.

Default rolling ASR window settings target about 3 seconds:

```bash
EM2MEM_AUDIO_ASR_WINDOW_MS=3000
EM2MEM_AUDIO_ASR_HOP_MS=3000
EM2MEM_AUDIO_ASR_MIN_WINDOW_MS=2000
EM2MEM_AUDIO_ASR_FLUSH_MIN_MS=1000
EM2MEM_AUDIO_ASR_MAX_WINDOW_MS=4000
EM2MEM_AUDIO_ASR_MAX_PENDING_WINDOWS=5
```

Normal windows are non-overlapping and roughly 2-4 seconds depending on chunk boundaries. A final stream stop can flush a tail window when at least 1 second of audio remains.

`stream/audio_chunks/` and `stream/audio_asr/windows/` are not one-to-one:

- `stream/audio_chunks/` contains the raw uploaded or live-ingested short slices.
- `stream/audio_asr/windows/` contains grouped ASR windows, concat manifests, temporary wav parts, window-level temporary sources, and final wav files.

Fragmented `.m4a`/`.mp4` chunks may not be independently decodable because only the first chunk has `ftyp`/`moov` initialization metadata while later chunks contain media fragments. The ASR worker first tries the normal per-chunk transcode path, then can fall back to a window-level source with an init segment for mp4-like audio containers.

## Timeline Contract

The connector must use one shared stream-start origin for frames and audio:

```text
relative_ts_ms = SystemClock.elapsedRealtime() - streamStartElapsedMs
```

`relative_ts_ms` is written into frame state, audio buffer state, ASR windows, current memory, and short-term frame events. If a timestamp is missing, the backend falls back to server receive elapsed time, clamps through existing monotonic stream logic, records a warning, and keeps accepting future media.

## Status

`GET /stream/<session_id>/status` includes the normal status plus `rokid` for Rokid sessions:

```json
{
  "rokid": {
    "enabled": true,
    "input_mode": "rokid_frame_audio",
    "device_type": "rokid",
    "transport": "phone_sdk",
    "video_source": "rokid_sdk_video",
    "audio_source": "rokid_sdk_audio",
    "frames_received": 12,
    "audio_chunks_received": 8,
    "latest_frame_relative_ts_ms": 12000,
    "latest_audio_relative_ts_ms": 11500,
    "audio_asr_window_ms": 3000,
    "timestamp_mode": "connector_relative_ts_ms",
    "warnings": []
  }
}
```

Status is read-only. It reads state files and does not trigger ASR, query, or heavy media probing.
