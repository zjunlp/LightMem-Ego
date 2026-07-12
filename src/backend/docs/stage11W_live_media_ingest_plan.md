# Stage 11W Live Media Ingest Plan

Current live media ingest accepts `live_pusher_rtmp` and `web_webrtc_whip` sessions, stores SRS/live source metadata, and uses `online_live_ingest_worker.py` to pull media and call the shared realtime ingest adapters.

Rokid backend adapter is intentionally not part of this live ingest worker path. `input_mode=rokid_frame_audio` uses direct HTTP frame/audio uploads:

```text
Rokid Glass
-> Rokid SDK / Phone Connector
-> /stream/start(input_mode=rokid_frame_audio)
-> /stream/<session_id>/frame
-> /stream/<session_id>/audio_chunk
-> ingest_frame / ingest_audio_chunk
```

No SRS, WHIP, RTMP, live source, or live ingest worker is required for Rokid direct connector input.

Live/WebRTC/RTMP audio and direct `frame_audio_stream` uploads still enter the backend as short audio chunks. The ASR worker does not transcribe every short chunk directly. Accepted short chunks are buffered and grouped into rolling ASR windows that target about 3 seconds:

```bash
EM2MEM_AUDIO_ASR_WINDOW_MS=3000
EM2MEM_AUDIO_ASR_HOP_MS=3000
EM2MEM_AUDIO_ASR_MIN_WINDOW_MS=2000
EM2MEM_AUDIO_ASR_FLUSH_MIN_MS=1000
EM2MEM_AUDIO_ASR_MAX_WINDOW_MS=4000
EM2MEM_AUDIO_ASR_MAX_PENDING_WINDOWS=5
```

Normal windows are usually about 2-4 seconds depending on chunk boundaries, and stream stop/finalize may flush a tail window when at least 1 second remains.

`stream/audio_chunks/` stores raw uploaded or live-ingested short slices. `stream/audio_asr/windows/` stores grouped ASR windows, concat manifests, temporary wav parts, window-level temporary sources, and final wav files, so the two directories are not one-to-one.

Fragmented `.m4a`/`.mp4` chunks may not be independently decodable when only the first chunk carries `ftyp`/`moov` initialization metadata. The `stream_asr` worker keeps the normal per-chunk transcode path first, then can fall back to a window-level mp4-like source with an init segment.
