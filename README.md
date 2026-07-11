# LightMem-Ego

LightMem-Ego is an end-to-end egocentric memory system for smart glasses. It connects a Rokid AI Glass app, a web frontend, and a backend service so that a user can stream first-person camera/audio context, build memory from daily experience, and ask questions about live or remembered moments.

The project is organized as three cooperating components:

- The glasses app captures first-person camera frames and microphone audio, controls live sessions, submits voice questions, and displays answers on the glasses.
- The backend receives live streams, manages sessions, extracts and stores memory, performs retrieval, and returns answers.
- The frontend provides a browser interface for reviewing sessions, interacting with memory, and using the system outside the glasses.

## Repository Layout

```text
src/
  frontend/       # Web UI for using and reviewing LightMem-Ego
  backend/        # API service, online workers, and memory-processing logic
  ai_glass_app/   # Rokid AI Glass Android app
```

## System Flow

```text
Rokid AI Glass app -> Backend API and workers -> Memory / retrieval / QA
                                      ^
                                      |
                              Web frontend
```

At runtime, the glasses app opens a live session with the backend, sends camera/audio data, and receives answers for voice questions. The frontend connects to the same backend service for browser-side interaction and session review.

## Components

### `src/ai_glass_app/`

Android app for Rokid AI Glass.

Current open-source features include:

- Real-time glasses capture session start/stop.
- Camera frame capture from the glasses camera.
- Microphone audio capture from the glasses microphone.
- RTMP live video push when a backend `push_url` is available.
- HTTP frame/audio upload fallback when RTMP is unavailable.
- Voice-question recording and answer display on the glasses UI.

This open-source version does not include local session recording, replay-from-file mode, preset-question UI, or standalone sample screens.

See [src/ai_glass_app/README.md](src/ai_glass_app/README.md) for build, install, configuration, controls, and permissions.

### `src/backend/`

Backend service for LightMem-Ego. It is responsible for API endpoints, online session management, stream ingestion, memory processing, retrieval, and answer generation.

See [src/backend/README.md](src/backend/README.md) for backend-specific setup and deployment notes.

### `src/frontend/`

Web frontend for LightMem-Ego. It provides the browser interface for interacting with the system, reviewing memory/session content, and using backend-powered QA outside the glasses.

See [src/frontend/README.md](src/frontend/README.md) for frontend-specific setup and deployment notes.

## Quick Start

Each subproject has its own setup and runtime requirements. Start with the README in the component you want to run.

For the glasses app:

```powershell
cd src\ai_glass_app
.\gradlew.bat assembleDebug
```

On macOS or Linux:

```bash
cd src/ai_glass_app
./gradlew assembleDebug
```

## Configuration

The glasses app API endpoint is configured in:

```text
src/ai_glass_app/app/src/main/java/cn/zjukg/lightmem/glass/worldmm/WorldMMConfig.kt
```

Set `API_BASE_URL` to the backend API address you want the glasses app to use.

## Privacy Notice

LightMem-Ego may process camera frames, microphone audio, transcripts, generated answers, and memory data depending on deployment configuration. Before deploying with real users, review the API endpoint configuration, data retention policy, access control, and user consent flow for your environment.

## License

See [LICENSE](LICENSE).
