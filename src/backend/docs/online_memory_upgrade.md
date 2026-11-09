# Online memory retrieval and evidence upgrade

This update retains the current, short-term, long-term, and interaction-cache
tiers. It improves their publication, retrieval, and answer-context boundaries;
it does not replace the offline implementation under `research/em2mem/`.

## Memory construction and publication

- Current-frame change scores are computed from the original source captures.
- Short-term refinement can inspect all eight retained keyframes by default.
  Set `EM2MEM_MST_REFINE_MAX_IMAGES` to trade coverage for model cost.
- Long-term `visual_summary` is generated separately from the event narrative.
  The OpenAI backend inspects selected event images; deterministic backends use
  the available structured evidence. Text-only aggregation does not silently
  masquerade as an independent visual observation.
- Timed frame descriptions retain source event IDs, versions, and capture
  paths for image selection. They supplement event narratives, rather than
  replacing the short-term tier with a second long-term detail store.
- Consolidation validates source versions before publication. Queries pin a
  published snapshot, and cache retirement waits for active users to finish.

## Retrieval and answer context

- Short-term retrieval can combine lexical retrieval with an incremental text
  embedding index. With `EM2MEM_TEXT_EMBED_BACKEND=remote`, it is enabled by
  default. `EM2MEM_MST_SEMANTIC_ENABLED=0` disables it. Missing or pending vectors
  retain the lexical path, rather than requiring model initialization during
  every question.
- Query time constraints are derived from the original question. A user's
  explicit time range and the arrived-video cutoff take precedence over cache
  hints. Identity-only follow-ups can inherit an entity without inheriting the
  previous event's time or transient appearance.
- Selected events use one evidence schema for identities, timestamps, frames,
  and provenance. The final text-event capacity follows enabled candidate
  budgets unless the caller or deployment supplies a cap; image limits remain
  independent.
- A soft text budget preserves selected event narratives before optional detail
  fields. Parent summaries remain broader context with their own time range.
- Image ranking uses saved descriptions, action timestamps, event priority, and
  coverage. It does not issue a new vision-model call to rank each image.
  Missing descriptions can receive bounded inspection opportunities. Image
  labels identify capture time and source, and duplicate captures retain their
  original provenance.
- Generation failures, local evidence summaries, text-only fallback, and
  successful multimodal answers have distinct states. Failed answers do not
  establish normal cache anchors for the next turn.
- `EM2MEM_LLM_EVENT_SELECTOR_ENABLED=0` bypasses the query-time LLM event
  selector; `1` retains the existing selector default. This setting is
  independent of the final answer model.

These are retrieval and context policies, not guarantees that every question
is answered correctly. Ranking heuristics and model answers still need
comparative evaluation. Passing the offline tests below does not establish an
accuracy gain or validate a particular remote model service.

## API compatibility

Existing capture, question, streaming, memory-view, edit, and rollback routes
remain available. New routes support processing uploaded historical videos and
serving session-bound video metadata and byte ranges:

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/video-uploads` | Receive a video with an idempotent upload ID and build its memory in the background |
| GET | `/video-uploads/{upload_id}` | Inspect upload and build progress |
| POST | `/video-uploads/{upload_id}/resume` | Resume a recoverable upload job |
| GET | `/session/{session_id}/demo-media` | Read the session's video binding and memory coverage |
| GET | `/session/{session_id}/demo-video` | Serve the bound video with HTTP byte-range support |

The upload path uses the existing full-video preprocessing, memory-building,
and retrieval-index functions. It does not make future video frames available
to a live prefix session. Camera sessions are separate from historical uploads.
The legacy optional `demo.api` wrapper is not part of this update.

The existing `PUT /session/{session_id}/title` route continues to update display
titles using optimistic revision checks.

The React frontend, glasses client, root Docker configuration, and offline
research tree are retained. UI controls for new routes can be integrated
separately. Existing memory-edit routes also expose additive editability and
title metadata to clients.

Task polling at `/query_task/{task_id}` and `/task/query/{task_id}` accepts
`compact=1` to omit repeated retrieval traces from the client response while
retaining the full answer, evidence images, timestamps, and failure indicators.
The default full response and stored task result remain unchanged.

## Configuration and verification

See `.env.example` for model endpoints and tuning variables. The model name,
API key, endpoint, and original image limits remain deployment choices; no
machine-specific credentials or existing session data are included.

From the repository root:

```bash
cd src/backend
python -m pip install -e '.[dev]'
python -m pytest -q tests
```

Regression fixtures contain compact text and evidence metadata. Tests create
their own temporary state; they do not need the source videos, private replay
controllers, API credentials, or live model services. The migration-specific
tests cover the interaction between the upstream timestamp decoder and the
new finite-second schema boundary, including long sessions and legacy cache
timestamps.

The original test helpers that globally replaced modules in `sys.modules`
have been replaced with regular imports or isolated source loaders so that
memory and HTTP tests can run in one process.
