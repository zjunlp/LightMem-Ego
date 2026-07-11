# Evaluation Recording Patch Notes

## Purpose

This patch adds evaluation-oriented logging for the EMNLP demo paper without changing the worker queue architecture.
It records future query traces in normal frontend, glasses, streaming, async, sync, and `scripts/ask_session_wait.py`
flows, then adds an offline script that exports a session-level `eval.json`.

## Files Changed

- `src/worldmm/llm/openai_gpt.py`
  - Records API request start, first token for streaming, response finish, duration, request path, model, attempt, and errors in `last_debug.api_timing`.

- `online_query/query_engine.py`
  - Adds `eval_trace` with query/retrieval/prompt/generation timestamps.
  - Adds eval metadata to final selected evidence: `eval_rank`, `eval_score`, `eval_score_source`, `eval_evidence_type`.
  - Stores LLM API timing under `result.eval_trace.llm_api`.

- `api_server.py`
  - Makes synchronous `/ask` also write a complete query task JSON to `online_tasks/query_done` or `online_tasks/query_failed`.
  - Adds that generated task id to `qa_history.jsonl`.

- `scripts/ask_session_wait.py`
  - Keeps using async `/session/{session_id}/ask`, so it benefits from worker/query trace.
  - Adds `--client-source` and `--input-method`, defaulting to `script` and `cli`.

- `scripts/build_session_eval.py`
  - New offline exporter.
  - Writes `online_sessions/{session_id}/eval.json` by default.
  - Automatically includes related Rokid parent/day-child sessions unless `--no-related-sessions` is passed.

## How To Apply

From `/zjunlp/chenyijun/worldmm-online-server-release`, review and apply:

```bash
git apply /path/to/eval_recording_changes.clean.diff
```

If the server working tree has local edits, review conflicts manually before applying.

## Verification

```bash
python -m py_compile api_server.py online_query/query_engine.py src/worldmm/llm/openai_gpt.py scripts/ask_session_wait.py scripts/build_session_eval.py
python scripts/build_session_eval.py --session-id <SESSION_ID> --pretty
```

Then inspect:

- `online_tasks/query_done/<task_id>.json` contains `result.eval_trace`.
- `result.raw.llm_debug` contains `api_timing` when the OpenAI wrapper is used.
- `online_sessions/<SESSION_ID>/eval.json` contains `questions`, `memory_tasks`, `latency_tables`, and metric placeholders.

## Known Historical Limitations

- Old synchronous QA records that never wrote full query task files cannot recover evidence or detailed latency.
- Old tasks without `eval_trace` use existing `latency.*_ms` fields as fallback.
- Gold evidence, gold answers, and scenario labels are intentionally not recorded.
