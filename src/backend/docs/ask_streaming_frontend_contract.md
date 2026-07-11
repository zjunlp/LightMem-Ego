# Ask Streaming Frontend Contract

This update keeps the old `/ask/{session_id}` polling flow and adds an optional streaming answer flow.

## What Changed

- Existing legacy mode is unchanged:
  - `POST /ask/{session_id}` with default `response_mode: "legacy"` still returns `202 queued` for async mode.
  - Frontend can keep polling `GET /query_task/{task_id}` exactly as before.
- New streaming mode:
  - `POST /ask/{session_id}` with `response_mode: "stream"` returns `text/event-stream`.
  - `POST /ask/{session_id}/stream` is an equivalent explicit streaming endpoint.
  - The backend streams LLM token deltas directly, then emits a final `done` event with the complete old-style result JSON.

## Request

Legacy polling:

```json
{
  "question": "刚才发生了什么？",
  "response_mode": "legacy",
  "mode": "async",
  "memory_mode": "auto"
}
```

Streaming:

```json
{
  "question": "刚才发生了什么？",
  "response_mode": "stream",
  "memory_mode": "auto"
}
```

The usual query options are still supported: `retrieval_mode`, `use_image_evidence`, `max_image_evidence`, `use_current`, `use_short_term`, `use_long_term`, `cache_mode`, etc.

## SSE Events

The stream uses standard Server-Sent Events format:

```text
event: delta
data: {"type":"delta","stage":"answer","delta":"这"}
```

Event types:

- `start`: request accepted by streaming path.
- `ping`: keepalive event while retrieval or LLM generation is still running; ignore for display.
- `delta`: append `data.delta` to the final-answer buffer.
- `final`: final text is ready earlier than the full result payload in some paths.
- `done`: contains `data.result`, the complete response object compatible with the old result JSON.
- `error`: terminal error; show `data.message`. The backend will still send a following `done` event so the UI can reset loading state.

Frontend display rule:

- Do not expect a draft/provisional answer.
- Once the first `delta` arrives, append deltas into a streaming answer buffer.
- When `done` arrives, use `done.result.answer` as the authoritative final answer and update evidence, latency, timestamps, images, and debug fields from `done.result`.
- Always reset the loading state on `done`, even when `data.status === "error"`.
- If the network closes before `done`, reset loading state and show a retryable connection error.

## Fetch Example

```js
const res = await fetch(`/ask/${sessionId}`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question,
    response_mode: "stream",
    memory_mode: "auto"
  })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";
let answer = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  const parts = buffer.split("\n\n");
  buffer = parts.pop() || "";

  for (const part of parts) {
    const eventLine = part.split("\n").find((line) => line.startsWith("event: "));
    const dataLine = part.split("\n").find((line) => line.startsWith("data: "));
    if (!dataLine) continue;

    const event = eventLine ? eventLine.slice(7) : "message";
    const data = JSON.parse(dataLine.slice(6));

    if (event === "ping") {
      continue;
    } else if (event === "delta") {
      answer += data.delta || "";
      renderStreamingAnswer(answer);
    } else if (event === "done") {
      renderFinalResult(data.result);
    } else if (event === "error") {
      showError(data.message);
    }
  }
}
```

## Frontend Toggle

Expose a setting such as:

- `legacy`: stable polling mode, waits for final complete JSON.
- `stream`: progressive mode, shows token streaming, then reconciles with final JSON.

The backend default is `legacy`, so existing clients remain compatible.

## Legacy Polling Notes

`GET /query_task/{task_id}` keeps returning the original task payload with `result`.
For frontend compatibility, successful done responses also mirror common fields such as `answer`, `answer_text`, `timestamps`, `evidence_frames`, `latency`, and `stream_context` to the top level.

Recommended legacy completion check:

```js
if (data.status === "done" && data.queue_state === "query_done") {
  const result = data.result || data;
  renderFinalResult(result);
}
```

## Reverse Proxy Note

If the API is behind Nginx or a cloud reverse proxy, response buffering must be disabled for the streaming endpoint. Otherwise the browser may still receive tokens only after the full answer finishes. The backend sends `X-Accel-Buffering: no`, but the proxy config should also allow streaming for `/ask/*`.
