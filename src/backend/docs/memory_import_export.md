# Memory import and export

The memory workspace accepts UTF-8 JSON or TXT files up to 10 MiB and 1000
records. Each import creates a separate `import_<task-id>` session. It does not
replace an existing video session or restore uploaded filesystem paths.

## Import

`POST /memory-imports` accepts multipart fields `file`, `format` (`auto`, `json`,
or `txt`) and optional `title`. The 202 response includes `status: queued`,
`state: queued`, `task_id`, `session_id`, `record_count`, and `query_ready: false`.

JSON may contain a record, an array of strings/records, `{ "records": [...] }`,
or the exported `{ "format": "lightmem-memory", "record": {...} }` envelope.
For example:

```json
{
  "records": [
    {"text": "The keys were placed on the passenger seat.", "start": 43, "end": 45},
    {"text": "The backpack was placed in the rear seat.", "start": 90, "end": 95}
  ]
}
```

TXT paragraphs become individual text memories. Missing observation times are
marked unknown; the internal zero used for indexing is not shown as a recorded
video timestamp. Imports contain textual annotations; media files and visual
embeddings are not carried by a JSON export.

`GET /memory-imports/{task_id}` returns `status: ok` and durable task state:
`queued`, `processing`, `ready`, or `failed`. Processing stages include
`normalizing`, `building_memory`, `building_retrieval`, and `warming_query`.
Only a published memory, completed retrieval indexes, and successful query
initialization permit `query_ready: true`. Interrupted queued/processing imports
are recovered when the API starts. Failed imports remain visible in task status
and do not enter the ready-session catalog.

The implementation uses the normal multiscale memory builder, configured graph
and semantic extraction, text embedding service, and query runtime. It does not
create placeholder video files or synthetic short-term events.

## Export

`GET /session/{session_id}/memories/{memory_id}/export?format=json` downloads one
episodic, semantic, or visual record from the currently published version.
The ID is the normalized record `id` returned by the session memories endpoint.
`GET /memories/{memory_id}/export?format=json` supports clients that only have
this globally scoped ID; clients with a session ID should use the scoped route.

Exports preserve full memory text and structured record data, rather than the
truncated display representation. They include source session, version, type,
granularity, and export time. Local media/model paths, resource URLs, embeddings,
and credential fields are omitted. Exported JSON can be imported into a new
session. Generated semantic facts and coarse summaries are rebuilt according
to the server's generation settings.

Responses include `X-Request-ID`; downloads set `Content-Disposition: attachment`
and `Cache-Control: no-store`. Structured errors use
`{"status":"error","error":{"code":"...","message":"...","details":{}}}`.

## Verification

```bash
.venv/bin/python -m pytest tests/test_memory_transfer.py tests/test_memory_editor_connection.py tests/test_video_uploads.py -q
```
