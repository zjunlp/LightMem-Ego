#!/usr/bin/env bash

# Shared defaults for dependency scheduling and the query client. Explicit
# local/remote selections and service URLs from .env or the caller are kept.
export EM2MEM_TEXT_EMBED_BACKEND="${EM2MEM_TEXT_EMBED_BACKEND:-remote}"
export EM2MEM_TEXT_EMBED_URL="${EM2MEM_TEXT_EMBED_URL:-http://${EM2MEM_TEXT_EMBED_HOST:-127.0.0.1}:${EM2MEM_TEXT_EMBED_PORT:-18096}}"
