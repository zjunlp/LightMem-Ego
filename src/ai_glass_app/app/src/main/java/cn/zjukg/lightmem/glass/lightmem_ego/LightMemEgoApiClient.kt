package cn.zjukg.lightmem.glass.lightmem_ego

import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.time.Instant
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.UUID

data class LightMemEgoStartResult(
    val sessionId: String,
    val parentSessionId: String,
    val childSessionId: String,
    val dayLabel: String,
    val dayIndex: Int,
    val runId: String,
    val streamId: String,
    val canAsk: Boolean,
    val inputMode: String,
    val pushUrl: String,
    val liveIngestStartPath: String,
    val liveIngestStopPath: String,
    val frameUploadPath: String,
    val audioUploadPath: String,
    val statusPath: String,
    val audioQuestionPath: String,
)

class LightMemEgoApiException(
    message: String,
    val statusCode: Int,
    val payload: JSONObject,
) : IllegalStateException(message)

data class LightMemEgoUploadResult(
    val status: String,
    val canAsk: Boolean,
)

data class LightMemEgoStatusResult(
    val streamStatus: String,
    val canAsk: Boolean,
    val memoryReady: Boolean,
    val framesReceived: Int,
    val audioChunksReceived: Int,
    val latestFrameTsMs: Long?,
    val latestAudioTsMs: Long?,
)

data class LightMemEgoAudioQuestionSubmitResult(
    val status: String,
    val question: String,
    val queued: Boolean,
    val taskId: String,
    val answer: String,
    val message: String,
)

data class LightMemEgoStreamEvent(
    val type: String,
    val status: String,
    val delta: String,
    val text: String,
    val answer: String,
    val question: String,
    val message: String,
)

data class LightMemEgoStreamResult(
    val status: String,
    val answer: String,
    val question: String,
    val message: String,
)

data class LightMemEgoAudioQuestionStreamResult(
    val status: String,
    val question: String,
    val answer: String,
    val message: String,
)

data class LightMemEgoQueryTaskResult(
    val status: String,
    val done: Boolean,
    val answer: String,
    val message: String,
)

class LightMemEgoApiClient(
    private val baseUrl: String = LightMemEgoConfig.API_BASE_URL,
) {
    private val frameUploadPaths = mutableMapOf<String, String>()
    private val audioUploadPaths = mutableMapOf<String, String>()
    private val statusPaths = mutableMapOf<String, String>()
    private val audioQuestionPaths = mutableMapOf<String, String>()
    private val startDateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
    private val dateLabelFormatter = DateTimeFormatter.ofPattern("yyyy.M.d")

    fun startRokidStream(
        inputMode: String = LightMemEgoConfig.INPUT_MODE,
        parentSessionId: String? = null,
        runId: String,
        createParentSession: Boolean,
    ): LightMemEgoStartResult {
        val requestedParentSessionId = parentSessionId?.trim().orEmpty()
        val startTsMs = System.currentTimeMillis()
        val zone = ZoneId.systemDefault()
        val startInstant = Instant.ofEpochMilli(startTsMs)
        val startDate = startInstant.atZone(zone).toLocalDate()
        val metadata = JSONObject()
            .put("source", "rokid_glass")
            .put("device_type", "rokid")
            .put("transport", if (inputMode == "rokid_live_rtmp") "rtmp_srs" else "glasses_bare_app")
            .put("sdk", if (inputMode == "rokid_live_rtmp") "android_rootencoder_rtmp" else "android_camera_x_audio_record")
            .put("timestamp_mode", "connector_relative_ts_ms")
            .put("run_id", runId)
            .put("client_session_start_ts_ms", startTsMs)
            .put("client_timezone_id", zone.id)
            .put("client_timezone_offset_minutes", zone.rules.getOffset(startInstant).totalSeconds / 60)
            .put("client_start_date", dateLabelFormatter.format(startDate))
            .put("client_start_datetime", startDateTimeFormatter.format(startInstant.atZone(zone)))
        if (requestedParentSessionId.isNotBlank()) {
            metadata.put("parent_session_id", requestedParentSessionId)
        }
        val body = JSONObject()
            .put("input_mode", inputMode)
            .put("chunk_duration", 1)
            .put("run_id", runId)
            .put("create_parent_session", createParentSession)
            .put("metadata", metadata)
        if (requestedParentSessionId.isNotBlank()) {
            body.put("session_id", requestedParentSessionId)
            body.put("parent_session_id", requestedParentSessionId)
        }

        val json = postJson("/rokid/stream/start", body)
        val result = parseStartResult(
            json = json,
            requestedParentSessionId = requestedParentSessionId,
            inputMode = inputMode,
            runId = runId,
        )
        if (result.sessionId.isNotBlank()) {
            frameUploadPaths[result.sessionId] = result.frameUploadPath
            audioUploadPaths[result.sessionId] = result.audioUploadPath
            statusPaths[result.sessionId] = result.statusPath
            audioQuestionPaths[result.sessionId] = result.audioQuestionPath
        }
        return result
    }

    internal fun parseStartResult(
        json: JSONObject,
        requestedParentSessionId: String,
        inputMode: String,
        runId: String,
    ): LightMemEgoStartResult {
        val sessionId = json.optString("session_id")
        val parentId = json.optString("parent_session_id", requestedParentSessionId)
        val childId = json.optString("child_session_id", sessionId)
        val dayContext = json.optJSONObject("day_context")
            ?: json.optJSONObject("dayContext")
            ?: JSONObject()
        val framePath = json.optString("frame_upload_url").ifBlank { "/rokid/$sessionId/frame" }
        val audioPath = json.optString("audio_upload_url").ifBlank { "/rokid/$sessionId/audio_chunk" }
        val statusPath = json.optString("status_url").ifBlank { "/rokid/$sessionId/status" }
        val audioQuestionPath = json.optString("audio_question_url").ifBlank { "/rokid/$sessionId/audio_question" }
        val dayLabel = dayContext.firstDateLabel()
            ?: json.firstDateLabel()
            ?: currentDateLabel()
        val dayIndex = dayContext.firstIntOrNull("day_index", "dayIndex", "index")
            ?: json.firstIntOrNull("day_index", "dayIndex")
            ?: 0
        return LightMemEgoStartResult(
            sessionId = sessionId,
            parentSessionId = parentId,
            childSessionId = childId.ifBlank { sessionId },
            dayLabel = dayLabel,
            dayIndex = dayIndex,
            runId = dayContext.optString("run_id", runId).ifBlank { runId },
            streamId = json.optString("stream_id"),
            canAsk = json.optBoolean("can_ask", false),
            inputMode = json.optString("input_mode", inputMode),
            pushUrl = json.optString("push_url"),
            liveIngestStartPath = json.optString("live_ingest_start_url"),
            liveIngestStopPath = json.optString("live_ingest_stop_url"),
            frameUploadPath = framePath,
            audioUploadPath = audioPath,
            statusPath = statusPath,
            audioQuestionPath = audioQuestionPath,
        )
    }

    fun startLiveIngest(sessionId: String, path: String): JSONObject {
        val effectivePath = path.ifBlank { "/rokid/$sessionId/live/ingest/start" }
        return postJson(effectivePath, JSONObject())
    }

    fun stopLiveIngest(sessionId: String, path: String): JSONObject {
        val effectivePath = path.ifBlank { "/rokid/$sessionId/live/ingest/stop" }
        return postJson(effectivePath, JSONObject())
    }

    fun uploadFrame(
        sessionId: String,
        frameBytes: ByteArray,
        frameIndex: Int,
        relativeTsMs: Long,
        width: Int,
        height: Int,
    ): LightMemEgoUploadResult {
        val json = postMultipart(
            path = frameUploadPaths[sessionId] ?: "/rokid/$sessionId/frame",
            fileField = "frame",
            filename = "rokid-frame-$frameIndex.jpg",
            contentType = "image/jpeg",
            payload = frameBytes,
            fields = mapOf(
                "frame_index" to frameIndex.toString(),
                "relative_ts_ms" to relativeTsMs.toString(),
                "client_ts_ms" to System.currentTimeMillis().toString(),
                "format" to "jpg",
                "source" to "rokid_sdk_video",
                "width" to width.toString(),
                "height" to height.toString(),
            ),
        )
        return LightMemEgoUploadResult(
            status = json.optString("status"),
            canAsk = json.optBoolean("can_ask", false) || json.optBoolean("mcur_ready", false),
        )
    }

    fun uploadAudioChunk(
        sessionId: String,
        wavBytes: ByteArray,
        audioIndex: Int,
        relativeTsMs: Long,
        durationMs: Long,
    ): LightMemEgoUploadResult {
        val json = postMultipart(
            path = audioUploadPaths[sessionId] ?: "/rokid/$sessionId/audio_chunk",
            fileField = "audio",
            filename = "rokid-audio-$audioIndex.wav",
            contentType = "audio/wav",
            payload = wavBytes,
            fields = mapOf(
                "audio_index" to audioIndex.toString(),
                "relative_ts_ms" to relativeTsMs.toString(),
                "client_ts_ms" to System.currentTimeMillis().toString(),
                "duration_ms" to durationMs.toString(),
                "format" to "wav",
                "sample_rate" to LightMemEgoConfig.AUDIO_SAMPLE_RATE.toString(),
                "channels" to "1",
                "source" to "rokid_sdk_audio",
            ),
        )
        return LightMemEgoUploadResult(
            status = json.optString("status"),
            canAsk = json.optBoolean("can_ask", false),
        )
    }

    fun getStreamStatus(sessionId: String): LightMemEgoStatusResult {
        val json = getJson(statusPaths[sessionId] ?: "/rokid/$sessionId/status")
        val rokid = json.optJSONObject("rokid") ?: JSONObject()
        val memoryReady = getMemoryReady(sessionId)
        return LightMemEgoStatusResult(
            streamStatus = json.optString("stream_status", json.optString("status", "")),
            canAsk = json.optBoolean("can_ask", false),
            memoryReady = memoryReady,
            framesReceived = rokid.optInt("frames_received", 0),
            audioChunksReceived = rokid.optInt("audio_chunks_received", 0),
            latestFrameTsMs = rokid.optLongOrNull("latest_frame_relative_ts_ms"),
            latestAudioTsMs = rokid.optLongOrNull("latest_audio_relative_ts_ms"),
        )
    }

    fun getSessionState(sessionId: String): JSONObject {
        return getJson("/session/$sessionId/status")
    }

    fun endStream(sessionId: String): JSONObject =
        postJson(
            "/stream/$sessionId/end",
            JSONObject()
                .put("close_open_event", true)
                .put("force_accept", true),
        )

    fun askAudioQuestion(
        sessionId: String,
        wavBytes: ByteArray,
        durationMs: Long,
    ): LightMemEgoAudioQuestionSubmitResult {
        val json = postMultipart(
            path = audioQuestionPaths[sessionId] ?: "/rokid/$sessionId/audio_question",
            fileField = "audio",
            filename = "rokid-audio-question.wav",
            contentType = "audio/wav",
            payload = wavBytes,
            fields = mapOf(
                "duration_ms" to durationMs.toString(),
                "format" to "wav",
                "sample_rate" to LightMemEgoConfig.AUDIO_SAMPLE_RATE.toString(),
                "channels" to "1",
                "mode" to "async",
                "retrieval_mode" to "auto",
                "memory_mode" to "auto",
                "top_k" to "5",
                "use_current" to "true",
                "use_image_evidence" to "auto",
                "max_image_evidence" to "6",
                "debug_router" to "true",
                "use_interaction_cache" to "true",
                "client_source" to "glasses",
                "input_method" to "voice",
            ),
        )
        val taskId = json.optString("task_id", json.optString("taskId"))
        val status = json.optString("status", "")
        val answer = extractAnswer(json)
        val question = json.firstNonBlankString("question", "transcript", "text").orEmpty()
        return LightMemEgoAudioQuestionSubmitResult(
            status = status,
            question = question,
            queued = status == "queued" || taskId.isNotBlank(),
            taskId = taskId,
            answer = answer,
            message = json.optString("message"),
        )
    }

    fun askAudioQuestionStream(
        sessionId: String,
        wavBytes: ByteArray,
        durationMs: Long,
        onEvent: (LightMemEgoStreamEvent) -> Unit,
    ): LightMemEgoAudioQuestionStreamResult {
        val result = postMultipartStream(
            path = audioQuestionStreamPath(sessionId),
            fileField = "audio",
            filename = "rokid-audio-question.wav",
            contentType = "audio/wav",
            payload = wavBytes,
            fields = mapOf(
                "duration_ms" to durationMs.toString(),
                "format" to "wav",
                "sample_rate" to LightMemEgoConfig.AUDIO_SAMPLE_RATE.toString(),
                "channels" to "1",
                "retrieval_mode" to "auto",
                "memory_mode" to "auto",
                "top_k" to "5",
                "use_current" to "true",
                "use_image_evidence" to "auto",
                "max_image_evidence" to "6",
                "debug_router" to "true",
                "use_interaction_cache" to "true",
                "client_source" to "glasses",
                "input_method" to "voice",
            ),
            onEvent = onEvent,
        )
        return LightMemEgoAudioQuestionStreamResult(
            status = result.status,
            question = result.question,
            answer = result.answer,
            message = result.message,
        )
    }
    fun getQueryTask(taskId: String): LightMemEgoQueryTaskResult {
        val json = getJson("/query_task/$taskId")
        val status = json.optString("status", "queued").lowercase()
        return LightMemEgoQueryTaskResult(
            status = status,
            done = status == "done",
            answer = extractAnswer(json),
            message = json.optString("message"),
        )
    }

    private fun getMemoryReady(sessionId: String): Boolean {
        return runCatching {
            val json = getJson("/session/$sessionId/pipeline_state")
            json.optJSONObject("short_term")
                ?.optBoolean("ready", false) == true
        }.getOrDefault(false)
    }

    private fun getJson(path: String): JSONObject {
        val conn = openConnection(path, "GET")
        return parseResponse(conn)
    }

    private fun postJson(path: String, body: JSONObject): JSONObject {
        val bytes = body.toString().toByteArray(Charsets.UTF_8)
        val conn = openConnection(path, "POST")
        conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
        conn.doOutput = true
        conn.outputStream.use { it.write(bytes) }
        return parseResponse(conn)
    }

    private fun postMultipart(
        path: String,
        fileField: String,
        filename: String,
        contentType: String,
        payload: ByteArray,
        fields: Map<String, String>,
    ): JSONObject {
        val boundary = "LightMemEgo-${UUID.randomUUID()}"
        val conn = openConnection(path, "POST")
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
        conn.doOutput = true
        BufferedOutputStream(conn.outputStream).use { out ->
            fields.forEach { (name, value) ->
                out.writeString("--$boundary\r\n")
                out.writeString("Content-Disposition: form-data; name=\"$name\"\r\n\r\n")
                out.writeString(value)
                out.writeString("\r\n")
            }
            out.writeString("--$boundary\r\n")
            out.writeString("Content-Disposition: form-data; name=\"$fileField\"; filename=\"$filename\"\r\n")
            out.writeString("Content-Type: $contentType\r\n\r\n")
            out.write(payload)
            out.writeString("\r\n--$boundary--\r\n")
        }
        return parseResponse(conn)
    }

    private fun postMultipartStream(
        path: String,
        fileField: String,
        filename: String,
        contentType: String,
        payload: ByteArray,
        fields: Map<String, String>,
        onEvent: (LightMemEgoStreamEvent) -> Unit,
    ): LightMemEgoStreamResult {
        val boundary = "LightMemEgo-${UUID.randomUUID()}"
        val conn = openConnection(path, "POST")
        conn.setRequestProperty("Accept", "text/event-stream")
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
        conn.doOutput = true
        BufferedOutputStream(conn.outputStream).use { out ->
            fields.forEach { (name, value) ->
                out.writeString("--$boundary\r\n")
                out.writeString("Content-Disposition: form-data; name=\"$name\"\r\n\r\n")
                out.writeString(value)
                out.writeString("\r\n")
            }
            out.writeString("--$boundary\r\n")
            out.writeString("Content-Disposition: form-data; name=\"$fileField\"; filename=\"$filename\"\r\n")
            out.writeString("Content-Type: $contentType\r\n\r\n")
            out.write(payload)
            out.writeString("\r\n--$boundary--\r\n")
        }
        return parseSseResponse(conn, onEvent)
    }

    private fun parseSseResponse(
        conn: HttpURLConnection,
        onEvent: (LightMemEgoStreamEvent) -> Unit,
    ): LightMemEgoStreamResult {
        val code = conn.responseCode
        val stream = if (code in 200..299) conn.inputStream else conn.errorStream
        if (code !in 200..299) {
            val text = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            val message = runCatching { JSONObject(text).optString("message") }.getOrNull()
                ?.takeIf { it.isNotBlank() }
                ?: text.ifBlank { "HTTP $code" }
            throw IllegalStateException(message)
        }

        val answerBuilder = StringBuilder()
        val dataLines = mutableListOf<String>()
        var eventName = ""
        var finalAnswer = ""
        var question = ""
        var status = ""
        var message = ""

        fun dispatchEvent() {
            if (dataLines.isEmpty()) return
            val data = dataLines.joinToString("\n")
            dataLines.clear()
            val json = runCatching { JSONObject(data) }.getOrNull() ?: return
            val event = parseStreamEvent(eventName, json)
            eventName = ""

            when (event.type) {
                "delta" -> if (event.delta.isNotBlank()) answerBuilder.append(event.delta)
                "transcript" -> if (event.question.isNotBlank()) question = event.question
                "final" -> if (event.answer.isNotBlank()) finalAnswer = event.answer
                "done" -> {
                    if (event.status.isNotBlank()) status = event.status
                    if (event.answer.isNotBlank()) finalAnswer = event.answer
                    if (event.message.isNotBlank()) message = event.message
                }
                "error" -> {
                    status = "error"
                    message = event.message.ifBlank { "Stream question failed" }
                }
            }
            if (event.question.isNotBlank()) question = event.question
            onEvent(event)
        }

        stream?.bufferedReader(Charsets.UTF_8)?.useLines { lines ->
            lines.forEach { rawLine ->
                val line = rawLine.trimEnd('\r')
                when {
                    line.isEmpty() -> dispatchEvent()
                    line.startsWith("event:") -> eventName = line.removePrefix("event:").trim()
                    line.startsWith("data:") -> dataLines.add(line.removePrefix("data:").trimStart())
                }
            }
        }
        dispatchEvent()

        val answer = finalAnswer.ifBlank { answerBuilder.toString() }
        if (status == "error" && answer.isBlank()) {
            throw IllegalStateException(message.ifBlank { "Stream question failed" })
        }
        return LightMemEgoStreamResult(
            status = status.ifBlank { "ok" },
            answer = answer,
            question = question,
            message = message,
        )
    }

    private fun parseStreamEvent(eventName: String, json: JSONObject): LightMemEgoStreamEvent {
        val result = json.optJSONObject("result")
        val resultAnswer = result?.let { extractAnswer(it) }.orEmpty()
        val answer = json.firstNonBlankString("answer", "final_answer", "finalAnswer", "response", "text")
            ?: resultAnswer.takeIf { it.isNotBlank() }
            ?: ""
        val question = json.firstNonBlankString("question", "transcript")
            ?: result?.firstNonBlankString("question", "transcript")
            ?: ""
        val text = json.firstNonBlankString("text", "response")
            ?: result?.firstNonBlankString("text", "response")
            ?: ""
        val message = json.firstNonBlankString("message", "error")
            ?: result?.firstNonBlankString("message", "error")
            ?: ""
        return LightMemEgoStreamEvent(
            type = json.optString("type").ifBlank { eventName.ifBlank { "message" } },
            status = json.firstNonBlankString("status") ?: result?.firstNonBlankString("status") ?: "",
            delta = json.optString("delta"),
            text = text,
            answer = answer,
            question = question,
            message = message,
        )
    }

    private fun audioQuestionStreamPath(sessionId: String): String {
        val base = audioQuestionPaths[sessionId]
            ?.trim()
            ?.takeIf { it.isNotBlank() }
            ?: "/rokid/$sessionId/audio_question"
        return if (base.endsWith("/stream")) base else base.trimEnd('/') + "/stream"
    }

    private fun openConnection(path: String, method: String): HttpURLConnection {
        val joined = if (path.startsWith("http://") || path.startsWith("https://")) {
            path
        } else {
            baseUrl.trimEnd('/') + path
        }
        return (URL(joined).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 12_000
            readTimeout = LightMemEgoConfig.HTTP_READ_TIMEOUT_MS
            useCaches = false
            setRequestProperty("Accept", "application/json")
        }
    }

    private fun parseResponse(conn: HttpURLConnection): JSONObject {
        val code = conn.responseCode
        val stream = if (code in 200..299) conn.inputStream else conn.errorStream
        val text = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
        if (code !in 200..299) {
            val payload = runCatching { if (text.isBlank()) JSONObject() else JSONObject(text) }.getOrDefault(JSONObject())
            val bodyMessage = payload.optString("message")
                .takeIf { it.isNotBlank() }
                ?: text.ifBlank { "HTTP $code" }
            val path = conn.url?.path.orEmpty()
            val message = "HTTP $code${path.ifBlank { "" }.let { if (it.isBlank()) "" else " $it" }}: $bodyMessage"
            throw LightMemEgoApiException(message, code, payload)
        }
        return if (text.isBlank()) JSONObject() else JSONObject(text)
    }

    private fun BufferedOutputStream.writeString(value: String) {
        write(value.toByteArray(Charsets.UTF_8))
    }

    private fun JSONObject.optLongOrNull(name: String): Long? =
        if (has(name) && !isNull(name)) optLong(name) else null

    private fun extractAnswer(json: JSONObject): String {
        val result = json.optJSONObject("result")
        val nestedResult = result?.optJSONObject("result")
        return listOfNotNull(nestedResult, result, json)
            .firstNotNullOfOrNull { item ->
                item.firstNonBlankString("answer", "final_answer", "finalAnswer", "response", "text")
            }
            .orEmpty()
    }

    private fun JSONObject.firstNonBlankString(vararg names: String): String? =
        names.asSequence()
            .map { optString(it) }
            .firstOrNull { it.isMeaningfulString() }

    private fun JSONObject.firstIntOrNull(vararg names: String): Int? =
        names.asSequence()
            .firstNotNullOfOrNull { name ->
                if (has(name) && !isNull(name)) optInt(name) else null
            }

    private fun JSONObject.firstDateLabel(): String? =
        firstNonBlankString(
            "actual_date",
            "actualDate",
            "date",
            "date_label",
            "dateLabel",
            "client_start_date",
            "clientStartDate",
            "client_start_datetime",
            "clientStartDatetime",
            "day_label",
            "dayLabel",
            "label",
            "display_label",
            "displayLabel",
        )?.toDateLabel()

    private fun String.toDateLabel(): String? {
        val value = trim()
        if (!value.isMeaningfulString() || value.startsWith("DAY", ignoreCase = true)) return null
        parseFlexibleDate(value)?.let { return dateLabelFormatter.format(it) }
        parseFlexibleDateTime(value)?.let { return dateLabelFormatter.format(it.toLocalDate()) }
        return value
    }

    private fun String.isMeaningfulString(): Boolean {
        val value = trim()
        return value.isNotBlank() &&
            !value.equals("null", ignoreCase = true) &&
            !value.equals("none", ignoreCase = true)
    }

    private fun currentDateLabel(): String =
        dateLabelFormatter.format(LocalDate.now())

    private fun parseFlexibleDate(value: String): LocalDate? {
        val normalized = value.trim()
        val directPatterns = listOf("yyyy.M.d", "yyyy-M-d", "yyyy/M/d", "yyyyMMdd")
        directPatterns.forEach { pattern ->
            runCatching { LocalDate.parse(normalized, DateTimeFormatter.ofPattern(pattern)) }
                .getOrNull()
                ?.let { return it }
        }
        val datePart = normalized
            .substringBefore('T')
            .substringBefore(' ')
        if (datePart != normalized) return parseFlexibleDate(datePart)
        return null
    }

    private fun parseFlexibleDateTime(value: String): LocalDateTime? {
        val normalized = value.trim().replace('T', ' ')
        val patterns = listOf(
            "yyyy-M-d H:m:s",
            "yyyy-M-d H:m",
            "yyyy/M/d H:m:s",
            "yyyy/M/d H:m",
            "yyyy.M.d H:m:s",
            "yyyy.M.d H:m",
        )
        patterns.forEach { pattern ->
            runCatching { LocalDateTime.parse(normalized, DateTimeFormatter.ofPattern(pattern)) }
                .getOrNull()
                ?.let { return it }
        }
        return null
    }
}
