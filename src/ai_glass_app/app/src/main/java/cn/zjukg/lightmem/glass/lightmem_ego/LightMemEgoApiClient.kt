package cn.zjukg.lightmem.glass.lightmem_ego

import cn.zjukg.lightmem.glass.BuildConfig
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

data class LightMemEgoStartResult(
    val sessionId: String,
    val dayLabel: String,
    val dayIndex: Int,
    val displayDayLabel: String,
    val runId: String,
    val streamId: String,
    val canAsk: Boolean,
    val inputMode: String,
    val frameUploadPath: String,
    val audioUploadPath: String,
    val statusPath: String,
    val audioQuestionPath: String,
    val nextFrameIndex: Int,
    val nextAudioIndex: Int,
    val relativeTsBaseMs: Long,
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

data class LightMemEgoAskSubmitResult(
    val status: String,
    val queued: Boolean,
    val taskId: String,
    val answer: String,
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

    fun startRokidStream(
        inputMode: String = LightMemEgoConfig.INPUT_MODE,
        sessionId: String? = null,
        runId: String,
    ): LightMemEgoStartResult {
        val requestedSessionId = sessionId?.trim().orEmpty()
        val metadata = JSONObject()
            .put("source", "rokid_glass")
            .put("device_type", "rokid")
            .put("transport", "http_frame_audio")
            .put("sdk", "android_camera_x_audio_record")
            .put("timestamp_mode", "connector_relative_ts_ms")
            .put("run_id", runId)
            .put("rokid_session_mode", "single_session")
        val body = JSONObject()
            .put("input_mode", inputMode)
            .put("chunk_duration", 1)
            .put("run_id", runId)
            .put("metadata", metadata)
        if (requestedSessionId.isNotBlank()) {
            body.put("session_id", requestedSessionId)
        }

        val json = postJson("/rokid/stream/start", body)
        val result = parseStartResult(
            json = json,
            requestedSessionId = requestedSessionId,
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
        requestedSessionId: String,
        inputMode: String,
        runId: String,
    ): LightMemEgoStartResult {
        val sessionId = json.optString("session_id")
        val dayContext = json.optJSONObject("day_context")
            ?: json.optJSONObject("dayContext")
            ?: throw IllegalStateException("Rokid stream start response missing day_context")
        val framePath = json.optString("frame_upload_url").ifBlank { "/rokid/$sessionId/frame" }
        val audioPath = json.optString("audio_upload_url").ifBlank { "/rokid/$sessionId/audio_chunk" }
        val statusPath = json.optString("status_url").ifBlank { "/rokid/$sessionId/status" }
        val audioQuestionPath = json.optString("audio_question_url").ifBlank { "/rokid/$sessionId/audio_question" }
        val dayLabel = dayContext.firstNonBlankString("day_label", "dayLabel")
            ?: throw IllegalStateException("Rokid stream start day_context missing day_label")
        val dayIndex = dayContext.firstIntOrNull("day_index", "dayIndex", "index")
            ?: throw IllegalStateException("Rokid stream start day_context missing day_index")
        val weekdayLabel = dayContext.firstNonBlankString("weekday_label", "weekdayLabel")
            ?: throw IllegalStateException("Rokid stream start day_context missing weekday_label")
        val displayDayLabel = dayContext.firstNonBlankString("display_day_label", "displayDayLabel", "date", "date_label", "dateLabel")
            ?: throw IllegalStateException("Rokid stream start day_context missing display_day_label")
        val nextFrameIndex = json.firstIntOrNull("next_frame_index", "nextFrameIndex")
            ?: dayContext.firstIntOrNull("next_frame_index", "nextFrameIndex")
            ?: 0
        val nextAudioIndex = json.firstIntOrNull("next_audio_index", "nextAudioIndex")
            ?: dayContext.firstIntOrNull("next_audio_index", "nextAudioIndex")
            ?: 0
        val relativeTsBaseMs = json.firstLongOrNull("relative_ts_base_ms", "relativeTsBaseMs")
            ?: dayContext.firstLongOrNull("relative_ts_base_ms", "relativeTsBaseMs")
            ?: 0L
        return LightMemEgoStartResult(
            sessionId = sessionId,
            dayLabel = dayLabel,
            dayIndex = dayIndex,
            displayDayLabel = displayDayLabel,
            runId = dayContext.optString("run_id", runId).ifBlank { runId },
            streamId = json.optString("stream_id"),
            canAsk = json.optBoolean("can_ask", false),
            inputMode = json.optString("input_mode", inputMode),
            frameUploadPath = framePath,
            audioUploadPath = audioPath,
            statusPath = statusPath,
            audioQuestionPath = audioQuestionPath,
            nextFrameIndex = nextFrameIndex,
            nextAudioIndex = nextAudioIndex,
            relativeTsBaseMs = relativeTsBaseMs,
        )
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

    fun askQuestion(sessionId: String, question: String): LightMemEgoAskSubmitResult {
        val json = postJson(
            "/ask/$sessionId",
            JSONObject()
                .put("question", question)
                .put("mode", "async")
                .put("retrieval_mode", "auto")
                .put("memory_mode", "auto")
                .put("top_k", 5)
                .put("use_current", true)
                .put("use_image_evidence", "auto")
                .put("max_image_evidence", 6)
                .put("debug_router", BuildConfig.LIGHTMEM_DEBUG_ROUTER)
                .put("use_interaction_cache", true)
                .put("client_source", "glasses")
                .put("input_method", "preset"),
        )
        return parseAskSubmitResult(json)
    }

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
                "debug_router" to BuildConfig.LIGHTMEM_DEBUG_ROUTER.toString(),
                "use_interaction_cache" to "true",
                "client_source" to "glasses",
                "input_method" to "voice",
            ),
        )
        return parseAudioQuestionSubmitResult(json)
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
                "debug_router" to BuildConfig.LIGHTMEM_DEBUG_ROUTER.toString(),
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
        return parseQueryTaskResult(json)
    }

    internal fun parseAskSubmitResult(json: JSONObject): LightMemEgoAskSubmitResult {
        val taskId = json.optString("task_id", json.optString("taskId"))
        val status = json.optString("status", "")
        return LightMemEgoAskSubmitResult(
            status = status,
            queued = status == "queued" || taskId.isNotBlank(),
            taskId = taskId,
            answer = extractAnswer(json),
        )
    }

    internal fun parseAudioQuestionSubmitResult(json: JSONObject): LightMemEgoAudioQuestionSubmitResult {
        val taskId = json.optString("task_id", json.optString("taskId"))
        val status = json.optString("status", "")
        val question = json.firstNonBlankString("question", "transcript", "text").orEmpty()
        return LightMemEgoAudioQuestionSubmitResult(
            status = status,
            question = question,
            queued = status == "queued" || taskId.isNotBlank(),
            taskId = taskId,
            answer = extractAnswer(json),
            message = json.optString("message"),
        )
    }

    internal fun parseQueryTaskResult(json: JSONObject): LightMemEgoQueryTaskResult {
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
        return json.firstNonBlankString("answer")
            ?: result?.firstNonBlankString("answer")
            ?: ""
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

    private fun JSONObject.firstLongOrNull(vararg names: String): Long? =
        names.asSequence()
            .firstNotNullOfOrNull { name ->
                if (has(name) && !isNull(name)) optLong(name) else null
            }

    private fun String.isMeaningfulString(): Boolean {
        val value = trim()
        return value.isNotBlank() &&
            !value.equals("null", ignoreCase = true) &&
            !value.equals("none", ignoreCase = true)
    }

}
