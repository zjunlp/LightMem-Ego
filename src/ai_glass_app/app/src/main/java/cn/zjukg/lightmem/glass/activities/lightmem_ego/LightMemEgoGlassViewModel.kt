package cn.zjukg.lightmem.glass.activities.lightmem_ego

import android.Manifest
import android.annotation.SuppressLint
import android.app.Application
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import cn.zjukg.lightmem.glass.app.CONSTANT
import cn.zjukg.lightmem.glass.utils.BarePermissions
import cn.zjukg.lightmem.glass.lightmem_ego.WavEncoder
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoApiClient
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoConfig
import cn.zjukg.lightmem.glass.lightmem_ego.StoredRokidSession
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoStartResult
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoStreamEvent
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoUploadResult
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoRokidSessionStore
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoApiException
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoDiagnostics
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.util.UUID

private val TERMINAL_STREAM_STATUSES = setOf("ended", "stopped", "aborted", "cancelled", "canceled", "done")

data class LightMemEgoGlassUiState(
    val status: String = "disconnected",
    val sessionId: String = "",
    val parentSessionId: String = "",
    val childSessionId: String = "",
    val dayLabel: String = "",
    val dayIndex: Int = 0,
    val runId: String = "",
    val streamStatus: String = "idle",
    val cameraGranted: Boolean = false,
    val audioGranted: Boolean = false,
    val cameraReady: Boolean = false,
    val running: Boolean = false,
    val canAsk: Boolean = false,
    val memoryReady: Boolean = false,
    val frameIndex: Int = 0,
    val audioIndex: Int = 0,
    val frameUploadedCount: Int = 0,
    val frameFailedCount: Int = 0,
    val frameDroppedCount: Int = 0,
    val audioUploadedCount: Int = 0,
    val audioFailedCount: Int = 0,
    val framesReceived: Int = 0,
    val audioChunksReceived: Int = 0,
    val lastError: String = "",
    val asking: Boolean = false,
    val queryStatus: String = "idle",
    val queryTaskId: String = "",
    val lastQuestion: String = "",
    val answer: String = "",
    val answerLatencyMs: Long? = null,
    val voiceQuestionRecording: Boolean = false,
    val voiceQuestionStatus: String = "idle",
    val voiceQuestionText: String = "",
    val voiceQuestionMessage: String = "",
    val voiceQuestionDurationMs: Long = 0L,
    val liveRtmpMode: Boolean = false,
    val livePushUrl: String = "",
    val liveIngestStartPath: String = "",
    val liveIngestStopPath: String = "",
    val liveRtmpStatus: String = "idle",
    val rtmpRestartToken: Int = 0,
)

class LightMemEgoGlassViewModel(application: Application) : AndroidViewModel(application) {
    private val api = LightMemEgoApiClient()
    private val sessionStore = LightMemEgoRokidSessionStore(application)

    private val _uiState = MutableStateFlow(
        LightMemEgoGlassUiState(
            cameraGranted = BarePermissions.hasCamera(application),
            audioGranted = BarePermissions.hasRecordAudio(application),
        ),
    )
    val uiState: StateFlow<LightMemEgoGlassUiState> = _uiState.asStateFlow()

    @Volatile
    private var streaming = false

    @Volatile
    private var frameUploadInFlight = false

    @Volatile
    private var stopping = false

    private var streamStartElapsedMs = 0L
    private var lastFrameCaptureElapsedMs = 0L
    private var audioJob: Job? = null
    private var statusJob: Job? = null
    private var queryJob: Job? = null
    private var rtmpRetryJob: Job? = null
    private var rtmpRetryAttempt = 0
    private var recorder: AudioRecord? = null
    private val voiceQuestionLock = Any()
    private var voiceQuestionPcm = ByteArrayOutputStream()
    private var voiceQuestionStartElapsedMs = 0L
    private var lastVoiceQuestionDurationUiElapsedMs = 0L
    private var attemptedStoredSessionResume = false

    @Volatile
    private var voiceQuestionCapturing = false

    fun refreshPermissions() {
        val context = getApplication<Application>()
        _uiState.update {
            it.copy(
                cameraGranted = BarePermissions.hasCamera(context),
                audioGranted = BarePermissions.hasRecordAudio(context),
            )
        }
    }

    fun onPermissionsResult(grants: Map<String, Boolean>) {
        _uiState.update {
            it.copy(
                cameraGranted = grants[Manifest.permission.CAMERA] ?: it.cameraGranted,
                audioGranted = grants[Manifest.permission.RECORD_AUDIO] ?: it.audioGranted,
                status = if ((grants[Manifest.permission.CAMERA] == false) || (grants[Manifest.permission.RECORD_AUDIO] == false)) {
                    "Camera and microphone permissions are required"
                } else {
                    it.status
                },
            )
        }
    }

    fun setCameraReady(ready: Boolean) {
        _uiState.update { it.copy(cameraReady = ready) }
    }

    fun toggleVoiceQuestionRecording() {
        if (voiceQuestionCapturing) {
            stopVoiceQuestionAndAsk()
        } else {
            startVoiceQuestionRecording()
        }
    }

    private fun startVoiceQuestionRecording() {
        val state = _uiState.value
        if (!streaming || state.sessionId.isBlank()) {
            _uiState.update { it.copy(lastError = "Hold Start first") }
            return
        }
        if (state.asking) {
            _uiState.update { it.copy(lastError = "Already answering. Please wait.") }
            return
        }
        synchronized(voiceQuestionLock) {
            voiceQuestionPcm = ByteArrayOutputStream()
            voiceQuestionStartElapsedMs = SystemClock.elapsedRealtime()
            lastVoiceQuestionDurationUiElapsedMs = voiceQuestionStartElapsedMs
            voiceQuestionCapturing = true
        }
        _uiState.update {
            it.copy(
                voiceQuestionRecording = true,
                voiceQuestionStatus = "recording",
                voiceQuestionText = "",
                voiceQuestionMessage = "",
                voiceQuestionDurationMs = 0L,
                answer = "",
                answerLatencyMs = null,
                lastError = "",
            )
        }
    }

    private fun stopVoiceQuestionAndAsk() {
        val sessionId = _uiState.value.sessionId
        val stoppedAt = SystemClock.elapsedRealtime()
        val durationMs = (stoppedAt - voiceQuestionStartElapsedMs).coerceAtLeast(0L)
        val pcmBytes = synchronized(voiceQuestionLock) {
            voiceQuestionCapturing = false
            val bytes = voiceQuestionPcm.toByteArray()
            voiceQuestionPcm = ByteArrayOutputStream()
            bytes
        }
        _uiState.update {
            it.copy(
                voiceQuestionRecording = false,
                voiceQuestionDurationMs = durationMs,
            )
        }

        if (sessionId.isBlank()) {
            _uiState.update { it.copy(voiceQuestionStatus = "failed", voiceQuestionMessage = "Capture session ended", lastError = "Capture session ended") }
            return
        }
        if (pcmBytes.size < LightMemEgoConfig.AUDIO_SAMPLE_RATE) {
            _uiState.update { it.copy(voiceQuestionStatus = "failed", voiceQuestionMessage = "Voice question was too short. Please try again.", lastError = "Voice question was too short. Please try again.") }
            return
        }
        if (!_uiState.value.canAsk) {
            _uiState.update { it.copy(voiceQuestionStatus = "waiting", voiceQuestionMessage = "Question service is not ready. Please wait.", lastError = "Question service is not ready. Please wait.") }
            refreshStatusOnce()
            return
        }

        val wavBytes = WavEncoder.mono16PcmToWav(pcmBytes)
        var recognizedVoiceQuestion = ""
        val answerStartedElapsedMs = SystemClock.elapsedRealtime()
        queryJob?.cancel()
        _uiState.update {
            it.copy(
                asking = true,
                queryStatus = "transcribing",
                queryTaskId = "",
                lastQuestion = "",
                answer = "",
                answerLatencyMs = null,
                voiceQuestionStatus = "transcribing",
                voiceQuestionText = "",
                voiceQuestionMessage = "transcribing",
                lastError = "",
            )
        }
        queryJob = viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val result = api.askAudioQuestion(sessionId, wavBytes, durationMs)
                recognizedVoiceQuestion = result.question.trim().ifBlank { recognizedVoiceQuestion }
                if (recognizedVoiceQuestion.isBlank()) {
                    throw IllegalStateException(result.message.ifBlank { "Backend did not recognize a voice question" })
                }
                _uiState.update {
                    it.copy(
                        lastQuestion = recognizedVoiceQuestion,
                        voiceQuestionText = recognizedVoiceQuestion,
                        voiceQuestionStatus = "asking",
                        voiceQuestionMessage = "Recognized: $recognizedVoiceQuestion",
                        queryStatus = result.status.ifBlank { "queued" },
                        queryTaskId = result.taskId,
                        lastError = "",
                    )
                }
                when {
                    result.answer.isNotBlank() -> result.answer
                    result.queued && result.taskId.isNotBlank() -> pollQueryAnswer(result.taskId)
                    else -> "Backend returned no answer"
                }
            }.onSuccess { answer ->
                val latencyMs = (SystemClock.elapsedRealtime() - answerStartedElapsedMs).coerceAtLeast(0L)
                _uiState.update {
                    it.copy(
                        asking = false,
                        queryStatus = "done",
                        queryTaskId = "",
                        voiceQuestionStatus = "done",
                        voiceQuestionMessage = "",
                        answer = answer,
                        answerLatencyMs = latencyMs,
                        lastQuestion = recognizedVoiceQuestion,
                        lastError = "",
                    )
                }
            }.onFailure { error ->
                if (error is CancellationException) return@launch
                _uiState.update {
                    it.copy(
                        asking = false,
                        queryStatus = "failed",
                        queryTaskId = "",
                        voiceQuestionStatus = "failed",
                        voiceQuestionMessage = error.message ?: "Voice question failed",
                        answer = "",
                        answerLatencyMs = null,
                        lastError = error.message ?: "Voice question failed",
                    )
                }
            }
        }
    }

    private fun applyQuestionStreamEvent(event: LightMemEgoStreamEvent, isVoice: Boolean) {
        when (event.type) {
            "transcribing" -> if (isVoice) {
                _uiState.update {
                    it.copy(
                        queryStatus = "transcribing",
                        voiceQuestionStatus = "transcribing",
                        voiceQuestionMessage = "transcribing",
                        lastError = "",
                    )
                }
            }
            "transcript" -> {
                val question = event.question.trim()
                if (question.isNotBlank()) {
                    _uiState.update {
                        it.copy(
                            lastQuestion = question,
                            voiceQuestionText = if (isVoice) question else it.voiceQuestionText,
                            voiceQuestionMessage = if (isVoice) "Recognized: $question" else it.voiceQuestionMessage,
                            voiceQuestionStatus = if (isVoice) "asking" else it.voiceQuestionStatus,
                            queryStatus = "streaming",
                            lastError = "",
                        )
                    }
                }
            }
            "delta" -> if (event.delta.isNotBlank()) {
                _uiState.update {
                    it.copy(
                        answer = it.answer + event.delta,
                        queryStatus = "streaming",
                        voiceQuestionStatus = if (isVoice) "asking" else it.voiceQuestionStatus,
                        lastError = "",
                    )
                }
            }
            "final" -> {
                val finalAnswer = event.answer.ifBlank { event.text }
                if (finalAnswer.isNotBlank()) {
                    _uiState.update {
                        it.copy(
                            answer = finalAnswer,
                            queryStatus = "streaming",
                            voiceQuestionStatus = if (isVoice) "asking" else it.voiceQuestionStatus,
                            lastError = "",
                        )
                    }
                }
            }
            "done" -> {
                val finalAnswer = event.answer.ifBlank { event.text }
                _uiState.update {
                    it.copy(
                        answer = finalAnswer.ifBlank { it.answer },
                        queryStatus = if (event.status == "error") "failed" else "done",
                        voiceQuestionStatus = if (isVoice) {
                            if (event.status == "error") "failed" else "done"
                        } else {
                            it.voiceQuestionStatus
                        },
                        lastError = if (event.status == "error") event.message else "",
                    )
                }
            }
            "error" -> {
                val message = event.message.ifBlank { "Question request failed" }
                _uiState.update {
                    it.copy(
                        queryStatus = "failed",
                        voiceQuestionStatus = if (isVoice) "failed" else it.voiceQuestionStatus,
                        voiceQuestionMessage = if (isVoice) message else it.voiceQuestionMessage,
                        lastError = message,
                    )
                }
            }
        }
    }

    fun startStreaming() {
        startLiveIngestSession()
    }

    private fun startLiveIngestSession() {
        val state = _uiState.value
        LightMemEgoDiagnostics.log(getApplication(), "start-live-ingest", "streaming=$streaming stopping=$stopping")
        if (!state.cameraGranted || !state.audioGranted) {
            _uiState.update { it.copy(status = "Please grant camera and microphone permissions", lastError = "") }
            return
        }
        if (streaming || stopping) return
        queryJob?.cancel()
        _uiState.update { it.copy(status = "connecting", asking = false, queryStatus = "idle", queryTaskId = "", answer = "", lastError = "") }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                startPreferredRokidSession()
            }.onSuccess { result ->
                LightMemEgoDiagnostics.log(
                    getApplication(),
                    "start-live-ingest-success",
                    "session=${result.sessionId} parent=${result.parentSessionId} mode=${result.inputMode} hasPush=${result.pushUrl.isNotBlank()}",
                )
                if (result.sessionId.isBlank()) {
                    failStart("Backend did not return session_id")
                    return@onSuccess
                }
                rememberActiveSession(result)
                streaming = true
                stopping = false
                streamStartElapsedMs = SystemClock.elapsedRealtime()
                lastFrameCaptureElapsedMs = 0L
                _uiState.update {
                    it.copy(
                        status = "capturing",
                        sessionId = result.sessionId,
                        parentSessionId = result.parentSessionId,
                        childSessionId = result.childSessionId,
                        dayLabel = result.dayLabel,
                        dayIndex = result.dayIndex,
                        runId = result.runId,
                        streamStatus = "streaming",
                        running = true,
                        canAsk = false,
                        memoryReady = false,
                        liveRtmpMode = result.inputMode == "rokid_live_rtmp" && result.pushUrl.isNotBlank(),
                        livePushUrl = result.pushUrl,
                        liveIngestStartPath = result.liveIngestStartPath,
                        liveIngestStopPath = result.liveIngestStopPath,
                        liveRtmpStatus = if (result.inputMode == "rokid_live_rtmp") "waiting" else "disabled",
                        frameIndex = 0,
                        audioIndex = 0,
                        frameUploadedCount = 0,
                        frameFailedCount = 0,
                        frameDroppedCount = 0,
                        audioUploadedCount = 0,
                        audioFailedCount = 0,
                        framesReceived = 0,
                        audioChunksReceived = 0,
                        asking = false,
                        queryStatus = "idle",
                        queryTaskId = "",
                        lastQuestion = "",
                        answer = "",
                        lastError = "",
                    )
                }
                startMicrophoneAudioLoop(result.sessionId, uploadToBackend = true)
                if (result.inputMode == "rokid_live_rtmp") startLiveIngest(result.sessionId, result.liveIngestStartPath)
                startStatusPolling(result.sessionId)
            }.onFailure { error ->
                LightMemEgoDiagnostics.logError(getApplication(), "start-live-ingest-failed", "", error)
                failStart(error.message ?: "Connection failed")
            }
        }
    }

    fun resumeStoredSessionIfAvailable(): Boolean {
        if (attemptedStoredSessionResume || streaming || stopping) return false
        attemptedStoredSessionResume = true
        if (LightMemEgoConfig.CREATE_NEW_PARENT_SESSION) {
            sessionStore.clear()
            LightMemEgoDiagnostics.log(getApplication(), "stored-session-clear", "CREATE_NEW_PARENT_SESSION=true")
            return false
        }
        val stored = sessionStore.load()
        if (stored == null) {
            LightMemEgoDiagnostics.log(getApplication(), "stored-session-missing")
            return false
        }
        LightMemEgoDiagnostics.log(
            getApplication(),
            "stored-session-resume",
            "session=${stored.sessionId} parent=${stored.parentSessionId} hasPush=${stored.pushUrl.isNotBlank()}",
        )
        resumeRokidSession(stored)
        return true
    }

    private fun resumeRokidSession(stored: StoredRokidSession) {
        val cleanSessionId = stored.sessionId.trim()
        val cleanPushUrl = stored.pushUrl.trim()
        if (cleanSessionId.isBlank() || cleanPushUrl.isBlank()) return
        val state = _uiState.value
        if (!state.cameraGranted || !state.audioGranted) {
            _uiState.update { it.copy(status = "Please grant camera and microphone permissions", lastError = "") }
            return
        }
        if (streaming && state.sessionId == cleanSessionId) {
            _uiState.update { it.copy(livePushUrl = cleanPushUrl, liveRtmpMode = true, lastError = "") }
            return
        }
        queryJob?.cancel()
        audioJob?.cancel()
        statusJob?.cancel()
        _uiState.update { it.copy(status = "resuming", asking = false, queryStatus = "idle", queryTaskId = "", answer = "", lastError = "") }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val status = api.getStreamStatus(cleanSessionId)
                if (status.streamStatus in TERMINAL_STREAM_STATUSES) {
                    sessionStore.clear()
                    throw IllegalStateException("Stored session already ${status.streamStatus}")
                }
                val latestAudioTs = status.latestAudioTsMs ?: status.latestFrameTsMs ?: 0L
                val nextAudioIndex = status.audioChunksReceived.coerceAtLeast(0)
                streaming = true
                stopping = false
                streamStartElapsedMs = SystemClock.elapsedRealtime() - latestAudioTs.coerceAtLeast(0L)
                lastFrameCaptureElapsedMs = 0L
                _uiState.update {
                    it.copy(
                        status = "capturing",
                        sessionId = cleanSessionId,
                        parentSessionId = stored.parentSessionId,
                        childSessionId = cleanSessionId,
                        dayLabel = stored.dayLabel,
                        dayIndex = stored.dayIndex,
                        runId = "",
                        streamStatus = status.streamStatus.ifBlank { "streaming" },
                        running = true,
                        canAsk = status.canAsk,
                        memoryReady = status.memoryReady,
                        liveRtmpMode = true,
                        livePushUrl = cleanPushUrl,
                        liveIngestStartPath = stored.liveIngestStartPath.ifBlank { "/rokid/$cleanSessionId/live/ingest/start" },
                        liveIngestStopPath = stored.liveIngestStopPath.ifBlank { "/rokid/$cleanSessionId/live/ingest/stop" },
                        liveRtmpStatus = "resuming",
                        rtmpRestartToken = it.rtmpRestartToken + 1,
                        frameIndex = status.framesReceived.coerceAtLeast(0),
                        audioIndex = nextAudioIndex,
                        frameUploadedCount = 0,
                        frameFailedCount = 0,
                        frameDroppedCount = 0,
                        audioUploadedCount = 0,
                        audioFailedCount = 0,
                        framesReceived = status.framesReceived,
                        audioChunksReceived = status.audioChunksReceived,
                        asking = false,
                        queryStatus = "idle",
                        queryTaskId = "",
                        lastQuestion = "",
                        answer = "",
                        lastError = "",
                    )
                }
                releaseRecorder()
                startMicrophoneAudioLoop(cleanSessionId, uploadToBackend = true)
                startLiveIngest(cleanSessionId, stored.liveIngestStartPath.ifBlank { "/rokid/$cleanSessionId/live/ingest/start" })
                startStatusPolling(cleanSessionId)
                LightMemEgoDiagnostics.log(
                    getApplication(),
                    "stored-session-resumed",
                    "session=$cleanSessionId status=${status.streamStatus} frames=${status.framesReceived} audio=${status.audioChunksReceived}",
                )
            }.onFailure { error ->
                LightMemEgoDiagnostics.logError(getApplication(), "stored-session-resume-failed", "session=$cleanSessionId", error)
                _uiState.update { it.copy(status = "resume failed", lastError = error.message ?: "Stored session resume failed") }
            }
        }
    }

    fun stopStreaming() {
        if (stopping) return
        val state = _uiState.value
        val sessionId = state.sessionId
        val liveStopPath = state.liveIngestStopPath
        LightMemEgoDiagnostics.log(getApplication(), "stop-streaming", "session=$sessionId liveStopPath=$liveStopPath")
        sessionStore.clear()
        stopping = true
        streaming = false
        frameUploadInFlight = false
        _uiState.update { it.copy(status = "stopping", running = false, asking = false) }
        viewModelScope.launch(Dispatchers.IO) {
            audioJob?.cancel()
            audioJob = null
            statusJob?.cancel()
            statusJob = null
            queryJob?.cancel()
            queryJob = null
            cancelRtmpRetry(resetAttempt = true)
            cancelVoiceQuestionCapture()
            releaseRecorder()
            if (sessionId.isNotBlank()) {
                runCatching { if (liveStopPath.isNotBlank()) api.stopLiveIngest(sessionId, liveStopPath) }
                runCatching { api.endStream(sessionId) }
            }
            stopping = false
            _uiState.update {
                it.copy(
                    status = "stopped",
                    sessionId = "",
                    childSessionId = "",
                    dayLabel = "",
                    dayIndex = 0,
                    runId = "",
                    streamStatus = "stopped",
                    canAsk = false,
                    memoryReady = false,
                    liveRtmpMode = false,
                    livePushUrl = "",
                    liveIngestStartPath = "",
                    liveIngestStopPath = "",
                    liveRtmpStatus = "stopped",
                )
            }
        }
    }

    @Synchronized
    fun reserveFrameCapture(): Boolean {
        if (!streaming || frameUploadInFlight) return false
        val now = SystemClock.elapsedRealtime()
        if (now - lastFrameCaptureElapsedMs < LightMemEgoConfig.FRAME_INTERVAL_MS) return false
        lastFrameCaptureElapsedMs = now
        return true
    }

    fun onFrameConverted(jpegBytes: ByteArray, width: Int, height: Int) {
        val sessionId = _uiState.value.sessionId
        if (!streaming || sessionId.isBlank()) return
        frameUploadInFlight = true
        val frameIndex = _uiState.value.frameIndex
        val relativeTsMs = relativeElapsedMs()
        _uiState.update { it.copy(frameIndex = frameIndex + 1) }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                api.uploadFrame(
                    sessionId = sessionId,
                    frameBytes = jpegBytes,
                    frameIndex = frameIndex,
                    relativeTsMs = relativeTsMs,
                    width = width,
                    height = height,
                )
            }.onSuccess {
                _uiState.update {
                    it.copy(
                        frameUploadedCount = it.frameUploadedCount + 1,
                        lastError = "",
                    )
                }
            }.onFailure { error ->
                _uiState.update {
                    it.copy(
                        frameFailedCount = it.frameFailedCount + 1,
                        lastError = error.message ?: "Frame upload failed",
                    )
                }
            }
            frameUploadInFlight = false
        }
    }

    fun onFrameConvertFailed(message: String) {
        _uiState.update { it.copy(frameFailedCount = it.frameFailedCount + 1, lastError = message) }
    }

    fun onFrameDropped() {
        _uiState.update { it.copy(frameDroppedCount = it.frameDroppedCount + 1) }
    }

    fun refreshStatusOnce() {
        val sessionId = _uiState.value.sessionId
        if (sessionId.isBlank()) {
            _uiState.update { it.copy(status = "No session yet") }
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            pollStatus(sessionId)
        }
    }

    @SuppressLint("MissingPermission")
    private fun startMicrophoneAudioLoop(sessionId: String, uploadToBackend: Boolean) {
        audioJob?.cancel()
        releaseRecorder()
        audioJob = viewModelScope.launch(Dispatchers.IO) {
            val audioFormat = AudioFormat.Builder()
                .setSampleRate(LightMemEgoConfig.AUDIO_SAMPLE_RATE)
                .setChannelMask(CONSTANT.AUDIO_CHANNEL_MASK)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .build()
            val bytesPerSecond = LightMemEgoConfig.AUDIO_SAMPLE_RATE * LightMemEgoConfig.ROKID_CHANNEL_COUNT * 2
            val rec = AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.MIC)
                .setAudioFormat(audioFormat)
                .setBufferSizeInBytes(bytesPerSecond * 2)
                .build()
            recorder = rec
            val readBuffer = ByteArray(4096 * LightMemEgoConfig.ROKID_CHANNEL_COUNT)
            var chunkStartElapsedMs = SystemClock.elapsedRealtime()
            var pcm = ByteArrayOutputStream()
            rec.startRecording()
            while (isActive && streaming) {
                val read = rec.read(readBuffer, 0, readBuffer.size)
                if (read > 0) {
                    val rawBytes = readBuffer.copyOf(read)
                    val monoBytes = WavEncoder.downmixRokidEightChannelToMono(rawBytes, rawBytes.size)
                    pcm.write(monoBytes)
                    appendVoiceQuestionAudio(monoBytes)
                }
                val now = SystemClock.elapsedRealtime()
                if (now - chunkStartElapsedMs >= LightMemEgoConfig.AUDIO_CHUNK_MS && pcm.size() > 0) {
                    val pcmBytes = pcm.toByteArray()
                    pcm = ByteArrayOutputStream()
                    val audioIndex = _uiState.value.audioIndex
                    val relativeTsMs = (chunkStartElapsedMs - streamStartElapsedMs).coerceAtLeast(0L)
                    val durationMs = now - chunkStartElapsedMs
                    chunkStartElapsedMs = now
                    _uiState.update { it.copy(audioIndex = audioIndex + 1) }
                    if (uploadToBackend && sessionId.isNotBlank()) {
                        launchAudioUpload(sessionId, audioIndex, relativeTsMs, durationMs, pcmBytes)
                    }
                }
            }
        }
    }

    private fun launchAudioUpload(
        sessionId: String,
        audioIndex: Int,
        relativeTsMs: Long,
        durationMs: Long,
        pcmBytes: ByteArray,
    ) {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                uploadAudioChunkNow(sessionId, audioIndex, relativeTsMs, durationMs, pcmBytes)
            }.onSuccess {
                _uiState.update { state ->
                    state.copy(audioUploadedCount = state.audioUploadedCount + 1, lastError = "")
                }
            }.onFailure { error ->
                _uiState.update { state ->
                    state.copy(
                        audioFailedCount = state.audioFailedCount + 1,
                        lastError = error.message ?: "Audio upload failed",
                    )
                }
            }
        }
    }

    private fun uploadAudioChunkNow(
        sessionId: String,
        audioIndex: Int,
        relativeTsMs: Long,
        durationMs: Long,
        pcmBytes: ByteArray,
    ): LightMemEgoUploadResult {
        val wavBytes = WavEncoder.mono16PcmToWav(pcmBytes)
        return api.uploadAudioChunk(sessionId, wavBytes, audioIndex, relativeTsMs, durationMs)
    }

    private suspend fun pollQueryAnswer(taskId: String): String {
        val maxAttempts = (LightMemEgoConfig.QUERY_POLL_TIMEOUT_MS / LightMemEgoConfig.QUERY_POLL_INTERVAL_MS).toInt().coerceAtLeast(1)
        repeat(maxAttempts) {
            val task = api.getQueryTask(taskId)
            _uiState.update { it.copy(queryStatus = task.status.ifBlank { "running" }) }
            if (task.done) {
                return task.answer.ifBlank { "Backend returned no answer" }
            }
            if (task.status in setOf("failed", "cancelled", "canceled", "aborted", "not_found")) {
                throw IllegalStateException(task.message.ifBlank { "Question task failed: ${task.status}" })
            }
            delay(LightMemEgoConfig.QUERY_POLL_INTERVAL_MS)
        }
        throw IllegalStateException("Question timed out")
    }

    private fun rememberActiveSession(result: LightMemEgoStartResult) {
        if (result.parentSessionId.isNotBlank()) {
            sessionStore.saveParentSessionId(result.parentSessionId)
        }
        if (result.inputMode == "rokid_live_rtmp" && result.sessionId.isNotBlank() && result.pushUrl.isNotBlank()) {
            LightMemEgoDiagnostics.log(
                getApplication(),
                "stored-session-save",
                "session=${result.sessionId} parent=${result.parentSessionId}",
            )
            sessionStore.save(
                StoredRokidSession(
                    sessionId = result.sessionId,
                    parentSessionId = result.parentSessionId,
                    dayLabel = result.dayLabel,
                    dayIndex = result.dayIndex,
                    pushUrl = result.pushUrl,
                    liveIngestStartPath = result.liveIngestStartPath,
                    liveIngestStopPath = result.liveIngestStopPath,
                ),
            )
        } else {
            LightMemEgoDiagnostics.log(getApplication(), "stored-session-clear", "inputMode=${result.inputMode}")
            sessionStore.clear()
        }
    }

    private fun startPreferredRokidSession(): LightMemEgoStartResult {
        val runId = UUID.randomUUID().toString()
        val parentSessionId = configuredParentSessionId()
        val createParentSession = LightMemEgoConfig.CREATE_NEW_PARENT_SESSION && parentSessionId.isNullOrBlank()
        if (!parentSessionId.isNullOrBlank()) {
            runCatching { api.getSessionState(parentSessionId) }.onSuccess { stateJson ->
                val streamStatus = stateJson.optString("stream_status", stateJson.optString("status", "unknown"))
                if (streamStatus in setOf("ended", "stopped", "aborted", "cancelled", "canceled", "done")) {
                    _uiState.update { it.copy(status = "parent $streamStatus, starting day...") }
                } else {
                    _uiState.update { it.copy(status = "parent $streamStatus, starting day...") }
                }
            }.onFailure {
                _uiState.update { it.copy(status = "starting parent day...") }
            }
        } else if (createParentSession) {
            _uiState.update { it.copy(status = "creating parent...") }
        }
        return runCatching {
            api.startRokidStream(
                inputMode = LightMemEgoConfig.INPUT_MODE,
                parentSessionId = parentSessionId,
                runId = runId,
                createParentSession = createParentSession,
            )
        }.getOrElse { liveError ->
            if (LightMemEgoConfig.INPUT_MODE == "rokid_live_rtmp") {
                val fallbackParentId = (liveError as? LightMemEgoApiException)
                    ?.payload
                    ?.optString("parent_session_id")
                    ?.takeIf { it.isNotBlank() }
                    ?: parentSessionId
                api.startRokidStream(
                    inputMode = LightMemEgoConfig.FALLBACK_INPUT_MODE,
                    parentSessionId = fallbackParentId,
                    runId = runId,
                    createParentSession = createParentSession && fallbackParentId.isNullOrBlank(),
                ).also {
                    _uiState.update { state ->
                        state.copy(lastError = "RTMP live start failed, using HTTP fallback: ${liveError.message ?: liveError.javaClass.simpleName}")
                    }
                }
            } else {
                throw liveError
            }
        }
    }

    private fun configuredParentSessionId(): String? {
        if (LightMemEgoConfig.CREATE_NEW_PARENT_SESSION) {
            return null
        }
        val sessionId = LightMemEgoConfig.PARENT_SESSION_ID.trim()
        if (sessionId.isBlank()) {
            throw IllegalStateException("PARENT_SESSION_ID is blank while CREATE_NEW_PARENT_SESSION is false")
        }
        return sessionId
    }

    private fun startLiveIngest(sessionId: String, path: String) {
        if (sessionId.isBlank()) return
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { api.startLiveIngest(sessionId, path) }
                .onSuccess {
                    _uiState.update { state ->
                        val shouldShowQueued = state.liveRtmpStatus in setOf("waiting", "resuming", "reconnecting")
                        state.copy(liveRtmpStatus = if (shouldShowQueued) "ingest_queued" else state.liveRtmpStatus)
                    }
                }
                .onFailure { error ->
                    _uiState.update { state ->
                        state.copy(liveRtmpStatus = "ingest_failed", lastError = error.message ?: "live ingest start failed")
                    }
                }
        }
    }

    fun onRtmpStatus(status: String, detail: String = "") {
        LightMemEgoDiagnostics.log(getApplication(), "rtmp-status", "status=$status detail=$detail")
        when (status) {
            "connected", "auth_success" -> {
                cancelRtmpRetry(resetAttempt = true)
                _uiState.update { state ->
                    state.copy(liveRtmpStatus = status, lastError = "")
                }
                val state = _uiState.value
                if (status == "connected" && streaming && state.liveRtmpMode && state.sessionId.isNotBlank()) {
                    startLiveIngest(state.sessionId, state.liveIngestStartPath)
                }
            }
            "failed", "disconnected", "auth_error" -> {
                scheduleRtmpReconnect(status, detail)
            }
            else -> {
                _uiState.update { state ->
                    state.copy(
                        liveRtmpStatus = status,
                        lastError = if (status == "stop_failed") detail.ifBlank { status } else state.lastError,
                    )
                }
            }
        }
    }

    private fun scheduleRtmpReconnect(status: String, detail: String) {
        if (!streaming || stopping) return
        val current = _uiState.value
        if (!current.liveRtmpMode || current.livePushUrl.isBlank()) return
        rtmpRetryJob?.cancel()
        val delayMs = rtmpReconnectDelayMs(rtmpRetryAttempt)
        rtmpRetryAttempt += 1
        _uiState.update { state ->
            state.copy(
                liveRtmpStatus = "retrying",
                lastError = detail.ifBlank { status },
            )
        }
        rtmpRetryJob = viewModelScope.launch(Dispatchers.IO) {
            delay(delayMs)
            val latest = _uiState.value
            if (!streaming || stopping || !latest.liveRtmpMode || latest.livePushUrl.isBlank()) return@launch
            _uiState.update { state ->
                state.copy(
                    liveRtmpStatus = "reconnecting",
                    rtmpRestartToken = state.rtmpRestartToken + 1,
                )
            }
        }
    }

    private fun rtmpReconnectDelayMs(attempt: Int): Long {
        val capped = attempt.coerceIn(0, 4)
        return (1_000L shl capped).coerceAtMost(10_000L)
    }

    private fun cancelRtmpRetry(resetAttempt: Boolean) {
        rtmpRetryJob?.cancel()
        rtmpRetryJob = null
        if (resetAttempt) rtmpRetryAttempt = 0
    }

    private fun startStatusPolling(sessionId: String) {
        statusJob?.cancel()
        statusJob = viewModelScope.launch(Dispatchers.IO) {
            while (isActive && streaming) {
                pollStatus(sessionId)
                delay(LightMemEgoConfig.STATUS_POLL_MS)
            }
        }
    }

    private fun pollStatus(sessionId: String) {
        runCatching {
            api.getStreamStatus(sessionId)
        }.onSuccess { result ->
            _uiState.update {
                it.copy(
                    streamStatus = result.streamStatus.ifBlank { it.streamStatus },
                    canAsk = result.canAsk,
                    memoryReady = result.memoryReady,
                    framesReceived = result.framesReceived,
                    audioChunksReceived = result.audioChunksReceived,
                    lastError = "",
                )
            }
        }.onFailure { error ->
            _uiState.update { it.copy(lastError = error.message ?: "Status refresh failed") }
        }
    }

    private fun failStart(message: String) {
        LightMemEgoDiagnostics.log(getApplication(), "start-failed", message)
        streaming = false
        cancelRtmpRetry(resetAttempt = true)
        releaseRecorder()
        _uiState.update {
            it.copy(status = "connection failed", running = false, canAsk = false, memoryReady = false, liveRtmpMode = false, livePushUrl = "", liveIngestStartPath = "", liveIngestStopPath = "", liveRtmpStatus = "failed", lastError = message)
        }
    }

    private fun relativeElapsedMs(): Long =
        (SystemClock.elapsedRealtime() - streamStartElapsedMs).coerceAtLeast(0L)

    private fun appendVoiceQuestionAudio(monoBytes: ByteArray) {
        if (!voiceQuestionCapturing) return
        val now = SystemClock.elapsedRealtime()
        synchronized(voiceQuestionLock) {
            if (voiceQuestionCapturing) {
                voiceQuestionPcm.write(monoBytes)
            }
        }
        if (now - lastVoiceQuestionDurationUiElapsedMs >= 500L) {
            lastVoiceQuestionDurationUiElapsedMs = now
            val durationMs = (now - voiceQuestionStartElapsedMs).coerceAtLeast(0L)
            _uiState.update { it.copy(voiceQuestionDurationMs = durationMs) }
        }
    }

    private fun cancelVoiceQuestionCapture() {
        synchronized(voiceQuestionLock) {
            voiceQuestionCapturing = false
            voiceQuestionPcm = ByteArrayOutputStream()
        }
        _uiState.update {
            it.copy(
                voiceQuestionRecording = false,
                voiceQuestionDurationMs = 0L,
                voiceQuestionStatus = "idle",
                voiceQuestionMessage = "",
            )
        }
    }
    private fun releaseRecorder() {
        runCatching {
            recorder?.stop()
        }
        runCatching {
            recorder?.release()
        }
        recorder = null
    }

    override fun onCleared() {
        streaming = false
        audioJob?.cancel()
        statusJob?.cancel()
        queryJob?.cancel()
        cancelRtmpRetry(resetAttempt = true)
        cancelVoiceQuestionCapture()
        releaseRecorder()
        super.onCleared()
    }
}
