package cn.zjukg.lightmem.glass.lightmem_ego

import android.content.Context
import com.pedro.common.ConnectChecker
import com.pedro.encoder.input.sources.audio.NoAudioSource
import com.pedro.encoder.input.sources.video.Camera2Source
import com.pedro.library.generic.GenericStream

private const val RTMP_RECONNECT_MAX_DELAY_MS = 10_000L
private const val RTMP_RECONNECT_SINGLE_RETRY = 1

class LightMemEgoRtmpStreamer(
    context: Context,
    private val listener: Listener,
) : ConnectChecker {
    interface Listener {
        fun onRtmpStatus(status: String, detail: String = "")
    }

    private val appContext = context.applicationContext
    private var stream: GenericStream? = null
    private var reconnectAttempt = 0
    private var reconnectPending = false

    val isStreaming: Boolean
        get() = stream?.isStreaming == true

    val isActive: Boolean
        get() = stream != null

    fun start(pushUrl: String) {
        if (pushUrl.isBlank()) throw IllegalArgumentException("RTMP push_url is empty")
        if (isActive) return
        LightMemEgoDiagnostics.log(appContext, "rtmp-start", "mode=live url=${pushUrl.safeRtmpUrl()}")
        resetReconnectState()
        val nextStream = GenericStream(appContext, this, Camera2Source(appContext), NoAudioSource())
        try {
            val videoReady = nextStream.prepareVideo(
                width = LightMemEgoConfig.RTMP_VIDEO_WIDTH,
                height = LightMemEgoConfig.RTMP_VIDEO_HEIGHT,
                bitrate = LightMemEgoConfig.RTMP_VIDEO_BITRATE,
                fps = LightMemEgoConfig.RTMP_VIDEO_FPS,
                iFrameInterval = 2,
                rotation = 0,
            )
            val audioReady = nextStream.prepareAudio(
                sampleRate = LightMemEgoConfig.AUDIO_SAMPLE_RATE,
                isStereo = false,
                bitrate = 32_000,
            )
            if (!videoReady || !audioReady) {
                throw IllegalStateException("RTMP encoder prepare failed: video=$videoReady audio=$audioReady")
            }
            stream = nextStream
            nextStream.startStream(pushUrl)
        } catch (error: Throwable) {
            LightMemEgoDiagnostics.logError(appContext, "rtmp-start-failed", "mode=live", error)
            if (stream === nextStream) stream = null
            runCatching { nextStream.release() }
            throw error
        }
    }

    fun stop() {
        val active = stream ?: return
        LightMemEgoDiagnostics.log(appContext, "rtmp-stop", "streaming=${active.isStreaming}")
        stream = null
        resetReconnectState()
        runCatching {
            if (active.isStreaming) active.stopStream()
        }.onFailure { error ->
            listener.onRtmpStatus("stop_failed", error.message ?: error.javaClass.simpleName)
        }
        runCatching { active.release() }
    }

    override fun onConnectionStarted(url: String) {
        LightMemEgoDiagnostics.log(appContext, "rtmp-connection-started", url.safeRtmpUrl())
        reconnectPending = false
        listener.onRtmpStatus("connecting", url)
    }

    override fun onConnectionSuccess() {
        LightMemEgoDiagnostics.log(appContext, "rtmp-connection-success")
        resetReconnectState()
        listener.onRtmpStatus("connected")
    }

    override fun onConnectionFailed(reason: String) {
        LightMemEgoDiagnostics.log(appContext, "rtmp-connection-failed", reason)
        if (retryKeepingStream(reason)) return
        releaseBrokenStream()
        listener.onRtmpStatus("failed", reason)
    }

    override fun onDisconnect() {
        LightMemEgoDiagnostics.log(appContext, "rtmp-disconnect")
        if (retryKeepingStream("disconnect")) return
        releaseBrokenStream()
        listener.onRtmpStatus("disconnected")
    }

    override fun onAuthError() {
        LightMemEgoDiagnostics.log(appContext, "rtmp-auth-error")
        resetReconnectState()
        releaseBrokenStream()
        listener.onRtmpStatus("auth_error")
    }

    override fun onAuthSuccess() {
        LightMemEgoDiagnostics.log(appContext, "rtmp-auth-success")
        listener.onRtmpStatus("auth_success")
    }

    override fun onNewBitrate(bitrate: Long) = Unit

    private fun releaseBrokenStream() {
        val active = stream ?: return
        LightMemEgoDiagnostics.log(appContext, "rtmp-release-broken", "streaming=${active.isStreaming}")
        stream = null
        resetReconnectState()
        runCatching {
            if (active.isStreaming) active.stopStream()
        }
        runCatching { active.release() }
    }

    private fun retryKeepingStream(reason: String): Boolean {
        val active = stream ?: return false
        if (reconnectPending) {
            LightMemEgoDiagnostics.log(
                appContext,
                "rtmp-retry-pending",
                "attempt=$reconnectAttempt reason=$reason streaming=${active.isStreaming}",
            )
            listener.onRtmpStatus("retrying", "attempt=$reconnectAttempt pending reason=$reason")
            return true
        }
        val delayMs = rtmpReconnectDelayMs(reconnectAttempt)
        val nextAttempt = reconnectAttempt + 1
        return runCatching {
            val streamClient = active.getStreamClient()
            streamClient.setReTries(RTMP_RECONNECT_SINGLE_RETRY)
            reconnectPending = true
            val accepted = streamClient.reTry(delayMs, reason, null)
            if (accepted) {
                reconnectAttempt = nextAttempt
                LightMemEgoDiagnostics.log(
                    appContext,
                    "rtmp-retry-keep-stream",
                    "attempt=$nextAttempt delayMs=$delayMs reason=$reason streaming=${active.isStreaming}",
                )
                listener.onRtmpStatus("retrying", "attempt=$nextAttempt delayMs=$delayMs reason=$reason")
            } else {
                reconnectPending = false
                LightMemEgoDiagnostics.log(
                    appContext,
                    "rtmp-retry-rejected",
                    "attempt=$nextAttempt reason=$reason streaming=${active.isStreaming}",
                )
            }
            accepted
        }.getOrElse { error ->
            reconnectPending = false
            LightMemEgoDiagnostics.logError(appContext, "rtmp-retry-failed", "reason=$reason", error)
            false
        }
    }

    private fun rtmpReconnectDelayMs(attempt: Int): Long {
        val capped = attempt.coerceIn(0, 4)
        return (1_000L shl capped).coerceAtMost(RTMP_RECONNECT_MAX_DELAY_MS)
    }

    private fun resetReconnectState() {
        reconnectAttempt = 0
        reconnectPending = false
    }
}

private fun String.safeRtmpUrl(): String =
    replace(Regex("(?i)(rtmp://[^/]+/).*"), "$1...")
