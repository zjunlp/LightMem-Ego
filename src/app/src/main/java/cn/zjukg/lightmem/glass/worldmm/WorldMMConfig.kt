package cn.zjukg.lightmem.glass.worldmm

object WorldMMConfig {
    const val API_BASE_URL = "https://lightmem-ego.zjukg.cn/api"
    const val INPUT_MODE = "rokid_live_rtmp"
    const val FALLBACK_INPUT_MODE = "rokid_frame_audio"
    const val CREATE_NEW_PARENT_SESSION = true
    const val PARENT_SESSION_ID = ""
    const val FRAME_INTERVAL_MS = 125L
    const val FRAME_CAPTURE_WIDTH = 1280
    const val FRAME_CAPTURE_HEIGHT = 720
    const val AUDIO_CHUNK_MS = 1_000L
    const val STATUS_POLL_MS = 2_000L
    const val HTTP_READ_TIMEOUT_MS = 120_000
    const val QUERY_POLL_TIMEOUT_MS = 120_000L
    const val QUERY_POLL_INTERVAL_MS = 900L
    const val JPEG_QUALITY = 90
    const val AUDIO_SAMPLE_RATE = 16_000
    const val ROKID_CHANNEL_COUNT = 8
    const val RTMP_VIDEO_WIDTH = 1280
    const val RTMP_VIDEO_HEIGHT = 720
    const val RTMP_VIDEO_FPS = 24
    const val RTMP_VIDEO_BITRATE = 6_000_000
}
