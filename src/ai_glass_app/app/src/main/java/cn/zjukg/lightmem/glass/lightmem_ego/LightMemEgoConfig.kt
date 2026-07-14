package cn.zjukg.lightmem.glass.lightmem_ego

object LightMemEgoConfig {
    const val API_BASE_URL = "https://lightmem-ego.zjukg.cn/api"
    const val INPUT_MODE = "rokid_frame_audio"
    const val CREATE_NEW_PARENT_SESSION = true
    const val PARENT_SESSION_ID = ""
    val PRESET_QUESTIONS = listOf(
        "What is in the current scene?",
        "What just happened",
        "What did they just say?",
        "Summarize everything so far.",
    )
    const val FRAME_INTERVAL_MS = 1_000L
    const val FRAME_CAPTURE_WIDTH = 640
    const val FRAME_CAPTURE_HEIGHT = 360
    const val AUDIO_CHUNK_MS = 1_000L
    const val STATUS_POLL_MS = 2_000L
    const val HTTP_READ_TIMEOUT_MS = 120_000
    const val QUERY_POLL_TIMEOUT_MS = 120_000L
    const val QUERY_POLL_INTERVAL_MS = 900L
    const val JPEG_QUALITY = 75
    const val AUDIO_SAMPLE_RATE = 16_000
    const val ROKID_CHANNEL_COUNT = 8
}
