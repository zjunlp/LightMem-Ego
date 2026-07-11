package cn.zjukg.lightmem.glass.worldmm

import android.content.Context

data class StoredRokidSession(
    val sessionId: String,
    val parentSessionId: String,
    val dayLabel: String,
    val dayIndex: Int,
    val pushUrl: String,
    val liveIngestStartPath: String,
    val liveIngestStopPath: String,
)

class WorldMMRokidSessionStore(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun save(session: StoredRokidSession) {
        prefs.edit()
            .putString(KEY_SESSION_ID, session.sessionId)
            .putString(KEY_PARENT_SESSION_ID, session.parentSessionId)
            .putString(KEY_DAY_LABEL, session.dayLabel)
            .putInt(KEY_DAY_INDEX, session.dayIndex)
            .putString(KEY_PUSH_URL, session.pushUrl)
            .putString(KEY_LIVE_INGEST_START_PATH, session.liveIngestStartPath)
            .putString(KEY_LIVE_INGEST_STOP_PATH, session.liveIngestStopPath)
            .apply()
    }

    fun load(): StoredRokidSession? {
        val sessionId = prefs.getString(KEY_SESSION_ID, "").orEmpty().trim()
        val pushUrl = prefs.getString(KEY_PUSH_URL, "").orEmpty().trim()
        if (sessionId.isBlank() || pushUrl.isBlank()) return null
        return StoredRokidSession(
            sessionId = sessionId,
            parentSessionId = prefs.getString(KEY_PARENT_SESSION_ID, "").orEmpty(),
            dayLabel = prefs.getString(KEY_DAY_LABEL, "").orEmpty(),
            dayIndex = prefs.getInt(KEY_DAY_INDEX, 0),
            pushUrl = pushUrl,
            liveIngestStartPath = prefs.getString(KEY_LIVE_INGEST_START_PATH, "").orEmpty(),
            liveIngestStopPath = prefs.getString(KEY_LIVE_INGEST_STOP_PATH, "").orEmpty(),
        )
    }

    fun saveParentSessionId(parentSessionId: String) {
        val clean = parentSessionId.trim()
        if (clean.isBlank()) return
        prefs.edit()
            .putString(KEY_PARENT_SESSION_ID, clean)
            .apply()
    }

    fun loadParentSessionId(): String =
        prefs.getString(KEY_PARENT_SESSION_ID, "").orEmpty().trim()

    fun clear() {
        prefs.edit().clear().apply()
    }

    private companion object {
        const val PREFS_NAME = "worldmm_rokid_session"
        const val KEY_SESSION_ID = "session_id"
        const val KEY_PARENT_SESSION_ID = "parent_session_id"
        const val KEY_DAY_LABEL = "day_label"
        const val KEY_DAY_INDEX = "day_index"
        const val KEY_PUSH_URL = "push_url"
        const val KEY_LIVE_INGEST_START_PATH = "live_ingest_start_path"
        const val KEY_LIVE_INGEST_STOP_PATH = "live_ingest_stop_path"
    }
}
