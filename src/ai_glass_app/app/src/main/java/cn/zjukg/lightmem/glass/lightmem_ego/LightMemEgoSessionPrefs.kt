package cn.zjukg.lightmem.glass.lightmem_ego

import android.content.Context

data class LightMemEgoSavedSession(
    val sessionId: String,
    val displayDayLabel: String,
    val dayIndex: Int,
)

enum class LightMemEgoSessionStartMode {
    NewSession,
    ContinueLast,
}

object LightMemEgoSessionPrefs {
    private const val PREFS_NAME = "lightmem_ego_session"
    private const val KEY_SESSION_ID = "last_session_id"
    private const val KEY_DISPLAY_DAY_LABEL = "last_display_day_label"
    private const val KEY_DAY_INDEX = "last_day_index"

    fun loadLastSession(context: Context): LightMemEgoSavedSession? {
        val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val sessionId = prefs.getString(KEY_SESSION_ID, "").orEmpty().trim()
        if (sessionId.isBlank()) return null
        return LightMemEgoSavedSession(
            sessionId = sessionId,
            displayDayLabel = prefs.getString(KEY_DISPLAY_DAY_LABEL, "").orEmpty(),
            dayIndex = prefs.getInt(KEY_DAY_INDEX, 0),
        )
    }

    fun saveLastSession(context: Context, result: LightMemEgoStartResult) {
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_SESSION_ID, result.sessionId)
            .putString(KEY_DISPLAY_DAY_LABEL, result.displayDayLabel)
            .putInt(KEY_DAY_INDEX, result.dayIndex)
            .apply()
    }

    fun clearLastSession(context: Context) {
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .clear()
            .apply()
    }
}
