package cn.zjukg.lightmem.glass.activities.lightmem_ego

import java.util.Locale

internal fun formatAnswerLatency(latencyMs: Long): String {
    val safeLatencyMs = latencyMs.coerceAtLeast(0L)
    return if (safeLatencyMs < 1_000L) {
        "${safeLatencyMs}ms"
    } else {
        String.format(Locale.US, "%.2fs", safeLatencyMs / 1_000.0)
    }
}
