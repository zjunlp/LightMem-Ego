package cn.zjukg.lightmem.glass.worldmm

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Debug
import android.os.Process
import android.os.SystemClock
import android.util.Log
import android.view.View
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object WorldMMDiagnostics {
    private const val TAG = "OmniSparkDiag"
    private const val PREFS_NAME = "worldmm_diagnostics"
    private const val KEY_LAST_EXIT_TS = "last_exit_ts"
    private const val DIR_NAME = "worldmm_diagnostics"
    private const val LOG_FILE = "diagnostics.log"
    private const val MAX_LOG_BYTES = 512_000L
    private const val MAX_TRACE_CHARS = 32_000

    fun recordStartup(context: Context, owner: String) {
        log(context, "startup", "owner=$owner pid=${Process.myPid()} uptimeMs=${SystemClock.uptimeMillis()}")
        recordMemory(context, "startup")
        recordHistoricalExitReasons(context)
    }

    fun logLifecycle(context: Context, owner: String, event: String, detail: String = "") {
        log(context, "lifecycle", "owner=$owner event=$event $detail")
    }

    fun logView(context: Context, owner: String, event: String, view: View?) {
        val detail = if (view == null) {
            "view=null"
        } else {
            "attached=${view.isAttachedToWindow} focused=${view.hasWindowFocus()} size=${view.width}x${view.height}"
        }
        log(context, "view", "owner=$owner event=$event $detail")
    }

    fun recordMemory(context: Context, label: String) {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        val runtime = Runtime.getRuntime()
        log(
            context,
            "memory",
            "label=$label pssKb=${info.totalPss} nativePssKb=${info.nativePss} " +
                "dalvikPssKb=${info.dalvikPss} heapUsedKb=${(runtime.totalMemory() - runtime.freeMemory()) / 1024}",
        )
    }

    fun log(context: Context, event: String, detail: String = "") {
        val line = "${timestamp()} $event $detail".trim()
        Log.i(TAG, line)
        appendLine(context, line)
    }

    fun logError(context: Context, event: String, detail: String, error: Throwable? = null) {
        val line = "${timestamp()} $event $detail error=${error?.javaClass?.simpleName.orEmpty()}:${error?.message.orEmpty()}"
        Log.e(TAG, line, error)
        appendLine(context, line)
    }

    fun diagnosticsLogPath(context: Context): String =
        File(File(context.applicationContext.filesDir, DIR_NAME), LOG_FILE).absolutePath

    private fun recordHistoricalExitReasons(context: Context) {
        val appContext = context.applicationContext
        val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val lastRecordedTs = prefs.getLong(KEY_LAST_EXIT_TS, 0L)
        val manager = appContext.getSystemService(ActivityManager::class.java) ?: return
        val exits = runCatching {
            manager.getHistoricalProcessExitReasons(appContext.packageName, 0, 8)
        }.getOrElse { error ->
            logError(appContext, "exit-history-read-failed", "", error)
            return
        }
        var newestTs = lastRecordedTs
        exits
            .filter { it.timestamp > lastRecordedTs }
            .sortedBy { it.timestamp }
            .forEach { exit ->
                newestTs = maxOf(newestTs, exit.timestamp)
                log(
                    appContext,
                    "previous-exit",
                    "time=${timestamp(exit.timestamp)} reason=${exit.reasonName()} " +
                        "status=${exit.status} importance=${exit.importance} pssKb=${exit.pss} rssKb=${exit.rss} " +
                        "description=${exit.description.orEmpty()}",
                )
                saveExitTrace(appContext, exit)
            }
        if (newestTs > lastRecordedTs) {
            prefs.edit().putLong(KEY_LAST_EXIT_TS, newestTs).apply()
        }
    }

    private fun saveExitTrace(context: Context, exit: ApplicationExitInfo) {
        val trace = readTrace(exit) ?: return
        val dir = File(context.filesDir, DIR_NAME).also { it.mkdirs() }
        val traceFile = File(dir, "exit_${exit.timestamp}_${exit.reasonName()}.txt")
        runCatching {
            traceFile.writeText(trace)
        }.onFailure { error ->
            logError(context, "exit-trace-save-failed", "path=${traceFile.absolutePath}", error)
        }
        val interesting = trace
            .lineSequence()
            .filter { line ->
                line.contains("Abort message", ignoreCase = true) ||
                    line.contains("EGL", ignoreCase = true) ||
                    line.contains("signal ", ignoreCase = true) ||
                    line.contains("RenderThread", ignoreCase = true) ||
                    line.contains("Cmdline:", ignoreCase = true) ||
                    line.contains("pid:", ignoreCase = true)
            }
            .take(12)
            .joinToString(" | ")
            .ifBlank { trace.take(800).replace('\n', ' ') }
        log(context, "previous-exit-trace", "path=${traceFile.absolutePath} snippet=$interesting")
    }

    private fun readTrace(exit: ApplicationExitInfo): String? {
        val input = runCatching { exit.traceInputStream }.getOrNull() ?: return null
        return input.bufferedReader().use { reader ->
            val buffer = CharArray(4096)
            val out = StringBuilder()
            while (out.length < MAX_TRACE_CHARS) {
                val read = reader.read(buffer, 0, minOf(buffer.size, MAX_TRACE_CHARS - out.length))
                if (read <= 0) break
                out.append(buffer, 0, read)
            }
            out.toString()
        }.ifBlank { null }
    }

    @Synchronized
    private fun appendLine(context: Context, line: String) {
        val dir = File(context.applicationContext.filesDir, DIR_NAME).also { it.mkdirs() }
        val file = File(dir, LOG_FILE)
        if (file.length() > MAX_LOG_BYTES) {
            runCatching { file.renameTo(File(dir, "diagnostics.old.log")) }
        }
        file.appendText(line + "\n")
    }

    private fun ApplicationExitInfo.reasonName(): String = when (reason) {
        ApplicationExitInfo.REASON_ANR -> "ANR"
        ApplicationExitInfo.REASON_CRASH -> "CRASH"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "CRASH_NATIVE"
        ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "DEPENDENCY_DIED"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "EXCESSIVE_RESOURCE_USAGE"
        ApplicationExitInfo.REASON_EXIT_SELF -> "EXIT_SELF"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "INITIALIZATION_FAILURE"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "LOW_MEMORY"
        ApplicationExitInfo.REASON_OTHER -> "OTHER"
        ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "PERMISSION_CHANGE"
        ApplicationExitInfo.REASON_SIGNALED -> "SIGNALED"
        ApplicationExitInfo.REASON_UNKNOWN -> "UNKNOWN"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "USER_REQUESTED"
        else -> "reason_$reason"
    }

    private fun timestamp(timeMs: Long = System.currentTimeMillis()): String =
        SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date(timeMs))
}
