package cn.zjukg.lightmem.glass.input

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.SystemClock
import android.util.Log
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.core.content.ContextCompat
import cn.zjukg.lightmem.glass.BuildConfig
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoDiagnostics

typealias BareKeyHandler = (BareKeyEvent) -> Boolean

/**
 * Centralized glasses input dispatcher: TouchPad KeyEvents plus ordered temple/TouchPad broadcasts.
 *
 * See the developer documentation chapter about keys, wear detection, and folding for event definitions.
 *
 * - One-finger click -> [dispatchEnterKey] with `KEYCODE_ENTER`
 * - One-finger double click -> [dispatchBackKey] with `KEYCODE_BACK`
 * - One-finger long press -> [dispatchLongKey] with `KEYCODE_PROG_BLUE`, or `ACTION_AI_START`
 * - Temple click -> broadcast mapped to [BareKeyEvent.SpriteClick]
 * - Two-finger long press -> [dispatchTwoFingerLongPressKey] with `KEYCODE_SETTINGS`, or `ACTION_SETTINGS_KEY`
 *
 * Abort-only events are not dispatched to UI, such as temple long/double press and unused two-finger gestures.
 * Ordered broadcasts must call `abortBroadcast()` inside `onReceive` to avoid default system AI,
 * settings, or power actions.
 */
class BareGlassesInputDispatcher(context: Context) {
    private val appContext = context.applicationContext
    private var handler: BareKeyHandler? = null
    private var interceptListener: ((String) -> Unit)? = null
    private var lastLongPressDispatchedAtMs = 0L
    private var lastSpriteClickDispatchedAtMs = 0L

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context?, intent: Intent?) {
            val action = intent?.action ?: return
            val label = keyEventActionLabel(action)
            val aborted = abortOrderedBroadcast(this)
            notifyIntercept(label, aborted)
            eventForBroadcastAction(action)?.let { event -> dispatchEvent(event, label) }
        }
    }

    init {
        val filter = IntentFilter().apply {
            priority = IntentFilter.SYSTEM_HIGH_PRIORITY
            addAction(KeyEventAction.CLICK.action)
            addAction(KeyEventAction.BUTTON_DOWN.action)
            addAction(KeyEventAction.BUTTON_UP.action)
            addAction(KeyEventAction.DOUBLE_CLICK.action)
            addAction(KeyEventAction.LONG_PRESS.action)
            addAction(KeyEventAction.VERY_LONG_PRESS.action)
            addAction(KeyEventAction.TWO_FINGER_SINGLE.action)
            addAction(KeyEventAction.TWO_FINGER_DOUBLE.action)
            addAction(KeyEventAction.SWIPE_FORWARD.action)
            addAction(KeyEventAction.SWIPE_BACK.action)
            addAction(KeyEventAction.AI_START.action)
            addAction(KeyEventAction.SETTINGS_KEY.action)
        }
        ContextCompat.registerReceiver(
            appContext,
            receiver,
            filter,
            ContextCompat.RECEIVER_EXPORTED,
        )
    }

    fun setHandler(h: BareKeyHandler?) {
        handler = h
    }

    fun setInterceptListener(listener: ((String) -> Unit)?) {
        interceptListener = listener
    }

    /** Clears only when the current handler is still [expected], avoiding page-switch races. */
    fun clearHandlerIf(expected: BareKeyHandler) {
        if (handler === expected) {
            handler = null
        }
    }

    /** One-finger TouchPad click with `KEYCODE_ENTER` -> [BareKeyEvent.Click]. */
    fun dispatchEnterKey() {
        notifyIntercept("Key ENTER", consumed = true)
        dispatchEvent(BareKeyEvent.Click, "Key ENTER")
    }

    /** One-finger TouchPad double click with `KEYCODE_BACK` -> [BareKeyEvent.DoubleClick]. */
    fun dispatchBackKey() {
        notifyIntercept("Key BACK", consumed = true)
        dispatchEvent(BareKeyEvent.DoubleClick, "Key BACK")
    }

    /** One-finger TouchPad long press with `KEYCODE_PROG_BLUE` -> [BareKeyEvent.LongPress]. */
    fun dispatchLongKey() {
        notifyIntercept("Key PROG_BLUE", consumed = true)
        dispatchEvent(BareKeyEvent.LongPress, "Key PROG_BLUE")
    }

    /** Consumes KeyEvent-only paths such as two-finger long press without dispatching to UI. */
    fun consumeSystemKey(label: String) {
        notifyIntercept(label, consumed = true)
        logConsumedKey(label)
    }

    /** Two-finger TouchPad long press with `KEYCODE_SETTINGS` -> [BareKeyEvent.TwoFingerLongPress]. */
    fun dispatchTwoFingerLongPressKey(label: String) {
        notifyIntercept(label, consumed = true)
        dispatchEvent(BareKeyEvent.TwoFingerLongPress, label)
    }

    /** Direction-key fallback for TouchPad forward swipe -> [BareKeyEvent.SwipeForward]. */
    fun dispatchSwipeForwardKey(label: String) {
        notifyIntercept(label, consumed = true)
        dispatchEvent(BareKeyEvent.SwipeForward, label)
    }

    /** Direction-key fallback for TouchPad backward swipe -> [BareKeyEvent.SwipeBack]. */
    fun dispatchSwipeBackKey(label: String) {
        notifyIntercept(label, consumed = true)
        dispatchEvent(BareKeyEvent.SwipeBack, label)
    }

    fun unregister(context: Context) {
        try {
            context.applicationContext.unregisterReceiver(receiver)
        } catch (_: Exception) {
        }
        handler = null
        interceptListener = null
    }

    private fun dispatchEvent(event: BareKeyEvent, label: String) {
        val recognizedAtUptimeMs = SystemClock.uptimeMillis()
        if (event == BareKeyEvent.LongPress && shouldDropDuplicateLongPress()) {
            logRecognizedAction(
                event = event,
                label = label,
                recognizedAtUptimeMs = recognizedAtUptimeMs,
                handled = false,
                dropped = true,
            )
            if (BuildConfig.DEBUG) {
                Log.d(TAG, "drop duplicate long press from $label")
            }
            return
        }
        if (event == BareKeyEvent.SpriteClick && shouldDropDuplicateSpriteClick()) {
            logRecognizedAction(
                event = event,
                label = label,
                recognizedAtUptimeMs = recognizedAtUptimeMs,
                handled = false,
                dropped = true,
            )
            if (BuildConfig.DEBUG) {
                Log.d(TAG, "drop duplicate sprite click from $label")
            }
            return
        }
        runCatching { handler?.invoke(event) ?: false }
            .onSuccess { handled ->
                logRecognizedAction(
                    event = event,
                    label = label,
                    recognizedAtUptimeMs = recognizedAtUptimeMs,
                    handled = handled,
                    dropped = false,
                )
            }
            .onFailure { error ->
                logRecognizedAction(
                    event = event,
                    label = label,
                    recognizedAtUptimeMs = recognizedAtUptimeMs,
                    handled = false,
                    dropped = false,
                    error = error.javaClass.simpleName,
                )
                Log.e(TAG, "key handler failed for $label: ${error.message}", error)
            }
    }

    private fun shouldDropDuplicateLongPress(): Boolean {
        val now = SystemClock.elapsedRealtime()
        if (now - lastLongPressDispatchedAtMs < LONG_PRESS_DEBOUNCE_MS) {
            return true
        }
        lastLongPressDispatchedAtMs = now
        return false
    }

    private fun shouldDropDuplicateSpriteClick(): Boolean {
        val now = SystemClock.elapsedRealtime()
        if (now - lastSpriteClickDispatchedAtMs < SPRITE_CLICK_DEBOUNCE_MS) {
            return true
        }
        lastSpriteClickDispatchedAtMs = now
        return false
    }

    private fun notifyIntercept(label: String, consumed: Boolean) {
        val suffix = when {
            label.startsWith("Key ") -> " consumed"
            consumed -> " aborted"
            else -> " not aborted (non-ordered broadcast?)"
        }
        interceptListener?.invoke(label + suffix)
    }

    private fun logConsumedKey(label: String) {
        LightMemEgoDiagnostics.log(
            appContext,
            "input-key-consumed",
            "source=${label.toLogValue()} uptimeMs=${SystemClock.uptimeMillis()}",
        )
    }

    private fun logRecognizedAction(
        event: BareKeyEvent,
        label: String,
        recognizedAtUptimeMs: Long,
        handled: Boolean,
        dropped: Boolean,
        error: String = "",
    ) {
        LightMemEgoDiagnostics.log(
            appContext,
            "input-action",
            inputActionLogDetail(
                event = event,
                label = label,
                recognizedAtUptimeMs = recognizedAtUptimeMs,
                handled = handled,
                dropped = dropped,
                error = error,
            ),
        )
    }

    companion object {
        private const val TAG = "BareGlassesInputDispatcher"
        private const val LONG_PRESS_DEBOUNCE_MS = 800L
        private const val SPRITE_CLICK_DEBOUNCE_MS = 350L

        internal fun eventForBroadcastAction(action: String): BareKeyEvent? = when (action) {
            KeyEventAction.CLICK.action -> BareKeyEvent.SpriteClick
            KeyEventAction.BUTTON_UP.action -> BareKeyEvent.SpriteClick
            KeyEventAction.AI_START.action -> BareKeyEvent.LongPress
            KeyEventAction.LONG_PRESS.action -> BareKeyEvent.LongPress
            KeyEventAction.TWO_FINGER_SINGLE.action -> BareKeyEvent.TwoFingerClick
            KeyEventAction.TWO_FINGER_DOUBLE.action -> BareKeyEvent.TwoFingerDoubleClick
            KeyEventAction.SETTINGS_KEY.action -> BareKeyEvent.TwoFingerLongPress
            KeyEventAction.SWIPE_FORWARD.action -> BareKeyEvent.SwipeForward
            KeyEventAction.SWIPE_BACK.action -> BareKeyEvent.SwipeBack
            else -> null
        }

        internal fun inputActionLogDetail(
            event: BareKeyEvent,
            label: String,
            recognizedAtUptimeMs: Long,
            handled: Boolean,
            dropped: Boolean,
            error: String = "",
        ): String =
            "event=${event.name} source=${label.toLogValue()} uptimeMs=$recognizedAtUptimeMs " +
                "handled=$handled dropped=$dropped error=${error.toLogValue()}"

        fun abortOrderedBroadcast(receiver: BroadcastReceiver): Boolean {
            if (!receiver.isOrderedBroadcast) {
                Log.w(TAG, "abort skipped: not ordered broadcast")
                return false
            }
            return try {
                receiver.abortBroadcast()
                true
            } catch (e: IllegalStateException) {
                Log.w(TAG, "abortBroadcast failed: ${e.message}")
                false
            }
        }
    }
}

private fun String.toLogValue(): String =
    if (isBlank()) "-" else replace('\n', ' ').replace('\r', ' ')

typealias BareSpriteKeyDispatcher = BareGlassesInputDispatcher

val LocalBareGlassesInputDispatcher = staticCompositionLocalOf<BareGlassesInputDispatcher?> { null }

val LocalBareSpriteKeyDispatcher = LocalBareGlassesInputDispatcher

@Composable
fun RegisterBareKeyHandler(handler: BareKeyHandler) {
    val dispatcher = LocalBareGlassesInputDispatcher.current ?: return
    val latestHandler by rememberUpdatedState(handler)
    DisposableEffect(dispatcher) {
        val delegate: BareKeyHandler = { event -> latestHandler(event) }
        dispatcher.setHandler(delegate)
        onDispose { dispatcher.clearHandlerIf(delegate) }
    }
}

@Composable
fun rememberBareGlassesInputDispatcher(context: Context): BareGlassesInputDispatcher {
    val appContext = context.applicationContext
    return remember(appContext) { BareGlassesInputDispatcher(appContext) }
}

@Composable
fun rememberBareSpriteKeyDispatcher(context: Context): BareGlassesInputDispatcher =
    rememberBareGlassesInputDispatcher(context)
