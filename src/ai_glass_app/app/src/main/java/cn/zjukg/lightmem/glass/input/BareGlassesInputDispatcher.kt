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
 *
 * Abort-only events are not dispatched to UI, such as temple long/double press and two-finger gestures.
 * Ordered broadcasts must call `abortBroadcast()` inside `onReceive` to avoid default system AI,
 * settings, or power actions.
 */
class BareGlassesInputDispatcher(context: Context) {
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
            context.applicationContext,
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
        if (event == BareKeyEvent.LongPress && shouldDropDuplicateLongPress()) {
            Log.d(TAG, "drop duplicate long press from $label")
            return
        }
        if (event == BareKeyEvent.SpriteClick && shouldDropDuplicateSpriteClick()) {
            Log.d(TAG, "drop duplicate sprite click from $label")
            return
        }
        runCatching { handler?.invoke(event) }
            .onFailure { error ->
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
            KeyEventAction.SWIPE_FORWARD.action -> BareKeyEvent.SwipeForward
            KeyEventAction.SWIPE_BACK.action -> BareKeyEvent.SwipeBack
            else -> null
        }

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
