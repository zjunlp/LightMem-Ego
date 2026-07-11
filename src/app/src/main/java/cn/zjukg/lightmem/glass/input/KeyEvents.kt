package cn.zjukg.lightmem.glass.input

/** Ordered broadcast actions for system keys, TouchPad gestures, and the temple button. */
enum class KeyEventAction(val action: String) {
    /** Temple button click. */
    CLICK("com.android.action.ACTION_SPRITE_BUTTON_CLICK"),

    /** Temple button down. */
    BUTTON_DOWN("com.android.action.ACTION_SPRITE_BUTTON_DOWN"),

    /** Temple button up. Not sent after a long press has already fired. */
    BUTTON_UP("com.android.action.ACTION_SPRITE_BUTTON_UP"),

    /** Temple button double click. */
    DOUBLE_CLICK("com.android.action.ACTION_SPRITE_BUTTON_DOUBLE_CLICK"),

    /** One-finger TouchPad long press. Abort it in samples to avoid launching system AI. */
    AI_START("com.android.action.ACTION_AI_START"),

    /** Temple button long press. Samples only abort it because the system uses it for power flow. */
    LONG_PRESS("com.android.action.ACTION_SPRITE_BUTTON_LONG_PRESS"),

    /** Temple button very long press. */
    VERY_LONG_PRESS("com.android.action.ACTION_SPRITE_BUTTON_VERY_VERY_LONG_PRESS"),

    /** Two-finger TouchPad click. */
    TWO_FINGER_SINGLE("com.android.action.ACTION_TWO_FINGER_SINGLE_TAP"),

    /** Two-finger TouchPad double click. */
    TWO_FINGER_DOUBLE("com.android.action.ACTION_TWO_FINGER_DOUBLE_TAP"),

    /** Two-finger TouchPad forward swipe. */
    SWIPE_FORWARD("com.android.action.ACTION_TWO_FINGER_SWIPE_FORWARD"),

    /** Two-finger TouchPad backward swipe. */
    SWIPE_BACK("com.android.action.ACTION_TWO_FINGER_SWIPE_BACK"),

    /** Two-finger TouchPad long press. Samples only abort it because the system uses it for settings. */
    SETTINGS_KEY("com.android.action.ACTION_SETTINGS_KEY"),
}

/** Short labels used by logs. */
fun KeyEventAction.logLabel(): String = when (this) {
    KeyEventAction.CLICK -> "broadcast: temple click"
    KeyEventAction.BUTTON_DOWN -> "broadcast: temple down"
    KeyEventAction.BUTTON_UP -> "broadcast: temple up"
    KeyEventAction.DOUBLE_CLICK -> "broadcast: temple double click"
    KeyEventAction.AI_START -> "broadcast: TouchPad long press"
    KeyEventAction.LONG_PRESS -> "broadcast: temple long press"
    KeyEventAction.VERY_LONG_PRESS -> "broadcast: temple very long press"
    KeyEventAction.TWO_FINGER_SINGLE -> "broadcast: two-finger click"
    KeyEventAction.TWO_FINGER_DOUBLE -> "broadcast: two-finger double click"
    KeyEventAction.SWIPE_FORWARD -> "broadcast: two-finger forward swipe"
    KeyEventAction.SWIPE_BACK -> "broadcast: two-finger backward swipe"
    KeyEventAction.SETTINGS_KEY -> "broadcast: two-finger long press"
}

fun keyEventActionLabel(action: String): String =
    KeyEventAction.entries.firstOrNull { it.action == action }?.logLabel()
        ?: action.substringAfterLast('.')
