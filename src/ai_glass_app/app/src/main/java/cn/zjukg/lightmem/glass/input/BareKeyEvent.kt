package cn.zjukg.lightmem.glass.input

/**
 * Shared UI semantics for the sample app.
 *
 * See the developer documentation chapter about keys, wear detection, and folding for event definitions.
 *
 * - [Click]: `KEYCODE_ENTER` from one-finger TouchPad click.
 * - [SpriteClick]: `ACTION_SPRITE_BUTTON_CLICK` from temple button click.
 * - [DoubleClick]: `KEYCODE_BACK` from one-finger TouchPad double click.
 * - [LongPress]: one-finger TouchPad long press via `KEYCODE_PROG_BLUE` or `ACTION_AI_START`.
 * - [SwipeForward]: two-finger TouchPad forward swipe via `ACTION_TWO_FINGER_SWIPE_FORWARD`.
 * - [SwipeBack]: two-finger TouchPad backward swipe via `ACTION_TWO_FINGER_SWIPE_BACK`.
 */
enum class BareKeyEvent {
    Click,
    SpriteClick,
    DoubleClick,
    LongPress,
    TwoFingerClick,
    TwoFingerDoubleClick,
    SwipeForward,
    SwipeBack,
}
