package cn.zjukg.lightmem.glass.activities.main

import android.view.KeyEvent

internal fun mainActivityConsumedKeyUpLabel(keyCode: Int): String? = when (keyCode) {
    KeyEvent.KEYCODE_BACK -> "Key BACK up"
    KeyEvent.KEYCODE_SETTINGS -> "Key SETTINGS up"
    KeyEvent.KEYCODE_DPAD_LEFT,
    KeyEvent.KEYCODE_DPAD_UP -> "Key-DPAD-PREV-up"
    KeyEvent.KEYCODE_DPAD_RIGHT,
    KeyEvent.KEYCODE_DPAD_DOWN -> "Key-DPAD-NEXT-up"
    else -> null
}
