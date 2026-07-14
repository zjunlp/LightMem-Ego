package cn.zjukg.lightmem.glass.activities.main

import android.view.KeyEvent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MainActivityKeyHandlingTest {
    @Test
    fun backKeyUpIsConsumedBecauseRokidMapsDoubleClickToBack() {
        assertEquals("Key BACK up", mainActivityConsumedKeyUpLabel(KeyEvent.KEYCODE_BACK))
    }

    @Test
    fun dpadFallbackKeyUpsAreConsumedAfterDispatchingSwipeOnKeyDown() {
        assertEquals("Key-DPAD-PREV-up", mainActivityConsumedKeyUpLabel(KeyEvent.KEYCODE_DPAD_LEFT))
        assertEquals("Key-DPAD-PREV-up", mainActivityConsumedKeyUpLabel(KeyEvent.KEYCODE_DPAD_UP))
        assertEquals("Key-DPAD-NEXT-up", mainActivityConsumedKeyUpLabel(KeyEvent.KEYCODE_DPAD_RIGHT))
        assertEquals("Key-DPAD-NEXT-up", mainActivityConsumedKeyUpLabel(KeyEvent.KEYCODE_DPAD_DOWN))
    }

    @Test
    fun unrelatedKeyUpFallsThroughToAndroid() {
        assertNull(mainActivityConsumedKeyUpLabel(KeyEvent.KEYCODE_A))
    }
}
