package cn.zjukg.lightmem.glass.input
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BareGlassesInputDispatcherTest {
    @Test
    fun touchpadLongPressDispatchesLongPress() {
        assertEquals(
            BareKeyEvent.LongPress,
            BareGlassesInputDispatcher.eventForBroadcastAction(KeyEventAction.AI_START.action),
        )
    }

    @Test
    fun templeLongPressDispatchesLongPress() {
        assertEquals(
            BareKeyEvent.LongPress,
            BareGlassesInputDispatcher.eventForBroadcastAction(KeyEventAction.LONG_PRESS.action),
        )
    }

    @Test
    fun templeButtonUpDispatchesSpriteClick() {
        assertEquals(
            BareKeyEvent.SpriteClick,
            BareGlassesInputDispatcher.eventForBroadcastAction(KeyEventAction.BUTTON_UP.action),
        )
    }

    @Test
    fun twoFingerSingleTapDispatchesTwoFingerClick() {
        assertEquals(
            BareKeyEvent.TwoFingerClick,
            BareGlassesInputDispatcher.eventForBroadcastAction(KeyEventAction.TWO_FINGER_SINGLE.action),
        )
    }

    @Test
    fun twoFingerDoubleTapDispatchesTwoFingerDoubleClick() {
        assertEquals(
            BareKeyEvent.TwoFingerDoubleClick,
            BareGlassesInputDispatcher.eventForBroadcastAction(KeyEventAction.TWO_FINGER_DOUBLE.action),
        )
    }

    @Test
    fun twoFingerLongPressDispatchesTwoFingerLongPress() {
        assertEquals(
            BareKeyEvent.TwoFingerLongPress,
            BareGlassesInputDispatcher.eventForBroadcastAction(KeyEventAction.SETTINGS_KEY.action),
        )
    }

    @Test
    fun inputActionLogDetailIncludesActionSourceAndRecognizedTime() {
        val detail = BareGlassesInputDispatcher.inputActionLogDetail(
            event = BareKeyEvent.DoubleClick,
            label = "Key BACK",
            recognizedAtUptimeMs = 1234L,
            handled = true,
            dropped = false,
        )

        assertTrue(detail.contains("event=DoubleClick"))
        assertTrue(detail.contains("source=Key BACK"))
        assertTrue(detail.contains("uptimeMs=1234"))
        assertTrue(detail.contains("handled=true"))
        assertTrue(detail.contains("dropped=false"))
    }
}
