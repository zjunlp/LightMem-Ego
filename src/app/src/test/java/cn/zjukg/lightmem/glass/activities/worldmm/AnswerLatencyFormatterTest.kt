package cn.zjukg.lightmem.glass.activities.worldmm

import org.junit.Assert.assertEquals
import org.junit.Test

class AnswerLatencyFormatterTest {
    @Test
    fun formatsLatencyForStandaloneDisplay() {
        assertEquals("1.53s", formatAnswerLatency(1_530L))
    }

    @Test
    fun formatsShortLatencyInMilliseconds() {
        assertEquals("999ms", formatAnswerLatency(999L))
    }
}
