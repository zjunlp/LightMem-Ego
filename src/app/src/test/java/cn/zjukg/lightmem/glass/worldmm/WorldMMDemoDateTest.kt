package cn.zjukg.lightmem.glass.worldmm

import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.Clock
import java.time.Instant
import java.time.ZoneId

class WorldMMDemoDateTest {
    @Test
    fun formatsActualDateFromClock() {
        val clock = Clock.fixed(
            Instant.parse("2026-07-11T08:30:00Z"),
            ZoneId.of("Asia/Shanghai"),
        )

        assertEquals("2026.7.11", WorldMMDemoDate.actualDateLabel(clock))
    }
}
