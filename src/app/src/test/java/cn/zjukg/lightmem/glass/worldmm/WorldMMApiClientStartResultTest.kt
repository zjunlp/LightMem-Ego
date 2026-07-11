package cn.zjukg.lightmem.glass.worldmm

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test
import java.time.LocalDate
import java.time.format.DateTimeFormatter

class WorldMMApiClientStartResultTest {
    private val dateLabelFormatter = DateTimeFormatter.ofPattern("yyyy.M.d")

    @Test
    fun parsesTopLevelDateLabelWhenDayContextIsMissing() {
        val result = WorldMMApiClient().parseStartResult(
            json = JSONObject()
                .put("session_id", "parent__day3")
                .put("date_label", "2026.7.11")
                .put("day_index", 3)
                .put("input_mode", "rokid_live_rtmp")
                .put("push_url", "rtmp://example/live"),
            requestedParentSessionId = "",
            inputMode = "rokid_live_rtmp",
            runId = "run-1",
        )

        assertEquals("2026.7.11", result.dayLabel)
        assertEquals(3, result.dayIndex)
    }

    @Test
    fun ignoresDayLabelFromChildSessionId() {
        val result = WorldMMApiClient().parseStartResult(
            json = JSONObject()
                .put("session_id", "parent")
                .put("child_session_id", "parent__day7")
                .put("input_mode", "rokid_live_rtmp")
                .put("push_url", "rtmp://example/live"),
            requestedParentSessionId = "",
            inputMode = "rokid_live_rtmp",
            runId = "run-1",
        )

        assertEquals(dateLabelFormatter.format(LocalDate.now()), result.dayLabel)
        assertNotEquals("DAY7", result.dayLabel)
        assertEquals(0, result.dayIndex)
    }

    @Test
    fun ignoresNumericDayLabel() {
        val result = WorldMMApiClient().parseStartResult(
            json = JSONObject()
                .put("session_id", "parent")
                .put("day", "4")
                .put("input_mode", "rokid_live_rtmp")
                .put("push_url", "rtmp://example/live"),
            requestedParentSessionId = "",
            inputMode = "rokid_live_rtmp",
            runId = "run-1",
        )

        assertEquals(dateLabelFormatter.format(LocalDate.now()), result.dayLabel)
        assertNotEquals("4", result.dayLabel)
        assertEquals(0, result.dayIndex)
    }

    @Test
    fun ignoresNullStringDateLabel() {
        val result = WorldMMApiClient().parseStartResult(
            json = JSONObject()
                .put("session_id", "parent")
                .put(
                    "day_context",
                    JSONObject()
                        .put("date_label", "null")
                        .put("dayLabel", "None"),
                )
                .put("date_label", "null")
                .put("input_mode", "rokid_live_rtmp")
                .put("push_url", "rtmp://example/live"),
            requestedParentSessionId = "",
            inputMode = "rokid_live_rtmp",
            runId = "run-1",
        )

        assertEquals(dateLabelFormatter.format(LocalDate.now()), result.dayLabel)
        assertNotEquals("null", result.dayLabel)
    }
}
