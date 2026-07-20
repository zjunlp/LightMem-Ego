package cn.zjukg.lightmem.glass.lightmem_ego

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class LightMemEgoApiClientStartResultTest {
    @Test
    fun parsesSingleSessionDayContextAndUploadOffsets() {
        val result = LightMemEgoApiClient().parseStartResult(
            json = JSONObject()
                .put("session_id", "session-123")
                .put("day_context", dayContextJson(dayLabel = "DAY2", weekdayLabel = "Tue", displayDayLabel = "DAY2 Tue", dayIndex = 2, runId = "run-2", relativeTsBaseMs = 90001))
                .put("next_frame_index", 42)
                .put("next_audio_index", 77)
                .put("input_mode", "rokid_frame_audio"),
            requestedSessionId = "session-123",
            inputMode = "rokid_frame_audio",
            runId = "run-fallback",
        )

        assertEquals("session-123", result.sessionId)
        assertEquals("DAY2", result.dayLabel)
        assertEquals(2, result.dayIndex)
        assertEquals("DAY2 Tue", result.displayDayLabel)
        assertEquals("run-2", result.runId)
        assertEquals(42, result.nextFrameIndex)
        assertEquals(77, result.nextAudioIndex)
        assertEquals(90001L, result.relativeTsBaseMs)
    }

    @Test
    fun rejectsStartResultWhenBackendOmitsDayContext() {
        try {
            LightMemEgoApiClient().parseStartResult(
                json = JSONObject()
                    .put("session_id", "session-123")
                    .put("input_mode", "rokid_frame_audio"),
                requestedSessionId = "",
                inputMode = "rokid_frame_audio",
                runId = "run-1",
            )
            fail("Expected missing day_context to fail")
        } catch (error: IllegalStateException) {
            assertTrue(error.message.orEmpty().contains("day_context"))
        }
    }

    @Test
    fun rejectsStartResultWhenDayContextOmitsRequiredFields() {
        try {
            LightMemEgoApiClient().parseStartResult(
                json = JSONObject()
                    .put("session_id", "session-123")
                    .put("day_context", JSONObject().put("mode", "single_session"))
                    .put("input_mode", "rokid_frame_audio"),
                requestedSessionId = "",
                inputMode = "rokid_frame_audio",
                runId = "run-1",
            )
            fail("Expected incomplete day_context to fail")
        } catch (error: IllegalStateException) {
            assertTrue(error.message.orEmpty().contains("day_label"))
        }
    }

    @Test
    fun defaultsToRokidHttpUploadPaths() {
        val result = LightMemEgoApiClient().parseStartResult(
            json = JSONObject()
                .put("session_id", "session-123")
                .put("day_context", dayContextJson())
                .put("input_mode", "rokid_frame_audio"),
            requestedSessionId = "",
            inputMode = "rokid_frame_audio",
            runId = "run-1",
        )

        assertEquals("rokid_frame_audio", result.inputMode)
        assertEquals("/rokid/session-123/frame", result.frameUploadPath)
        assertEquals("/rokid/session-123/audio_chunk", result.audioUploadPath)
        assertEquals("/rokid/session-123/status", result.statusPath)
        assertEquals("/rokid/session-123/audio_question", result.audioQuestionPath)
    }

    private fun dayContextJson(
        dayLabel: String = "DAY1",
        weekdayLabel: String = "Mon",
        displayDayLabel: String = "DAY1 Mon",
        dayIndex: Int = 1,
        runId: String = "run-1",
        relativeTsBaseMs: Long = 0L,
    ): JSONObject =
        JSONObject()
            .put("mode", "single_session")
            .put("day_label", dayLabel)
            .put("weekday_label", weekdayLabel)
            .put("display_day_label", displayDayLabel)
            .put("day_index", dayIndex)
            .put("run_id", runId)
            .put("relative_ts_base_ms", relativeTsBaseMs)
}
