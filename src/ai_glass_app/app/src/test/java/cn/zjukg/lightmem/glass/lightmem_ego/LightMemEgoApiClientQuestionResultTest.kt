package cn.zjukg.lightmem.glass.lightmem_ego

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LightMemEgoApiClientQuestionResultTest {
    @Test
    fun audioQuestionTextIsParsedAsQuestionNotAnswer() {
        val result = LightMemEgoApiClient().parseAudioQuestionSubmitResult(
            JSONObject()
                .put("status", "queued")
                .put("task_id", "task-1")
                .put("text", "What did they just say?"),
        )

        assertEquals("What did they just say?", result.question)
        assertEquals("", result.answer)
        assertTrue(result.queued)
    }

    @Test
    fun nestedTextIsNotTreatedAsTaskAnswer() {
        val result = LightMemEgoApiClient().parseQueryTaskResult(
            JSONObject()
                .put("status", "done")
                .put(
                    "result",
                    JSONObject()
                        .put("text", "What is in the current scene?"),
                ),
        )

        assertEquals("", result.answer)
    }

    @Test
    fun topLevelAnswerIsAccepted() {
        val result = LightMemEgoApiClient().parseQueryTaskResult(
            JSONObject()
                .put("status", "done")
                .put("answer", "There is a laptop on the desk."),
        )

        assertEquals("There is a laptop on the desk.", result.answer)
    }

    @Test
    fun nestedResultAnswerIsAccepted() {
        val result = LightMemEgoApiClient().parseQueryTaskResult(
            JSONObject()
                .put("status", "done")
                .put(
                    "result",
                    JSONObject()
                        .put("answer", "There is a bottle on the table."),
                ),
        )

        assertEquals("There is a bottle on the table.", result.answer)
    }

    @Test
    fun legacyResponseFieldIsNotTreatedAsAnswer() {
        val result = LightMemEgoApiClient().parseQueryTaskResult(
            JSONObject()
                .put("status", "done")
                .put("response", "This should not be used as an answer."),
        )

        assertEquals("", result.answer)
    }
}
