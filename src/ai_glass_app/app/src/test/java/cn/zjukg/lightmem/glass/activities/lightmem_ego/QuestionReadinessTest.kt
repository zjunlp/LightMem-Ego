package cn.zjukg.lightmem.glass.activities.lightmem_ego

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QuestionReadinessTest {
    @Test
    fun firstUploadedFrameAllowsQuestionBeforeBackendReportsReady() {
        assertTrue(
            isQuestionReadyAfterUpload(
                backendCanAsk = false,
                frameUploadedCount = 1,
                audioUploadedCount = 0,
            ),
        )
    }

    @Test
    fun firstUploadedAudioChunkAllowsQuestionBeforeBackendReportsReady() {
        assertTrue(
            isQuestionReadyAfterUpload(
                backendCanAsk = false,
                frameUploadedCount = 0,
                audioUploadedCount = 1,
            ),
        )
    }

    @Test
    fun noSuccessfulUploadKeepsQuestionBlocked() {
        assertFalse(
            isQuestionReadyAfterUpload(
                backendCanAsk = false,
                frameUploadedCount = 0,
                audioUploadedCount = 0,
            ),
        )
    }
}
