package cn.zjukg.lightmem.glass.activities.lightmem_ego

import org.junit.Assert.assertEquals
import org.junit.Test

class AnswerPageNavigationTest {
    @Test
    fun swipeForwardDoesNotWrapFirstPageToLastPage() {
        assertEquals(0, answerPageIndexAfterSwipeForward(currentIndex = 0, pageCount = 8))
    }

    @Test
    fun swipeBackAdvancesFromFirstPageToSecondPage() {
        assertEquals(1, answerPageIndexAfterSwipeBack(currentIndex = 0, pageCount = 8))
    }

    @Test
    fun swipeBackDoesNotWrapLastPageToFirstPage() {
        assertEquals(7, answerPageIndexAfterSwipeBack(currentIndex = 7, pageCount = 8))
    }

    @Test
    fun swipeForwardMovesLastPageToPreviousPage() {
        assertEquals(6, answerPageIndexAfterSwipeForward(currentIndex = 7, pageCount = 8))
    }

    @Test
    fun emptyPageListFallsBackToFirstPageIndex() {
        assertEquals(0, answerPageIndexAfterSwipeBack(currentIndex = 3, pageCount = 0))
        assertEquals(0, answerPageIndexAfterSwipeForward(currentIndex = 3, pageCount = 0))
    }

    @Test
    fun answerLabelShowsCurrentAndTotalForSinglePageAnswer() {
        assertEquals("Answer 1/1", answerLabelFor(showingAnswer = true, currentIndex = 0, pageCount = 1))
    }

    @Test
    fun answerLabelShowsClampedCurrentAndTotalForMultiPageAnswer() {
        assertEquals("Answer 3/3", answerLabelFor(showingAnswer = true, currentIndex = 9, pageCount = 3))
    }

    @Test
    fun answerLabelFallsBackWhenAnswerIsNotShowing() {
        assertEquals("Answer", answerLabelFor(showingAnswer = false, currentIndex = 0, pageCount = 3))
        assertEquals("Answer", answerLabelFor(showingAnswer = true, currentIndex = 0, pageCount = 0))
    }
}
