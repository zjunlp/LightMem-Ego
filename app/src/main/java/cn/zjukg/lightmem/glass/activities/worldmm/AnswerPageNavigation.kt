package cn.zjukg.lightmem.glass.activities.worldmm

internal enum class AnswerPageMove {
    Previous,
    Next,
}

internal fun moveAnswerPageIndex(
    currentIndex: Int,
    pageCount: Int,
    move: AnswerPageMove,
): Int {
    if (pageCount <= 0) return 0
    val lastIndex = pageCount - 1
    val clampedIndex = currentIndex.coerceIn(0, lastIndex)
    return when (move) {
        AnswerPageMove.Previous -> (clampedIndex - 1).coerceAtLeast(0)
        AnswerPageMove.Next -> (clampedIndex + 1).coerceAtMost(lastIndex)
    }
}

internal fun answerPageIndexAfterSwipeForward(currentIndex: Int, pageCount: Int): Int =
    moveAnswerPageIndex(currentIndex, pageCount, AnswerPageMove.Previous)

internal fun answerPageIndexAfterSwipeBack(currentIndex: Int, pageCount: Int): Int =
    moveAnswerPageIndex(currentIndex, pageCount, AnswerPageMove.Next)

internal fun answerLabelFor(
    showingAnswer: Boolean,
    currentIndex: Int,
    pageCount: Int,
): String {
    if (!showingAnswer || pageCount <= 0) return "Answer"
    val visibleIndex = currentIndex.coerceIn(0, pageCount - 1)
    return "Answer ${visibleIndex + 1}/$pageCount"
}
