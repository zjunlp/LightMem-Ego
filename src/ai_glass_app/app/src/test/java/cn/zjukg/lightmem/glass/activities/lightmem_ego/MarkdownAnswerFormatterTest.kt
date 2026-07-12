package cn.zjukg.lightmem.glass.activities.lightmem_ego

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MarkdownAnswerFormatterTest {
    @Test
    fun removesInlineMarkersAndKeepsBoldSpan() {
        val pages = "**Important** text".toMarkdownAnswerPages(charsPerLine = 21, linesPerPage = 3)

        assertEquals("Important text", pages.single().single().text)
        assertEquals(MarkdownAnswerSpan(0, 9, MarkdownAnswerStyle.Bold), pages.single().single().spans.single())
    }

    @Test
    fun convertsMarkdownListsToReadableLines() {
        val pages = """
            - first item
            2. second item
        """.trimIndent().toMarkdownAnswerPages(charsPerLine = 21, linesPerPage = 3)

        assertEquals(listOf("- first item", "2. second item"), pages.single().map { it.text })
    }

    @Test
    fun chunksLongStyledLineWithoutLeakingMarkerCharacters() {
        val pages = "**abcdef** ghij".toMarkdownAnswerPages(charsPerLine = 4, linesPerPage = 3)
        val flattened = pages.flatten()

        assertEquals(listOf("abcd", "ef g", "hij"), flattened.map { it.text })
        assertTrue(flattened[0].spans.contains(MarkdownAnswerSpan(0, 4, MarkdownAnswerStyle.Bold)))
        assertTrue(flattened[1].spans.contains(MarkdownAnswerSpan(0, 2, MarkdownAnswerStyle.Bold)))
    }

    @Test
    fun defaultLineWidthUsesMostOfGlassesScreen() {
        val line = "abcdefghijklmnopqrstuvwxyz1234567890ABCDEF"
        val pages = line.toMarkdownAnswerPages(linesPerPage = 3)

        assertEquals(line, pages.single().single().text)
    }

    @Test
    fun headingBecomesBoldPlainText() {
        val pages = "# Title".toMarkdownAnswerPages(charsPerLine = 21, linesPerPage = 3)

        assertEquals("Title", pages.single().single().text)
        assertEquals(MarkdownAnswerSpan(0, 5, MarkdownAnswerStyle.Bold), pages.single().single().spans.single())
    }

    @Test
    fun demoFirstAnswerSpillsOntoSecondDisplayedPage() {
        val answer = "Yesterday, you mainly answered Ethan's call at your office workstation. " +
            "In the call, he asked you to prepare a brief progress report for the team meeting at " +
            "\"10 o'clock this morning\", focusing on the progress of API integration and the current " +
            "blockers encountered; you promised to prepare it and attend on time, and then both of " +
            "you said goodbye."

        val pages = answer.toMarkdownAnswerPages()

        assertEquals(2, pages.size)
        assertEquals(7, pages.first().size)
        assertEquals(answer, pages.flatten().joinToString("") { it.text })
    }
}
