package cn.zjukg.lightmem.glass.activities.worldmm

import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight

internal data class MarkdownAnswerLine(
    val text: String,
    val spans: List<MarkdownAnswerSpan> = emptyList(),
)

internal data class MarkdownAnswerSpan(
    val start: Int,
    val end: Int,
    val style: MarkdownAnswerStyle,
)

internal enum class MarkdownAnswerStyle {
    Bold,
    Italic,
    Code,
}

internal fun String.toMarkdownAnswerPages(
    charsPerLine: Int = 43,
    linesPerPage: Int = 7,
): List<List<MarkdownAnswerLine>> {
    val displayLines = parseMarkdownAnswerLines()
        .flatMap { it.chunkForGlasses(charsPerLine) }

    return displayLines
        .chunked(linesPerPage)
}

internal fun MarkdownAnswerLine.toAnnotatedString(): AnnotatedString {
    val annotated = AnnotatedString.Builder(text)
    spans.forEach { span ->
        if (span.start < span.end && span.start >= 0 && span.end <= text.length) {
            annotated.addStyle(
                style = when (span.style) {
                    MarkdownAnswerStyle.Bold -> SpanStyle(fontWeight = FontWeight.Bold)
                    MarkdownAnswerStyle.Italic -> SpanStyle(fontStyle = FontStyle.Italic)
                    MarkdownAnswerStyle.Code -> SpanStyle(fontFamily = FontFamily.Monospace)
                },
                start = span.start,
                end = span.end,
            )
        }
    }
    return annotated.toAnnotatedString()
}

private fun String.parseMarkdownAnswerLines(): List<MarkdownAnswerLine> {
    val lines = mutableListOf<MarkdownAnswerLine>()
    var inCodeBlock = false

    replace("\r\n", "\n")
        .trim()
        .lineSequence()
        .forEach { rawLine ->
            val clean = rawLine.trim()
            if (clean.isBlank()) return@forEach
            if (clean.startsWith("```")) {
                inCodeBlock = !inCodeBlock
                return@forEach
            }

            lines += if (inCodeBlock) {
                MarkdownAnswerLine(
                    text = clean,
                    spans = listOf(MarkdownAnswerSpan(0, clean.length, MarkdownAnswerStyle.Code)),
                )
            } else {
                parseMarkdownBlockLine(clean)
            }
        }

    return lines
}

private fun parseMarkdownBlockLine(line: String): MarkdownAnswerLine {
    val heading = Regex("^#{1,6}\\s+(.+)$").matchEntire(line)
    if (heading != null) {
        return parseMarkdownInline(heading.groupValues[1]).withFullLineStyle(MarkdownAnswerStyle.Bold)
    }

    val unorderedList = Regex("^[-*+]\\s+(.+)$").matchEntire(line)
    if (unorderedList != null) {
        return parseMarkdownInline(unorderedList.groupValues[1]).withPrefix("- ")
    }

    val orderedList = Regex("^(\\d+)[.)]\\s+(.+)$").matchEntire(line)
    if (orderedList != null) {
        return parseMarkdownInline(orderedList.groupValues[2]).withPrefix("${orderedList.groupValues[1]}. ")
    }

    val quote = Regex("^>\\s+(.+)$").matchEntire(line)
    if (quote != null) {
        return parseMarkdownInline(quote.groupValues[1]).withPrefix("> ")
    }

    return parseMarkdownInline(line)
}

private fun parseMarkdownInline(text: String): MarkdownAnswerLine {
    val output = StringBuilder()
    val spans = mutableListOf<MarkdownAnswerSpan>()
    var index = 0

    while (index < text.length) {
        val match = nextMarkdownDelimiter(text, index)
        if (match == null) {
            output.append(text[index])
            index += 1
            continue
        }

        val closeIndex = text.indexOf(match.delimiter, startIndex = index + match.delimiter.length)
        if (closeIndex <= index + match.delimiter.length) {
            output.append(text[index])
            index += 1
            continue
        }

        val parsedInner = parseMarkdownInline(
            text.substring(index + match.delimiter.length, closeIndex),
        )
        val start = output.length
        output.append(parsedInner.text)
        parsedInner.spans.forEach { inner ->
            spans += inner.copy(start = inner.start + start, end = inner.end + start)
        }
        spans += MarkdownAnswerSpan(start, output.length, match.style)
        index = closeIndex + match.delimiter.length
    }

    return MarkdownAnswerLine(output.toString(), spans)
}

private data class MarkdownDelimiter(
    val delimiter: String,
    val style: MarkdownAnswerStyle,
)

private fun nextMarkdownDelimiter(text: String, index: Int): MarkdownDelimiter? =
    when {
        text.startsWith("**", index) -> MarkdownDelimiter("**", MarkdownAnswerStyle.Bold)
        text.startsWith("__", index) -> MarkdownDelimiter("__", MarkdownAnswerStyle.Bold)
        text.startsWith("`", index) -> MarkdownDelimiter("`", MarkdownAnswerStyle.Code)
        text.startsWith("*", index) -> MarkdownDelimiter("*", MarkdownAnswerStyle.Italic)
        text.startsWith("_", index) -> MarkdownDelimiter("_", MarkdownAnswerStyle.Italic)
        else -> null
    }

private fun MarkdownAnswerLine.withPrefix(prefix: String): MarkdownAnswerLine =
    copy(
        text = prefix + text,
        spans = spans.map { it.copy(start = it.start + prefix.length, end = it.end + prefix.length) },
    )

private fun MarkdownAnswerLine.withFullLineStyle(style: MarkdownAnswerStyle): MarkdownAnswerLine =
    copy(spans = spans + MarkdownAnswerSpan(0, text.length, style))

private fun MarkdownAnswerLine.chunkForGlasses(charsPerLine: Int): List<MarkdownAnswerLine> {
    if (text.length <= charsPerLine) return listOf(this)

    val chunks = mutableListOf<MarkdownAnswerLine>()
    var start = 0
    while (start < text.length) {
        val end = (start + charsPerLine).coerceAtMost(text.length)
        chunks += MarkdownAnswerLine(
            text = text.substring(start, end),
            spans = spans.mapNotNull { span ->
                val clippedStart = span.start.coerceAtLeast(start)
                val clippedEnd = span.end.coerceAtMost(end)
                if (clippedStart >= clippedEnd) {
                    null
                } else {
                    span.copy(start = clippedStart - start, end = clippedEnd - start)
                }
            },
        )
        start = end
    }
    return chunks
}
