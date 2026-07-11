package cn.zjukg.lightmem.glass.ui.design

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import cn.zjukg.lightmem.glass.app.CONSTANT
import cn.zjukg.lightmem.glass.ui.theme.NeonGreen
import cn.zjukg.lightmem.glass.ui.theme.PitchBlack

/**
 * Full-screen 480x640 layout. Main content and wireframes stay in the safe area y=80..560,
 * with structured key guidance at the bottom.
 */
@Composable
fun BareScreenLayout(
    title: String,
    keyGuide: BareKeyGuide,
    subtitle: String? = null,
    drawSafeAreaFrame: Boolean = true,
    body: @Composable ColumnScope.() -> Unit,
) {
    val topPad = CONSTANT.SAFE_AREA_TOP_PX.dp
    val bottomPad = (CONSTANT.SCREEN_HEIGHT_PX - CONSTANT.SAFE_AREA_BOTTOM_PX).dp
    val screenPadV = 2.dp

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(PitchBlack),
    ) {
        if (drawSafeAreaFrame) {
            SafeAreaFrame(modifier = Modifier.fillMaxSize())
        }
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(
                    start = BareTokens.ScreenPadH,
                    end = BareTokens.ScreenPadH,
                    top = topPad + screenPadV,
                    bottom = bottomPad + screenPadV,
                ),
        ) {
            BareScreenHeader(
                title = title,
                subtitle = subtitle,
            )
            BareContentPanel(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                content = body,
            )
            BareKeyLegendBar(guide = keyGuide)
        }
    }
}

@Composable
fun BareDevScaffold(
    title: String,
    keyHint: String,
    subtitle: String? = null,
    drawSafeArea: Boolean = true,
    content: @Composable ColumnScope.() -> Unit,
) {
    BareScreenLayout(
        title = title,
        subtitle = subtitle,
        keyGuide = parseLegacyKeyHint(keyHint),
        drawSafeAreaFrame = drawSafeArea,
        body = content,
    )
}

private fun parseLegacyKeyHint(hint: String): BareKeyGuide {
    var click: String? = null
    var doubleClick: String? = null
    var longPress: String? = null
    hint.split("·").map { it.trim() }.forEach { part ->
        when {
            part.startsWith("单击") ->
                click = part.removePrefix("单击").removePrefix("：").trim().ifEmpty { "确认" }
            part.startsWith("双击") ->
                doubleClick = part.removePrefix("双击").removePrefix("：").trim().ifEmpty { "返回" }
            part.startsWith("长按") ->
                longPress = part.removePrefix("长按").removePrefix("：").trim()
        }
    }
    if (click == null && doubleClick == null && longPress == null && hint.isNotBlank()) {
        click = hint
    }
    return BareKeyGuide(click = click, doubleClick = doubleClick, longPress = longPress)
}

@Composable
private fun BareScreenHeader(
    title: String,
    subtitle: String?,
) {
    val headerHeight = if (subtitle.isNullOrBlank()) BareTokens.HeaderH else 68.dp
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(headerHeight),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = title,
                color = NeonGreen,
                fontSize = BareTokens.TitleSp,
                fontWeight = FontWeight.SemiBold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )
            subtitle?.let {
                Text(
                    text = it,
                    color = NeonGreen.copy(alpha = 0.75f),
                    fontSize = BareTokens.SubtitleSp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

@Composable
fun BareContentPanel(
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit,
) {
    Box(modifier = modifier.padding(vertical = 2.dp)) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(BareTokens.PanelPad),
            verticalArrangement = Arrangement.spacedBy(BareTokens.LineGap),
            content = content,
        )
    }
}

@Composable
fun SafeAreaFrame(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val top = CONSTANT.SAFE_AREA_TOP_PX.toFloat()
        val bottom = CONSTANT.SAFE_AREA_BOTTOM_PX.toFloat()
        drawRect(
            color = NeonGreen.copy(alpha = 0.45f),
            style = Stroke(width = BareTokens.STROKE_THIN),
            topLeft = Offset(0.5f, top),
            size = Size(size.width - 1f, bottom - top),
        )
    }
}

@Composable
fun BareHeroText(
    text: String,
    modifier: Modifier = Modifier,
    hint: String? = null,
) {
    Column(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = text,
            color = NeonGreen,
            fontSize = BareTokens.HeroSp,
            fontWeight = FontWeight.SemiBold,
            lineHeight = BareTokens.HeroSp * 1.2f,
            textAlign = TextAlign.Center,
            maxLines = 3,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.fillMaxWidth(),
        )
        hint?.let {
            Text(
                text = it,
                color = NeonGreen.copy(alpha = 0.7f),
                fontSize = BareTokens.BodySp,
                lineHeight = BareTokens.BodySp * 1.2f,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 6.dp),
            )
        }
    }
}

@Composable
fun BareInfoBlock(
    label: String,
    lines: List<String>,
    modifier: Modifier = Modifier,
    maxLineCount: Int = 3,
    maxLinesPerItem: Int = 2,
) {
    Column(modifier = modifier.fillMaxWidth()) {
        if (label.isNotBlank()) {
            Text(
                text = label,
                color = NeonGreen,
                fontSize = BareTokens.BodySp,
                fontWeight = FontWeight.Medium,
                lineHeight = BareTokens.BodySp * 1.2f,
                modifier = Modifier.fillMaxWidth(),
            )
            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 2.dp)
                    .height(1.dp),
            ) {
                drawLine(
                    NeonGreen.copy(alpha = 0.5f),
                    Offset(0f, 0f),
                    Offset(size.width, 0f),
                    strokeWidth = BareTokens.STROKE_THIN,
                )
            }
        }
        lines.filter { it.isNotBlank() }.take(maxLineCount).forEach { line ->
            Text(
                text = line,
                color = NeonGreen,
                fontSize = BareTokens.BodySp,
                maxLines = maxLinesPerItem,
                overflow = TextOverflow.Ellipsis,
                lineHeight = BareTokens.BodySp * 1.2f,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
fun BareRichInfoBlock(
    label: String,
    lines: List<AnnotatedString>,
    modifier: Modifier = Modifier,
    maxLineCount: Int = 3,
    maxLinesPerItem: Int = 2,
) {
    Column(modifier = modifier.fillMaxWidth()) {
        if (label.isNotBlank()) {
            Text(
                text = label,
                color = NeonGreen,
                fontSize = BareTokens.BodySp,
                fontWeight = FontWeight.Medium,
                lineHeight = BareTokens.BodySp * 1.2f,
                modifier = Modifier.fillMaxWidth(),
            )
            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 2.dp)
                    .height(1.dp),
            ) {
                drawLine(
                    NeonGreen.copy(alpha = 0.5f),
                    Offset(0f, 0f),
                    Offset(size.width, 0f),
                    strokeWidth = BareTokens.STROKE_THIN,
                )
            }
        }
        lines.filter { it.text.isNotBlank() }.take(maxLineCount).forEach { line ->
            Text(
                text = line,
                color = NeonGreen,
                fontSize = BareTokens.BodySp,
                maxLines = maxLinesPerItem,
                overflow = TextOverflow.Ellipsis,
                lineHeight = BareTokens.BodySp * 1.2f,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
fun BarePagedViewport(
    pageIndex: Int,
    pageCount: Int,
    modifier: Modifier = Modifier,
    content: @Composable (pageIndex: Int) -> Unit,
) {
    Box(modifier = modifier.fillMaxSize()) {
        content(pageIndex.coerceIn(0, (pageCount - 1).coerceAtLeast(0)))
    }
}

/** @deprecated Use [BareHeroText] / [BareInfoBlock]. */
@Composable
fun BareActionHint(
    label: String,
    modifier: Modifier = Modifier,
    emphasized: Boolean = false,
) {
    BareHeroText(text = label, modifier = modifier, hint = if (emphasized) null else null)
}

/** @deprecated Use [BareInfoBlock]. */
@Composable
fun BareStatusCard(title: String, lines: List<String>, modifier: Modifier = Modifier) {
    BareInfoBlock(label = title, lines = lines, modifier = modifier)
}
