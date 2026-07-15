package cn.zjukg.lightmem.glass.ui.design

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import cn.zjukg.lightmem.glass.ui.theme.NeonGreen

data class BareKeyGuide(
    val click: String? = null,
    val spriteClick: String? = null,
    val doubleClick: String? = null,
    val longPress: String? = null,
    val twoFingerClick: String? = null,
    val twoFingerDoubleClick: String? = null,
    val twoFingerLongPress: String? = null,
    val swipeForward: String? = null,
    val swipeBack: String? = null,
) {
    companion object {
        val Hub = BareKeyGuide(
            click = "Next",
            doubleClick = "Open",
        )
    }
}

@Composable
fun BareKeyLegendBar(
    guide: BareKeyGuide,
    modifier: Modifier = Modifier,
) {
    val mainRows = buildList {
        guide.click?.let { add(KeyRowKind.Click to it) }
        guide.twoFingerLongPress?.let { add(KeyRowKind.TwoFingerLongPress to it) }
        guide.twoFingerClick?.let { add(KeyRowKind.TwoFingerClick to it) }
        guide.twoFingerDoubleClick?.let { add(KeyRowKind.TwoFingerDoubleClick to it) }
        guide.swipeForward?.let { add(KeyRowKind.SwipeForward to it) }
        guide.swipeBack?.let { add(KeyRowKind.SwipeBack to it) }
        guide.doubleClick?.let { add(KeyRowKind.DoubleClick to it) }
        guide.longPress?.let { add(KeyRowKind.LongPress to it) }
    }
    val spriteRow = guide.spriteClick?.let { KeyRowKind.SpriteClick to it }

    val totalItems = mainRows.size + (if (spriteRow != null) 1 else 0)
    if (totalItems == 0) return

    val rowH = if (totalItems >= 4) 27.dp else 30.dp
    val mainDisplayRows = (mainRows.size + 1) / 2
    val spriteDisplayRows = if (spriteRow != null) 1 else 0
    val displayRows = mainDisplayRows + spriteDisplayRows
    val totalH = (4 + displayRows * rowH.value).dp.coerceAtMost(BareTokens.LegendH)
    Column(
        modifier = modifier
            .fillMaxWidth()
            .height(totalH),
    ) {
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp),
        ) {
            drawLine(
                color = NeonGreen,
                start = Offset(0f, 0f),
                end = Offset(size.width, 0f),
                strokeWidth = BareTokens.STROKE_THIN,
            )
        }
        mainRows.chunked(2).forEach { rowItems ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(rowH),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                rowItems.forEach { (kind, action) ->
                    BareKeyLegendRow(
                        kind = kind,
                        action = action,
                        modifier = Modifier.weight(1f),
                    )
                }
                if (rowItems.size == 1) {
                    androidx.compose.foundation.layout.Spacer(modifier = Modifier.weight(1f))
                }
            }
        }
        if (spriteRow != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(rowH),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                BareKeyLegendRow(
                    kind = spriteRow.first,
                    action = spriteRow.second,
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }
}

private enum class KeyRowKind(val badge: String) {
    Click("Click"),
    SpriteClick("Button click"),
    TwoFingerClick("2F click"),
    TwoFingerDoubleClick("2F double"),
    TwoFingerLongPress("2F hold"),
    SwipeForward("Forward"),
    SwipeBack("Back"),
    DoubleClick("Double"),
    LongPress("Hold"),
}

@Composable
private fun BareKeyLegendRow(kind: KeyRowKind, action: String, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        BareKeyBadge(kind = kind)
        Text(
            text = action,
            color = NeonGreen,
            fontSize = BareTokens.LegendSp,
            lineHeight = BareTokens.LegendSp * 1.15f,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier
                .padding(start = 4.dp)
                .weight(1f),
        )
    }
}

@Composable
private fun BareKeyBadge(kind: KeyRowKind) {
    val width = when (kind) {
        KeyRowKind.SpriteClick -> 98.dp
        else -> 78.dp
    }
    Text(
        text = kind.badge,
        color = NeonGreen,
        fontSize = BareTokens.CaptionSp,
        fontWeight = FontWeight.Medium,
        lineHeight = BareTokens.CaptionSp * 1.15f,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
        modifier = Modifier.width(width),
    )
}
