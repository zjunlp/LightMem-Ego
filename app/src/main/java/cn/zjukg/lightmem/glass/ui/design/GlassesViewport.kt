package cn.zjukg.lightmem.glass.ui.design

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.layout.layout
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Density
import cn.zjukg.lightmem.glass.app.CONSTANT
import cn.zjukg.lightmem.glass.ui.theme.PitchBlack

/**
 * Physical glasses display: 480x640 px. Background must be #FF000000 because black pixels do not emit light.
 * The content area stays fixed at this resolution; larger devices center it with pure black outside.
 */
@Composable
fun GlassesDisplayFrame(content: @Composable BoxScope.() -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(PitchBlack),
        contentAlignment = Alignment.Center,
    ) {
        CompositionLocalProvider(
            LocalDensity provides Density(density = 1f, fontScale = 1f),
        ) {
            Box(
                modifier = Modifier
                    .glassesScreenPx()
                    .background(PitchBlack)
                    .clipToBounds(),
                content = content,
            )
        }
    }
}

private fun Modifier.glassesScreenPx(): Modifier = layout { measurable, constraints ->
    val width = CONSTANT.SCREEN_WIDTH_PX
    val height = CONSTANT.SCREEN_HEIGHT_PX
    val placeable = measurable.measure(Constraints.fixed(width, height))
    layout(width, height) {
        placeable.place(0, 0)
    }
}
