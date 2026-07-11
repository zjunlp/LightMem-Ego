package cn.zjukg.lightmem.glass.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val BareColorScheme = darkColorScheme(
    primary = NeonGreen,
    onPrimary = PitchBlack,
    background = PitchBlack,
    onBackground = NeonGreen,
    surface = PitchBlack,
    onSurface = NeonGreen,
    outline = NeonGreen,
)

@Composable
fun LightMemGlassTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = BareColorScheme,
        content = content,
    )
}
