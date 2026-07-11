package cn.zjukg.lightmem.glass.input

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember

/** Ignores duplicate [BareKeyEvent.DoubleClick] events shortly after entering a child page from the hub. */
@Composable
fun rememberSubPageEnterDebounce(windowMs: Long = 400L): () -> Boolean {
    val enteredAt = remember { System.currentTimeMillis() }
    return remember(windowMs) {
        { System.currentTimeMillis() - enteredAt < windowMs }
    }
}
