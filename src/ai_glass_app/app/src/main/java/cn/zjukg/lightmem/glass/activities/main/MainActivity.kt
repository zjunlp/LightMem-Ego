package cn.zjukg.lightmem.glass.activities.main

import android.os.Bundle
import android.view.KeyEvent
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import cn.zjukg.lightmem.glass.activities.lightmem_ego.LightMemEgoGlassScreen
import cn.zjukg.lightmem.glass.activities.lightmem_ego.LightMemEgoGlassViewModel
import cn.zjukg.lightmem.glass.input.BareGlassesInputDispatcher
import cn.zjukg.lightmem.glass.input.LocalBareGlassesInputDispatcher
import cn.zjukg.lightmem.glass.input.rememberBareGlassesInputDispatcher
import cn.zjukg.lightmem.glass.ui.design.GlassesDisplayFrame
import cn.zjukg.lightmem.glass.ui.theme.LightMemGlassTheme
import cn.zjukg.lightmem.glass.ui.theme.PitchBlack
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoDiagnostics

class MainActivity : ComponentActivity() {
    private val lightMemEgoGlassViewModel by viewModels<LightMemEgoGlassViewModel>()

    private var keyDispatcher: BareGlassesInputDispatcher? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        LightMemEgoDiagnostics.recordStartup(this, "MainActivity.onCreate")
        setupFullscreen()
        setContent {
            val context = LocalContext.current
            val dispatcher = rememberBareGlassesInputDispatcher(context)
            remember { keyDispatcher = dispatcher }
            LightMemGlassTheme {
                CompositionLocalProvider(LocalBareGlassesInputDispatcher provides dispatcher) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(PitchBlack),
                    ) {
                        GlassesDisplayFrame {
                            BareNavApp(
                                lightMemEgoGlassViewModel = lightMemEgoGlassViewModel,
                                onExit = { finish() },
                            )
                        }
                    }
                }
            }
        }
    }

    override fun onStart() {
        super.onStart()
        LightMemEgoDiagnostics.logLifecycle(this, "MainActivity", "onStart")
    }

    override fun onResume() {
        super.onResume()
        LightMemEgoDiagnostics.logLifecycle(this, "MainActivity", "onResume")
        LightMemEgoDiagnostics.logView(this, "MainActivity", "onResume", window.decorView)
    }

    override fun onPause() {
        LightMemEgoDiagnostics.logLifecycle(this, "MainActivity", "onPause")
        super.onPause()
    }

    override fun onStop() {
        LightMemEgoDiagnostics.logLifecycle(this, "MainActivity", "onStop")
        LightMemEgoDiagnostics.recordMemory(this, "MainActivity.onStop")
        super.onStop()
    }

    override fun onTrimMemory(level: Int) {
        LightMemEgoDiagnostics.logLifecycle(this, "MainActivity", "onTrimMemory", "level=$level")
        LightMemEgoDiagnostics.recordMemory(this, "trim-$level")
        super.onTrimMemory(level)
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        LightMemEgoDiagnostics.logView(this, "MainActivity", "windowFocus=$hasFocus", window.decorView)
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_ENTER -> return true
            KeyEvent.KEYCODE_BACK -> {
                keyDispatcher?.dispatchBackKey()
                return true
            }
            KeyEvent.KEYCODE_PROG_BLUE -> {
                if (event?.repeatCount == 0) {
                    keyDispatcher?.dispatchLongKey()
                } else {
                    keyDispatcher?.consumeSystemKey("Key-PROG_BLUE-repeat")
                }
                return true
            }
            KeyEvent.KEYCODE_SETTINGS -> {
                if (event?.repeatCount == 0) {
                    keyDispatcher?.dispatchTwoFingerLongPressKey("Key-SETTINGS")
                } else {
                    keyDispatcher?.consumeSystemKey("Key-SETTINGS-repeat")
                }
                return true
            }
            KeyEvent.KEYCODE_DPAD_LEFT,
            KeyEvent.KEYCODE_DPAD_UP -> {
                if (event?.repeatCount == 0) {
                    keyDispatcher?.dispatchSwipeForwardKey("Key-DPAD-PREV")
                } else {
                    keyDispatcher?.consumeSystemKey("Key-DPAD-PREV-repeat")
                }
                return true
            }
            KeyEvent.KEYCODE_DPAD_RIGHT,
            KeyEvent.KEYCODE_DPAD_DOWN -> {
                if (event?.repeatCount == 0) {
                    keyDispatcher?.dispatchSwipeBackKey("Key-DPAD-NEXT")
                } else {
                    keyDispatcher?.consumeSystemKey("Key-DPAD-NEXT-repeat")
                }
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_ENTER && event != null && event.repeatCount == 0) {
            keyDispatcher?.dispatchEnterKey()
            return true
        }
        return super.onKeyUp(keyCode, event)
    }

    override fun onDestroy() {
        LightMemEgoDiagnostics.logLifecycle(this, "MainActivity", "onDestroy", "finishing=$isFinishing")
        keyDispatcher?.unregister(applicationContext)
        keyDispatcher = null
        super.onDestroy()
    }

    private fun setupFullscreen() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        WindowInsetsControllerCompat(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }
}

@Composable
private fun BareNavApp(
    lightMemEgoGlassViewModel: LightMemEgoGlassViewModel,
    onExit: () -> Unit,
) {
    LightMemEgoGlassScreen(
        onBack = onExit,
        viewModel = lightMemEgoGlassViewModel,
    )
}
