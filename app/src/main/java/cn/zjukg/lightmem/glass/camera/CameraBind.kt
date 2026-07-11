package cn.zjukg.lightmem.glass.camera

import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.UseCase
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner

private const val TAG = "BareCameraBind"

/**
 * Asynchronously binds CameraX. This matches CXRSSDKSamples by binding business UseCases only, with no Preview.
 */
@Composable
fun rememberCameraBound(
    context: Context,
    lifecycleOwner: LifecycleOwner,
    enabled: Boolean,
    onReady: () -> Unit,
    onError: (String) -> Unit,
    onUnbind: () -> Unit = {},
    onBound: (Array<out UseCase>) -> Unit = {},
    useCases: () -> Array<out UseCase>,
): Boolean {
    var ready by remember { mutableStateOf(false) }

    DisposableEffect(enabled, lifecycleOwner) {
        if (!enabled) {
            ready = false
            return@DisposableEffect onDispose { }
        }
        ready = false
        var provider: ProcessCameraProvider? = null
        var cancelled = false
        val mainExecutor = ContextCompat.getMainExecutor(context)
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener(
            {
                if (cancelled) return@addListener
                try {
                    val p = future.get()
                    provider = p
                    val extras = useCases()
                    p.unbindAll()
                    p.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        *extras,
                    )
                    onBound(extras)
                    ready = true
                    onReady()
                    Log.d(TAG, "Camera bound: ${extras.map { it.javaClass.simpleName }}")
                } catch (e: Exception) {
                    Log.e(TAG, "Camera bind failed", e)
                    onError(e.message ?: "相机初始化失败")
                }
            },
            mainExecutor,
        )
        onDispose {
            cancelled = true
            ready = false
            provider?.unbindAll()
            onUnbind()
        }
    }

    return ready
}
