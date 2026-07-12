package cn.zjukg.lightmem.glass.lightmem_ego

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

data class EncodedJpegFrame(
    val bytes: ByteArray,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
)

object ImageProxyJpegConverter {
    fun toJpeg(image: ImageProxy, quality: Int = LightMemEgoConfig.JPEG_QUALITY): ByteArray? =
        toJpegFrame(image, quality)?.bytes

    fun toJpegFrame(image: ImageProxy, quality: Int = LightMemEgoConfig.JPEG_QUALITY): EncodedJpegFrame? {
        if (image.format != ImageFormat.YUV_420_888) return null
        val nv21 = yuv420ToNv21(image)
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        val ok = yuv.compressToJpeg(Rect(0, 0, image.width, image.height), quality, out)
        if (!ok) return null

        val rotationDegrees = image.imageInfo.rotationDegrees.normalizedRightAngle()
        val jpegBytes = out.toByteArray()
        if (rotationDegrees == 0) {
            return EncodedJpegFrame(
                bytes = jpegBytes,
                width = image.width,
                height = image.height,
                rotationDegrees = rotationDegrees,
            )
        }

        return rotateJpeg(jpegBytes, rotationDegrees, quality)
    }

    private fun rotateJpeg(jpegBytes: ByteArray, rotationDegrees: Int, quality: Int): EncodedJpegFrame? {
        val source = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        var rotated: Bitmap? = null
        return try {
            rotated = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
            val output = ByteArrayOutputStream()
            if (!rotated.compress(Bitmap.CompressFormat.JPEG, quality, output)) return null
            EncodedJpegFrame(
                bytes = output.toByteArray(),
                width = rotated.width,
                height = rotated.height,
                rotationDegrees = rotationDegrees,
            )
        } finally {
            rotated?.recycle()
            source.recycle()
        }
    }

    private fun Int.normalizedRightAngle(): Int =
        (((this % 360) + 360) % 360).let { if (it in setOf(0, 90, 180, 270)) it else 0 }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val nv21 = ByteArray(ySize + ySize / 2)
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        var outputOffset = 0
        for (row in 0 until height) {
            val rowStart = row * yPlane.rowStride
            yBuffer.position(rowStart)
            yBuffer.get(nv21, outputOffset, width)
            outputOffset += width
        }

        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        var chromaOffset = ySize
        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val vIndex = row * vPlane.rowStride + col * vPlane.pixelStride
                val uIndex = row * uPlane.rowStride + col * uPlane.pixelStride
                nv21[chromaOffset++] = vBuffer.get(vIndex)
                nv21[chromaOffset++] = uBuffer.get(uIndex)
            }
        }
        return nv21
    }
}
