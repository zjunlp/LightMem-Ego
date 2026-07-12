package cn.zjukg.lightmem.glass.lightmem_ego

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object WavEncoder {
    fun mono16PcmToWav(pcmBytes: ByteArray, sampleRate: Int = LightMemEgoConfig.AUDIO_SAMPLE_RATE): ByteArray {
        val out = ByteArrayOutputStream(44 + pcmBytes.size)
        val byteRate = sampleRate * 2
        out.writeAscii("RIFF")
        out.writeIntLe(36 + pcmBytes.size)
        out.writeAscii("WAVE")
        out.writeAscii("fmt ")
        out.writeIntLe(16)
        out.writeShortLe(1)
        out.writeShortLe(1)
        out.writeIntLe(sampleRate)
        out.writeIntLe(byteRate)
        out.writeShortLe(2)
        out.writeShortLe(16)
        out.writeAscii("data")
        out.writeIntLe(pcmBytes.size)
        out.write(pcmBytes)
        return out.toByteArray()
    }

    fun downmixRokidEightChannelToMono(input: ByteArray, bytesRead: Int): ByteArray {
        val frameBytes = LightMemEgoConfig.ROKID_CHANNEL_COUNT * 2
        val frames = bytesRead / frameBytes
        val output = ByteArray(frames * 2)
        var outIndex = 0
        for (frame in 0 until frames) {
            val base = frame * frameBytes
            val ch2 = readShortLe(input, base + 4).toInt()
            val ch3 = readShortLe(input, base + 6).toInt()
            val ch4 = readShortLe(input, base + 8).toInt()
            val ch5 = readShortLe(input, base + 10).toInt()
            val mixed = ((ch2 + ch3 + ch4 + ch5) / 4).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            output[outIndex++] = (mixed and 0xFF).toByte()
            output[outIndex++] = ((mixed ushr 8) and 0xFF).toByte()
        }
        return output
    }

    private fun readShortLe(bytes: ByteArray, offset: Int): Short =
        ByteBuffer.wrap(bytes, offset, 2).order(ByteOrder.LITTLE_ENDIAN).short

    private fun ByteArrayOutputStream.writeAscii(value: String) {
        write(value.toByteArray(Charsets.US_ASCII))
    }

    private fun ByteArrayOutputStream.writeIntLe(value: Int) {
        write(value and 0xFF)
        write((value ushr 8) and 0xFF)
        write((value ushr 16) and 0xFF)
        write((value ushr 24) and 0xFF)
    }

    private fun ByteArrayOutputStream.writeShortLe(value: Int) {
        write(value and 0xFF)
        write((value ushr 8) and 0xFF)
    }
}
