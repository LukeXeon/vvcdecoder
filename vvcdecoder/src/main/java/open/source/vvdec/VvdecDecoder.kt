package open.source.vvdec

import android.os.Handler
import android.os.HandlerThread
import android.os.Message
import android.view.Surface
import androidx.annotation.WorkerThread
import androidx.collection.ArrayMap
import androidx.core.util.Pools
import com.google.android.exoplayer2.C
import com.google.android.exoplayer2.C.VideoOutputMode
import com.google.android.exoplayer2.decoder.Decoder
import com.google.android.exoplayer2.decoder.DecoderInputBuffer
import com.google.android.exoplayer2.decoder.DecoderOutputBuffer
import com.google.android.exoplayer2.decoder.VideoDecoderOutputBuffer
import java.lang.ref.WeakReference
import java.nio.ByteBuffer
import java.util.*


@Suppress("UNCHECKED_CAST")
class VvdecDecoder(
    availableInputBufferCount: Int,
    availableOutputBufferCount: Int,
) : Decoder<DecoderInputBuffer, VideoDecoderOutputBuffer, VvdecDecoderException>,
    DecoderOutputBuffer.Owner<VideoDecoderOutputBuffer> {

    private class DecodeThreadHandler(decoder: VvdecDecoder) : Handler(
        HandlerThread("${decoder.name}-${System.identityHashCode(this)}")
            .apply { start() }.looper
    ) {

        private val reference = WeakReference(decoder)

        override fun handleMessage(msg: Message) {
            val decoder = reference.get()
            if (decoder != null) {
                when (msg.what) {
                    MSG_NATIVE_DECODE_FRAME -> decoder.decodeFrame(msg.obj as DecoderInputBuffer)
                    MSG_NATIVE_FLUSH_FRAME -> decoder.flushFrame()
                    MSG_NATIVE_RESET_DECODER -> decoder.resetDecoder()
                }
            } else {
                looper.quit()
            }

        }
    }

    @Volatile
    internal var outputMode: @VideoOutputMode Int = 0
    private val nativeInstance = nInit(Runtime.getRuntime().availableProcessors())
    private val dequeueInputBuffers = Collections.synchronizedMap(
        ArrayMap<DecoderInputBuffer, Boolean>(
            availableInputBufferCount
        )
    )
    private val availableInputBuffers = Pools.SynchronizedPool<DecoderInputBuffer>(
        availableInputBufferCount
    ).apply {
        repeat(availableInputBufferCount) {
            release(DecoderInputBuffer(DecoderInputBuffer.BUFFER_REPLACEMENT_MODE_DIRECT))
        }
    }
    private val availableOutputBuffers = Pools.SynchronizedPool<VideoDecoderOutputBuffer>(
        availableOutputBufferCount
    ).apply {
        repeat(availableOutputBufferCount) {
            release(VideoDecoderOutputBuffer(this@VvdecDecoder))
        }
    }
    private val decodeThread by lazy { DecodeThreadHandler(this) }

    override fun getName(): String {
        return "libvvdec-${nGetVersion()}"
    }

    override fun dequeueInputBuffer(): DecoderInputBuffer? {
        val buffer = availableInputBuffers.acquire() ?: return null
        dequeueInputBuffers[buffer] = false
        return buffer
    }

    override fun queueInputBuffer(inputBuffer: DecoderInputBuffer) {
        require(dequeueInputBuffers.put(inputBuffer, true) == false)
        decodeThread.obtainMessage(MSG_NATIVE_DECODE_FRAME, inputBuffer).sendToTarget()
    }

    override fun dequeueOutputBuffer(): VideoDecoderOutputBuffer? {
        val buffer = availableOutputBuffers.acquire() ?: return null
        if (nDequeueOutputBuffer(nativeInstance, buffer) == 0) {
            return buffer
        }
        availableOutputBuffers.release(buffer)
        return null
    }

    override fun flush() {
        dequeueInputBuffers.clear()
        decodeThread.sendEmptyMessageAtTime(MSG_NATIVE_RESET_DECODER, 0)
    }

    override fun releaseOutputBuffer(outputBuffer: VideoDecoderOutputBuffer) {
        if (outputMode == C.VIDEO_OUTPUT_MODE_SURFACE_YUV && !outputBuffer.isDecodeOnly) {
            nReleaseFrame(nativeInstance, outputBuffer.decoderPrivate)
        }
        outputBuffer.clear()
        availableOutputBuffers.release(outputBuffer)
    }

    fun renderToSurface(buffer: VideoDecoderOutputBuffer, surface: Surface) {
        nRenderFrame(nativeInstance, surface, buffer.decoderPrivate)
    }

    override fun release() {
        nRelease(nativeInstance)
    }

    @WorkerThread
    private fun resetDecoder() {
        decodeThread.removeCallbacksAndMessages(null)
        nReset(nativeInstance)
    }

    @WorkerThread
    private fun flushFrame() {
        val ret = nFlushFrame(nativeInstance)
        if (ret != -50) {
            decodeThread.sendEmptyMessage(MSG_NATIVE_FLUSH_FRAME)
        }
    }

    @WorkerThread
    private fun decodeFrame(inputBuffer: DecoderInputBuffer) {
        val prepared = dequeueInputBuffers[inputBuffer]
        val inputData = inputBuffer.data
        if (prepared == true && inputData != null) {
            if (inputBuffer.isEndOfStream) {
                flushFrame()
            } else {
                nDecodeFrame(nativeInstance, inputData, inputData.limit())
            }
        }
        inputBuffer.clear()
        dequeueInputBuffers.remove(inputBuffer)
        availableInputBuffers.release(inputBuffer)
    }

    private companion object {

        private const val MSG_NATIVE_DECODE_FRAME = 1
        private const val MSG_NATIVE_FLUSH_FRAME = 2
        private const val MSG_NATIVE_RESET_DECODER = 3

        init {
            System.loadLibrary("vvcdecoder_jni")
        }

        @JvmStatic
        private external fun nGetVersion(): String

        @JvmStatic
        private external fun nGetLastError(nativeInstance: Long): String

        @JvmStatic
        external fun nInit(threads: Int): Long

        @JvmStatic
        external fun nRelease(nativeInstance: Long)

        @JvmStatic
        private external fun nReleaseFrame(
            nativeInstance: Long,
            frameId: Int,
        ): Int

        @JvmStatic
        external fun nRenderFrame(
            nativeInstance: Long,
            surface: Surface,
            frameId: Int,
        ): Int

        @JvmStatic
        external fun nDecodeFrame(
            nativeInstance: Long,
            inputData: ByteBuffer?,
            length: Int,
        ): Int

        @JvmStatic
        external fun nFlushFrame(nativeInstance: Long): Int

        @JvmStatic
        external fun nReset(nativeInstance: Long): Int

        @JvmStatic
        external fun nDequeueOutputBuffer(
            nativeInstance: Long,
            outputBuffer: VideoDecoderOutputBuffer,
        ): Int
    }
}