package open.source.vvdec

import android.os.Handler
import android.view.Surface
import com.google.android.exoplayer2.C
import com.google.android.exoplayer2.Format
import com.google.android.exoplayer2.RendererCapabilities
import com.google.android.exoplayer2.decoder.*
import com.google.android.exoplayer2.util.MimeTypes
import com.google.android.exoplayer2.video.DecoderVideoRenderer
import com.google.android.exoplayer2.video.VideoRendererEventListener

class VvdecDecoderVideoRenderer(
    allowedJoiningTimeMs: Long,
    eventHandler: Handler,
    eventListener: VideoRendererEventListener,
    maxDroppedFramesToNotify: Int,
) : DecoderVideoRenderer(
    allowedJoiningTimeMs,
    eventHandler,
    eventListener,
    maxDroppedFramesToNotify
) {

    companion object {
        private const val VIDEO_H266 = MimeTypes.BASE_TYPE_VIDEO + "/vvc"
        private const val TAG = "VvcDecoderVideoRenderer"
    }

    /** The number of input buffers.  */
    private val numInputBuffers = 0

    /**
     * The number of output buffers. The renderer may limit the minimum possible value due to
     * requiring multiple output buffers to be dequeued at a time for it to make progress.
     */
    private val numOutputBuffers = 0

    private lateinit var decoder: VvdecDecoder

    override fun getName(): String {
        return TAG
    }

    override fun supportsFormat(format: Format): @RendererCapabilities.Capabilities Int {
        if (!VIDEO_H266.equals(format.sampleMimeType, ignoreCase = true)) {
            return RendererCapabilities.create(C.FORMAT_UNSUPPORTED_TYPE)
        }
        return if (format.cryptoType != C.CRYPTO_TYPE_NONE) {
            RendererCapabilities.create(C.FORMAT_UNSUPPORTED_DRM)
        } else {
            RendererCapabilities.create(
                C.FORMAT_HANDLED,
                ADAPTIVE_SEAMLESS,
                TUNNELING_NOT_SUPPORTED
            )
        }
    }

    override fun createDecoder(
        format: Format,
        cryptoConfig: CryptoConfig?,
    ): VvdecDecoder {
        decoder = VvdecDecoder(numInputBuffers, numOutputBuffers)
        return decoder
    }

    override fun renderOutputBufferToSurface(
        outputBuffer: VideoDecoderOutputBuffer,
        surface: Surface,
    ) {
        decoder.renderToSurface(outputBuffer, surface)
    }

    override fun setDecoderOutputMode(outputMode: Int) {
        decoder.outputMode = outputMode
    }
}