package open.source.vvdec

import com.google.android.exoplayer2.decoder.DecoderException

class VvdecDecoderException : DecoderException {
    constructor(message: String) : super(message)
    constructor(cause: Throwable?) : super(cause)
    constructor(message: String, cause: Throwable?) : super(message, cause)
}