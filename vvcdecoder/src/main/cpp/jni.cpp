#include <jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <algorithm>
#include <mutex>
#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include "cpufeatures/cpu-features.h"
#include "vvdec/vvdec.h"
#include <android/looper.h>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

//
// Created by Luke on 2022/3/24.
//

#define MAX_CODED_PICTURE_SIZE  (20*1024)
#define LOG_TAG "VvcDecoderJNI"

#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

static jmethodID method_init_for_yuv_frame;
static jmethodID method_init_for_private_frame;
static jfieldID field_data;
static jfieldID field_output_mode;
static jfieldID field_decoder_private;
// Android YUV format. See:
// https://developer.android.com/reference/android/graphics/ImageFormat.html#YV12.
static constexpr int IMAGE_FORMAT_YV12 = 0x32315659;
//
static constexpr int COLOR_SPACE_UNKNOWN = 0;
//
static constexpr int OUTPUT_MODE_YUV = 0;
static constexpr int OUTPUT_MODE_SURFACE_YUV = 1;
//

struct frame_buffer {
    struct plane {
        unsigned char *ptr;                  // pointer to plane buffer
        uint32_t width;                // width of the plane
        uint32_t height;               // height of the plane
        uint32_t stride;               // stride (width + left margin + right margins) of plane in samples

        explicit plane(const vvdecPlane &plane) {
            width = plane.width;
            height = plane.height;
            stride = plane.stride;
            const uint64_t length = plane.stride * plane.height;
            ptr = new unsigned char[length];
            memcpy(ptr, plane.ptr, length);
        }

        plane() = delete;

        plane(const plane &) = delete;

        plane(plane &&) = default;

        ~plane() {
            delete[] ptr;
        }

    };

    explicit frame_buffer(const vvdecFrame &frame) {
        width = frame.width;
        height = frame.height;
        bit_depth = frame.bitDepth;
        for (int i = 0; i < frame.numPlanes; ++i) {
            planes.emplace_back(frame.planes[i]);
        }
    }

    frame_buffer() = delete;

    frame_buffer(const frame_buffer &) = delete;

    frame_buffer(frame_buffer &&) = default;

    std::vector<plane> planes;     // component plane for yuv
    uint32_t width;           // width of the luminance plane
    uint32_t height;          // height of the luminance plane
    uint32_t bit_depth;        // bit depth of input signal (8: depth 8 bit, 10: depth 10 bit  )

};

#ifdef __ARM_NEON__

static int convert_16_to_8_neon(
        const frame_buffer *const img,
        jbyte *const data,
        const int32_t uvHeight,
        const int32_t yLength,
        const int32_t uvLength
) {
    if (!(android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_NEON)) return 0;
    uint32x2_t lcg_val = vdup_n_u32(random());
    lcg_val = vset_lane_u32(random(), lcg_val, 1);
    // LCG values recommended in good ol' "Numerical Recipes"
    const uint32x2_t LCG_MULT = vdup_n_u32(1664525);
    const uint32x2_t LCG_INCR = vdup_n_u32(1013904223);

    const uint16_t *srcBase = reinterpret_cast<uint16_t *>(img->planes[VVDEC_CT_Y].ptr);
    uint8_t *dstBase = reinterpret_cast<uint8_t *>(data);
    // In units of uint16_t, so /2 from raw stride
    const int srcStride = img->planes[VVDEC_CT_Y].stride / 2;
    const int dstStride = img->planes[VVDEC_CT_Y].stride;

    for (int y = 0; y < img->height; y++) {
        const uint16_t *src = srcBase;
        uint8_t *dst = dstBase;

        // Each read consumes 4 2-byte samples, but to reduce branches and
        // random steps we unroll to four rounds, so each loop consumes 16
        // samples.
        const int imax = img->width & ~15;
        int i;
        for (i = 0; i < imax; i += 16) {
            // Run a round of the RNG.
            lcg_val = vmla_u32(LCG_INCR, lcg_val, LCG_MULT);

            // The lower two bits of this LCG parameterization are garbage,
            // leaving streaks on the image. We access the upper bits of each
            // 16-bit lane by shifting. (We use this both as an 8- and 16-bit
            // vector, so the choice of which one to keep it as is arbitrary.)
            uint8x8_t randvec =
                    vreinterpret_u8_u16(vshr_n_u16(vreinterpret_u16_u32(lcg_val), 8));

            // We retrieve the values and shift them so that the bits we'll
            // shift out (after biasing) are in the upper 8 bits of each 16-bit
            // lane.
            uint16x4_t values = vshl_n_u16(vld1_u16(src), 6);
            src += 4;

            // We add the bias bits in the lower 8 to the shifted values to get
            // the final values in the upper 8 bits.
            uint16x4_t added1 = vqadd_u16(values, vreinterpret_u16_u8(randvec));

            // Shifting the randvec bits left by 2 bits, as an 8-bit vector,
            // should leave us with enough bias to get the needed rounding
            // operation.
            randvec = vshl_n_u8(randvec, 2);

            // Retrieve and sum the next 4 pixels.
            values = vshl_n_u16(vld1_u16(src), 6);
            src += 4;
            uint16x4_t added2 = vqadd_u16(values, vreinterpret_u16_u8(randvec));

            // Reinterpret the two added vectors as 8x8, zip them together, and
            // discard the lower portions.
            uint8x8_t zipped =
                    vuzp_u8(vreinterpret_u8_u16(added1), vreinterpret_u8_u16(added2))
                            .val[1];
            vst1_u8(dst, zipped);
            dst += 8;

            // Run it again with the next two rounds using the remaining
            // entropy in randvec.
            randvec = vshl_n_u8(randvec, 2);
            values = vshl_n_u16(vld1_u16(src), 6);
            src += 4;
            added1 = vqadd_u16(values, vreinterpret_u16_u8(randvec));
            randvec = vshl_n_u8(randvec, 2);
            values = vshl_n_u16(vld1_u16(src), 6);
            src += 4;
            added2 = vqadd_u16(values, vreinterpret_u16_u8(randvec));
            zipped = vuzp_u8(vreinterpret_u8_u16(added1), vreinterpret_u8_u16(added2))
                    .val[1];
            vst1_u8(dst, zipped);
            dst += 8;
        }

        uint32_t randval = 0;
        // For the remaining pixels in each row - usually none, as most
        // standard sizes are divisible by 32 - convert them "by hand".
        while (i < img->width) {
            if (!randval) randval = random();
            dstBase[i] = (srcBase[i] + (randval & 3)) >> 2;
            i++;
            randval >>= 2;
        }

        srcBase += srcStride;
        dstBase += dstStride;
    }

    const uint16_t *srcUBase =
            reinterpret_cast<uint16_t *>(img->planes[VVDEC_CT_U].ptr);
    const uint16_t *srcVBase =
            reinterpret_cast<uint16_t *>(img->planes[VVDEC_CT_V].ptr);
    const int32_t uvWidth = (img->width + 1) / 2;
    uint8_t *dstUBase = reinterpret_cast<uint8_t *>(data + yLength);
    uint8_t *dstVBase = reinterpret_cast<uint8_t *>(data + yLength + uvLength);
    const int srcUVStride = img->planes[VVDEC_CT_V].stride / 2;
    const int dstUVStride = img->planes[VVDEC_CT_V].stride;

    for (int y = 0; y < uvHeight; y++) {
        const uint16_t *srcU = srcUBase;
        const uint16_t *srcV = srcVBase;
        uint8_t *dstU = dstUBase;
        uint8_t *dstV = dstVBase;

        // As before, each i++ consumes 4 samples (8 bytes). For simplicity we
        // don't unroll these loops more than we have to, which is 8 samples.
        const int imax = uvWidth & ~7;
        int i;
        for (i = 0; i < imax; i += 8) {
            lcg_val = vmla_u32(LCG_INCR, lcg_val, LCG_MULT);
            uint8x8_t randvec =
                    vreinterpret_u8_u16(vshr_n_u16(vreinterpret_u16_u32(lcg_val), 8));
            uint16x4_t uVal1 = vqadd_u16(vshl_n_u16(vld1_u16(srcU), 6),
                                         vreinterpret_u16_u8(randvec));
            srcU += 4;
            randvec = vshl_n_u8(randvec, 2);
            uint16x4_t vVal1 = vqadd_u16(vshl_n_u16(vld1_u16(srcV), 6),
                                         vreinterpret_u16_u8(randvec));
            srcV += 4;
            randvec = vshl_n_u8(randvec, 2);
            uint16x4_t uVal2 = vqadd_u16(vshl_n_u16(vld1_u16(srcU), 6),
                                         vreinterpret_u16_u8(randvec));
            srcU += 4;
            randvec = vshl_n_u8(randvec, 2);
            uint16x4_t vVal2 = vqadd_u16(vshl_n_u16(vld1_u16(srcV), 6),
                                         vreinterpret_u16_u8(randvec));
            srcV += 4;
            vst1_u8(dstU,
                    vuzp_u8(vreinterpret_u8_u16(uVal1), vreinterpret_u8_u16(uVal2))
                            .val[1]);
            dstU += 8;
            vst1_u8(dstV,
                    vuzp_u8(vreinterpret_u8_u16(vVal1), vreinterpret_u8_u16(vVal2))
                            .val[1]);
            dstV += 8;
        }

        uint32_t randval = 0;
        while (i < uvWidth) {
            if (!randval) randval = random();
            dstUBase[i] = (srcUBase[i] + (randval & 3)) >> 2;
            randval >>= 2;
            dstVBase[i] = (srcVBase[i] + (randval & 3)) >> 2;
            randval >>= 2;
            i++;
        }

        srcUBase += srcUVStride;
        srcVBase += srcUVStride;
        dstUBase += dstUVStride;
        dstVBase += dstUVStride;
    }

    return 1;
}

#endif

static void convert_16_to_8_standard(
        const frame_buffer *const img,
        jbyte *const data,
        const int32_t uvHeight,
        const int32_t yLength,
        const int32_t uvLength
) {
    // Y
    int sampleY = 0;
    for (int y = 0; y < img->height; y++) {
        const uint16_t *srcBase = reinterpret_cast<uint16_t *>(
                img->planes[VVDEC_CT_Y].ptr + img->planes[VVDEC_CT_Y].stride * y);
        int8_t *destBase = data + img->planes[VVDEC_CT_Y].stride * y;
        for (int x = 0; x < img->width; x++) {
            // Lightweight dither. Carryover the remainder of each 10->8 bit
            // conversion to the next pixel.
            sampleY += *srcBase++;
            *destBase++ = sampleY >> 2;
            sampleY = sampleY & 3;  // Remainder.
        }
    }
    // UV
    int sampleU = 0;
    int sampleV = 0;
    const int32_t uvWidth = (img->width + 1) / 2;
    for (int y = 0; y < uvHeight; y++) {
        const uint16_t *srcUBase = reinterpret_cast<uint16_t *>(
                img->planes[VVDEC_CT_U].ptr + img->planes[VVDEC_CT_U].stride * y);
        const uint16_t *srcVBase = reinterpret_cast<uint16_t *>(
                img->planes[VVDEC_CT_V].ptr + img->planes[VVDEC_CT_V].stride * y);
        int8_t *destUBase = data + yLength + img->planes[VVDEC_CT_U].stride * y;
        int8_t *destVBase =
                data + yLength + uvLength + img->planes[VVDEC_CT_V].stride * y;
        for (int x = 0; x < uvWidth; x++) {
            // Lightweight dither. Carryover the remainder of each 10->8 bit
            // conversion to the next pixel.
            sampleU += *srcUBase++;
            *destUBase++ = sampleU >> 2;
            sampleU = sampleU & 3;  // Remainder.
            sampleV += *srcVBase++;
            *destVBase++ = sampleV >> 2;
            sampleV = sampleV & 3;  // Remainder.
        }
    }
}

static void copy_frame_to_data_buffer(frame_buffer *decoder_buffer, jbyte *data) {
    for (auto &plane : decoder_buffer->planes) {
        const uint64_t length = plane.stride * plane.height;
        memcpy(data, plane.ptr, length);
        data += length;
    }
}

class decode_context {
    std::mutex frames_lock;
    std::unordered_map<jint, frame_buffer *> frames;
    jint frame_number = 0;
    std::mutex queue_lock;
    std::queue<frame_buffer *> queue;
    jobject surface = nullptr;
    ANativeWindow *native_window = nullptr;
    vvdecAccessUnit *access_unit = nullptr;
    vvdecDecoder *decoder = nullptr;
    vvdecParams *params;

    static void logging_callback(void *decoder, int level, const char *msg, va_list ap) {

    }

    void release_frames() {
        std::lock_guard<std::mutex> guard(frames_lock);
        for (auto &frame : frames) {
            delete frame.second;
        }
        frames.clear();
    }

    void release_queue() {
        std::lock_guard<std::mutex> guard(queue_lock);
        while (!queue.empty()) {
            auto frame = queue.front();
            delete frame;
            queue.pop();
        }
    }

    frame_buffer *find_frame_by_id(jint id) {
        std::lock_guard<std::mutex> guard(frames_lock);
        return frames[id];
    }

    jint unregister_frame(frame_buffer *frame) {
        std::lock_guard<std::mutex> guard(frames_lock);
        while (frames.find(frame_number) != frames.end()) {
            ++frame_number;
        }
        frames[frame_number] = frame;
        return frame_number;
    }

    frame_buffer *unregister_frame(jint id) {
        std::lock_guard<std::mutex> guard(frames_lock);
        auto frame = frames[id];
        if (frame) {
            frames.erase(id);
            return frame;
        } else {
            return nullptr;
        }
    }

    void queue_frame(frame_buffer *frame) {
        std::lock_guard<std::mutex> guard(queue_lock);
        queue.push(frame);
    }

    frame_buffer *dequeue_frame() {
        std::lock_guard<std::mutex> guard(queue_lock);
        auto frame = queue.front();
        queue.pop();
        return frame;
    }

public:
    decode_context() = delete;

    decode_context(const decode_context &) = delete;

    decode_context(decode_context &&) = delete;

    explicit decode_context(int threads) {
        params = vvdec_params_alloc();
        params->parseThreads = threads;
        params->logLevel = VVDEC_INFO;
        decoder = vvdec_decoder_open(params);
        vvdec_set_logging_callback(decoder, &logging_callback);
        access_unit = vvdec_accessUnit_alloc();
        vvdec_accessUnit_default(access_unit);
    }

    ~decode_context() {
        release_frames();
        release_queue();
        if (native_window) {
            ANativeWindow_release(native_window);
        }
        vvdec_decoder_close(decoder);
        vvdec_params_free(params);
        vvdec_accessUnit_free(access_unit);
    }

    jint release_frame(jint frame_id) {
        auto frame = unregister_frame(frame_id);
        if (frame) {
            delete frame;
            return 0;
        } else {
            return -1;
        }
    }

    jint render_frame(
            JNIEnv *env,
            jobject new_surface,
            jint frame_id
    ) {
        auto frame = find_frame_by_id(frame_id);
        if (!frame) {
            return -1;
        }

        if (!env->IsSameObject(surface, new_surface)) {
            if (native_window) {
                ANativeWindow_release(native_window);
            }
            native_window = ANativeWindow_fromSurface(env, new_surface);
            if (surface) {
                env->DeleteWeakGlobalRef(surface);
            }
            surface = env->NewWeakGlobalRef(new_surface);
        };
        if (native_window == nullptr) {
            return -1;
        }
        int surface_width = ANativeWindow_getWidth(native_window);
        int surface_height = ANativeWindow_getHeight(native_window);
        if (surface_width != frame->width || surface_height != frame->height) {
            ANativeWindow_setBuffersGeometry(
                    native_window,
                    frame->width,
                    frame->height,
                    IMAGE_FORMAT_YV12
            );
        }
        ANativeWindow_Buffer buffer;
        int result = ANativeWindow_lock(native_window, &buffer, nullptr);
        if (buffer.bits == nullptr || result) {
            return -1;
        }
        // Y
        const size_t src_y_stride = frame->planes[VVDEC_CT_Y].stride;
        int stride = frame->width;
        const uint8_t *src_base = reinterpret_cast<uint8_t *>(frame->planes[VVDEC_CT_Y].ptr);
        auto dest_base = (uint8_t *) buffer.bits;
        for (int y = 0; y < frame->height; y++) {
            memcpy(dest_base, src_base, stride);
            src_base += src_y_stride;
            dest_base += buffer.stride;
        }
        // UV
        const int src_uv_stride = frame->planes[VVDEC_CT_U].stride;
        const int dest_uv_stride = (buffer.stride / 2 + 15) & (~15);
        const int32_t buffer_uv_height = (buffer.height + 1) / 2;
        const int32_t height = std::min((int32_t) (frame->height + 1) / 2, buffer_uv_height);
        stride = (frame->width + 1) / 2; // NOLINT(cppcoreguidelines-narrowing-conversions)
        src_base = reinterpret_cast<uint8_t *>(frame->planes[VVDEC_CT_U].ptr);
        const uint8_t *src_v_base = reinterpret_cast<uint8_t *>(frame->planes[VVDEC_CT_V].ptr);
        uint8_t *dest_v_base = ((uint8_t *) buffer.bits) + buffer.stride * buffer.height;
        dest_base = dest_v_base + buffer_uv_height * dest_uv_stride;
        for (int y = 0; y < height; y++) {
            memcpy(dest_base, src_base, stride);
            memcpy(dest_v_base, src_v_base, stride);
            src_base += src_uv_stride;
            src_v_base += src_uv_stride;
            dest_base += dest_uv_stride;
            dest_v_base += dest_uv_stride;
        }
        return ANativeWindow_unlockAndPost(native_window);
    }

    jstring get_last_error(JNIEnv *env) {
        auto message = vvdec_get_last_error(decoder);
        return message == nullptr ? nullptr : env->NewStringUTF(message);
    }

    jint decode(
            JNIEnv *env,
            jobject input_data,
            jint input_length
    ) {
        auto data = reinterpret_cast<unsigned char *>(env->GetDirectBufferAddress(input_data));
        if (input_length > access_unit->payloadSize) {
            vvdec_accessUnit_free_payload(access_unit);
            vvdec_accessUnit_alloc_payload(access_unit, input_length);
        }
        memcpy(access_unit->payload, data, input_length);
        access_unit->payloadUsedSize = input_length;
        vvdecFrame *frame = nullptr;
        auto ret = vvdec_decode(decoder, access_unit, &frame);
        if (ret == VVDEC_OK && frame != nullptr) {
            auto *fb = new frame_buffer(*frame);
            vvdec_frame_unref(decoder, frame);
            queue_frame(fb);
        }
        return ret;
    }

    jint flush() {
        vvdecFrame *frame = nullptr;
        auto ret = vvdec_flush(decoder, &frame);
        if (ret == VVDEC_OK && frame != nullptr) {
            auto *fb = new frame_buffer(*frame);
            vvdec_frame_unref(decoder, frame);
            queue_frame(fb);
        }
        return ret;
    }

    jint reset() {
        release_frames();
        release_queue();
        auto ret = vvdec_decoder_close(decoder);
        if (ret == VVDEC_OK) {
            decoder = vvdec_decoder_open(params);
        }
        return ret;
    }

    jint dequeue_output_buffer(
            JNIEnv *env,
            jobject output_buffer
    ) {
        auto frame = dequeue_frame();
        if (!frame) {
            return -1;
        }
        auto frame_id = unregister_frame(frame);
        int output_mode = env->GetIntField(output_buffer, field_output_mode);
        if (output_mode == OUTPUT_MODE_YUV) {
            jboolean init_result = env->CallBooleanMethod(
                    output_buffer, method_init_for_yuv_frame,
                    static_cast<jint>(frame->width),
                    static_cast<jint>(frame->height),
                    static_cast<jint>(frame->planes[VVDEC_CT_Y].stride),
                    static_cast<jint>(frame->planes[VVDEC_CT_U].stride),
                    COLOR_SPACE_UNKNOWN
            );
            if (env->ExceptionCheck() || !init_result) {
                return -1;
            }
            // get pointer to the data buffer.
            auto data_object = env->GetObjectField(output_buffer, field_data);
            auto output_data = reinterpret_cast<jbyte *>(
                    env->GetDirectBufferAddress(data_object)
            );
            if (frame->bit_depth == 8) {
                copy_frame_to_data_buffer(frame, output_data);
            } else if (frame->bit_depth == 10) {
                uint64_t yLength = frame->planes[VVDEC_CT_Y].stride
                                   * frame->planes[VVDEC_CT_Y].height;
                uint64_t uvHeight = frame->planes[VVDEC_CT_U].height;
                uint64_t uvLength = frame->planes[VVDEC_CT_U].stride
                                    * frame->planes[VVDEC_CT_U].height;
#ifdef __ARM_NEON__
                if (!convert_16_to_8_neon(frame, output_data, uvHeight, yLength, uvLength)) {
                    return -1;
                }
#endif
                convert_16_to_8_standard(frame, output_data, uvHeight, yLength, uvLength);

            } else {
                return -1;
            }
        } else if (output_mode == OUTPUT_MODE_SURFACE_YUV) {
            env->CallVoidMethod(
                    output_buffer,
                    method_init_for_private_frame,
                    static_cast<jint>(frame->width),
                    static_cast<jint>(frame->height)
            );
            if (env->ExceptionCheck()) {
                return -1;
            }
            env->SetIntField(
                    output_buffer,
                    field_decoder_private,
                    frame_id
            );
        }
        return 0;
    }
};

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *env = nullptr;
    vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_4);
    jclass output_buffer_class = env->FindClass(
            "com/google/android/exoplayer2/decoder/VideoDecoderOutputBuffer"
    );
    method_init_for_yuv_frame = env->GetMethodID(
            output_buffer_class,
            "initForYuvFrame",
            "(IIIII)Z"
    );
    method_init_for_private_frame = env->GetMethodID(
            output_buffer_class,
            "initForPrivateFrame",
            "(II)V"
    );
    field_data = env->GetFieldID(
            output_buffer_class,
            "data",
            "Ljava/nio/ByteBuffer;"
    );
    field_output_mode = env->GetFieldID(
            output_buffer_class,
            "mode",
            "I"
    );
    field_decoder_private = env->GetFieldID(
            output_buffer_class,
            "decoderPrivate",
            "I"
    );
    return JNI_VERSION_1_4;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_open_source_vvdec_VvdecDecoder_nInit(
        JNIEnv *env,
        jclass clazz,
        jint threads
) {
    return reinterpret_cast<jlong>(new decode_context(threads));
}

extern "C"
JNIEXPORT void JNICALL
Java_open_source_vvdec_VvdecDecoder_nRelease(
        JNIEnv *env,
        jclass clazz,
        jlong native_instance
) {
    delete reinterpret_cast<decode_context *>(native_instance);
}
extern "C"
JNIEXPORT jint JNICALL
Java_open_source_vvdec_VvdecDecoder_nRenderFrame(
        JNIEnv *env,
        jclass clazz,
        jlong native_instance,
        jobject surface,
        jint frame_id
) {
    auto context = reinterpret_cast<decode_context *>(native_instance);
    return context->render_frame(env, surface, frame_id);
}

extern "C"
JNIEXPORT jint JNICALL
Java_open_source_vvdec_VvdecDecoder_nReleaseFrame(
        JNIEnv *env,
        jclass clazz,
        jlong native_instance,
        jint frame_id
) {
    auto context = reinterpret_cast<decode_context *>(native_instance);
    return context->release_frame(frame_id);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_open_source_vvdec_VvdecDecoder_nGetVersion(
        JNIEnv *env,
        jclass clazz
) {
    return env->NewStringUTF(vvdec_get_version());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_open_source_vvdec_VvdecDecoder_nGetLastError(
        JNIEnv *env,
        jclass clazz,
        jlong native_instance
) {
    auto context = reinterpret_cast<decode_context *>(native_instance);
    return context->get_last_error(env);
}

extern "C"
JNIEXPORT jint JNICALL
Java_open_source_vvdec_VvdecDecoder_nDequeueOutputBuffer(
        JNIEnv *env,
        jclass clazz,
        jlong native_instance,
        jobject output_buffer
) {
    auto context = reinterpret_cast<decode_context *>(native_instance);
    return context->dequeue_output_buffer(env, output_buffer);
}

extern "C"
JNIEXPORT jint JNICALL
Java_open_source_vvdec_VvdecDecoder_nDecodeFrame(
        JNIEnv *env, jclass clazz,
        jlong native_instance,
        jobject input_data,
        jint length) {
    auto context = reinterpret_cast<decode_context *>(native_instance);
    return context->decode(env, input_data, length);
}

extern "C"
JNIEXPORT jint JNICALL
Java_open_source_vvdec_VvdecDecoder_nFlushFrame(
        JNIEnv *env,
        jclass clazz,
        jlong native_instance
) {
    auto context = reinterpret_cast<decode_context *>(native_instance);
    return context->flush();
}

extern "C"
JNIEXPORT jint JNICALL
Java_open_source_vvdec_VvdecDecoder_nReset(JNIEnv *env, jclass clazz, jlong native_instance) {
    auto context = reinterpret_cast<decode_context *>(native_instance);
    return context->reset();
}
