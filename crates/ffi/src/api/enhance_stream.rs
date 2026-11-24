//! Live voice isolation streams (`ts_audio_enhance_stream_*`) over the inworld
//! realtime install: 48 kHz mono in, the isolated voice out as it becomes
//! final.
//!
//! Like live transcription, a stream takes no heavy-work guard, so chat and
//! image work keep running. Push and flush run synchronously on the caller's
//! thread; released PCM goes to the callback as kind 2 (borrowed for the call)
//! and the result JSON `{"released","processing_ms","late"}` to `*out` (free
//! with `ts_string_free`). `late` is true when the push took longer than the
//! audio it carried: a host feeding the voice onward (for example to live
//! transcription) should fall back to the raw input while pushes run late.
use crate::abi::{self, guard_result};
#[cfg(target_os = "macos")]
use crate::strings;
use std::os::raw::{c_char, c_int, c_void};

use super::{TsAudioCallback, TsAudioOpenToken};

#[cfg(target_os = "macos")]
mod native {
    use super::*;
    use runtime::voice_isolator_live::{Backend, LiveDenoiser};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn invalid(e: impl ToString) -> (c_int, String) {
        (abi::TS_ERR_INVALID_ARGUMENT, e.to_string())
    }

    pub struct TsAudioEnhanceStream {
        session: LiveDenoiser,
        in_flight: AtomicUsize,
    }

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct OpenOptions {
        #[serde(default = "auto")]
        backend: Backend,
    }

    fn auto() -> Backend {
        Backend::Auto
    }

    /// Opens a realtime voice-isolator install (`denoiser_realtime.safetensors`,
    /// optionally with `VoiceIsolatorStep.mlmodelc`). `options` may be NULL or
    /// `{"backend":"auto"|"coreml"|"metal"|"cpu"}`. A managed install is
    /// verified first with kind 1 progress and cancellation, as
    /// `ts_audio_session_open_cancellable` does.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_enhance_stream_open(
        path: *const c_char,
        options: *const c_char,
        token: *const TsAudioOpenToken,
        callback: TsAudioCallback,
        userdata: *mut c_void,
        out: *mut *mut TsAudioEnhanceStream,
    ) -> c_int {
        guard_result(|| {
            if out.is_null() {
                return Err(invalid("out must not be null"));
            }
            *out = std::ptr::null_mut();
            let options: OpenOptions =
                match strings::optional(options, "options").map_err(invalid)? {
                    Some(json) => serde_json::from_str(json).map_err(invalid)?,
                    None => OpenOptions {
                        backend: Backend::Auto,
                    },
                };
            let path = strings::required(path, "path").map_err(invalid)?;
            super::super::verify_open(path, token.as_ref(), callback, userdata)?;
            let session = LiveDenoiser::open(std::path::Path::new(path), options.backend)
                .map_err(|e| abi::audio_error(e, false, abi::TS_ERR_OPEN))?;
            *out = Box::into_raw(Box::new(TsAudioEnhanceStream {
                session,
                in_flight: AtomicUsize::new(0),
            }));
            Ok(())
        })
    }

    /// The network the stream runs, as UTF-8 JSON `{"backend":...}`.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_enhance_stream_info(
        handle: *const TsAudioEnhanceStream,
        out: *mut *mut c_char,
    ) -> c_int {
        guard_result(|| {
            if out.is_null() {
                return Err(invalid("out must not be null"));
            }
            *out = std::ptr::null_mut();
            let stream = handle
                .as_ref()
                .ok_or_else(|| invalid("stream must not be null"))?;
            let json = serde_json::json!({ "backend": stream.session.backend() }).to_string();
            strings::emit(&json, out).map_err(invalid)
        })
    }

    struct InFlight<'a>(&'a AtomicUsize);
    impl<'a> InFlight<'a> {
        /// Refuses a second call while one runs on the same stream.
        fn enter(counter: &'a AtomicUsize) -> Result<Self, (c_int, String)> {
            if counter.fetch_add(1, Ordering::AcqRel) > 0 {
                counter.fetch_sub(1, Ordering::AcqRel);
                return Err((
                    abi::TS_ERR_BUSY,
                    "another call is running on this stream".into(),
                ));
            }
            Ok(Self(counter))
        }
    }
    impl Drop for InFlight<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::AcqRel);
        }
    }

    unsafe fn deliver(callback: TsAudioCallback, userdata: *mut c_void, pcm: &[f32]) {
        if let (Some(cb), false) = (callback, pcm.is_empty()) {
            cb(userdata, 2, std::ptr::null(), 0, pcm.as_ptr(), pcm.len());
        }
    }

    /// Pushes 1..32000 samples of 48 kHz mono f32.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_enhance_stream_push(
        handle: *const TsAudioEnhanceStream,
        pcm: *const f32,
        count: usize,
        callback: TsAudioCallback,
        userdata: *mut c_void,
        out: *mut *mut c_char,
    ) -> c_int {
        guard_result(|| {
            if out.is_null() {
                return Err(invalid("out must not be null"));
            }
            *out = std::ptr::null_mut();
            if pcm.is_null() || count == 0 || count > runtime::native_audio::MAX_PCM_CHUNK {
                return Err(invalid("invalid PCM chunk pointer or sample count"));
            }
            let stream = handle
                .as_ref()
                .ok_or_else(|| invalid("stream must not be null"))?;
            let _guard = InFlight::enter(&stream.in_flight)?;
            let samples = std::slice::from_raw_parts(pcm, count);
            let push = stream
                .session
                .push(samples)
                .map_err(|e| (abi::TS_ERR_GENERATE, e))?;
            deliver(callback, userdata, &push.samples);
            strings::emit(&serde_json::to_string(&push).map_err(invalid)?, out).map_err(invalid)
        })
    }

    /// Ends the stream: delivers the remaining voice as kind 2 PCM. Later
    /// pushes are refused; close and reopen for a new stream.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_enhance_stream_flush(
        handle: *const TsAudioEnhanceStream,
        callback: TsAudioCallback,
        userdata: *mut c_void,
    ) -> c_int {
        guard_result(|| {
            let stream = handle
                .as_ref()
                .ok_or_else(|| invalid("stream must not be null"))?;
            let _guard = InFlight::enter(&stream.in_flight)?;
            let rest = stream
                .session
                .flush()
                .map_err(|e| (abi::TS_ERR_GENERATE, e))?;
            deliver(callback, userdata, &rest);
            Ok(())
        })
    }

    /// Frees the stream. Refused while a push or flush is running.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_enhance_stream_close(
        handle: *mut TsAudioEnhanceStream,
    ) -> c_int {
        guard_result(|| {
            if let Some(stream) = handle.as_ref() {
                if stream.in_flight.load(Ordering::Acquire) > 0 {
                    return Err(invalid("a push or flush is still running on this stream"));
                }
                drop(Box::from_raw(handle));
            }
            Ok(())
        })
    }
}
#[cfg(target_os = "macos")]
pub use native::*;

#[cfg(not(target_os = "macos"))]
mod unsupported {
    use super::*;
    type TsAudioEnhanceStream = c_void;
    fn refused() -> Result<(), (c_int, String)> {
        Err((
            abi::TS_ERR_UNSUPPORTED,
            "native audio inference requires macOS".into(),
        ))
    }
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_enhance_stream_open(
        _path: *const c_char,
        _options: *const c_char,
        _token: *const TsAudioOpenToken,
        _callback: TsAudioCallback,
        _userdata: *mut c_void,
        out: *mut *mut TsAudioEnhanceStream,
    ) -> c_int {
        guard_result(|| {
            if !out.is_null() {
                *out = std::ptr::null_mut();
            }
            refused()
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_enhance_stream_info(
        _h: *const TsAudioEnhanceStream,
        out: *mut *mut c_char,
    ) -> c_int {
        guard_result(|| {
            if !out.is_null() {
                *out = std::ptr::null_mut();
            }
            refused()
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_enhance_stream_push(
        _h: *const TsAudioEnhanceStream,
        _pcm: *const f32,
        _count: usize,
        _callback: TsAudioCallback,
        _userdata: *mut c_void,
        out: *mut *mut c_char,
    ) -> c_int {
        guard_result(|| {
            if !out.is_null() {
                *out = std::ptr::null_mut();
            }
            refused()
        })
    }
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_enhance_stream_flush(
        _h: *const TsAudioEnhanceStream,
        _callback: TsAudioCallback,
        _userdata: *mut c_void,
    ) -> c_int {
        guard_result(refused)
    }
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_enhance_stream_close(_h: *mut TsAudioEnhanceStream) -> c_int {
        guard_result(|| Ok(()))
    }
}
#[cfg(not(target_os = "macos"))]
pub use unsupported::*;
