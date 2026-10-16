//! Live transcription streams (`ts_audio_stream_*`) over a streaming speech
//! install, today Parakeet Unified EN on Core ML.
//!
//! Separate from `ts_audio_session_*`: an audio session takes the
//! process-wide heavy-work guard for every job, which chat generation also
//! holds for a whole reply, so it cannot serve hours of live audio. A stream
//! never touches Metal and takes no guard. `ts_stt_stream_*` is not reused
//! either: its `finish` returns the whole transcript by contract, and a
//! stream commits text as it goes.
//!
//! Append and finish run synchronously on the caller's thread and return
//! UTF-8 JSON (free with `ts_string_free`):
//! `{"appended_text","closed_words":[{"text","start_seconds","end_seconds"}],
//! "open_word","received_seconds","committed_through_seconds","is_final"}`.
//! Joining `appended_text` across calls gives the transcript; committed text
//! never changes.
use crate::abi::{self, guard_result};
#[cfg(target_os = "macos")]
use crate::strings;
use std::os::raw::{c_char, c_int, c_void};

use super::{TsAudioCallback, TsAudioOpenToken};

#[cfg(target_os = "macos")]
mod native {
    use super::*;
    use runtime::native_audio::AudioStreamSession;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn invalid(e: impl ToString) -> (c_int, String) {
        (abi::TS_ERR_INVALID_ARGUMENT, e.to_string())
    }

    pub struct TsAudioStream {
        session: AudioStreamSession,
        /// Appends and finishes running now, so close can refuse.
        in_flight: AtomicUsize,
    }

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct OpenOptions {
        #[serde(default)]
        expected_family: Option<String>,
    }

    /// Opens a streaming speech install. `options` may be NULL or
    /// `{"expected_family":"parakeet_unified"}`. A managed install is
    /// verified first, with kind 1 progress and cancellation exactly as
    /// `ts_audio_session_open_cancellable`; the model load itself (a Neural
    /// Engine compile on first use, which can take minutes) is not
    /// interruptible. `token` and `callback` may be NULL.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_stream_open(
        path: *const c_char,
        options: *const c_char,
        token: *const TsAudioOpenToken,
        callback: TsAudioCallback,
        userdata: *mut c_void,
        out: *mut *mut TsAudioStream,
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
                        expected_family: None,
                    },
                };
            let path = strings::required(path, "path").map_err(invalid)?;
            super::super::verify_open(path, token.as_ref(), callback, userdata)?;
            let session = AudioStreamSession::open(std::path::Path::new(path))
                .map_err(|e| abi::audio_error(e, false, abi::TS_ERR_OPEN))?;
            if let Some(expected) = options.expected_family.as_deref() {
                if session.family() != expected {
                    return Err(invalid(format!(
                        "selected audio family {expected} does not match loaded family {}",
                        session.family()
                    )));
                }
            }
            *out = Box::into_raw(Box::new(TsAudioStream {
                session,
                in_flight: AtomicUsize::new(0),
            }));
            Ok(())
        })
    }

    /// Counts a call in flight for the life of the guard.
    struct InFlight<'a>(&'a AtomicUsize);
    impl<'a> InFlight<'a> {
        fn enter(counter: &'a AtomicUsize) -> Self {
            counter.fetch_add(1, Ordering::AcqRel);
            Self(counter)
        }
    }
    impl Drop for InFlight<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn stream_error(e: String) -> (c_int, String) {
        if e == "cancelled" {
            (abi::TS_ERR_CANCELLED, e)
        } else if e.contains("is busy") {
            (abi::TS_ERR_BUSY, e)
        } else {
            (abi::TS_ERR_GENERATE, e)
        }
    }

    /// Appends 16 kHz mono f32 samples, 1..32000 per call, and returns what
    /// they committed. A second call while one runs is refused with
    /// `TS_ERR_BUSY` rather than queued.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_stream_append(
        handle: *const TsAudioStream,
        pcm: *const f32,
        count: usize,
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
            if pcm.is_null() || count == 0 {
                return Err(invalid("invalid PCM chunk pointer or sample count"));
            }
            let _guard = InFlight::enter(&stream.in_flight);
            let samples = std::slice::from_raw_parts(pcm, count);
            let update = stream.session.append(samples).map_err(stream_error)?;
            strings::emit(&serde_json::to_string(&update).map_err(invalid)?, out).map_err(invalid)
        })
    }

    /// Commits the held-back tail and closes the last word (`is_final` is
    /// true). The stream then restarts at time zero, so a host can begin a
    /// new segment after a gap in its audio.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_stream_finish(
        handle: *const TsAudioStream,
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
            let _guard = InFlight::enter(&stream.in_flight);
            let update = stream.session.finish().map_err(stream_error)?;
            strings::emit(&serde_json::to_string(&update).map_err(invalid)?, out).map_err(invalid)
        })
    }

    /// Stops an append or finish in flight before its next encoder step,
    /// from any thread. Terminal: later calls fail; close and reopen.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_stream_cancel(handle: *const TsAudioStream) -> c_int {
        guard_result(|| {
            if let Some(stream) = handle.as_ref() {
                stream.session.cancel();
            }
            Ok(())
        })
    }

    /// Frees the stream. Refused while an append or finish is running;
    /// cancel, wait for it to return, then close. A host must not start a
    /// new call concurrently with close.
    #[no_mangle]
    pub unsafe extern "C" fn ts_audio_stream_close(handle: *mut TsAudioStream) -> c_int {
        guard_result(|| {
            if let Some(stream) = handle.as_ref() {
                if stream.in_flight.load(Ordering::Acquire) > 0 {
                    return Err(invalid(
                        "an append or finish is still running on this stream",
                    ));
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
    type TsAudioStream = c_void;
    macro_rules! refused {($name:ident($($arg:ident:$ty:ty),*) $(,$out:ident)?)=>{
 #[no_mangle] pub unsafe extern "C" fn $name($($arg:$ty),*)->c_int {guard_result(||{let _=($($arg),*);$(if !$out.is_null(){*$out=std::ptr::null_mut();})?Err((abi::TS_ERR_UNSUPPORTED,"native audio inference requires macOS".into()))})}
 };}
    refused!(ts_audio_stream_open(path:*const c_char,options:*const c_char,token:*const TsAudioOpenToken,callback:TsAudioCallback,userdata:*mut c_void,out:*mut *mut TsAudioStream),out);
    refused!(ts_audio_stream_append(handle:*const TsAudioStream,pcm:*const f32,count:usize,out:*mut *mut c_char),out);
    refused!(ts_audio_stream_finish(handle:*const TsAudioStream,out:*mut *mut c_char),out);
    refused!(ts_audio_stream_cancel(handle:*const TsAudioStream));
    refused!(ts_audio_stream_close(handle:*mut TsAudioStream));
}
#[cfg(not(target_os = "macos"))]
pub use unsupported::*;

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn open_refuses_null_out_and_a_non_streaming_directory() {
        unsafe {
            let path = CString::new("/nonexistent-ts-audio-stream").unwrap();
            assert_eq!(
                ts_audio_stream_open(
                    path.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    None,
                    std::ptr::null_mut(),
                    std::ptr::null_mut()
                ),
                abi::TS_ERR_INVALID_ARGUMENT
            );
            let mut out: *mut TsAudioStream = std::ptr::dangling_mut();
            let code = ts_audio_stream_open(
                path.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                None,
                std::ptr::null_mut(),
                &mut out,
            );
            assert_ne!(code, abi::TS_OK);
            assert!(out.is_null(), "a failed open leaves *out NULL");
            let options = CString::new(r#"{"unknown":1}"#).unwrap();
            let code = ts_audio_stream_open(
                path.as_ptr(),
                options.as_ptr(),
                std::ptr::null(),
                None,
                std::ptr::null_mut(),
                &mut out,
            );
            assert_eq!(
                code,
                abi::TS_ERR_INVALID_ARGUMENT,
                "unknown options are refused"
            );
        }
    }

    #[test]
    fn null_handles_are_refused_or_ignored() {
        unsafe {
            let mut json: *mut c_char = std::ptr::null_mut();
            let pcm = [0.0f32; 4];
            assert_eq!(
                ts_audio_stream_append(std::ptr::null(), pcm.as_ptr(), 4, &mut json),
                abi::TS_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                ts_audio_stream_finish(std::ptr::null(), &mut json),
                abi::TS_ERR_INVALID_ARGUMENT
            );
            assert!(json.is_null());
            assert_eq!(ts_audio_stream_cancel(std::ptr::null()), abi::TS_OK);
            assert_eq!(ts_audio_stream_close(std::ptr::null_mut()), abi::TS_OK);
        }
    }
}
