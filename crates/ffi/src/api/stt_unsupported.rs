//! Preserve the speech ABI on platforms without the Whisper runtime.

use std::os::raw::{c_char, c_int};

use crate::abi::{self, guard_result};
use crate::{TsSttModel, TsSttStream};

macro_rules! unsupported {
    ($name:ident ($($arg:ident: $ty:ty),*) $(, $out:ident)?) => {
        /// This platform cannot run speech inference.
        ///
        /// # Safety
        /// Non-null output pointers must be writable.
        #[no_mangle]
        pub unsafe extern "C" fn $name($($arg: $ty),*) -> c_int {
            guard_result(|| {
                let _ = ($($arg),*);
                $(if !$out.is_null() { *$out = std::ptr::null_mut(); })?
                Err((abi::TS_ERR_UNSUPPORTED, "speech inference is supported on macOS only".into()))
            })
        }
    };
}

unsupported!(ts_stt_open(model_dir: *const c_char, out: *mut *mut TsSttModel), out);
unsupported!(ts_stt_stream_open(model: *mut TsSttModel, options_json: *const c_char, out: *mut *mut TsSttStream), out);
unsupported!(ts_stt_stream_append(stream: *mut TsSttStream, pcm_base64: *const c_char, out_json: *mut *mut c_char), out_json);
unsupported!(ts_stt_stream_finish(stream: *mut TsSttStream, out_json: *mut *mut c_char), out_json);
unsupported!(ts_stt_close(model: *mut TsSttModel));
unsupported!(ts_stt_stream_cancel(stream: *mut TsSttStream));
unsupported!(ts_stt_stream_close(stream: *mut TsSttStream));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_speech_off_macos_returns_a_typed_refusal_and_clears_output() {
        let mut out = std::ptr::NonNull::<TsSttModel>::dangling().as_ptr();
        unsafe {
            assert_eq!(
                ts_stt_open(std::ptr::null(), &mut out),
                abi::TS_ERR_UNSUPPORTED
            );
        }
        assert!(out.is_null());
    }
}
