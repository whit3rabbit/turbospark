//! C ABI for the whisper speech-to-text runtime.
//!
//! One resident model per handle; streams buffer bounded PCM chunks and
//! decode at finish. All entry points run inside the ABI guard and cannot
//! unwind; null required handles return [`abi::TS_ERR_INVALID_ARGUMENT`].

use std::os::raw::{c_char, c_int};
use std::ptr;

use crate::abi::{self, guard_result, parse_json_or_default};
use crate::strings;
use crate::stt_session::SttStreamOptions;
use crate::{TsSttModel, TsSttStream};

/// Opens a speech install directory as a resident STT model.
#[no_mangle]
pub unsafe extern "C" fn ts_stt_open(model_dir: *const c_char, out: *mut *mut TsSttModel) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
        }
        *out = ptr::null_mut();
        let dir = strings::required(model_dir, "modelDir")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let model = TsSttModel::open(dir).map_err(|e| (abi::TS_ERR_OPEN, e))?;
        *out = Box::into_raw(Box::new(model));
        Ok(())
    })
}

/// Closes a resident STT model. Streams against it must be closed first;
/// this refuses (without dropping) when any stream still references it.
///
/// # Safety
/// `model` must be a handle [`ts_stt_open`] returned, not yet closed.
#[no_mangle]
pub unsafe extern "C" fn ts_stt_close(model: *mut TsSttModel) -> c_int {
    guard_result(|| {
        let Some(model_ref) = model.as_ref() else {
            return Ok(());
        };
        if model_ref.has_streams() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "close every stream against this model before closing the model".into(),
            ));
        }
        drop(Box::from_raw(model));
        Ok(())
    })
}

/// Opens a PCM stream against a resident model.
///
/// # Safety
/// `model` must be a live handle; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ts_stt_stream_open(
    model: *mut TsSttModel,
    options_json: *const c_char,
    out: *mut *mut TsSttStream,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "model and out must not be null".into(),
            ));
        }
        *out = ptr::null_mut();
        let options = parse_json_or_default::<SttStreamOptions>(options_json, "options")?;
        let model_ref = model.as_ref().ok_or((
            abi::TS_ERR_INVALID_ARGUMENT,
            "model must not be null".into(),
        ))?;
        *out = Box::into_raw(Box::new(TsSttStream::open(model_ref, &options)));
        Ok(())
    })
}

/// Appends one bounded PCM chunk (base64 f32 LE 16 kHz mono, at most two
/// seconds) and reports the buffered extent.
///
/// # Safety
/// `stream` must be a live handle; `out_json` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ts_stt_stream_append(
    stream: *mut TsSttStream,
    pcm_base64: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if out_json.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "outJson must not be null".into(),
            ));
        }
        *out_json = ptr::null_mut();
        let stream_ref = stream.as_mut().ok_or((
            abi::TS_ERR_INVALID_ARGUMENT,
            "stream must not be null".into(),
        ))?;
        let pcm =
            strings::required(pcm_base64, "pcm").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let status = stream_ref
            .append(pcm)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        strings::emit(&status.to_string(), out_json)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        Ok(())
    })
}

/// Finishes the stream: transcribes everything buffered and writes the
/// segment JSON (`segments` + `languageDetected`).
///
/// # Safety
/// `stream` must be a live handle; `out_json` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ts_stt_stream_finish(
    stream: *mut TsSttStream,
    out_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if out_json.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "outJson must not be null".into(),
            ));
        }
        *out_json = ptr::null_mut();
        let stream_ref = stream.as_mut().ok_or((
            abi::TS_ERR_INVALID_ARGUMENT,
            "stream must not be null".into(),
        ))?;
        let transcription = stream_ref
            .finish()
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        strings::emit(
            &crate::stt_session::transcription_to_json(&transcription).to_string(),
            out_json,
        )
        .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        Ok(())
    })
}

/// Cancels the stream: the buffered audio is discarded. Idempotent.
///
/// # Safety
/// `stream` must be a live handle or null.
#[no_mangle]
pub unsafe extern "C" fn ts_stt_stream_cancel(stream: *mut TsSttStream) -> c_int {
    guard_result(|| {
        if let Some(stream_ref) = stream.as_mut() {
            crate::stt_session::cancel(stream_ref);
        }
        Ok(())
    })
}

/// Closes a stream. A null handle is a no-op.
///
/// # Safety
/// `stream` must be a handle [`ts_stt_stream_open`] returned and not yet
/// closed.
#[no_mangle]
pub unsafe extern "C" fn ts_stt_stream_close(stream: *mut TsSttStream) -> c_int {
    guard_result(|| {
        if !stream.is_null() {
            drop(Box::from_raw(stream));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_handle() -> *mut TsSttModel {
        Box::into_raw(Box::new(crate::stt_session::model_for_testing()))
    }

    #[test]
    fn stt_model_close_only_refuses_its_own_open_streams() {
        unsafe {
            let first = model_handle();
            let second = model_handle();
            let mut stream = ptr::null_mut();
            assert_eq!(
                ts_stt_stream_open(first, ptr::null(), &mut stream),
                abi::TS_OK
            );
            assert_eq!(ts_stt_close(second), abi::TS_OK);
            assert_eq!(ts_stt_close(first), abi::TS_ERR_INVALID_ARGUMENT);
            assert!(abi::read_last_error(ptr::null_mut(), 0) > 0);
            assert_eq!(ts_stt_stream_cancel(stream), abi::TS_OK);
            assert_eq!(ts_stt_close(first), abi::TS_ERR_INVALID_ARGUMENT);
            assert_eq!(ts_stt_stream_close(stream), abi::TS_OK);
            assert_eq!(ts_stt_close(first), abi::TS_OK);
        }
    }

    #[test]
    fn stt_stream_lifetime_is_preserved_across_thread_handoffs() {
        unsafe {
            let model = model_handle();
            let mut stream = ptr::null_mut();
            assert_eq!(
                ts_stt_stream_open(model, ptr::null(), &mut stream),
                abi::TS_OK
            );
            // Transfer exclusive handle use to another thread, matching an
            // app moving its blocking work off the main thread.
            let model_address = model as usize;
            let stream_address = stream as usize;
            std::thread::spawn(move || {
                let model = model_address as *mut TsSttModel;
                let stream = stream_address as *mut TsSttStream;
                assert_eq!(ts_stt_close(model), abi::TS_ERR_INVALID_ARGUMENT);
                assert!(abi::read_last_error(ptr::null_mut(), 0) > 0);
                assert_eq!(ts_stt_stream_close(stream), abi::TS_OK);
                assert_eq!(ts_stt_close(model), abi::TS_OK);
            })
            .join()
            .expect("cross-thread handle lifecycle");
        }
    }

    #[test]
    fn stt_failed_opens_clear_the_output_handle() {
        unsafe {
            let mut model = std::ptr::NonNull::<TsSttModel>::dangling().as_ptr();
            assert_eq!(
                ts_stt_open(ptr::null(), &mut model),
                abi::TS_ERR_INVALID_ARGUMENT
            );
            assert!(model.is_null());
            let mut stream = std::ptr::NonNull::<TsSttStream>::dangling().as_ptr();
            assert_eq!(
                ts_stt_stream_open(ptr::null_mut(), ptr::null(), &mut stream),
                abi::TS_ERR_INVALID_ARGUMENT
            );
            assert!(stream.is_null());
        }
    }
}
