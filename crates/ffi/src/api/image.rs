//! C ABI for the native image-generation runtime.

use std::os::raw::{c_char, c_int, c_void};

use crate::abi::{self, guard_result, parse_json_or_default};
use crate::image_session::{ImageGenerateOptions, ImageSession};
use crate::strings;
use crate::{TsImageEventCallback, TsImageSession, TS_IMAGE_EVENT_FINISH, TS_IMAGE_EVENT_STAGE};

/// Opens a verified image install and writes a session handle to `out`.
///
/// The image session is separate from a text session because its backend,
/// memory envelope, and output ownership are different. The handle is still
/// serialized internally and can be canceled from another thread.
#[no_mangle]
pub unsafe extern "C" fn ts_image_session_open(
    model_dir: *const c_char,
    out: *mut *mut TsImageSession,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
        }
        let dir = strings::required(model_dir, "modelDir")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let session = ImageSession::open(dir).map_err(|e| (abi::TS_ERR_OPEN, e))?;
        *out = Box::into_raw(Box::new(session));
        Ok(())
    })
}

/// Closes an image session. The caller must not close it while generation is
/// in flight, matching the text-session ABI contract.
#[no_mangle]
pub unsafe extern "C" fn ts_image_session_close(ptr: *mut TsImageSession) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Requests cancellation without taking the backend lock.
#[no_mangle]
pub unsafe extern "C" fn ts_image_session_cancel(ptr: *const TsImageSession) {
    if let Some(session) = ptr.as_ref() {
        session.cancel();
    }
}

/// Generates one PNG. The PNG is returned in an explicitly owned byte buffer;
/// release it with `ts_image_buffer_free`. Metadata is an owned JSON string
/// released with `ts_string_free`.
#[no_mangle]
pub unsafe extern "C" fn ts_image_generate(
    ptr: *const TsImageSession,
    options_json: *const c_char,
    cb: TsImageEventCallback,
    userdata: *mut c_void,
    out_png: *mut *mut u8,
    out_png_len: *mut usize,
    out_metadata_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if out_png.is_null() || out_png_len.is_null() || out_metadata_json.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "image output pointers must not be null".into(),
            ));
        }
        *out_png = std::ptr::null_mut();
        *out_png_len = 0;
        *out_metadata_json = std::ptr::null_mut();
        let session = ptr.as_ref().ok_or((
            abi::TS_ERR_INVALID_ARGUMENT,
            "image session must not be null".into(),
        ))?;
        let options = parse_json_or_default::<ImageGenerateOptions>(options_json, "options")?;
        let output = session
            .generate(options, |progress| {
                if let Some(callback) = cb {
                    let stage = format_stage(progress.stage);
                    callback(
                        userdata,
                        TS_IMAGE_EVENT_STAGE,
                        stage.as_ptr() as *const c_char,
                        stage.len(),
                        progress.completed,
                        progress.total,
                    );
                }
            })
            .map_err(|e| (abi::TS_ERR_GENERATE, e))?;
        let metadata =
            serde_json::to_string(&output.info).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&metadata, out_metadata_json)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        if !output.png.is_empty() {
            let png = output.png.into_boxed_slice();
            *out_png_len = png.len();
            *out_png = Box::into_raw(png) as *mut u8;
        }
        if let Some(callback) = cb {
            let finish = "finished";
            callback(
                userdata,
                TS_IMAGE_EVENT_FINISH,
                finish.as_ptr() as *const c_char,
                finish.len(),
                1,
                1,
            );
        }
        Ok(())
    })
}

/// Reclaims a PNG byte buffer returned by `ts_image_generate`.
#[no_mangle]
pub unsafe extern "C" fn ts_image_buffer_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        let slice = std::ptr::slice_from_raw_parts_mut(ptr, len);
        drop(Box::from_raw(slice));
    }
}

fn format_stage(stage: image::ImageStage) -> String {
    match stage {
        image::ImageStage::TextEncoder => "text_encoder",
        image::ImageStage::Transformer => "transformer",
        image::ImageStage::VaeDecoder => "vae_decoder",
        image::ImageStage::PngEncode => "png_encode",
    }
    .to_string()
}
