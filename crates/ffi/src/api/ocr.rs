//! C ABI for document OCR on Core ML (Unlimited OCR, macOS 15+).
//!
//! A session holds the loaded model and a list of pending pages. Callers
//! add pages (decoded RGB or an image file), then read them as one
//! document; the pending list is cleared by every read.

use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(target_os = "macos")]
use std::sync::Mutex;

use crate::abi::{self, guard_result};
use crate::strings;

/// The opaque OCR handle. `TsOcrSession *` in C.
pub struct OcrSession {
    #[cfg(target_os = "macos")]
    model: runtime::coreml::UnlimitedOcrCoreMl,
    #[cfg(target_os = "macos")]
    pending: Mutex<Vec<runtime::coreml::OcrPage>>,
    cancel: AtomicBool,
}

pub type TsOcrSession = OcrSession;

#[cfg(target_os = "macos")]
#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ReadOptions {
    #[serde(default)]
    max_new_tokens: Option<usize>,
}

#[cfg(not(target_os = "macos"))]
fn unsupported<T>() -> Result<T, (c_int, String)> {
    Err((
        abi::TS_ERR_UNSUPPORTED,
        "OCR runs on Core ML and needs macOS 15 or newer".into(),
    ))
}

/// Opens a Core ML Unlimited OCR directory and writes a handle to `out`.
#[no_mangle]
pub unsafe extern "C" fn ts_ocr_open(
    model_path: *const c_char,
    out: *mut *mut TsOcrSession,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
        }
        let path = strings::required(model_path, "modelPath")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        #[cfg(target_os = "macos")]
        {
            let dir = catalog::resolve_model_arg(path);
            let model = runtime::coreml::UnlimitedOcrCoreMl::open(
                &dir,
                &runtime::coreml::OcrOptions::default(),
            )
            .map_err(|e| (abi::TS_ERR_OPEN, e))?;
            *out = Box::into_raw(Box::new(OcrSession {
                model,
                pending: Mutex::new(Vec::new()),
                cancel: AtomicBool::new(false),
            }));
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = path;
            unsupported()
        }
    })
}

/// Closes a session. Not while a read is in flight.
#[no_mangle]
pub unsafe extern "C" fn ts_ocr_close(ptr: *mut TsOcrSession) {
    abi::guard_value((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

/// Asks an in-flight read to stop. Safe from any thread; never blocks.
#[no_mangle]
pub unsafe extern "C" fn ts_ocr_cancel(ptr: *const TsOcrSession) {
    abi::guard_value((), || {
        if let Some(s) = ptr.as_ref() {
            s.cancel.store(true, Ordering::SeqCst);
        }
    })
}

/// Adds one page from interleaved RGB8 pixels. `row_bytes` is the stride
/// of one row (at least `width * 3`).
#[no_mangle]
pub unsafe extern "C" fn ts_ocr_add_page_rgb(
    ptr: *const TsOcrSession,
    rgb: *const u8,
    width: usize,
    height: usize,
    row_bytes: usize,
) -> c_int {
    guard_result(|| {
        let s = ptr.as_ref().ok_or((
            abi::TS_ERR_INVALID_ARGUMENT,
            "session must not be null".into(),
        ))?;
        if rgb.is_null() || width == 0 || height == 0 {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "rgb, width and height are required".into(),
            ));
        }
        let packed = width.checked_mul(3).filter(|&w| row_bytes >= w).ok_or((
            abi::TS_ERR_INVALID_ARGUMENT,
            "row_bytes is smaller than width * 3".into(),
        ))?;
        let total = row_bytes
            .checked_mul(height)
            .filter(|&t| t <= isize::MAX as usize)
            .ok_or((abi::TS_ERR_INVALID_ARGUMENT, "image is too large".into()))?;
        #[cfg(target_os = "macos")]
        {
            let src = std::slice::from_raw_parts(rgb, total);
            let mut data = Vec::with_capacity(packed * height);
            for row in src.chunks_exact(row_bytes) {
                data.extend_from_slice(&row[..packed]);
            }
            let page = runtime::coreml::OcrPage::from_rgb(&data, width, height)
                .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
            s.pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(page);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (s, packed, total);
            unsupported()
        }
    })
}

/// Adds one page from a PNG or JPEG file.
#[no_mangle]
pub unsafe extern "C" fn ts_ocr_add_page_file(
    ptr: *const TsOcrSession,
    path: *const c_char,
) -> c_int {
    guard_result(|| {
        let s = ptr.as_ref().ok_or((
            abi::TS_ERR_INVALID_ARGUMENT,
            "session must not be null".into(),
        ))?;
        let path =
            strings::required(path, "path").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        #[cfg(target_os = "macos")]
        {
            let page = runtime::coreml::OcrPage::from_file(std::path::Path::new(path))
                .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
            s.pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(page);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (s, path);
            unsupported()
        }
    })
}

/// Reads the pending pages as one document and clears them. `options_json`
/// may be NULL or `{"max_new_tokens": N}`. Writes `{"text", "raw_text",
/// "pages", "prompt_tokens", "output_tokens", "stopped_at_eos"}` to `*out`.
/// A cancelled read returns TS_ERR_CANCELLED.
#[no_mangle]
pub unsafe extern "C" fn ts_ocr_read_json(
    ptr: *const TsOcrSession,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let s = ptr.as_ref().ok_or((
            abi::TS_ERR_INVALID_ARGUMENT,
            "session must not be null".into(),
        ))?;
        if out.is_null() {
            return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
        }
        #[cfg(target_os = "macos")]
        {
            let opts: ReadOptions = if options_json.is_null() {
                ReadOptions::default()
            } else {
                let text = strings::required(options_json, "optionsJson")
                    .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
                serde_json::from_str(text)
                    .map_err(|e| (abi::TS_ERR_JSON, format!("optionsJson: {e}")))?
            };
            let pages = std::mem::take(&mut *s.pending.lock().unwrap_or_else(|p| p.into_inner()));
            if pages.is_empty() {
                return Err((
                    abi::TS_ERR_INVALID_ARGUMENT,
                    "add at least one page first".into(),
                ));
            }
            // Cleared on entry so a Stop pressed between reads does not
            // cancel the next one.
            s.cancel.store(false, Ordering::SeqCst);
            let prompt = if pages.len() == 1 { "single" } else { "multi" };
            let result = s
                .model
                .read_limited(&pages, prompt, opts.max_new_tokens, |_| {
                    !s.cancel.load(Ordering::SeqCst)
                })
                .map_err(|e| {
                    if s.cancel.load(Ordering::SeqCst) {
                        (abi::TS_ERR_CANCELLED, e)
                    } else {
                        (abi::TS_ERR_GENERATE, e)
                    }
                })?;
            let json = serde_json::json!({
                "text": result.plain_text(),
                "raw_text": result.text,
                "pages": result.stats.pages,
                "prompt_tokens": result.stats.prompt_tokens,
                "output_tokens": result.stats.output_tokens,
                "stopped_at_eos": result.stats.stopped_at_eos,
            })
            .to_string();
            strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (s, options_json);
            unsupported()
        }
    })
}
