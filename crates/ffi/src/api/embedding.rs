//! Standalone embedding model inference and similarity C ABI entry points.

use std::os::raw::{c_char, c_int};

use crate::abi::{self, guard_result};
use crate::strings;

/// Computes vector embeddings for a JSON array of strings using an encoder model.
/// `texts_json` is a JSON array of strings, e.g. `["text 1", "text 2"]`.
/// Writes a JSON array of float vectors to `*out`, e.g. `[[0.1, ...], [0.2, ...]]`.
#[no_mangle]
pub unsafe extern "C" fn ts_embedding_encode_json(
    model_path: *const c_char,
    texts_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let model_path = strings::required(model_path, "modelPath")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let texts_json = strings::required(texts_json, "textsJson")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;

        #[cfg(target_os = "macos")]
        {
            let texts: Vec<String> = serde_json::from_str(texts_json)
                .map_err(|e| (abi::TS_ERR_JSON, format!("textsJson: {e}")))?;

            let dir = catalog::resolve_model_arg(model_path);
            let runner = runtime::EncoderRunner::open(&dir).map_err(|e| {
                (
                    abi::TS_ERR_OPEN,
                    format!("failed to open encoder from {}: {e}", dir.display()),
                )
            })?;

            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let embeddings = runner.encode_batch_text(&text_refs).map_err(|e| {
                (
                    abi::TS_ERR_GENERATE,
                    format!("embedding encoding failed: {e}"),
                )
            })?;

            let json = serde_json::to_string(&embeddings)
                .map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
            strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (model_path, texts_json, out);
            Err((
                abi::TS_ERR_UNSUPPORTED_PLATFORM,
                "embedding model inference is supported on macOS only".to_string(),
            ))
        }
    })
}

/// Computes cosine similarity between two float vectors of length `len`.
/// Returns 0.0 if either pointer is null, len is 0, or an internal panic is
/// caught at the boundary (read `ts_last_error` to see why; there is no
/// status code here to carry `TS_ERR_PANIC` through, so the sentinel is the
/// only signal a caller gets, same as the null/zero-length case).
#[no_mangle]
pub unsafe extern "C" fn ts_cosine_similarity(a: *const f32, b: *const f32, len: usize) -> f32 {
    abi::guard_value(0.0, || {
        if a.is_null() || b.is_null() || len == 0 {
            return 0.0;
        }
        let slice_a = std::slice::from_raw_parts(a, len);
        let slice_b = std::slice::from_raw_parts(b, len);
        compute::cosine_similarity(slice_a, slice_b)
    })
}
