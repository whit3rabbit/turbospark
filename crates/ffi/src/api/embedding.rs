//! Standalone embedding model inference and similarity C ABI entry points.

use std::os::raw::{c_char, c_int};
#[cfg(target_os = "macos")]
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::sync::{Mutex, OnceLock};

use crate::abi::{self, guard_result};
use crate::strings;

#[cfg(target_os = "macos")]
struct CachedEncoder {
    dir: PathBuf,
    file_stamps: [(u64, Option<std::time::SystemTime>); 3],
    runner: runtime::EncoderRunner,
}

#[cfg(target_os = "macos")]
fn file_stamp(path: &Path) -> (u64, Option<std::time::SystemTime>) {
    std::fs::metadata(path)
        .map(|metadata| (metadata.len(), metadata.modified().ok()))
        .unwrap_or((0, None))
}

#[cfg(target_os = "macos")]
static ENCODER_CACHE: OnceLock<Mutex<Option<CachedEncoder>>> = OnceLock::new();

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
        // Checked before any work: encoding a batch can take a while, and a
        // null out-pointer discovered only at the end means the encode ran
        // for nothing.
        if out.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "out must not be null".to_string(),
            ));
        }
        let model_path = strings::required(model_path, "modelPath")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let texts_json = strings::required(texts_json, "textsJson")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;

        #[cfg(target_os = "macos")]
        {
            let texts: Vec<String> = serde_json::from_str(texts_json)
                .map_err(|e| (abi::TS_ERR_JSON, format!("textsJson: {e}")))?;

            let dir = catalog::resolve_model_arg(model_path);
            let dir = dir.canonicalize().unwrap_or(dir);
            let file_stamps = ["config.json", "model.safetensors", "tokenizer.json"]
                .map(|name| file_stamp(&dir.join(name)));
            let mutex = ENCODER_CACHE.get_or_init(|| Mutex::new(None));
            let mut cache = match mutex.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    let mut guard = poisoned.into_inner();
                    *guard = None;
                    guard
                }
            };
            let needs_open = cache.as_ref().map_or(true, |entry| {
                entry.dir != dir || entry.file_stamps != file_stamps
            });
            if needs_open {
                *cache = None;
                let runner = runtime::EncoderRunner::open(&dir).map_err(|e| {
                    (
                        abi::TS_ERR_OPEN,
                        format!("failed to open encoder from {}: {e}", dir.display()),
                    )
                })?;
                *cache = Some(CachedEncoder {
                    dir: dir.clone(),
                    file_stamps,
                    runner,
                });
            }
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let embeddings = match cache.as_ref().unwrap().runner.encode_batch_text(&text_refs) {
                Ok(embeddings) => embeddings,
                Err(error) => {
                    *cache = None;
                    return Err((
                        abi::TS_ERR_GENERATE,
                        format!("embedding encoding failed: {error}"),
                    ));
                }
            };
            drop(cache);

            let json = serde_json::to_string(&embeddings)
                .map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
            strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (model_path, texts_json, out);
            Err((
                abi::TS_ERR_UNSUPPORTED,
                "embedding model inference is supported on macOS only".to_string(),
            ))
        }
    })
}

/// Computes the cosine similarity between two float vectors of length `len`.
/// Both vectors are L2-normalized first, so arbitrary nonzero inputs give a
/// true cosine in [-1, 1]; a zero vector yields 0.0. Returns 0.0 if either
/// pointer is null, len is 0, or an internal panic is caught at the boundary
/// (read `ts_last_error` to see why; there is no status code here to carry
/// `TS_ERR_PANIC` through, so the sentinel is the only signal a caller gets,
/// same as the null/zero-length case).
#[no_mangle]
pub unsafe extern "C" fn ts_cosine_similarity(a: *const f32, b: *const f32, len: usize) -> f32 {
    abi::guard_value(0.0, || {
        if a.is_null() || b.is_null() || len == 0 {
            return 0.0;
        }
        // `from_raw_parts` needs the byte length to fit in `isize`; no real
        // caller approaches it, but a bad `len` from an FFI caller would
        // otherwise be undefined behaviour rather than an error.
        if len > isize::MAX as usize / std::mem::size_of::<f32>() {
            return 0.0;
        }
        let slice_a = std::slice::from_raw_parts(a, len);
        let slice_b = std::slice::from_raw_parts(b, len);
        compute::cosine_similarity(slice_a, slice_b)
    })
}
