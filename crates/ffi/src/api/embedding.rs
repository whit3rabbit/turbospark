//! Standalone embedding model inference and similarity C ABI entry points.

use std::os::raw::{c_char, c_int};
#[cfg(target_os = "macos")]
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::sync::{Mutex, OnceLock};

use super::libraries::refuse_while_moving;
use crate::abi::{self, guard_result};
use crate::strings;

/// The resident embedding model: a safetensors BERT-class encoder, or the
/// Core ML EmbeddingGemma 2 export (recognized by its file layout).
#[cfg(target_os = "macos")]
enum EmbeddingBackend {
    Encoder(Box<runtime::EncoderRunner>),
    Gemma2(Box<runtime::coreml::EmbeddingGemma2CoreMl>),
}

#[cfg(target_os = "macos")]
impl EmbeddingBackend {
    fn open(dir: &Path) -> Result<Self, String> {
        if runtime::coreml::detect_layout(dir)
            == Some(runtime::coreml::CoreMlLayout::EmbeddingGemma2)
        {
            let opts = runtime::coreml::EmbeddingGemma2Options::from_env()?;
            return runtime::coreml::EmbeddingGemma2CoreMl::open(dir, &opts)
                .map(|m| Self::Gemma2(Box::new(m)));
        }
        runtime::EncoderRunner::open(dir).map(|r| Self::Encoder(Box::new(r)))
    }

    /// The files whose size and mtime decide whether the cached model is
    /// stale, per layout.
    fn stamp_files(dir: &Path) -> [PathBuf; 3] {
        if runtime::coreml::detect_layout(dir)
            == Some(runtime::coreml::CoreMlLayout::EmbeddingGemma2)
        {
            [
                dir.join("config.json"),
                dir.join("embeddings.bf16"),
                dir.join("EmbeddingGemma2Text.mlpackage/Manifest.json"),
            ]
        } else {
            ["config.json", "model.safetensors", "tokenizer.json"].map(|n| dir.join(n))
        }
    }

    fn encode_texts(
        &self,
        texts: &[&str],
        task: Option<&str>,
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, String> {
        match self {
            Self::Gemma2(m) => m.embed_texts(texts, task, dimensions),
            Self::Encoder(r) => {
                if task.is_some() {
                    return Err("this encoder model has no task prompts".into());
                }
                let mut out = r.encode_batch_text(texts)?;
                if let Some(d) = dimensions {
                    for v in &mut out {
                        if d == 0 || d > v.len() {
                            return Err(format!("dimensions {d} is outside 1..={}", v.len()));
                        }
                        v.truncate(d);
                        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                        v.iter_mut().for_each(|x| *x /= norm);
                    }
                }
                Ok(out)
            }
        }
    }
}

#[cfg(target_os = "macos")]
struct CachedEncoder {
    dir: PathBuf,
    file_stamps: [(u64, Option<std::time::SystemTime>); 3],
    backend: EmbeddingBackend,
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

            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let embeddings = with_backend(model_path, |b| b.encode_texts(&text_refs, None, None))?;
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

/// Runs `f` on the cached backend for `model_path`, opening (or reopening,
/// when its files changed) as needed. A failed call drops the cache so the
/// next one reopens from disk.
#[cfg(target_os = "macos")]
fn with_backend<T>(
    model_path: &str,
    f: impl FnOnce(&EmbeddingBackend) -> Result<T, String>,
) -> Result<T, (c_int, String)> {
    let probe = crate::embedding_probe::probe(model_path);
    let dir = match (probe.runnable, probe.resolved_path) {
        (true, Some(dir)) => dir,
        (_, resolved) => {
            let what = resolved.map_or_else(|| model_path.to_string(), |d| d.display().to_string());
            return Err((
                abi::TS_ERR_OPEN,
                format!(
                    "failed to open embedding model from {what}: {}",
                    probe
                        .refused_because
                        .unwrap_or_else(|| "not a supported embedding model".into())
                ),
            ));
        }
    };
    let file_stamps = EmbeddingBackend::stamp_files(&dir).map(|p| file_stamp(&p));
    let mutex = ENCODER_CACHE.get_or_init(|| Mutex::new(None));
    let mut cache = match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            *guard = None;
            guard
        }
    };
    let needs_open = cache
        .as_ref()
        .is_none_or(|entry| entry.dir != dir || entry.file_stamps != file_stamps);
    if needs_open {
        *cache = None;
        let backend = EmbeddingBackend::open(&dir).map_err(|e| {
            (
                abi::TS_ERR_OPEN,
                format!("failed to open embedding model from {}: {e}", dir.display()),
            )
        })?;
        *cache = Some(CachedEncoder {
            dir: dir.clone(),
            file_stamps,
            backend,
        });
    }
    match f(&cache.as_ref().expect("opened above").backend) {
        Ok(v) => Ok(v),
        Err(error) => {
            *cache = None;
            Err((
                abi::TS_ERR_GENERATE,
                format!("embedding encoding failed: {error}"),
            ))
        }
    }
}

/// One embedding request: texts with an optional task prompt and
/// Matryoshka size, plus image and audio files for models with those
/// towers (EmbeddingGemma 2 on Core ML).
#[cfg(target_os = "macos")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingRequest {
    #[serde(default)]
    texts: Vec<String>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    image_paths: Vec<String>,
    #[serde(default)]
    image_budget: Option<usize>,
    #[serde(default)]
    audio_paths: Vec<String>,
}

/// Computes embeddings for a JSON request. `request_json`:
/// `{"texts": [...], "task": "search_query", "dimensions": 256,
///   "image_paths": [...], "image_budget": 280, "audio_paths": [...]}`
/// (every field optional). Writes `{"texts": [[...]], "images": [[...]],
/// "audio": [[{"start_seconds", "duration_seconds", "embedding"}]]}` to
/// `*out`. Task, images and audio need a model that supports them.
#[no_mangle]
pub unsafe extern "C" fn ts_embedding_encode_request_json(
    model_path: *const c_char,
    request_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "out must not be null".to_string(),
            ));
        }
        let model_path = strings::required(model_path, "modelPath")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let request_json = strings::required(request_json, "requestJson")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;

        #[cfg(target_os = "macos")]
        {
            let req: EmbeddingRequest = serde_json::from_str(request_json)
                .map_err(|e| (abi::TS_ERR_JSON, format!("requestJson: {e}")))?;
            let result = with_backend(model_path, |b| {
                let refs: Vec<&str> = req.texts.iter().map(String::as_str).collect();
                let texts = if refs.is_empty() {
                    Vec::new()
                } else {
                    b.encode_texts(&refs, req.task.as_deref(), req.dimensions)?
                };
                let (images, audio) = match b {
                    EmbeddingBackend::Gemma2(m) => {
                        let images = req
                            .image_paths
                            .iter()
                            .map(|p| {
                                m.embed_image_file(Path::new(p), req.image_budget, req.dimensions)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let audio = req
                            .audio_paths
                            .iter()
                            .map(|p| {
                                m.embed_audio_file(Path::new(p), req.dimensions).map(|ws| {
                                    ws.into_iter()
                                        .map(|w| {
                                            serde_json::json!({
                                                "start_seconds": w.start_seconds,
                                                "duration_seconds": w.duration_seconds,
                                                "embedding": w.embedding,
                                            })
                                        })
                                        .collect::<Vec<_>>()
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        (images, audio)
                    }
                    EmbeddingBackend::Encoder(_) => {
                        if !req.image_paths.is_empty() || !req.audio_paths.is_empty() {
                            return Err("this encoder model embeds text only".into());
                        }
                        (Vec::new(), Vec::new())
                    }
                };
                Ok(serde_json::json!({"texts": texts, "images": images, "audio": audio}))
            })?;
            let json =
                serde_json::to_string(&result).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
            strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (model_path, request_json, out);
            Err((
                abi::TS_ERR_UNSUPPORTED,
                "embedding model inference is supported on macOS only".to_string(),
            ))
        }
    })
}

/// Writes the pinned embedding profiles as a JSON array to `*out`:
/// `[{"alias", "display_name", "family", "backend", "repository",
///   "revision", "license", "modalities", "dimensions", "matryoshka_dims",
///   "status", "gates", "download_bytes", "installed_path", "install_state",
///   "install_problem", "can_run", "unavailable_reason"}]`.
/// `installed_path` is null until an install verifies against the current
/// pin. `install_state` is `current`, `stale` (an older pin or damaged; a
/// download replaces it), `foreign` (a download refuses it) or `absent`,
/// and `install_problem` says why in words.
/// `status` is `verified` only when the named `gates` ran on this pin.
#[no_mangle]
pub unsafe extern "C" fn ts_embedding_catalog_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
        }
        let cat = catalog::EmbeddingCatalog::embedded().map_err(|e| (abi::TS_ERR_JSON, e))?;
        let store = catalog::Store::default_store().ok();
        let rows: Vec<serde_json::Value> = cat
            .profiles()
            .iter()
            .map(|p| {
                let (can_run, reason) = p.availability();
                serde_json::json!({
                    "alias": p.alias,
                    "display_name": p.display_name,
                    "family": p.family,
                    "backend": p.backend,
                    "repository": p.repository,
                    "revision": p.revision,
                    "license": p.license,
                    "modalities": p.modalities,
                    "dimensions": p.dimensions,
                    "matryoshka_dims": p.matryoshka_dims,
                    "status": p.status.as_str(),
                    "gates": p.gates,
                    "download_bytes": p.download_bytes(),
                    "installed_path": store
                        .as_ref()
                        .and_then(|s| cat.installed_path(s, &p.alias)),
                    "install_state": store
                        .as_ref()
                        .and_then(|s| cat.install_state(s, &p.alias)),
                    "install_problem": store
                        .as_ref()
                        .and_then(|s| cat.install_problem(s, &p.alias)),
                    "can_run": can_run,
                    "unavailable_reason": reason,
                })
            })
            .collect();
        let json = serde_json::to_string(&rows).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Writes the verdict for one embedding model argument (a pinned alias or a
/// directory) to `*out`, without loading it:
/// `{"input", "resolved_path", "runnable", "family", "backend", "verified",
///   "profile_alias", "refused_because"}`. A catalog alias resolves only to
/// an install of its current pin. `verified` is true only for the verified
/// install of a catalog profile whose status is `verified`.
#[no_mangle]
pub unsafe extern "C" fn ts_embedding_probe_json(
    model_path: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
        }
        let model_path = strings::required(model_path, "modelPath")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json = serde_json::to_string(&crate::embedding_probe::probe(model_path))
            .map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Downloads, verifies and installs one pinned embedding profile into the
/// default store. `revision` must be the catalog's current pin for `alias`
/// (the one the host showed), so a retry never adopts a different pin.
/// Refuses an install that already verifies and a folder TurboSpark did not
/// write; replaces an older pin of the same alias. Progress arrives through
/// `callback` as `TS_INSTALL_BYTES`; `ts_install_cancel` stops it. Writes
/// `{"alias", "path"}` to `*out`.
#[no_mangle]
pub unsafe extern "C" fn ts_embedding_install(
    alias: *const c_char,
    revision: *const c_char,
    callback: crate::TsInstallCallback,
    userdata: *mut std::ffi::c_void,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
        }
        *out = std::ptr::null_mut();
        let alias =
            strings::required(alias, "alias").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let revision = strings::required(revision, "revision")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        refuse_while_moving()?;
        let cat = catalog::EmbeddingCatalog::embedded().map_err(|e| (abi::TS_ERR_JSON, e))?;
        let profile = cat.get(alias).ok_or_else(|| {
            (
                abi::TS_ERR_INVALID_ARGUMENT,
                format!("no embedding profile named {alias:?}"),
            )
        })?;
        if profile.revision != revision {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                format!(
                    "{alias} is pinned at {}, not {revision}; refresh the model list",
                    profile.revision
                ),
            ));
        }
        let active = crate::models::ActiveInstall::register();
        let store =
            catalog::Store::default_store().map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let path = cat
            .install(
                &catalog::Client::new(),
                &catalog::Catalog::embedded().map_err(|e| (abi::TS_ERR_JSON, e))?,
                &store,
                alias,
                revision,
                &mut |progress| {
                    if let Some(cb) = callback {
                        let stage = progress
                            .current_path
                            .as_deref()
                            .unwrap_or("embedding download")
                            .to_string();
                        cb(
                            userdata,
                            crate::TS_INSTALL_BYTES,
                            stage.as_ptr().cast(),
                            stage.len(),
                            progress.completed_bytes,
                            progress.total_bytes,
                        );
                    }
                },
                active.cancel_flag(),
            )
            .map_err(|e| {
                if active.cancel_flag().is_cancelled() {
                    (abi::TS_ERR_CANCELLED, e)
                } else {
                    (abi::TS_ERR_OPEN, e)
                }
            })?;
        let json = serde_json::json!({"alias": alias, "path": path}).to_string();
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
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
