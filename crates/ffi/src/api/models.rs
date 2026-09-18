//! Catalog, model inspection, recommendations, and installation C ABI entry points.

use std::os::raw::{c_char, c_int, c_void};
use std::sync::Arc;

use crate::abi::{self, guard_result};
use crate::models::{self, TS_INSTALL_BYTES, TS_INSTALL_STAGE};
use crate::strings;
use crate::TsInstallCallback;

/// The catalog, as a JSON array, each row carrying an `installed` flag.
#[no_mangle]
pub unsafe extern "C" fn ts_catalog_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let json = models::catalog_json().map_err(|e| (abi::TS_ERR_JSON, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// What is installed in `~/.turbospark`, as a JSON array.
#[no_mangle]
pub unsafe extern "C" fn ts_installed_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let json = models::installed_json().map_err(|e| (abi::TS_ERR_JSON, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// What valid image-generation installs are present in `~/.turbospark`.
#[no_mangle]
pub unsafe extern "C" fn ts_image_installed_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let json = models::image_installed_json().map_err(|e| (abi::TS_ERR_JSON, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Deletes one valid image-generation install from the shared image store.
#[no_mangle]
pub unsafe extern "C" fn ts_image_delete(alias: *const c_char) -> c_int {
    guard_result(|| {
        let alias =
            strings::required(alias, "alias").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        models::delete_image(alias).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// The curated image-generation catalog, including the pinned source rows.
#[no_mangle]
pub unsafe extern "C" fn ts_image_catalog_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let json = models::image_catalog_json().map_err(|e| (abi::TS_ERR_JSON, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Downloads and packs one curated image-generation catalog row.
#[no_mangle]
pub unsafe extern "C" fn ts_image_install(
    alias: *const c_char,
    cb: TsInstallCallback,
    userdata: *mut c_void,
    result_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if result_json.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "resultJson must not be null".to_string(),
            ));
        }
        let alias =
            strings::required(alias, "alias").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        struct Sink(TsInstallCallback, *mut c_void);
        unsafe impl Send for Sink {}
        unsafe impl Sync for Sink {}
        let sink = Arc::new(Sink(cb, userdata));
        let bytes_sink = Arc::clone(&sink);
        let on_bytes: Arc<dyn Fn(u64, u64) + Send + Sync> = Arc::new(move |done, total| {
            if let Some(f) = bytes_sink.0 {
                unsafe {
                    f(
                        bytes_sink.1,
                        TS_INSTALL_BYTES,
                        std::ptr::null(),
                        0,
                        done,
                        total,
                    )
                };
            }
        });
        let json = models::image_install(
            alias,
            |line| {
                if let Some(f) = sink.0 {
                    unsafe {
                        f(
                            sink.1,
                            TS_INSTALL_STAGE,
                            line.as_ptr() as *const c_char,
                            line.len(),
                            0,
                            0,
                        )
                    };
                }
            },
            on_bytes,
        )
        .map_err(|e| (abi::TS_ERR_GENERATE, e))?;
        strings::emit(&json, result_json).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Deletes an installed model from `~/.turbospark` and drops its directory.
#[no_mangle]
pub unsafe extern "C" fn ts_model_delete(alias: *const c_char) -> c_int {
    guard_result(|| {
        let alias =
            strings::required(alias, "alias").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        models::delete(alias).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Ranks curated models by hardware fit on this machine.
#[no_mangle]
pub unsafe extern "C" fn ts_recommend_json(
    context_window: u32,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let ctx = if context_window == 0 {
            None
        } else {
            Some(context_window)
        };
        // NULL, empty and `{}` all mean "everything default", which is
        // `relaxed` -- the tier every frozen row in `models.json` was measured
        // under, so a caller that passes nothing gets rankings that match
        // those rows. Read through the same helper `ts_session_open` uses, so
        // the two options bags cannot disagree about what absent means.
        let options =
            abi::parse_json_or_default::<crate::wire::RecommendOptions>(options_json, "options")?;
        let guard = crate::wire::load_guard(&options.load_guard)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let slots = crate::wire::expert_cache_slots(&options.expert_cache_slots)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json =
            models::recommend_json(ctx, slots, guard).map_err(|e| (abi::TS_ERR_GENERATE, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Probes a Hugging Face repository by header alone: kilobytes, seconds, no
/// download. `repo` is `owner/name` or `owner/name@revision`.
#[no_mangle]
pub unsafe extern "C" fn ts_probe_json(
    repo: *const c_char,
    file: *const c_char,
    sidecar_repo: *const c_char,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let repo =
            strings::required(repo, "repo").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let file =
            strings::optional(file, "file").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let sidecars = strings::optional(sidecar_repo, "sidecarRepo")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        // NULL, empty and `{}` mean the same defaults `ts_recommend_json`
        // takes, so a host that fits a curated row and a probed repository
        // side by side compares two numbers rather than two configurations.
        let options =
            abi::parse_json_or_default::<crate::wire::ProbeOptions>(options_json, "options")?;
        let guard = crate::wire::load_guard(&options.load_guard)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let slots = crate::wire::expert_cache_slots(&options.expert_cache_slots)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let context = options.context_window.filter(|c| *c > 0);
        let json = models::probe_json(repo, file, sidecars, context, slots, guard)
            .map_err(|e| (abi::TS_ERR_JSON, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// What a longer context window would cost an INSTALLED model.
///
/// A different question from `ts_recommend_json`'s and NOT derivable from it:
/// KV is not linear in the window, so multiplying one figure is 3.5x high on
/// a sliding-window family.
#[no_mangle]
pub unsafe extern "C" fn ts_context_ladder_json(
    model_path: *const c_char,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let path = strings::required(model_path, "modelPath")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let options =
            abi::parse_json_or_default::<crate::wire::ProbeOptions>(options_json, "options")?;
        let guard = crate::wire::load_guard(&options.load_guard)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let slots = crate::wire::expert_cache_slots(&options.expert_cache_slots)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json = models::context_ladder_json(path, slots, guard)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Every `.gguf` a Hugging Face repository publishes, as JSON, best quality
/// first. One API call, no header reads, no download.
///
/// Carries no fit: see `models::repo_variants_json`. Pair it with
/// `ts_probe_json` on the file the user picks.
#[no_mangle]
pub unsafe extern "C" fn ts_repo_variants_json(
    repo: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let repo =
            strings::required(repo, "repo").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json = models::repo_variants_json(repo).map_err(|e| (abi::TS_ERR_JSON, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// What a `.gguf` control vector declares: hidden size, covered blocks, and
/// the mode and architecture the file itself names. No model, no session, no
/// network -- a vector is ~1.3 MB and this is milliseconds.
///
/// A GUI needs this BEFORE it offers a vector against an install, so it can
/// say "this file is 4096 wide and your model is 5120" rather than letting
/// `ts_session_open` fail minutes into a load. It reads the same parser
/// `ts_session_open` reads, so the two cannot disagree about what a file
/// means.
#[no_mangle]
pub unsafe extern "C" fn ts_control_vector_info_json(
    path: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let path =
            strings::required(path, "path").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json = models::control_vector_info_json(path)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// The `(downloadBytes, installBytes)` an install of `alias` will cost, as
/// JSON, so a GUI can warn about space and show a determinate bar before the
/// walk starts.
#[no_mangle]
pub unsafe extern "C" fn ts_install_bytes_json(
    alias: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let alias =
            strings::required(alias, "alias").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let (download, install) =
            models::install_bytes(alias).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json = format!(r#"{{"downloadBytes":{download},"installBytes":{install}}}"#);
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Installs the catalog row `alias`. Blocks for the whole walk (minutes to
/// tens of minutes) and **cannot resume**: a failure restarts it.
///
/// `cb` receives `TS_INSTALL_STAGE` lines on the calling thread and
/// `TS_INSTALL_BYTES` updates **from worker threads, concurrently**. A
/// caller whose callback touches shared state must synchronise it.
#[no_mangle]
pub unsafe extern "C" fn ts_install(
    alias: *const c_char,
    cb: TsInstallCallback,
    userdata: *mut c_void,
    result_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        // Checked before any work: an install streams a whole checkpoint (it
        // is never written to disk twice) and cannot resume, so a null
        // out-pointer discovered only after the download completes means
        // re-streaming the entire thing to try again.
        if result_json.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "resultJson must not be null".to_string(),
            ));
        }
        let alias =
            strings::required(alias, "alias").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;

        // The raw pointer has to cross into worker threads for the byte
        // callback. `Send + Sync` is asserted here rather than assumed, and
        // the header states the matching obligation on the caller: the
        // callback must tolerate concurrent invocation.
        struct Sink(TsInstallCallback, *mut c_void);
        unsafe impl Send for Sink {}
        unsafe impl Sync for Sink {}
        let sink = Arc::new(Sink(cb, userdata));

        let bytes_sink = Arc::clone(&sink);
        let on_bytes: Arc<dyn Fn(u64) + Send + Sync> = Arc::new(move |done: u64| {
            if let Some(f) = bytes_sink.0 {
                unsafe { f(bytes_sink.1, TS_INSTALL_BYTES, std::ptr::null(), 0, done, 0) };
            }
        });

        let json = models::install(
            alias,
            |line| {
                if let Some(f) = sink.0 {
                    unsafe {
                        f(
                            sink.1,
                            TS_INSTALL_STAGE,
                            line.as_ptr() as *const c_char,
                            line.len(),
                            0,
                            0,
                        )
                    };
                }
            },
            on_bytes,
        )
        .map_err(|e| (abi::TS_ERR_GENERATE, e))?;
        strings::emit(&json, result_json).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Probes and installs an arbitrary Hugging Face repository.
#[no_mangle]
pub unsafe extern "C" fn ts_install_repo(
    repo: *const c_char,
    alias: *const c_char,
    file: *const c_char,
    sidecar_repo: *const c_char,
    cb: TsInstallCallback,
    userdata: *mut c_void,
    result_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        // Checked before any work, for `ts_install`'s reason: an install
        // cannot resume, so a null out-pointer found only at the end means
        // re-streaming the whole checkpoint to try again.
        if result_json.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "resultJson must not be null".to_string(),
            ));
        }
        let repo =
            strings::required(repo, "repo").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let alias =
            strings::required(alias, "alias").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let file =
            strings::optional(file, "file").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let sidecars = strings::optional(sidecar_repo, "sidecarRepo")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;

        struct Sink(TsInstallCallback, *mut c_void);
        unsafe impl Send for Sink {}
        unsafe impl Sync for Sink {}
        let sink = Arc::new(Sink(cb, userdata));

        let bytes_sink = Arc::clone(&sink);
        let on_bytes: Arc<dyn Fn(u64) + Send + Sync> = Arc::new(move |done: u64| {
            if let Some(f) = bytes_sink.0 {
                unsafe { f(bytes_sink.1, TS_INSTALL_BYTES, std::ptr::null(), 0, done, 0) };
            }
        });

        let json = models::install_repo(
            repo,
            alias,
            file,
            sidecars,
            |line| {
                if let Some(f) = sink.0 {
                    unsafe {
                        f(
                            sink.1,
                            TS_INSTALL_STAGE,
                            line.as_ptr() as *const c_char,
                            line.len(),
                            0,
                            0,
                        )
                    };
                }
            },
            on_bytes,
        )
        .map_err(|e| (abi::TS_ERR_GENERATE, e))?;
        strings::emit(&json, result_json).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Signals every in-flight install walk (`ts_install`, `ts_install_repo`)
/// to stop. Returns 1 when at least one walk was running and has been
/// signalled, 0 when nothing was running.
///
/// The walk notices at its next step boundary or ranged chunk read --
/// seconds, not tensor boundaries -- and returns `install cancelled`
/// through its own callback/error path. Cancellation does NOT resume or
/// keep progress: the walk dies the same death a network failure gives it,
/// and the partial install directory is exactly as usable (it is not).
/// An install in flight when this fires still runs its own completion
/// path, so a caller that needs to know the walk has exited can compare
/// `ts_installs_finished` before and after.
#[no_mangle]
pub unsafe extern "C" fn ts_install_cancel() -> c_int {
    if models::cancel_active_installs() > 0 {
        1
    } else {
        0
    }
}

/// How many install walks have finished (any outcome) since process start.
/// Read it twice around a `ts_install_cancel` to confirm the cancelled
/// walk actually exited rather than being wedged inside a blocking read.
#[no_mangle]
pub unsafe extern "C" fn ts_installs_finished() -> u32 {
    models::installs_finished() as u32
}

/// Reads the currently resolved Hugging Face token. If a token is found,
/// writes a newly-allocated string to `*out` (free with `ts_string_free`).
/// If no token is set, sets `*out` to NULL and returns `TS_OK`.
#[no_mangle]
pub unsafe extern "C" fn ts_hf_token_get(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "output pointer must not be null".to_string(),
            ));
        }
        if let Some(token) = catalog::resolve_hf_token(None) {
            strings::emit(&token, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
        } else {
            *out = std::ptr::null_mut();
            Ok(())
        }
    })
}

/// Reads the currently resolved Hugging Face token and its source origin.
/// If a token is found, writes JSON to `*out` (free with `ts_string_free`):
///   {"token": "...", "source": "..."}
/// If no token is set, sets `*out` to NULL and returns `TS_OK`.
#[no_mangle]
pub unsafe extern "C" fn ts_hf_token_info_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "output pointer must not be null".to_string(),
            ));
        }
        if let Some((token, source)) = catalog::resolve_hf_token_with_source(None) {
            let json = serde_json::json!({
                "token": token,
                "source": source.label(),
            });
            let text =
                serde_json::to_string(&json).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
            strings::emit(&text, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
        } else {
            *out = std::ptr::null_mut();
            Ok(())
        }
    })
}

/// Saves a Hugging Face token to the local store (~/.turbospark/hf_token).
#[no_mangle]
pub unsafe extern "C" fn ts_hf_token_set(token: *const c_char) -> c_int {
    guard_result(|| {
        let token =
            strings::required(token, "token").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let store =
            catalog::Store::default_store().map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        store
            .set_hf_token(token)
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Clears the Hugging Face token from the local store.
#[no_mangle]
pub unsafe extern "C" fn ts_hf_token_clear() -> c_int {
    guard_result(|| {
        let store =
            catalog::Store::default_store().map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        store
            .clear_hf_token()
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Validates a Hugging Face token against the whoami-v2 API.
#[no_mangle]
pub unsafe extern "C" fn ts_hf_token_validate_json(
    token: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let token =
            strings::required(token, "token").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let status = catalog::validate_hf_token(token);
        let json = serde_json::to_string(&status).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Reads the current Hugging Face mirror base URL into `*out`.
#[no_mangle]
pub unsafe extern "C" fn ts_hf_endpoint_get(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let endpoint = catalog::hf_endpoint();
        strings::emit(&endpoint, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Sets or clears the Hugging Face mirror base URL ($HF_ENDPOINT).
/// Passing NULL or an empty string removes the override, resetting to default.
#[no_mangle]
pub unsafe extern "C" fn ts_hf_endpoint_set(endpoint: *const c_char) -> c_int {
    guard_result(|| {
        let ep = strings::optional(endpoint, "endpoint")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        match ep {
            Some(url) if !url.trim().is_empty() => {
                catalog::set_hf_endpoint_override(Some(url.trim().to_string()));
            }
            _ => {
                catalog::set_hf_endpoint_override(None);
            }
        }
        Ok(())
    })
}

/// Resolves a model alias or relative directory path into its canonical on-disk install path.
/// If the model exists on disk, writes the path string to `*out` (free with `ts_string_free`).
/// If the model cannot be resolved or does not exist, returns `TS_ERR_OPEN`.
#[no_mangle]
pub unsafe extern "C" fn ts_model_resolve_path(
    model_or_alias: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let arg = strings::required(model_or_alias, "modelOrAlias")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let path = catalog::resolve_model_arg(arg);
        if path.exists() {
            let s = path.to_string_lossy().into_owned();
            strings::emit(&s, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
        } else {
            Err((
                abi::TS_ERR_OPEN,
                format!("model '{arg}' could not be resolved or directory does not exist"),
            ))
        }
    })
}
