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
pub unsafe extern "C" fn ts_recommend_json(context_window: u32, out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let ctx = if context_window == 0 {
            None
        } else {
            Some(context_window)
        };
        let json = models::recommend_json(ctx).map_err(|e| (abi::TS_ERR_GENERATE, e))?;
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
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let repo =
            strings::required(repo, "repo").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let file =
            strings::optional(file, "file").map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let sidecars = strings::optional(sidecar_repo, "sidecarRepo")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json = models::probe_json(repo, file, sidecars).map_err(|e| (abi::TS_ERR_JSON, e))?;
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
