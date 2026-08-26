//! C ABI over the inference engine, for a native GUI host.
//!
//! The contract in four sentences. Every fallible call returns 0 on success
//! and a non-zero [`abi`] code otherwise, with a message retrievable through
//! [`ts_last_error`] on the SAME thread. A `const char *` argument is
//! borrowed for the duration of the call; a `char **` out-parameter is an
//! allocation the caller returns through [`ts_string_free`]. Options and
//! results are JSON, so adding a knob is never an ABI break. And a session
//! is single-threaded except for [`ts_session_cancel`], which is safe from
//! any thread and never blocks.
//!
//! **THIS CRATE CARRIES `unsafe`**, joining `model-io` and `streaming`; the
//! rest of the workspace forbids it. Every `extern "C"` body below is a call
//! to [`abi::guard`] and nothing else, because unwinding across the boundary
//! is undefined behaviour and this workspace cannot use `panic = "abort"`
//! (see that function's docs).
//!
//! The canonical description of the surface is `include/turbospark.h`. It is
//! hand-written rather than generated: the surface is small, and the SwiftPM
//! target that compiles against it is a stronger check that the two agree
//! than a generator would be, since a generator only ever restates the Rust
//! side to itself.

// This is the ABI layer. `unsafe` is the point of it, and every use is
// justified at its site rather than by a crate-level allowance.
#![allow(clippy::missing_safety_doc)]

use std::os::raw::{c_char, c_int, c_void};
use std::sync::Arc;

/// C ABI error codes and exception-safe wrapper boundary.
pub mod abi;
mod generate;
mod models;
#[cfg(target_os = "macos")]
mod open;
mod session;
mod strings;
mod telemetry;
/// JSON wire structures exchanged across the C ABI.
pub mod wire;

pub use generate::{TS_EVENT_CONTENT, TS_EVENT_PREFILL, TS_EVENT_REASONING};
pub use models::{TS_INSTALL_BYTES, TS_INSTALL_STAGE};
pub use session::Session;

/// The opaque handle a caller holds. `TsSession *` in C.
pub type TsSession = Session;

/// One streamed event. `kind` is one of the `TS_EVENT_*` constants; `text`
/// is UTF-8 of length `len` and is NOT NUL-terminated and NOT owned by the
/// callee. For `TS_EVENT_PREFILL`, `a` is tokens done and `b` the total.
pub type TsEventCallback =
    Option<unsafe extern "C" fn(*mut c_void, c_int, *const c_char, usize, u32, u32)>;

/// One install-progress event. `kind` is one of the `TS_INSTALL_*`
/// constants. **May be called concurrently from worker threads** for
/// `TS_INSTALL_BYTES`; see `models::install`.
pub type TsInstallCallback =
    Option<unsafe extern "C" fn(*mut c_void, c_int, *const c_char, usize, u64, u64)>;

/// Copies this thread's last error message into `buf`, returning the
/// message's own length in bytes excluding the NUL.
///
/// Pass a null `buf` to ask for the length alone. The return value is the
/// message's length rather than the number of bytes written, so a truncated
/// read tells the caller what buffer to allocate.
#[no_mangle]
pub unsafe extern "C" fn ts_last_error(buf: *mut c_char, cap: usize) -> usize {
    // NOT wrapped in `guard`: it returns a length rather than a code, and it
    // is what a caller reaches for when something has already gone wrong, so
    // it must not itself be able to clear the slot it is reading.
    abi::read_last_error(buf, cap)
}

/// Frees a string this library handed out through a `char **`.
#[no_mangle]
pub unsafe extern "C" fn ts_string_free(ptr: *mut c_char) {
    strings::free(ptr);
}

/// Opens `model_dir` (a path, or a `turbospark-model` alias) and writes a
/// session handle to `out`.
///
/// `options_json` may be null or `{}`. Recognised keys: `maxContext`,
/// `expertCacheSlots` (a number or `"auto"`), `powerProfile`,
/// `maxTokensPerSec`.
#[no_mangle]
pub unsafe extern "C" fn ts_session_open(
    model_dir: *const c_char,
    options_json: *const c_char,
    out: *mut *mut TsSession,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
        }
        let dir = strings::required(model_dir, "modelDir")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let options = parse_json_or_default::<wire::OpenOptions>(options_json, "options")?;
        let session = open_platform(dir, &options).map_err(|e| (abi::TS_ERR_OPEN, e))?;
        *out = Box::into_raw(Box::new(session));
        Ok(())
    })
}

#[cfg(target_os = "macos")]
fn open_platform(dir: &str, options: &wire::OpenOptions) -> Result<Session, String> {
    open::open(dir, options)
}

#[cfg(not(target_os = "macos"))]
fn open_platform(_dir: &str, _options: &wire::OpenOptions) -> Result<Session, String> {
    // Refused BY NAME rather than by failing to build, so a caller on
    // another platform gets a sentence instead of a link error.
    Err(
        "the inference engine is macOS-only; this platform can browse the catalog \
         and install models but cannot open one"
            .to_string(),
    )
}

/// Closes a session and frees it. Null is a no-op.
///
/// Must not be called while a generation is in flight on another thread.
#[no_mangle]
pub unsafe extern "C" fn ts_session_close(ptr: *mut TsSession) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Asks the in-flight generation to stop.
///
/// **Safe from any thread and never blocks**, which is the whole point: the
/// flag it raises lives outside the session's mutex, so a Stop button on the
/// main thread does not queue behind the generation it is trying to stop.
/// Cleared at the start of every generation, so a Stop pressed between turns
/// does not cancel the next one.
#[no_mangle]
pub unsafe extern "C" fn ts_session_cancel(ptr: *const TsSession) {
    if let Some(session) = ptr.as_ref() {
        session.cancel();
    }
}

/// Everything resolved at open, as JSON. See `wire::SessionInfo`.
#[no_mangle]
pub unsafe extern "C" fn ts_session_info_json(
    ptr: *const TsSession,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json =
            serde_json::to_string(&session.info).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// The decode phase breakdown, as JSON. See `wire::PhaseReport`.
#[no_mangle]
pub unsafe extern "C" fn ts_session_phases_json(
    ptr: *const TsSession,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json = telemetry::phases_json(session).map_err(|e| (abi::TS_ERR_JSON, e))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// This process's peak physical footprint in bytes, or 0 where unavailable.
#[no_mangle]
pub unsafe extern "C" fn ts_peak_footprint_bytes() -> u64 {
    // No `guard`: nothing here can fail or panic, and there is no code to
    // return through.
    telemetry::peak_footprint_bytes()
}

/// Generates one assistant turn.
///
/// `messages_json` is `[{"role":"user","content":"..."}]`, the same shape
/// `--messages-file` takes, rendered through the checkpoint's own chat
/// template. `options_json` may be null or `{}`. `cb` may be null, in which
/// case nothing streams and the whole turn arrives in `result_json`.
///
/// Blocks for the whole turn. Call it from a background thread.
#[no_mangle]
pub unsafe extern "C" fn ts_generate(
    ptr: *const TsSession,
    messages_json: *const c_char,
    options_json: *const c_char,
    cb: TsEventCallback,
    userdata: *mut c_void,
    result_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let raw = strings::required(messages_json, "messagesJson")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let messages: Vec<wire::WireMessage> = serde_json::from_str(raw)
            .map_err(|e| (abi::TS_ERR_JSON, format!("messagesJson: {e}")))?;
        let options = parse_json_or_default::<wire::GenerateOptions>(options_json, "options")?;

        let result = generate::generate(session, &messages, &options, |kind, text, a, b| {
            if let Some(f) = cb {
                // The pointer is into `text`'s own buffer and is valid only
                // for this call, which is why the header says the callee
                // must copy before returning.
                f(
                    userdata,
                    kind,
                    text.as_ptr() as *const c_char,
                    text.len(),
                    a,
                    b,
                );
            }
        })
        .map_err(|e| (abi::TS_ERR_GENERATE, e))?;

        let json = serde_json::to_string(&result).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, result_json).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Evaluates the prompt token count without running generation.
#[no_mangle]
pub unsafe extern "C" fn ts_session_count_tokens(
    ptr: *const TsSession,
    messages_json: *const c_char,
    reasoning: *const c_char,
    out_count: *mut u32,
) -> c_int {
    guard_result(|| {
        if out_count.is_null() {
            return Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                "out_count must not be null".into(),
            ));
        }
        let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let raw = strings::required(messages_json, "messagesJson")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let messages: Vec<wire::WireMessage> = serde_json::from_str(raw)
            .map_err(|e| (abi::TS_ERR_JSON, format!("messagesJson: {e}")))?;
        let reasoning_str = strings::optional(reasoning, "reasoning")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?
            .unwrap_or("off");
        let count = generate::count_tokens(session, &messages, reasoning_str)
            .map_err(|e| (abi::TS_ERR_GENERATE, e))?;
        *out_count = count;
        Ok(())
    })
}

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

/// Hardware and power telemetry for this machine, as JSON.
#[no_mangle]
pub unsafe extern "C" fn ts_system_info_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let json = telemetry::system_info_json().map_err(|e| (abi::TS_ERR_JSON, e))?;
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

/// Shared spine: run `body` under the panic guard, recording its error.
fn guard_result(body: impl FnOnce() -> Result<(), (c_int, String)>) -> c_int {
    abi::guard(|| match body() {
        Ok(()) => abi::TS_OK,
        Err((code, message)) => abi::fail(code, message),
    })
}

/// Parses an optional JSON argument, defaulting when it is null or empty.
unsafe fn parse_json_or_default<T: Default + serde::de::DeserializeOwned>(
    ptr: *const c_char,
    name: &str,
) -> Result<T, (c_int, String)> {
    match strings::optional(ptr, name).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))? {
        None => Ok(T::default()),
        Some(text) => {
            serde_json::from_str(text).map_err(|e| (abi::TS_ERR_JSON, format!("{name}: {e}")))
        }
    }
}

/// A session over a scripted producer.
///
/// **Deliberately absent from `turbospark.h`, so it is unreachable from C.**
/// It exists so this crate's own tests can drive the whole generate path --
/// the channel split, the cancel plumbing, the event callback, the result
/// JSON -- on any platform and with no multi-gigabyte install. Without it the
/// threading contract this crate's design turns on would be covered by
/// nothing but a manual click of a Stop button.
#[doc(hidden)]
pub fn session_for_testing(
    tokenizer: tokenizer::MfTokenizer,
    steps: Vec<Vec<foundation::LogitValue>>,
    vocab_size: usize,
    max_context: u32,
) -> Session {
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;
    Session {
        engine: Mutex::new(session::Engine::Scripted(Box::new(
            runtime::ScriptedLogitProducer::new(steps),
        ))),
        cancel: Arc::new(AtomicBool::new(false)),
        max_context,
        rate: runtime::RateControl::default(),
        // A scripted producer replays logits and implements no drafter, so
        // there is nothing to speculate WITH. `None` rather than a policy
        // decision: this is the absence of a capability, not a caller's
        // choice, which is why the reported `reason` is null too.
        speculation_block: None,
        info: wire::SessionInfo {
            model_path: "<scripted>".to_string(),
            family: "<scripted>".to_string(),
            max_context,
            trained_context: None,
            past_trained_context: false,
            expert_cache_slots: 0,
            vocab_size,
            dialect: format!("{:?}", tokenizer.dialect),
            reasoning_support: "none".to_string(),
            steering: wire::SteeringInfo::default(),
            speculation: wire::SpeculationInfo::default(),
        },
        tokenizer,
    }
}
