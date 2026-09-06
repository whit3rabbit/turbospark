//! Session lifecycle and introspection C ABI entry points.

use std::os::raw::{c_char, c_int};

use crate::abi::{self, guard_result, parse_json_or_default};
use crate::session::{self, Session};
use crate::strings;
use crate::telemetry;
use crate::wire;
use crate::TsSession;

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
    crate::open::open(dir, options)
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

/// Frees the vision tower's open resources on this session (vision memory
/// sidecar, Part C) -- see `runtime::RealForwardRunner::release_vision_tower`.
///
/// Takes the SAME engine lock a generation turn holds for its whole
/// duration, since this mutates the runner directly and must not race an
/// in-flight image encode. Refused with `TS_ERR_UNSUPPORTED` on a SCRIPTED
/// session (opened only through the test-only `session_for_testing`, never
/// reachable from C) and off macOS -- there is no tower to release in
/// either case. Idempotent: a session whose tower is already closed, or
/// whose install declares no vision tower at all, succeeds and does
/// nothing.
///
/// Does NOT forget an attached sidecar directory or un-declare the
/// install's own vision capability -- `ts_session_info_json`'s
/// `vision.active` (resolved once at open, from the install's own
/// declaration and its preprocessor config, never from whether a tower
/// happens to be open right now) is unaffected. The next image sent
/// through `ts_generate` reopens the tower from wherever it would have
/// opened from before this call.
#[no_mangle]
pub unsafe extern "C" fn ts_session_release_vision(ptr: *const TsSession) -> c_int {
    guard_result(|| {
        let session = session::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let mut engine = session.engine.lock().map_err(|_| {
            (
                abi::TS_ERR_GENERATE,
                "the session is poisoned by an earlier panic".to_string(),
            )
        })?;
        crate::generate::release_vision(&mut engine).map_err(|e| (abi::TS_ERR_UNSUPPORTED, e))
    })
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
