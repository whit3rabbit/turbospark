//! In-process HTTP server C ABI entry points.

use std::os::raw::{c_char, c_int};

use crate::abi::{self, guard_result, parse_json_or_default};
use crate::server::{self, Server};
use crate::session;
use crate::strings;
use crate::wire;
use crate::{TsServer, TsSession};

/// Starts an in-process HTTP server sharing `session`'s already-open model,
/// and writes a handle to `out`.
///
/// `options_json` may be null or `{}`. Recognised keys: `port` (0, the
/// default, asks the OS for one), `apiKey`.
///
/// The server keeps the SESSION'S ENGINE ALIVE independently of `session`:
/// `ts_session_close` on `session` after this call drops only the caller's
/// own reference, not the model the server is still serving. Stop the
/// server with `ts_server_stop` to release it.
#[no_mangle]
pub unsafe extern "C" fn ts_server_start(
    session: *const TsSession,
    options_json: *const c_char,
    out: *mut *mut TsServer,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err((abi::TS_ERR_INVALID_ARGUMENT, "out must not be null".into()));
        }
        let session = session::borrow(session).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let options = parse_json_or_default::<wire::ServerOptions>(options_json, "options")?;
        // The install directory's own file name, matching
        // `RealChatModel::open`'s `model_id` convention -- `model_path` is
        // a full path and not something a `GET /v1/models` row should echo.
        let model_id = std::path::Path::new(&session.info.model_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| session.info.model_path.clone());
        let server = Server::start(session.core(), model_id, options.port, options.api_key)
            .map_err(|e| (abi::TS_ERR_OPEN, e))?;
        *out = Box::into_raw(Box::new(server));
        Ok(())
    })
}

/// Signals the server to stop, blocks until its background thread has
/// actually exited, and frees the handle. Null is a no-op.
#[no_mangle]
pub unsafe extern "C" fn ts_server_stop(ptr: *mut TsServer) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// `{ "port", "modelId", "authEnabled" }`. `port` is the ACTUALLY bound
/// port, never the requested one -- `port: 0` in `ts_server_start`'s options
/// asks the OS to choose.
#[no_mangle]
pub unsafe extern "C" fn ts_server_info_json(ptr: *const TsServer, out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let server = server::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json =
            serde_json::to_string(&server.info()).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}
