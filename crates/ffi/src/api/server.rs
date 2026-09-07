//! In-process HTTP server C ABI entry points.

use std::os::raw::{c_char, c_int};

use crate::abi::{self, guard_result, parse_json_or_default};
use crate::server::{self, Server};
use crate::session;
use crate::strings;
use crate::wire;
use crate::{TsServer, TsSession};

/// The install directory's own file name, which is the id a client puts in
/// a request's `model` field.
///
/// Matches `RealChatModel::open`'s convention rather than using the full
/// `model_path`: a path is not something a `GET /v1/models` row should echo,
/// and it is not something anyone would type into a client's model field.
fn model_id_of(session: &session::Session) -> String {
    std::path::Path::new(&session.info.model_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| session.info.model_path.clone())
}

/// Starts an in-process HTTP server and writes a handle to `out`.
///
/// `session` MAY BE NULL, meaning start with no model attached: the server
/// binds and answers `GET /health` (reporting `"state": "empty"`), and every
/// generation route returns 503 until `ts_server_attach_session` adds one.
/// That is the state a GUI starts a server in before the user has chosen
/// what to load. A non-null `session` is exactly equivalent to starting with
/// null and calling `ts_server_attach_session` immediately.
///
/// `options_json` may be null or `{}`. Recognised keys: `port` (0, the
/// default, asks the OS for one), `apiKey`.
///
/// The server keeps EVERY ATTACHED SESSION'S ENGINE ALIVE independently of
/// the `TsSession *` it came from: `ts_session_close` drops only the
/// caller's own reference, not the model the server is still serving. Stop
/// the server, or detach that one model, to release it.
/// `"on"` | `"off"` | absent, with the grammar and the meanings taken from
/// `turbospark-server --guardrails` rather than restated.
///
/// An unrecognized spelling is an ERROR and never a silent fallback: a host
/// sending `"disabled"` and getting guardrails anyway would have no way to
/// tell, and the whole point of the option is that the served path agrees
/// with what the host thinks it configured.
fn guardrails_config(raw: Option<&str>) -> Result<turbospark_server::GuardrailConfig, String> {
    match raw {
        None => Ok(turbospark_server::GuardrailConfig::default()),
        Some("on") => Ok(turbospark_server::GuardrailConfig::default()),
        Some("off") => Ok(turbospark_server::GuardrailConfig::OFF),
        Some(other) => Err(format!(
            "guardrails must be \"on\" or \"off\", got \"{other}\""
        )),
    }
}

fn reasoning_effort(raw: Option<&str>) -> Result<tokenizer::ReasoningEffort, String> {
    match raw {
        None => Ok(tokenizer::ReasoningEffort::Off),
        Some(s) => match s.trim().to_lowercase().as_str() {
            "off" => Ok(tokenizer::ReasoningEffort::Off),
            "low" => Ok(tokenizer::ReasoningEffort::Low),
            "medium" => Ok(tokenizer::ReasoningEffort::Medium),
            "high" => Ok(tokenizer::ReasoningEffort::High),
            "xhigh" | "extra_high" | "extrahigh" => Ok(tokenizer::ReasoningEffort::XHigh),
            other => Err(format!(
                "reasoning must be off, low, medium, high or xhigh, got \"{other}\""
            )),
        },
    }
}

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
        let options = parse_json_or_default::<wire::ServerOptions>(options_json, "options")?;
        let guardrails = guardrails_config(options.guardrails.as_deref())
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let default_reasoning = reasoning_effort(options.default_reasoning.as_deref())
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let default_system = options.default_system.filter(|s| !s.trim().is_empty());
        let server = Server::start(
            options.port,
            options.api_key,
            guardrails,
            default_system,
            default_reasoning,
        )
        .map_err(|e| (abi::TS_ERR_OPEN, e))?;
        if let Some(endpoint) = &options.hf_endpoint {
            if !endpoint.trim().is_empty() {
                catalog::set_hf_endpoint_override(Some(endpoint.trim().to_string()));
            }
        }
        // Attached AFTER the bind, through the same call a later attach
        // takes, so there is one code path rather than two. A failure here
        // drops `server`, which stops the thread it just started.
        if !session.is_null() {
            let session =
                session::borrow(session).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
            server
                .attach(session.core(), model_id_of(session))
                .map_err(|e| (abi::TS_ERR_OPEN, e))?;
        }
        if let Some(emb) = &options.embedding_model {
            server
                .attach_embedding_model(emb)
                .map_err(|e| (abi::TS_ERR_OPEN, e))?;
        }
        *out = Box::into_raw(Box::new(server));
        Ok(())
    })
}

/// Adds an open session's model to a running server, and writes the id
/// clients address it by to `out` (the install directory's own name).
///
/// **REFUSES A DUPLICATE ID RATHER THAN RENAMING IT.** The id is what a
/// request's `model` field names and what `ts_server_detach_model` keys on,
/// so a silently-suffixed second copy would be addressable under a name the
/// caller never learned, and a detach under the name they DO know would
/// remove the wrong one. Two sessions on one install directory is a caller
/// mistake and is reported as one.
///
/// The server holds its own reference to the session's engine from here on
/// (see `ts_server_start`).
#[no_mangle]
pub unsafe extern "C" fn ts_server_attach_session(
    server: *const TsServer,
    session: *const TsSession,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let server = server::borrow(server).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let session = session::borrow(session).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let id = server
            .attach(session.core(), model_id_of(session))
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        strings::emit(&id, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Adds an embedding model to a running server, enabling `/v1/embeddings`,
/// `/api/embeddings`, and `/api/embed`.
///
/// Writes the attached model id to `out`.
#[no_mangle]
pub unsafe extern "C" fn ts_server_attach_embedding_model(
    server: *const TsServer,
    model_path: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let server = server::borrow(server).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let path = strings::required(model_path, "modelPath")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let id = server
            .attach_embedding_model(path)
            .map_err(|e| (abi::TS_ERR_OPEN, e))?;
        strings::emit(&id, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Removes a model from a running server by id, releasing the server's
/// reference to its engine.
///
/// Returns `TS_OK` when one was attached under `model_id` and
/// `TS_ERR_INVALID_ARGUMENT` when none was -- a detach of something that is
/// not there is reported rather than silently succeeding, because the
/// caller's own model list and this one have gone out of step and the whole
/// point of the call is to keep them together.
#[no_mangle]
pub unsafe extern "C" fn ts_server_detach_model(
    server: *const TsServer,
    model_id: *const c_char,
) -> c_int {
    guard_result(|| {
        let server = server::borrow(server).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let id = strings::required(model_id, "modelId")
            .map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        if server.detach(id) {
            Ok(())
        } else {
            Err((
                abi::TS_ERR_INVALID_ARGUMENT,
                format!("no model with id '{id}' is attached to this server"),
            ))
        }
    })
}

/// Signals the server to stop, blocks until its background thread has
/// actually exited, and frees the handle. Null is a no-op.
///
/// This is also what releases every attached model.
#[no_mangle]
pub unsafe extern "C" fn ts_server_stop(ptr: *mut TsServer) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// `{ "port", "host", "modelId", "models", "authEnabled", "uptimeSeconds" }`.
///
/// `port` and `host` are the ACTUALLY bound pair, never the requested one --
/// `port: 0` in `ts_server_start`'s options asks the OS to choose. `modelId`
/// is the FIRST attached model and is there for a one-model reader; `models`
/// is the list that stays correct.
#[no_mangle]
pub unsafe extern "C" fn ts_server_info_json(ptr: *const TsServer, out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        let server = server::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let json =
            serde_json::to_string(&server.info()).map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

/// Takes up to `max` buffered request events and writes
/// `{ "events": [...], "dropped": N }` to `out`.
///
/// **DRAINING, NOT PEEKING: an event is returned exactly once.** A host polls
/// this on a timer and appends what it gets to its own log.
///
/// `max` bounds ONE call rather than the buffer. Anything over it stays
/// queued for the next poll, so a burst arrives late rather than being lost.
/// `max == 0` is read as "no bound", which is what a host draining before it
/// shuts down wants.
///
/// `dropped` counts events the ring discarded since the PREVIOUS poll,
/// oldest first, and is nonzero only for a host that stopped draining long
/// enough to overrun ~2,000 events. It is reported rather than swallowed
/// because a silently lossy log is indistinguishable from an idle server.
#[no_mangle]
pub unsafe extern "C" fn ts_server_poll_events_json(
    ptr: *const TsServer,
    max: u32,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        let server = server::borrow(ptr).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))?;
        let limit = if max == 0 { usize::MAX } else { max as usize };
        let json = serde_json::to_string(&server.drain_events(limit))
            .map_err(|e| (abi::TS_ERR_JSON, e.to_string()))?;
        strings::emit(&json, out).map_err(|e| (abi::TS_ERR_INVALID_ARGUMENT, e))
    })
}

#[cfg(test)]
mod guardrails_option_tests {
    use super::guardrails_config;

    /// Absent is the ENGINE default, not `OFF`. A host that sends `{}` -- the
    /// documented "everything automatic" options bag, and what every caller
    /// predating this option sends -- must keep the guardrails it has always
    /// had.
    #[test]
    fn absent_is_the_engine_default_and_not_off() {
        let resolved = guardrails_config(None).expect("absent is legal");
        assert_eq!(resolved, turbospark_server::GuardrailConfig::default());
        assert!(resolved.active(), "the engine default is on");
    }

    #[test]
    fn on_and_off_take_the_servers_own_spellings() {
        assert_eq!(
            guardrails_config(Some("on")).expect("on is legal"),
            turbospark_server::GuardrailConfig::default()
        );
        let off = guardrails_config(Some("off")).expect("off is legal");
        assert_eq!(off, turbospark_server::GuardrailConfig::OFF);
        assert!(!off.active());
    }

    /// **AN UNRECOGNIZED SPELLING IS AN ERROR, NOT A FALLBACK.** A host
    /// sending "disabled" and getting guardrails anyway has no way to tell,
    /// and the whole point of the option is that the served path agrees with
    /// what the host thinks it configured. The message names the value back.
    #[test]
    fn an_unrecognized_spelling_is_refused_by_name() {
        let err = guardrails_config(Some("disabled")).expect_err("must refuse");
        assert!(err.contains("disabled"), "{err}");
        assert!(err.contains("\"on\""), "{err}");
        // Case matters, deliberately: `crates/server/src/args.rs` matches
        // these two literals and nothing else, so accepting "OFF" here would
        // make the ABI more permissive than the flag it mirrors.
        assert!(guardrails_config(Some("OFF")).is_err());
        assert!(guardrails_config(Some("")).is_err());
    }
}
