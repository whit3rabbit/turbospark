//! The ACP (Agent Client Protocol) subset this adapter speaks, as serde types
//! and message builders.
//!
//! Pinned to ACP protocol version 1 (the integer `protocolVersion` that
//! `initialize` negotiates; the spec lives at agentclientprotocol.com). The
//! shapes below are written from that spec, not from a captured editor
//! session: the fixtures under `tests/fixtures` are hand-written spec-shaped
//! messages, and no real editor has been run against them.
//!
//! v1 method set (the draft's M4 cut):
//!
//! | direction        | method                       | kind         |
//! |------------------|------------------------------|--------------|
//! | client to agent  | `initialize`                 | request      |
//! | client to agent  | `session/new`                | request      |
//! | client to agent  | `session/prompt`             | request      |
//! | client to agent  | `session/cancel`             | notification |
//! | agent to client  | `session/update`             | notification |
//! | agent to client  | `session/request_permission` | request      |
//! | agent to client  | `fs/read_text_file`          | request      |
//! | agent to client  | `fs/write_text_file`         | request      |
//!
//! Not implemented: `authenticate` (no login, by program policy),
//! `session/load`, `session/set_mode`, terminals, MCP servers, images, audio.

use serde::Deserialize;
use serde_json::{json, Value};

pub const PROTOCOL_VERSION: u64 = 1;

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;
/// ACP/LSP-style "request issued before initialize".
pub const NOT_INITIALIZED: i64 = -32002;
/// Implementation-defined: a prompt arrived while one is already running.
pub const SESSION_BUSY: i64 = -32000;

#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

impl RpcError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        RpcError {
            code,
            message: message.into(),
        }
    }
}

/// One decoded inbound message.
#[derive(Debug)]
pub enum Inbound {
    Request {
        id: Value,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
    Response {
        id: Value,
        result: Result<Value, RpcError>,
    },
}

/// Decodes one frame. The error carries the id to answer with (Null when the
/// frame had none or was not JSON) and the error to send.
pub fn parse_inbound(bytes: &[u8]) -> Result<Inbound, (Value, RpcError)> {
    let v: Value = serde_json::from_slice(bytes)
        .map_err(|_| (Value::Null, RpcError::new(PARSE_ERROR, "invalid JSON")))?;
    let Some(obj) = v.as_object() else {
        // Batches are not part of ACP.
        return Err((
            Value::Null,
            RpcError::new(INVALID_REQUEST, "expected a JSON object"),
        ));
    };
    let id = obj.get("id").cloned();
    if let Some(i) = &id {
        if !(i.is_string() || i.is_number()) {
            return Err((
                Value::Null,
                RpcError::new(INVALID_REQUEST, "id must be a string or a number"),
            ));
        }
    }
    if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err((
            id.unwrap_or(Value::Null),
            RpcError::new(INVALID_REQUEST, "jsonrpc must be \"2.0\""),
        ));
    }
    if let Some(m) = obj.get("method") {
        let Some(method) = m.as_str() else {
            return Err((
                id.unwrap_or(Value::Null),
                RpcError::new(INVALID_REQUEST, "method must be a string"),
            ));
        };
        let params = obj.get("params").cloned().unwrap_or(Value::Null);
        return Ok(match id {
            Some(id) => Inbound::Request {
                id,
                method: method.to_string(),
                params,
            },
            None => Inbound::Notification {
                method: method.to_string(),
                params,
            },
        });
    }
    let Some(id) = id else {
        return Err((
            Value::Null,
            RpcError::new(INVALID_REQUEST, "no method and no id"),
        ));
    };
    if let Some(e) = obj.get("error") {
        let code = e
            .get("code")
            .and_then(Value::as_i64)
            .unwrap_or(INTERNAL_ERROR);
        let message = e
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        return Ok(Inbound::Response {
            id,
            result: Err(RpcError { code, message }),
        });
    }
    match obj.get("result") {
        Some(r) => Ok(Inbound::Response {
            id,
            result: Ok(r.clone()),
        }),
        None => Err((
            id,
            RpcError::new(INVALID_REQUEST, "response has neither result nor error"),
        )),
    }
}

pub fn response_ok(id: &Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

pub fn response_err(id: &Value, e: &RpcError) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": e.code, "message": e.message}})
}

pub fn request(id: u64, method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

pub fn notification(method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "method": method, "params": params})
}

// ---- client to agent parameter types ------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub protocol_version: u64,
    #[serde(default)]
    pub client_capabilities: ClientCapabilities,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(default)]
    pub fs: FsCapabilities,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FsCapabilities {
    #[serde(default)]
    pub read_text_file: bool,
    #[serde(default)]
    pub write_text_file: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionParams {
    pub cwd: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptParams {
    pub session_id: String,
    pub prompt: Vec<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelParams {
    pub session_id: String,
}

/// Flattens prompt content blocks to text. Baseline ACP agents must accept
/// `text` and `resource_link`; everything else is refused by name because
/// `initialize` advertised no image, audio, or embedded-context support.
pub fn prompt_to_text(blocks: &[Value]) -> Result<String, RpcError> {
    let mut out = String::new();
    for b in blocks {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => {
                let t = b.get("text").and_then(Value::as_str).ok_or_else(|| {
                    RpcError::new(INVALID_PARAMS, "text block without a text string")
                })?;
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(t);
            }
            Some("resource_link") => {
                let uri = b.get("uri").and_then(Value::as_str).unwrap_or("");
                let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                if !out.is_empty() {
                    out.push('\n');
                }
                // Only the reference is passed on; the adapter never fetches it.
                out.push_str(&format!("[linked resource: {name} {uri}]"));
            }
            Some(other) => {
                return Err(RpcError::new(
                    INVALID_PARAMS,
                    format!("unsupported prompt content type: {}", clip(other, 32)),
                ))
            }
            None => return Err(RpcError::new(INVALID_PARAMS, "prompt block without a type")),
        }
    }
    Ok(out)
}

fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

// ---- agent to client message builders ------------------------------------

pub fn initialize_result(agent_version: &str) -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "agentCapabilities": {
            "loadSession": false,
            "promptCapabilities": {"image": false, "audio": false, "embeddedContext": false},
            "mcpCapabilities": {"http": false, "sse": false}
        },
        "agentInfo": {"name": "turbospark", "title": "TurboSpark", "version": agent_version},
        "authMethods": []
    })
}

pub fn text_chunk_update(session_id: &str, text: &str) -> Value {
    notification(
        "session/update",
        json!({
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text}
            }
        }),
    )
}

pub fn tool_call_update(session_id: &str, call: &Value) -> Value {
    let mut u = call.clone();
    u["sessionUpdate"] = json!("tool_call");
    notification(
        "session/update",
        json!({"sessionId": session_id, "update": u}),
    )
}

pub fn tool_call_status_update(
    session_id: &str,
    tool_call_id: &str,
    status: &str,
    text: &str,
) -> Value {
    notification(
        "session/update",
        json!({
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_call_id,
                "status": status,
                "content": [{"type": "content", "content": {"type": "text", "text": text}}]
            }
        }),
    )
}

pub const OPT_ALLOW_ONCE: &str = "allow_once";
pub const OPT_ALLOW_ALWAYS: &str = "allow_always";
pub const OPT_REJECT_ONCE: &str = "reject_once";
pub const OPT_REJECT_ALWAYS: &str = "reject_always";

pub fn permission_params(session_id: &str, tool_call: &Value) -> Value {
    json!({
        "sessionId": session_id,
        "toolCall": tool_call,
        "options": [
            {"optionId": OPT_ALLOW_ONCE, "name": "Allow once", "kind": "allow_once"},
            {"optionId": OPT_ALLOW_ALWAYS, "name": "Always allow this session", "kind": "allow_always"},
            {"optionId": OPT_REJECT_ONCE, "name": "Reject", "kind": "reject_once"},
            {"optionId": OPT_REJECT_ALWAYS, "name": "Always reject this session", "kind": "reject_always"}
        ]
    })
}

/// What a `session/request_permission` result selected, or None for anything
/// else (cancelled, malformed, an option id we never offered). None is deny.
pub fn selected_option(result: &Value) -> Option<&str> {
    let o = result.get("outcome")?;
    if o.get("outcome").and_then(Value::as_str) != Some("selected") {
        return None;
    }
    let id = o.get("optionId").and_then(Value::as_str)?;
    [
        OPT_ALLOW_ONCE,
        OPT_ALLOW_ALWAYS,
        OPT_REJECT_ONCE,
        OPT_REJECT_ALWAYS,
    ]
    .into_iter()
    .find(|k| *k == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_request_notification_response() {
        let r = parse_inbound(br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#);
        assert!(matches!(r, Ok(Inbound::Request { .. })));
        let n = parse_inbound(br#"{"jsonrpc":"2.0","method":"session/cancel","params":{}}"#);
        assert!(matches!(n, Ok(Inbound::Notification { .. })));
        let ok = parse_inbound(br#"{"jsonrpc":"2.0","id":"a","result":{}}"#);
        assert!(matches!(ok, Ok(Inbound::Response { result: Ok(_), .. })));
        let er = parse_inbound(br#"{"jsonrpc":"2.0","id":2,"error":{"code":-1,"message":"m"}}"#);
        assert!(matches!(er, Ok(Inbound::Response { result: Err(_), .. })));
    }

    #[test]
    fn rejects_malformed() {
        assert_eq!(parse_inbound(b"not json").unwrap_err().1.code, PARSE_ERROR);
        assert_eq!(parse_inbound(b"[1]").unwrap_err().1.code, INVALID_REQUEST);
        assert_eq!(
            parse_inbound(br#"{"id":1,"method":"x"}"#)
                .unwrap_err()
                .1
                .code,
            INVALID_REQUEST
        );
        assert_eq!(
            parse_inbound(br#"{"jsonrpc":"2.0","id":null,"method":"x"}"#)
                .unwrap_err()
                .1
                .code,
            INVALID_REQUEST
        );
        assert_eq!(
            parse_inbound(br#"{"jsonrpc":"2.0","id":3}"#).unwrap_err().0,
            json!(3)
        );
    }

    #[test]
    fn prompt_text_and_refusals() {
        let t = prompt_to_text(&[
            json!({"type":"text","text":"a"}),
            json!({"type":"resource_link","name":"n","uri":"file:///x"}),
        ])
        .unwrap();
        assert!(t.starts_with("a\n[linked resource: n file:///x"));
        assert_eq!(
            prompt_to_text(&[json!({"type":"image","data":"","mimeType":"image/png"})])
                .unwrap_err()
                .code,
            INVALID_PARAMS
        );
        assert!(prompt_to_text(&[json!({"text":"x"})]).is_err());
    }

    #[test]
    fn selected_option_only_accepts_offered_ids() {
        let ok = json!({"outcome":{"outcome":"selected","optionId":"allow_once"}});
        assert_eq!(selected_option(&ok), Some("allow_once"));
        let bad = json!({"outcome":{"outcome":"selected","optionId":"allow_everything"}});
        assert_eq!(selected_option(&bad), None);
        let cancelled = json!({"outcome":{"outcome":"cancelled"}});
        assert_eq!(selected_option(&cancelled), None);
        assert_eq!(selected_option(&json!({})), None);
    }

    #[test]
    fn initialize_result_advertises_text_only() {
        let r = initialize_result("0.8.0");
        assert_eq!(r["protocolVersion"], 1);
        assert_eq!(r["agentCapabilities"]["promptCapabilities"]["image"], false);
        assert_eq!(r["authMethods"], json!([]));
    }
}
