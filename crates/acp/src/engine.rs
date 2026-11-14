//! The seam between the adapter and whatever generates tokens.
//!
//! The adapter owns the protocol, the workspace root, the file tools and the
//! permission gate. An [`Engine`] owns the model: it runs one turn, streams
//! text through [`TurnHost::text_delta`], and asks for file tools through
//! [`TurnHost::call_tool`] (which blocks until the tool has been authorized
//! and run, or refused). The engine decides how to render tool results back
//! to the model; the adapter only guarantees that nothing executes without
//! passing the gate.
//!
//! This is the Rust analogue of the Swift agent loop, not a wrapper over it:
//! the Swift loop (`AppToolPermissionEngine`) is not reachable without the
//! app process. See the draft's Implementation Record for what that means for
//! "same risk classification" (a narrower Rust ladder, stated there).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

#[derive(Debug)]
pub struct TurnRequest {
    pub session_id: String,
    /// Canonical workspace root.
    pub cwd: PathBuf,
    /// The conversation so far, with the new user prompt last.
    pub history: Vec<Message>,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Unique within the turn; echoed to the editor as `toolCallId`.
    pub id: String,
    /// `read_file`, `list_dir`, `write_file` or `edit_file`.
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    pub text: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
}

impl StopReason {
    pub fn as_acp(self) -> &'static str {
        match self {
            StopReason::EndTurn => "end_turn",
            StopReason::MaxTokens => "max_tokens",
            StopReason::MaxTurnRequests => "max_turn_requests",
            StopReason::Refusal => "refusal",
        }
    }
}

/// What the adapter lends an engine for the duration of one turn.
pub trait TurnHost {
    /// Streams assistant text to the editor.
    fn text_delta(&mut self, text: &str);
    /// Runs one file tool through the permission gate. Always returns: a
    /// refusal, a path escape, or a failure is a `ToolOutput` with
    /// `is_error` set, so the model can see why and continue.
    fn call_tool(&mut self, call: ToolCall) -> ToolOutput;
    /// True once the editor cancelled the turn or went away. Engines must
    /// poll this between tokens and between tool calls; the adapter reports
    /// `cancelled` to the editor regardless of what the engine returns.
    fn cancelled(&self) -> bool;
}

pub trait Engine: Send + Sync {
    /// Recorded in the session record. Not sent to the editor.
    fn model_id(&self) -> String;
    fn run_turn(&self, req: &TurnRequest, host: &mut dyn TurnHost) -> Result<StopReason, String>;
}

/// The engine the `turbospark-acp` binary ships with until a real checkpoint
/// is wired in: every turn fails with an explanatory error. It exists so the
/// protocol, session and permission layers can be driven end to end (and
/// tested as a process) without pretending a model answered.
pub struct UnwiredEngine;

impl Engine for UnwiredEngine {
    fn model_id(&self) -> String {
        "unwired".to_string()
    }
    fn run_turn(&self, _req: &TurnRequest, _host: &mut dyn TurnHost) -> Result<StopReason, String> {
        Err("no local model is wired to the ACP adapter in this build; \
             the protocol layer is live but generation is not"
            .to_string())
    }
}
