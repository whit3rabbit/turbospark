//! Agent Client Protocol adapter for TurboSpark: JSON-RPC over stdio so an
//! editor (Zed and others) can start and drive a local session.
//!
//! Stdio only. No listening socket, no network, no login. See
//! `docs/ACP.md` and `docs/designs/opencode2-parity/OC-12-acp-adapter.md`.

pub mod engine;
pub mod framing;
pub mod paths;
pub mod protocol;
pub mod server;
pub mod sessions;
pub mod tools;

pub use engine::{
    Engine, Message, Role, StopReason, ToolCall, ToolOutput, TurnHost, TurnRequest, UnwiredEngine,
};
pub use server::{serve, AcpConfig};
