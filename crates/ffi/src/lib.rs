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
//! rest of the workspace forbids it. Every `extern "C"` body in [`api`] is a
//! call to [`abi::guard`] or [`abi::guard_result`] and nothing else, because
//! unwinding across the boundary is undefined behaviour and this workspace
//! cannot use `panic = "abort"` (see that function's docs).
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

/// C ABI error codes and exception-safe wrapper boundary.
pub mod abi;
/// C ABI entry points organized by domain.
pub mod api;
mod generate;
mod models;
#[cfg(target_os = "macos")]
mod open;
mod server;
mod server_model;
mod session;
mod strings;
mod telemetry;
mod testing;
/// macOS only, for the reason `open` is: everything here ends at a
/// `RealForwardRunner`. The portable refusal lives in `generate`.
#[cfg(target_os = "macos")]
mod vision;
/// JSON wire structures exchanged across the C ABI.
pub mod wire;

pub use api::*;
pub use generate::{TS_EVENT_CONTENT, TS_EVENT_PREFILL, TS_EVENT_REASONING};
pub use models::{TS_INSTALL_BYTES, TS_INSTALL_STAGE};
pub use server::Server;
pub use session::Session;
#[doc(hidden)]
pub use testing::session_for_testing;

/// The opaque handle a caller holds. `TsSession *` in C.
pub type TsSession = Session;

/// The opaque in-process-server handle a caller holds. `TsServer *` in C.
pub type TsServer = Server;

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
