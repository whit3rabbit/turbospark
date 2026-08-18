//! The opaque session handle, and the one piece of this crate whose SHAPE is
//! load-bearing rather than incidental.
//!
//! **THE CANCEL FLAG LIVES OUTSIDE THE MUTEX, AND THAT IS THE WHOLE DESIGN.**
//! A GUI runs generation on a background thread, which holds the engine lock
//! for the entire turn, and presses Stop on the main thread. If the flag were
//! inside the lock, `ts_session_cancel` would block until the generation it
//! is trying to stop had finished -- a Stop button that works only once the
//! model is done, i.e. a deadlock the user experiences as a frozen window.
//! The `AtomicBool` is reachable without the lock, so cancelling is
//! wait-free.
//!
//! The corollary is that `Session` must be `Sync` while `RealForwardRunner`
//! is not. `Mutex` supplies that, and it is the same arrangement
//! `crates/server`'s `RealChatModel` uses for the same reason: one runner per
//! process, `&mut self` to decode, so callers queue rather than run.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use selection::ShapingConfig;
use tokenizer::MfTokenizer;

use crate::wire::SessionInfo;

/// What a session decodes through.
///
/// `Scripted` is NOT in `turbospark.h` and is deliberately not reachable
/// from C. It exists so this crate's own tests can drive the whole generate
/// path -- the channel split, the cancel plumbing, the event callback,
/// the result JSON -- on any platform and with no multi-gigabyte install.
/// The alternative was testing the FFI layer only against a real model,
/// which would mean the threading contract above was covered by nothing.
pub(crate) enum Engine {
    #[cfg(target_os = "macos")]
    Real(Box<runtime::RealForwardRunner>),
    Scripted(Box<runtime::ScriptedLogitProducer>),
}

/// Everything one opened model needs, and the thing `TsSession *` points at.
pub struct Session {
    /// The engine, serialized. A turn holds this for its whole duration.
    pub(crate) engine: Mutex<Engine>,
    pub(crate) tokenizer: MfTokenizer,
    /// Read by the decode loop's predicate, set by `ts_session_cancel` from
    /// any thread. `Arc` because the predicate closure outlives the borrow
    /// of the session inside `ts_generate`.
    pub(crate) cancel: Arc<AtomicBool>,
    /// Resolved sampling defaults are per CALL, so nothing is cached here;
    /// what IS cached is everything resolved at open, which a later call
    /// must not re-derive.
    pub(crate) info: SessionInfo,
    /// The RESOLVED context window. Read from here and never from a
    /// request: under `auto` the request carries no number, and the KV cache
    /// was allocated at this one.
    pub(crate) max_context: u32,
    /// Resolved once at open, because resolving per turn would let Low Power
    /// Mode toggling mid-conversation change the pace for reasons the caller
    /// never asked about.
    pub(crate) rate: runtime::RateControl,
}

impl Session {
    /// Raises the cancel flag. Wait-free, and safe from any thread.
    pub fn cancel(&self) {
        // `Release` pairs with the decode loop's `Acquire`: everything this
        // thread did before pressing Stop is visible to the thread that
        // observes the flag. Nothing here depends on that today, but a
        // `Relaxed` store would be a promise this type should not make.
        self.cancel.store(true, Ordering::Release);
    }

    /// Clears the flag. Called at the start of every generation, so a Stop
    /// pressed after the last turn ended cannot cancel the next one before
    /// it has produced a token.
    pub(crate) fn arm(&self) -> Arc<AtomicBool> {
        self.cancel.store(false, Ordering::Release);
        Arc::clone(&self.cancel)
    }

    /// A validated sampling configuration from the per-call options.
    pub(crate) fn shaping(
        &self,
        o: &crate::wire::GenerateOptions,
    ) -> Result<ShapingConfig, String> {
        ShapingConfig::new(
            o.temperature,
            o.top_k,
            Some(o.top_p),
            o.repetition_penalty,
            o.seed,
        )
        .map_err(|e| e.to_string())
    }
}

/// Turns a `TsSession *` back into a borrow.
///
/// # Safety
/// `ptr` must be null, or a pointer returned by `ts_session_open` and not
/// yet closed.
pub(crate) unsafe fn borrow<'a>(ptr: *const Session) -> Result<&'a Session, String> {
    ptr.as_ref()
        .ok_or_else(|| "session must not be null".to_string())
}
