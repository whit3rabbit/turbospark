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
//!
//! **`Session` IS A THIN, CLONEABLE HANDLE OVER `SessionCore`, AND THAT SPLIT
//! IS WHAT LETS `ts_server_start` SHARE THE MODEL WITHOUT OPENING A SECOND
//! ONE.** `SessionCore` carries the engine, the tokenizer and everything
//! resolved at open; `Session` is an `Arc<SessionCore>` newtype. Field access
//! through `Session` reaches `SessionCore` by ordinary `Deref` autoref, so
//! every existing `session.tokenizer` / `session.engine` / ... call site
//! needed no change. What the split buys: `Session::core` hands out an
//! `Arc<SessionCore>` clone that `crate::server::Server` can hold on its own
//! background thread, independent of the opaque `TsSession *`'s lifetime --
//! `ts_session_close` drops the caller's `Arc` reference, and the underlying
//! engine stays alive for as long as a running server (or any other clone)
//! still holds one. Opening a SECOND `RealForwardRunner` to serve the same
//! install through HTTP was considered and declined: it would double the
//! resident mapping and the Metal pipeline compile for an install that can
//! already be double-digit gigabytes, on a machine the GUI itself is running
//! on.

use std::ops::Deref;
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

/// Everything one opened model needs. Shared, never mutated by value after
/// construction -- interior mutability (`Mutex`, the cancel `AtomicBool`) is
/// how a clone reaches the live state.
///
/// `pub` rather than `pub(crate)` only because `Session`'s `Deref::Target`
/// must be at least as visible as `Session` itself; every FIELD stays
/// `pub(crate)`, so nothing outside this crate can actually read one.
pub struct SessionCore {
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
    /// How many tokens a speculative round proposes, or `None` when this
    /// session does not draft ahead. Resolved at open, where the drafter's
    /// state is allocated.
    ///
    /// **A block here is necessary and not sufficient**: acceptance is
    /// `argmax(target) == proposal`, exact only at temperature 0, so
    /// `generate` gates on the TURN's shaping as well. A plain `usize`
    /// rather than a `runtime::SpeculationPlan` because that type is
    /// macOS-only and this struct is not; the human-readable half of the
    /// plan is already carried in `info.speculation`.
    pub(crate) speculation_block: Option<usize>,
    /// The guard tier this session actually opened under (vision memory
    /// sidecar Part B3). Carried so a later image's pixel-budget clamp
    /// resolves against the SAME tier `maxContext` did, never a second,
    /// possibly different one (Gotcha 12's rule, applied to a third call).
    pub(crate) load_policy: runtime::LoadPolicy,
    /// What this install already commits before KV --
    /// `runtime::committed_bytes(dir)`, the same value `maxContext`
    /// resolved against.
    pub(crate) committed_bytes: u64,
    /// This session's own KV cache at [`Self::max_context`], i.e.
    /// `ContextPlan::kv_bytes`.
    pub(crate) kv_bytes: u64,
}

impl SessionCore {
    /// Raises the cancel flag. Wait-free, and safe from any thread.
    pub(crate) fn cancel(&self) {
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

/// The opaque handle `TsSession *` points at. See the module doc for why
/// this is a thin `Arc` wrapper rather than the state itself.
pub struct Session(pub(crate) Arc<SessionCore>);

impl Session {
    pub(crate) fn new(core: SessionCore) -> Self {
        Session(Arc::new(core))
    }

    /// Clones the shared core out from under this handle's own lifetime, for
    /// `ts_server_start` to hold on a background thread. The clone keeps the
    /// engine alive even after `ts_session_close` drops this `Session`.
    pub(crate) fn core(&self) -> Arc<SessionCore> {
        Arc::clone(&self.0)
    }
}

impl Deref for Session {
    type Target = SessionCore;

    fn deref(&self) -> &SessionCore {
        &self.0
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
