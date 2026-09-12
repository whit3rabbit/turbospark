//! The FIFO generation gate (ROADMAP P1 item 5): admission ordering for the
//! one-generation-at-a-time server.
//!
//! # What problem this solves that the runner mutex did not
//!
//! `RealChatModel` serializes generation on a `std::sync::Mutex`, which
//! gives NO fairness guarantee: a waiter that just arrived can barge past
//! one that has been waiting, so under client concurrency the queue order
//! is arbitrary. Worse, every waiter parks a tokio BLOCKING thread while it
//! spins on that mutex, because the generation itself runs under
//! `spawn_blocking` and blocks on the lock from inside it. Enough
//! concurrent requests pin every blocking thread and the async executor
//! starves.
//!
//! The gate fixes both: admission is a `tokio::sync::Semaphore` of ONE
//! permit whose waiters are granted in the order they arrived (pinned by
//! test, not assumed), and the wait is an ASYNC wait -- a queued request
//! holds no thread at all until its permit arrives, and only then enters
//! `spawn_blocking`, where the runner mutex (kept: it guards the runner,
//! not the order) is uncontended by construction.
//!
//! # Cancellation, and why `acquire` takes the cancel flag
//!
//! A request that disconnects while queued must not wake up into a
//! generation nobody will read. `acquire` checks the flag AFTER the permit
//! arrives and returns `None` when it is already set: the permit drops
//! immediately and the caller folds into the same silent-discard path a
//! mid-generation cancel already uses. This is the queued half of
//! `cancel.rs`'s contract; the in-flight half (the lock is held until the
//! cancelled generation actually stops) is unchanged.
//!
//! # Who is gated
//!
//! The gate is reached through [`crate::model::ChatModel::generation_queue`],
//! which defaults to `None`: scripted backends stay exactly as concurrent
//! as they are today (they serialize on nothing, and the integration suite
//! keeps its timing), and `RealChatModel` constructs one at open -- one per
//! RUNNER, so a future multi-runner registry serializes each runner's
//! traffic on its own gate rather than one global one. The acquisition
//! points are a closed, NON-NESTED set: `run_guarded` (which wraps the
//! guardrail retry loop, so a retry keeps its queue position rather than
//! going back to the tail), the five live-stream spawn sites that bypass
//! it, ollama's `run`, `completions::run_full`, and the embeddings pass.
//! `handler::exec::run_full` itself NEVER acquires -- every caller above it
//! already holds the gate, and a nested acquire on a one-permit semaphore
//! would deadlock.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::cancel::Cancel;

/// One permit's worth of admission. Holds the semaphore permit for as long
/// as it lives, so dropping it (end of request, abort of the awaiting task)
/// is what admits the next waiter.
pub struct GenerationPermit {
    _permit: OwnedSemaphorePermit,
}

/// A one-at-a-time FIFO admission gate. See the module docs.
pub struct GenerationQueue {
    permits: Arc<Semaphore>,
    queued: AtomicUsize,
}

impl GenerationQueue {
    /// A fresh gate with one permit, behind an `Arc` because
    /// `acquire` hands out OWNED permits (they must outlive `&self`
    /// across a `spawn_blocking` move).
    pub fn shared() -> Arc<Self> {
        Arc::new(Self {
            permits: Arc::new(Semaphore::new(1)),
            queued: AtomicUsize::new(0),
        })
    }

    /// Waits for admission in ARRIVAL order, or returns `None` when the
    /// request was cancelled while queued (client gone: fold into the
    /// silent-discard path rather than starting a generation nobody will
    /// read).
    ///
    /// The cancel check is AFTER the wait, deliberately: checking before
    /// would race a disconnect that lands during the wait, which is exactly
    /// the window the check exists for.
    pub async fn acquire(self: &Arc<Self>, cancel: &Cancel) -> Option<GenerationPermit> {
        self.queued.fetch_add(1, Ordering::Release);
        let permit = self.permits.clone().acquire_owned().await;
        self.queued.fetch_sub(1, Ordering::Release);
        match permit {
            // The semaphore is never closed here, so the `Err` arm is
            // unreachable in practice; treating it as "no admission" keeps
            // this total rather than propagating an invariant nothing can
            // trigger.
            Ok(permit) if !cancel.load(Ordering::Acquire) => {
                Some(GenerationPermit { _permit: permit })
            }
            _ => None,
        }
    }

    /// How many requests are waiting for admission right now. For tests
    /// (deterministic arrival ordering) and for an operator reading a
    /// future metrics surface; not asserted on by production code.
    pub fn queued(&self) -> usize {
        self.queued.load(Ordering::Acquire)
    }
}

/// The live-stream sites' wrap: wait for admission (an ASYNC wait, holding
/// no blocking thread), re-check cancellation, then run the blocking body
/// with the permit held for its whole duration -- matching how long the
/// runner mutex underneath is held, which is the hold the fairness is about.
///
/// Returns `None` when the request was cancelled while queued (the caller
/// does nothing: the client is gone, and an unsent event is the correct
/// silence), `Some(Err)` when the blocking task itself died, and
/// `Some(Ok(v))` with the body's value otherwise. The UNGATED arm (a
/// backend with no queue, i.e. the scripted one) still runs the body under
/// `spawn_blocking` -- these bodies can run for a whole generation, and
/// moving them onto the async executor would park a worker instead of a
/// blocking thread, which is the starvation this module exists to end.
pub(crate) async fn run_gated<T, F>(
    queue: Option<std::sync::Arc<GenerationQueue>>,
    cancel: &Cancel,
    body: F,
) -> Option<std::result::Result<T, tokio::task::JoinError>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let _permit = match queue {
        // The `?` is the cancelled-while-queued case: fold to the caller's
        // silence rather than starting a generation nobody will read.
        Some(queue) => Some(queue.acquire(cancel).await?),
        None => None,
    };
    Some(tokio::task::spawn_blocking(body).await)
}
