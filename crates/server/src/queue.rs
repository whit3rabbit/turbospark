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
#[derive(Clone)]
pub struct GenerationPermit {
    // The async request and detached blocking body can each own a clone. The
    // semaphore slot returns only after the last owner exits, so cancellation
    // cannot admit another model run while this body is still using it.
    _permit: Arc<OwnedSemaphorePermit>,
}

/// A one-at-a-time FIFO admission gate. See the module docs.
pub struct GenerationQueue {
    permits: Arc<Semaphore>,
    queued: AtomicUsize,
}

struct QueuedCountGuard<'a>(&'a AtomicUsize);

impl Drop for QueuedCountGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
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
        let queued = QueuedCountGuard(&self.queued);
        let permit = self.permits.clone().acquire_owned().await;
        drop(queued);
        match permit {
            // The semaphore is never closed here, so the `Err` arm is
            // unreachable in practice; treating it as "no admission" keeps
            // this total rather than propagating an invariant nothing can
            // trigger.
            Ok(permit) if !cancel.load(Ordering::Acquire) => Some(GenerationPermit {
                _permit: Arc::new(permit),
            }),
            _ => None,
        }
    }

    /// How many requests are waiting for admission right now. For tests
    /// (deterministic arrival ordering) and for an operator reading a
    /// future metrics surface; not asserted on by production code.
    pub fn queued(&self) -> usize {
        self.queued.load(Ordering::Acquire)
    }

    /// How many requests either hold the generation permit or are waiting
    /// for it. This is the pool dispatcher's advisory load metric; unlike
    /// [`Self::queued`], it does not mistake an active runner for an idle one.
    pub fn load(&self) -> usize {
        self.queued.load(Ordering::Acquire) + usize::from(self.permits.available_permits() == 0)
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
    let permit = match queue {
        // The `?` is the cancelled-while-queued case: fold to the caller's
        // silence rather than starting a generation nobody will read.
        Some(queue) => Some(queue.acquire(cancel).await?),
        None => None,
    };
    Some(run_blocking(permit, body).await)
}

/// Runs blocking model work while retaining any permit acquired by an outer
/// request flow. Cloned permits let a guardrail retry loop keep its FIFO place
/// while each detached generation body independently protects active work.
pub(crate) async fn run_blocking<T, F>(
    permit: Option<GenerationPermit>,
    body: F,
) -> std::result::Result<T, tokio::task::JoinError>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // A dropped JoinHandle does not stop spawn_blocking work that has
    // already started. Keep queue load and admission blocked until that
    // work exits, even if the async response waiter is cancelled.
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        body()
    })
    .await
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::{run_blocking, run_gated, GenerationQueue};
    use crate::cancel::new_cancel;

    #[tokio::test]
    async fn dropping_a_waiter_decrements_queue_load() {
        let queue = GenerationQueue::shared();
        let cancel = new_cancel();
        let active = queue
            .acquire(&cancel)
            .await
            .expect("queue admits first work");

        let waiting_queue = queue.clone();
        let waiting_cancel = cancel.clone();
        let waiter = tokio::spawn(async move { waiting_queue.acquire(&waiting_cancel).await });
        tokio::time::timeout(Duration::from_secs(5), async {
            while queue.queued() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("second request reaches the queue");
        assert_eq!(queue.queued(), 1);

        waiter.abort();
        assert!(matches!(waiter.await, Err(error) if error.is_cancelled()));
        assert_eq!(queue.queued(), 0, "cancellation releases its queued count");
        assert_eq!(queue.load(), 1, "the active request still owns the permit");

        drop(active);
        assert_eq!(queue.load(), 0, "the idle queue has no stale load");
    }

    #[tokio::test]
    async fn dropping_run_gated_waiter_keeps_permit_until_blocking_body_exits() {
        let queue = GenerationQueue::shared();
        let cancel = new_cancel();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let waiting_queue = queue.clone();
        let waiter = tokio::spawn(async move {
            run_gated(Some(waiting_queue), &cancel, move || {
                let _ = started_tx.send(());
                let _ = release_rx.recv_timeout(Duration::from_secs(10));
                let _ = finished_tx.send(());
            })
            .await
        });

        tokio::time::timeout(Duration::from_secs(5), started_rx)
            .await
            .expect("blocking body starts")
            .expect("start signal arrives");
        assert_eq!(queue.load(), 1, "body owns the generation permit");

        waiter.abort();
        assert!(waiter.await.expect_err("waiter was aborted").is_cancelled());
        let load_while_body_is_held = queue.load();

        release_tx.send(()).expect("blocking body is still waiting");
        tokio::time::timeout(Duration::from_secs(5), finished_rx)
            .await
            .expect("blocking body exits")
            .expect("finish signal arrives");
        tokio::time::timeout(Duration::from_secs(5), async {
            while queue.load() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("permit is released after the body exits");

        assert_eq!(
            load_while_body_is_held, 1,
            "dropping the waiter must not release capacity used by a still-running blocking body"
        );
        assert_eq!(queue.load(), 0, "body exit releases the permit");
    }

    #[tokio::test]
    async fn dropping_direct_waiter_keeps_cloned_permit_until_blocking_body_exits() {
        let queue = GenerationQueue::shared();
        let cancel = new_cancel();
        let permit = queue
            .acquire(&cancel)
            .await
            .expect("queue should admit work");
        let blocking_permit = permit.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let waiter = tokio::spawn(async move {
            run_blocking(Some(blocking_permit), move || {
                let _ = started_tx.send(());
                let _ = release_rx.recv_timeout(Duration::from_secs(10));
                let _ = finished_tx.send(());
            })
            .await
        });

        tokio::time::timeout(Duration::from_secs(5), started_rx)
            .await
            .expect("blocking body starts")
            .expect("start signal arrives");
        waiter.abort();
        assert!(waiter.await.expect_err("waiter was aborted").is_cancelled());
        drop(permit);
        assert_eq!(queue.load(), 1, "blocking body still owns the last clone");

        release_tx.send(()).expect("blocking body is still waiting");
        tokio::time::timeout(Duration::from_secs(5), finished_rx)
            .await
            .expect("blocking body exits")
            .expect("finish signal arrives");
        tokio::time::timeout(Duration::from_secs(5), async {
            while queue.load() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("permit is released after the body exits");
    }
}
