//! Per-request cancellation: the mechanism a client's mid-generation
//! disconnect uses to shorten a running or queued generation rather than
//! let it finish for nobody.
//!
//! [`Cancel`] is `Arc<AtomicBool>` under the name this crate threads through
//! its async call chain (`guardrails::run_guarded`, `handler::exec::run_full`,
//! every SSE handler). `runtime::CancelFlag`, the borrowed `&dyn Fn() -> bool`
//! the decode loop actually polls once per prefill and decoded token, is
//! built from it exactly once, on the blocking thread the generation itself
//! runs on ([`as_cancel_flag`]) -- `CancelFlag` is not `Send`, so it cannot
//! cross a `spawn_blocking` `move` closure the way the `Arc` it reads from
//! can.
//!
//! Two independent detectors set the flag, matched to the two shapes a
//! generation is awaited in across this crate. [`CancelOnDrop`] wraps the SSE
//! byte stream axum hands back: a client that goes away mid-stream is
//! observed by axum DROPPING that stream, which fires even when no chunk was
//! ever in flight to fail a send on (a long prefill sends none at all).
//! [`CancelGuard`] covers the non-streaming and buffered paths, where it is
//! the HANDLER's own `async fn` future that gets dropped when the client
//! disconnects (tokio simply stops polling it) while the generation it
//! `.await`s keeps running on a detached `spawn_blocking` task that a dropped
//! `JoinHandle` does not stop by itself.
//!
//! Cancelling is not an error and this module does not report one: the
//! runtime returns a normal [`runtime::RawDecodeResult`] with
//! [`runtime::StopReason::Cancelled`] and whatever it had generated, and
//! every caller here discards that result silently -- the client that would
//! have read it is the same one that is already gone.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;

/// Per-request cancellation flag. Cheap to clone (an `Arc` bump); every
/// clone observes the same underlying bool.
pub(crate) type Cancel = Arc<AtomicBool>;

pub(crate) fn new_cancel() -> Cancel {
    Arc::new(AtomicBool::new(false))
}

/// The `runtime::CancelFlag` the decode loop polls, built from `cancel`.
/// Call this ONCE, on the blocking thread the generation itself runs on, and
/// pass `&flag` down -- not `cancel` itself, which is not the type
/// `ChatModel::run_completion` takes.
pub(crate) fn as_cancel_flag(cancel: &Cancel) -> impl Fn() -> bool + '_ {
    move || cancel.load(Ordering::Relaxed)
}

/// Sets `cancel` when the wrapped stream is DROPPED -- the signal axum gives
/// for "the client is gone" on an SSE response, independent of whether any
/// chunk was ever attempted. Setting it again on a stream that ends
/// normally is harmless: by then the generation it would have shortened is
/// already over.
pub(crate) struct CancelOnDrop<S> {
    inner: S,
    cancel: Cancel,
}

impl<S> CancelOnDrop<S> {
    pub(crate) fn new(inner: S, cancel: Cancel) -> Self {
        Self { inner, cancel }
    }
}

impl<S: Stream + Unpin> Stream for CancelOnDrop<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Sets `cancel` if dropped before [`CancelGuard::defuse`] is called --
/// exactly what happens when tokio drops a non-streaming handler's
/// `async fn` future because the client disconnected before the response
/// was ready, since the generation it was `.await`ing keeps running on a
/// detached blocking task otherwise.
pub(crate) struct CancelGuard {
    cancel: Cancel,
    defused: bool,
}

impl CancelGuard {
    pub(crate) fn new(cancel: Cancel) -> Self {
        Self {
            cancel,
            defused: false,
        }
    }

    /// Call once the generation this guard was covering has actually
    /// finished (successfully or not) -- there is nothing left for a later
    /// drop to usefully cancel.
    pub(crate) fn defuse(&mut self) {
        self.defused = true;
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if !self.defused {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;

    #[test]
    fn as_cancel_flag_reads_the_live_value() {
        let cancel = new_cancel();
        let flag = as_cancel_flag(&cancel);
        assert!(!flag());
        cancel.store(true, Ordering::Relaxed);
        assert!(flag());
    }

    #[tokio::test]
    async fn dropping_the_wrapped_stream_sets_the_flag() {
        let cancel = new_cancel();
        let inner = stream::iter(vec![1, 2, 3]);
        let wrapped = CancelOnDrop::new(inner, cancel.clone());
        assert!(!cancel.load(Ordering::Relaxed));
        drop(wrapped);
        assert!(cancel.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn draining_the_wrapped_stream_still_yields_every_item() {
        let cancel = new_cancel();
        let inner = stream::iter(vec![1, 2, 3]);
        let mut wrapped = CancelOnDrop::new(inner, cancel);
        let mut items = Vec::new();
        while let Some(item) = wrapped.next().await {
            items.push(item);
        }
        assert_eq!(items, vec![1, 2, 3]);
    }

    #[test]
    fn a_defused_guard_does_not_set_the_flag_on_drop() {
        let cancel = new_cancel();
        let mut guard = CancelGuard::new(cancel.clone());
        guard.defuse();
        drop(guard);
        assert!(!cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn an_undefused_guard_sets_the_flag_on_drop() {
        let cancel = new_cancel();
        let guard = CancelGuard::new(cancel.clone());
        drop(guard);
        assert!(cancel.load(Ordering::Relaxed));
    }
}
