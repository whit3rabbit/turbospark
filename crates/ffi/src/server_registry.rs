//! The models a running server is serving, and the events it has recorded.
//!
//! Both are mutable while the server runs, which is the whole difference
//! between this and `turbospark_server::registry::StaticRegistry`: a GUI
//! starts a server before it has loaded anything, then attaches and detaches
//! models from it without rebinding the socket. So the router holds an `Arc`
//! of each of these and every read goes through a lock.
//!
//! **THE REGISTRY IS WHAT KEEPS A MODEL RESIDENT, AND DETACHING IS WHAT
//! RELEASES IT.** Each entry owns an `FfiChatModel`, which owns an
//! `Arc<SessionCore>` (`session.rs`'s module doc). `ts_session_close` on the
//! session an entry was built from drops only the CALLER's reference; the
//! engine, its KV cache and its compiled Metal pipelines stay mapped until
//! this map lets go. That is `ts_server_stop` for the whole set, or
//! `ts_server_detach_model` for one -- and a host that clears a session
//! without doing either has leaked a multi-gigabyte mapping with nothing in
//! its own UI still showing the model as loaded.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use turbospark_server::observe::{ServerEvent, ServerObserver};
use turbospark_server::registry::{resolve_among, ModelRegistry, ModelRow, Resolution};
use turbospark_server::ChatModel;

/// The models attached to one running server.
#[derive(Default)]
pub(crate) struct LiveRegistry {
    // A `Vec` rather than a `HashMap` because attachment ORDER is what a
    // host lists them in and what `/v1/models` reports, and the set is a
    // handful of entries at most -- a machine that can hold ten resident
    // models at once does not exist. Uniqueness is enforced on insert.
    models: Mutex<Vec<Arc<dyn ChatModel>>>,
}

impl LiveRegistry {
    /// Adds a model, or refuses if its id is already attached.
    ///
    /// **A DUPLICATE ID IS REFUSED BY NAME RATHER THAN SUFFIXED.** The id is
    /// what a client puts in a request's `model` field and what
    /// `detach` keys on, so inventing an `id #2` would create a second name
    /// for a thing the host believes it attached once -- and then `detach`
    /// on the name the host knows would remove the wrong one. Two sessions
    /// on one install directory is a caller mistake, and this is where it is
    /// cheapest to say so.
    pub(crate) fn attach(&self, model: Arc<dyn ChatModel>) -> Result<String, String> {
        let id = model.model_id().to_string();
        let mut models = self.models.lock().unwrap_or_else(|p| p.into_inner());
        if models.iter().any(|m| m.model_id() == id) {
            return Err(format!(
                "a model with id '{id}' is already attached to this server; \
                 detach it first, or open the second install under a different directory name"
            ));
        }
        models.push(model);
        Ok(id)
    }

    /// Removes a model by id. Returns whether one was there.
    pub(crate) fn detach(&self, id: &str) -> bool {
        let mut models = self.models.lock().unwrap_or_else(|p| p.into_inner());
        let before = models.len();
        models.retain(|m| m.model_id() != id);
        models.len() != before
    }

    pub(crate) fn ids(&self) -> Vec<String> {
        self.models
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|m| m.model_id().to_string())
            .collect()
    }
}

impl ModelRegistry for LiveRegistry {
    fn resolve(&self, requested: Option<&str>) -> Resolution {
        // The POLICY is `turbospark_server`'s, not this crate's: the
        // single-model fallback that keeps Claude Code working is stated
        // once, in `registry.rs`, and every registry defers to it. A second
        // copy here would be a second place for it to drift.
        let models = self.models.lock().unwrap_or_else(|p| p.into_inner());
        resolve_among(&models, requested)
    }

    fn rows(&self) -> Vec<ModelRow> {
        self.models
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|m| ModelRow {
                id: m.model_id().to_string(),
                max_context: m.max_context(),
            })
            .collect()
    }
}

/// How many events one server buffers before the oldest are dropped.
///
/// A host polling at a couple of hertz drains this long before it fills;
/// the bound exists for the host that stops polling (its window closed, it
/// is stuck behind a modal) so a busy server cannot grow this without limit.
const RING_CAPACITY: usize = 2_000;

/// A bounded event buffer a host drains.
///
/// **AN OVERFLOW IS COUNTED AND REPORTED, NEVER SILENT.** A console that
/// quietly loses rows reads as "nothing happened in that window", which is
/// the same thing it reads as when the server genuinely was idle -- and the
/// two are the states a person is trying to tell apart. `drain` returns the
/// count alongside the events so the host can say so.
#[derive(Default)]
pub(crate) struct EventRing {
    inner: Mutex<RingInner>,
}

#[derive(Default)]
struct RingInner {
    events: VecDeque<ServerEvent>,
    /// Dropped since the last drain, not since the server started: a host
    /// reports the gap it just experienced, and a lifetime total would keep
    /// re-reporting one that has already been shown.
    dropped: u64,
}

impl EventRing {
    /// Takes everything buffered, up to `max`, with the number dropped since
    /// the last call.
    ///
    /// `max` bounds ONE call rather than the buffer: anything left over stays
    /// queued for the next poll rather than being discarded, so a burst is
    /// delivered late and never lost.
    pub(crate) fn drain(&self, max: usize) -> (Vec<ServerEvent>, u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let take = inner.events.len().min(max);
        let events: Vec<_> = inner.events.drain(..take).collect();
        let dropped = std::mem::take(&mut inner.dropped);
        (events, dropped)
    }
}

impl ServerObserver for EventRing {
    fn record(&self, event: ServerEvent) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.events.len() >= RING_CAPACITY {
            inner.events.pop_front();
            inner.dropped += 1;
        }
        inner.events.push_back(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(id: u64) -> ServerEvent {
        ServerEvent::RequestStarted {
            id,
            at_ms: 0,
            method: "GET".to_string(),
            path: "/health".to_string(),
        }
    }

    fn id_of(event: &ServerEvent) -> u64 {
        serde_json::to_value(event).unwrap()["id"].as_u64().unwrap()
    }

    #[test]
    fn a_drain_bounded_by_max_leaves_the_rest_queued() {
        let ring = EventRing::default();
        for i in 0..5 {
            ring.record(event(i));
        }
        let (first, dropped) = ring.drain(2);
        assert_eq!(first.iter().map(id_of).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(dropped, 0);

        let (rest, _) = ring.drain(100);
        assert_eq!(rest.iter().map(id_of).collect::<Vec<_>>(), vec![2, 3, 4]);
    }

    /// The oldest go, and the COUNT of them survives to be reported. A ring
    /// that dropped silently would be indistinguishable from an idle server.
    #[test]
    fn an_overflow_drops_the_oldest_and_counts_them() {
        let ring = EventRing::default();
        for i in 0..(RING_CAPACITY as u64 + 3) {
            ring.record(event(i));
        }
        let (events, dropped) = ring.drain(usize::MAX);
        assert_eq!(dropped, 3);
        assert_eq!(events.len(), RING_CAPACITY);
        assert_eq!(id_of(&events[0]), 3, "the three oldest went");
    }

    /// Reported per DRAIN, so a gap that has already been shown to the user
    /// is not shown again on the next poll.
    #[test]
    fn the_dropped_count_resets_on_every_drain() {
        let ring = EventRing::default();
        for i in 0..(RING_CAPACITY as u64 + 1) {
            ring.record(event(i));
        }
        assert_eq!(ring.drain(usize::MAX).1, 1);
        ring.record(event(9_999));
        assert_eq!(ring.drain(usize::MAX).1, 0);
    }
}
