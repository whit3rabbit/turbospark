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

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use turbospark_server::observe::{ServerEvent, ServerObserver};
use turbospark_server::registry::{
    model_row, resolve_among, resolve_embedding_among, ModelRegistry, ModelRow, RequestUseLease,
    Resolution, ResolvedModel,
};
use turbospark_server::ChatModel;

/// The models attached to one running server.
pub(crate) struct LiveRegistry {
    // A `Vec` rather than a `HashMap` because attachment ORDER is what a
    // host lists them in and what `/v1/models` reports, and the set is a
    // handful of entries at most -- a machine that can hold ten resident
    // models at once does not exist. Uniqueness is enforced on insert.
    models: Mutex<Vec<Arc<LiveModelEntry>>>,
    clock: Arc<dyn Fn() -> Instant + Send + Sync>,
}

struct LiveModelEntry {
    model: Arc<dyn ChatModel>,
    generation_queue: Option<Arc<turbospark_server::GenerationQueue>>,
    use_state: Mutex<ModelUseState>,
    clock: Arc<dyn Fn() -> Instant + Send + Sync>,
}

struct ModelUseState {
    active_requests: usize,
    last_used: Instant,
}

impl LiveModelEntry {
    fn acquire_request(&self) {
        let mut state = self.use_state.lock().unwrap_or_else(|p| p.into_inner());
        state.last_used = (self.clock)();
        state.active_requests += 1;
    }

    fn release_request(&self) {
        let mut state = self.use_state.lock().unwrap_or_else(|p| p.into_inner());
        state.last_used = (self.clock)();
        debug_assert!(
            state.active_requests > 0,
            "request lease count cannot underflow"
        );
        state.active_requests = state.active_requests.saturating_sub(1);
    }
}

struct LiveRequestLease {
    entry: Arc<LiveModelEntry>,
}

impl RequestUseLease for LiveRequestLease {}

impl Drop for LiveRequestLease {
    fn drop(&mut self) {
        self.entry.release_request();
    }
}

impl Default for LiveRegistry {
    fn default() -> Self {
        Self::with_clock(Arc::new(Instant::now))
    }
}

impl LiveRegistry {
    pub(crate) fn with_clock(clock: Arc<dyn Fn() -> Instant + Send + Sync>) -> Self {
        Self {
            models: Mutex::new(Vec::new()),
            clock,
        }
    }

    /// Adds a model, or refuses if any public identity is already attached.
    ///
    /// **A DUPLICATE ID IS REFUSED BY NAME RATHER THAN SUFFIXED.** The id is
    /// what a client puts in a request's `model` field and what
    /// `detach` keys on, so inventing an `id #2` would create a second name
    /// for a thing the host believes it attached once -- and then `detach`
    /// on the name the host knows would remove the wrong one. Two sessions
    /// on one install directory is a caller mistake, and this is where it is
    /// cheapest to say so.
    pub(crate) fn attach(&self, model: Arc<dyn ChatModel>) -> Result<String, String> {
        let row = model_row(&*model);
        let id = row.id.clone();
        let generation_queue = model.generation_queue();
        let clock = Arc::clone(&self.clock);
        let last_used = (clock)();
        let mut models = self.models.lock().unwrap_or_else(|p| p.into_inner());
        let mut identities = HashSet::new();
        for existing in models.iter() {
            identities.extend(model_row(&*existing.model).ids().map(str::to_string));
        }
        for identity in row.ids() {
            if !identities.insert(identity.to_string()) {
                return Err(format!(
                    "a model with id or alias '{identity}' is already attached to this server; \
                     detach it first, or open the second install under a different directory name"
                ));
            }
        }
        models.push(Arc::new(LiveModelEntry {
            model,
            generation_queue,
            use_state: Mutex::new(ModelUseState {
                active_requests: 0,
                last_used,
            }),
            clock,
        }));
        Ok(id)
    }

    /// Removes a model by id. Returns whether one was there.
    pub(crate) fn detach(&self, id: &str) -> bool {
        let mut models = self.models.lock().unwrap_or_else(|p| p.into_inner());
        let Some(index) = models.iter().position(|entry| entry.model.model_id() == id) else {
            return false;
        };
        if models[index]
            .use_state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .active_requests
            != 0
        {
            return false;
        }
        // Cancelled response waiters may leave blocking model work alive.
        // Its permit protects residency until that work actually exits.
        if models[index]
            .generation_queue
            .as_ref()
            .is_some_and(|queue| queue.load() != 0)
        {
            return false;
        }
        models.remove(index);
        true
    }

    /// Atomically detaches entries whose last request use is older than `ttl`.
    /// The request lease count and queue load are checked while holding the
    /// same registry lock that resolution uses to acquire a lease.
    pub(crate) fn detach_idle_expired(&self, ttl: Duration) -> Vec<String> {
        let now = (self.clock)();
        let mut models = self.models.lock().unwrap_or_else(|p| p.into_inner());
        let mut detached = Vec::new();
        models.retain(|entry| {
            let expired_and_unused = {
                let state = entry.use_state.lock().unwrap_or_else(|p| p.into_inner());
                state.active_requests == 0 && now.saturating_duration_since(state.last_used) >= ttl
            };
            let generation_active = entry
                .generation_queue
                .as_ref()
                .is_some_and(|queue| queue.load() != 0);
            if expired_and_unused && !generation_active {
                detached.push(entry.model.model_id().to_string());
                false
            } else {
                true
            }
        });
        detached
    }

    /// Returns a model only when the requested id or alias is attached.
    ///
    /// Unlike ModelRegistry::resolve, this does not apply the single-model
    /// fallback, which would mistake any request for an exact match.
    pub(crate) fn exact_model(&self, requested: &str) -> Option<Arc<dyn ChatModel>> {
        let models = self.models.lock().unwrap_or_else(|p| p.into_inner());
        models.iter().find_map(|entry| {
            model_row(&*entry.model)
                .ids()
                .any(|id| id == requested)
                .then(|| Arc::clone(&entry.model))
        })
    }

    fn resolve_for_capability(&self, requested: Option<&str>, embedding: bool) -> Resolution {
        // The POLICY is `turbospark_server`'s, not this crate's: the
        // single-model fallback that keeps Claude Code working is stated
        // once, in `registry.rs`, and every registry defers to it. A second
        // copy here would be a second place for it to drift.
        let models = self.models.lock().unwrap_or_else(|p| p.into_inner());
        let model_arcs: Vec<_> = models
            .iter()
            .map(|entry| Arc::clone(&entry.model))
            .collect();
        let resolution = if embedding {
            resolve_embedding_among(&model_arcs, requested)
        } else {
            resolve_among(&model_arcs, requested)
        };
        match resolution {
            Resolution::Model(resolved) => {
                // Resolution and lease acquisition happen under the same lock
                // as detach, so a resolved model cannot become detached
                // before its request-use guard is visible.
                let entry = models
                    .iter()
                    .find(|entry| Arc::ptr_eq(&entry.model, &resolved.model))
                    .expect("resolved models originate in this live registry");
                entry.acquire_request();
                Resolution::Model(ResolvedModel::with_lease(
                    resolved.model,
                    Some(Box::new(LiveRequestLease {
                        entry: Arc::clone(entry),
                    })),
                ))
            }
            other => other,
        }
    }

    pub(crate) fn ids(&self) -> Vec<String> {
        self.models
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|entry| entry.model.model_id().to_string())
            .collect()
    }
}

impl ModelRegistry for LiveRegistry {
    fn resolve(&self, requested: Option<&str>) -> Resolution {
        self.resolve_for_capability(requested, false)
    }

    fn resolve_embedding(&self, requested: Option<&str>) -> Resolution {
        self.resolve_for_capability(requested, true)
    }

    fn rows(&self) -> Vec<ModelRow> {
        self.models
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|entry| model_row(&*entry.model))
            .collect()
    }
}

/// How many events one server buffers before the oldest are dropped.
///
/// A host polling at a couple of hertz drains this long before it fills;
/// the bound exists for the host that stops polling (its window closed, it
/// is stuck behind a modal) so a busy server cannot grow this without limit.
const RING_CAPACITY: usize = 2_000;
/// Maximum serialized event bytes retained by one server.
///
/// Request fields can be nearly as large as Axum's body limit, so the count
/// cap alone is not a memory bound. This also bounds an unbounded poll's JSON
/// allocation to roughly this size.
const RING_BYTE_CAPACITY: usize = 1 << 20;

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
    events: VecDeque<BufferedEvent>,
    bytes: usize,
    /// Dropped since the last drain, not since the server started: a host
    /// reports the gap it just experienced, and a lifetime total would keep
    /// re-reporting one that has already been shown.
    dropped: u64,
}

struct BufferedEvent {
    event: ServerEvent,
    serialized_bytes: usize,
}

fn serialized_len(event: &ServerEvent) -> usize {
    serde_json::to_vec(event)
        .map(|json| json.len())
        .unwrap_or(usize::MAX)
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
        let drained: Vec<_> = inner.events.drain(..take).collect();
        let drained_bytes: usize = drained.iter().map(|event| event.serialized_bytes).sum();
        inner.bytes -= drained_bytes;
        let events = drained.into_iter().map(|event| event.event).collect();
        let dropped = std::mem::take(&mut inner.dropped);
        (events, dropped)
    }
}

impl ServerObserver for EventRing {
    fn record(&self, event: ServerEvent) {
        let serialized_bytes = serialized_len(&event);
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        while !inner.events.is_empty()
            && (inner.events.len() >= RING_CAPACITY
                || inner.bytes.saturating_add(serialized_bytes) > RING_BYTE_CAPACITY)
        {
            let removed = inner.events.pop_front().expect("ring was not empty");
            inner.bytes -= removed.serialized_bytes;
            inner.dropped += 1;
        }
        if serialized_bytes > RING_BYTE_CAPACITY {
            inner.dropped += 1;
            return;
        }
        inner.bytes += serialized_bytes;
        inner.events.push_back(BufferedEvent {
            event,
            serialized_bytes,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokenizer::MfTokenizer;
    use turbospark_server::registry::{ModelRegistry, Resolution};
    use turbospark_server::{ChatModel, ScriptedChatModel};

    struct ChatOnlyModel(ScriptedChatModel);

    impl ChatModel for ChatOnlyModel {
        fn tokenizer(&self) -> &tokenizer::MfTokenizer {
            self.0.tokenizer()
        }

        fn vocab_size(&self) -> usize {
            self.0.vocab_size()
        }

        fn max_context(&self) -> u32 {
            self.0.max_context()
        }

        fn model_id(&self) -> &str {
            "live-chat"
        }

        fn supports_embeddings(&self) -> bool {
            false
        }

        fn with_producer(
            &self,
            f: &mut dyn FnMut(
                &mut dyn runtime::LogitProducer,
            )
                -> Result<runtime::RawDecodeResult, runtime::RuntimeError>,
        ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError> {
            self.0.with_producer(f)
        }
    }

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

    #[test]
    fn byte_overflow_drops_oldest_events() {
        let ring = EventRing::default();
        let requested = "x".repeat(RING_BYTE_CAPACITY / 2);
        for id in 0..3 {
            ring.record(ServerEvent::RequestRouted {
                id,
                requested: Some(requested.clone()),
                served: "model".to_string(),
                stream: false,
            });
        }

        let (events, dropped) = ring.drain(usize::MAX);
        assert_eq!(dropped, 2);
        assert_eq!(events.len(), 1);
        assert_eq!(id_of(&events[0]), 2);
    }

    #[test]
    fn one_event_larger_than_the_byte_budget_is_dropped() {
        let ring = EventRing::default();
        ring.record(ServerEvent::RequestRouted {
            id: 1,
            requested: Some("x".repeat(RING_BYTE_CAPACITY)),
            served: "model".to_string(),
            stream: false,
        });

        let (events, dropped) = ring.drain(usize::MAX);
        assert!(events.is_empty());
        assert_eq!(dropped, 1);
    }

    #[test]
    fn an_active_request_lease_prevents_live_model_detach() {
        let registry = LiveRegistry::default();
        let tokenizer_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tokenizer/tests/fixtures/ChatMLTokenizer");
        let tokenizer =
            MfTokenizer::load_from_dir(&tokenizer_dir).expect("fixture tokenizer should load");
        let model: Arc<dyn ChatModel> = Arc::new(ChatOnlyModel(ScriptedChatModel::new(
            tokenizer,
            4096,
            Vec::new(),
        )));
        let id = registry.attach(model).expect("model should attach");

        let resolved = match registry.resolve(Some(&id)) {
            Resolution::Model(resolved) => resolved,
            _ => panic!("the attached model should resolve"),
        };
        let request_lease = resolved
            .request_lease
            .expect("live resolution should acquire a request-use lease");

        assert!(
            !registry.detach(&id),
            "an active request must keep its model attached"
        );

        drop(request_lease);
        assert!(
            registry.detach(&id),
            "dropping the final request lease should allow detach"
        );
    }

    #[test]
    fn embedding_resolution_refuses_a_chat_only_live_registry() {
        let registry = LiveRegistry::default();
        let tokenizer_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tokenizer/tests/fixtures/ChatMLTokenizer");
        let tokenizer = MfTokenizer::load_from_dir(&tokenizer_dir).unwrap();
        registry
            .attach(Arc::new(ChatOnlyModel(ScriptedChatModel::new(
                tokenizer,
                4096,
                Vec::new(),
            ))))
            .unwrap();

        assert!(matches!(
            registry.resolve_embedding(Some("live-chat")),
            Resolution::Empty
        ));
    }

    #[test]
    fn embedding_resolution_selects_and_leases_the_capable_live_model() {
        let registry = LiveRegistry::default();
        let tokenizer_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tokenizer/tests/fixtures/ChatMLTokenizer");
        let tokenizer = || MfTokenizer::load_from_dir(&tokenizer_dir).unwrap();
        registry
            .attach(Arc::new(ChatOnlyModel(ScriptedChatModel::new(
                tokenizer(),
                4096,
                Vec::new(),
            ))))
            .unwrap();
        let embedding_id = registry
            .attach(Arc::new(ScriptedChatModel::new(
                tokenizer(),
                4096,
                Vec::new(),
            )))
            .unwrap();

        let resolved = match registry.resolve_embedding(Some("text-embedding-3-small")) {
            Resolution::Model(resolved) => resolved,
            _ => panic!("the only embedding model should resolve by default name"),
        };
        assert_eq!(resolved.model.model_id(), embedding_id);
        assert!(resolved.request_lease.is_some());
        assert!(!registry.detach(&embedding_id));
        assert!(registry.detach("live-chat"));
        drop(resolved);
        assert!(registry.detach(&embedding_id));
    }
}
