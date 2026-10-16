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

/// Admission ceiling a server enforces on its resident model set.
///
/// **THE COUNT IS THE DEFAULT BOUND; THE BYTES ARE OPT-IN.** A count cap
/// bounds what a client that keeps naming installed models can accumulate
/// when request-triggered loading is enabled. A byte ceiling over
/// committed weights plus KV is available for hosts that want an exact
/// aggregate budget, but it is NOT derived by default: `committed_bytes`
/// counts mapped weight address space that streamed or mapped expert
/// residency deliberately keeps mostly UNRESIDENT, so a physical-memory
/// default would refuse large streamed installs that run fine.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidencyPolicy {
    /// Refuse an attach that would exceed this many simultaneously
    /// attached models.
    pub(crate) max_models: usize,
    /// Cumulative ceiling in bytes over every attached model's
    /// committed weights plus KV cache. `u64::MAX` disables it.
    pub(crate) max_bytes: u64,
}

impl Default for ResidencyPolicy {
    fn default() -> Self {
        Self {
            max_models: DEFAULT_MAX_ATTACHED_MODELS,
            max_bytes: u64::MAX,
        }
    }
}

/// A machine that can hold this many resident text models at once does not
/// exist; every model is also individually load-guarded at open.
pub(crate) const DEFAULT_MAX_ATTACHED_MODELS: usize = 8;

/// Why a server refused to admit a model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AttachError {
    DuplicateIdentity(String),
    ModelCountExceeded { attached: usize, max: usize },
    ByteBudgetExceeded { held: u64, needs: u64, ceiling: u64 },
}

impl std::fmt::Display for AttachError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateIdentity(identity) => write!(
                f,
                "a model with id or alias '{identity}' is already attached to this server; \
                 detach it first, or open the second install under a different directory name"
            ),
            Self::ModelCountExceeded { attached, max } => write!(
                f,
                "attaching another model would exceed the server's limit of {max} \
                 attached models ({attached} attached); detach one first"
            ),
            Self::ByteBudgetExceeded {
                held,
                needs,
                ceiling,
            } => write!(
                f,
                "attached models hold {held} bytes and this model needs {needs} more, \
                 over the {ceiling}-byte ceiling; detach a model or raise maxAttachedModelBytes"
            ),
        }
    }
}

/// How much resident state an admitted model holds, and who asked for it.
///
/// `on_demand` marks a model the HTTP request path opened (rather than one
/// the host attached deliberately): those are what the default idle sweep
/// may release, because no operator ever decided they should stay.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelResidency {
    pub(crate) bytes: u64,
    pub(crate) on_demand: bool,
}

impl ModelResidency {
    /// A host-attached model whose byte cost this registry does not track
    /// (embedding encoders, scripted test fixtures).
    pub(crate) fn host() -> Self {
        Self {
            bytes: 0,
            on_demand: false,
        }
    }
}

/// The models attached to one running server.
pub(crate) struct LiveRegistry {
    // A `Vec` rather than a `HashMap` because attachment ORDER is what a
    // host lists them in and what `/v1/models` reports, and the set is a
    // handful of entries at most -- a machine that can hold ten resident
    // models at once does not exist. Uniqueness is enforced on insert.
    models: Mutex<Vec<Arc<LiveModelEntry>>>,
    clock: Arc<dyn Fn() -> Instant + Send + Sync>,
    policy: ResidencyPolicy,
}

struct LiveModelEntry {
    model: Arc<dyn ChatModel>,
    generation_queue: Option<Arc<turbospark_server::GenerationQueue>>,
    residency: ModelResidency,
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
        Self::with_policy(ResidencyPolicy::default())
    }
}

impl LiveRegistry {
    pub(crate) fn with_policy(policy: ResidencyPolicy) -> Self {
        Self::with_clock_and_policy(Arc::new(Instant::now), policy)
    }

    #[cfg(test)]
    pub(crate) fn with_clock(clock: Arc<dyn Fn() -> Instant + Send + Sync>) -> Self {
        Self::with_clock_and_policy(clock, ResidencyPolicy::default())
    }

    fn with_clock_and_policy(
        clock: Arc<dyn Fn() -> Instant + Send + Sync>,
        policy: ResidencyPolicy,
    ) -> Self {
        Self {
            models: Mutex::new(Vec::new()),
            clock,
            policy,
        }
    }

    /// Adds a model, or refuses if any public identity is already attached
    /// or the residency policy has no room left.
    ///
    /// **A DUPLICATE ID IS REFUSED BY NAME RATHER THAN SUFFIXED.** The id is
    /// what a client puts in a request's `model` field and what
    /// `detach` keys on, so inventing an `id #2` would create a second name
    /// for a thing the host believes it attached once -- and then `detach`
    /// on the name the host knows would remove the wrong one. Two sessions
    /// on one install directory is a caller mistake, and this is where it is
    /// cheapest to say so.
    ///
    /// **THE RESIDENCY CEILINGS ARE CHECKED UNDER THE SAME LOCK.** A request
    /// that names an installed model is how an ordinary inference client can
    /// grow this server's footprint, so admission -- count and, when
    /// configured, cumulative bytes -- is decided atomically with the insert
    /// rather than re-derivable from a stale snapshot.
    pub(crate) fn attach(
        &self,
        model: Arc<dyn ChatModel>,
        residency: ModelResidency,
    ) -> Result<String, AttachError> {
        let row = model_row(&*model);
        let id = row.id.clone();
        let generation_queue = model.generation_queue();
        let clock = Arc::clone(&self.clock);
        let last_used = (clock)();
        let mut models = self.models.lock().unwrap_or_else(|p| p.into_inner());
        let mut identities = HashSet::new();
        let mut held_bytes = 0u64;
        for existing in models.iter() {
            identities.extend(model_row(&*existing.model).ids().map(str::to_string));
            held_bytes = held_bytes.saturating_add(existing.residency.bytes);
        }
        for identity in row.ids() {
            if !identities.insert(identity.to_string()) {
                return Err(AttachError::DuplicateIdentity(identity.to_string()));
            }
        }
        if models.len() >= self.policy.max_models {
            return Err(AttachError::ModelCountExceeded {
                attached: models.len(),
                max: self.policy.max_models,
            });
        }
        if residency.bytes > 0 {
            let total = held_bytes.saturating_add(residency.bytes);
            if total > self.policy.max_bytes {
                return Err(AttachError::ByteBudgetExceeded {
                    held: held_bytes,
                    needs: residency.bytes,
                    ceiling: self.policy.max_bytes,
                });
            }
        }
        models.push(Arc::new(LiveModelEntry {
            model,
            generation_queue,
            residency,
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
    ///
    /// With `on_demand_only` the sweep is scoped to request-loaded models:
    /// that is the DEFAULT policy, where a host-attached model is a
    /// deliberate act that no timer undoes, while a model an HTTP request
    /// caused to open is bounded residency the server took back.
    pub(crate) fn detach_idle_expired(&self, ttl: Duration, on_demand_only: bool) -> Vec<String> {
        let now = (self.clock)();
        let mut models = self.models.lock().unwrap_or_else(|p| p.into_inner());
        let mut detached = Vec::new();
        models.retain(|entry| {
            if on_demand_only && !entry.residency.on_demand {
                return true;
            }
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

    /// The subset of [`Self::ids`] that serves `/v1/embeddings`, in
    /// attachment order, so a host can show them apart from chat models.
    pub(crate) fn embedding_ids(&self) -> Vec<String> {
        self.models
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter(|entry| entry.model.supports_embeddings())
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
        let id = registry
            .attach(model, ModelResidency::host())
            .expect("model should attach");

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
            .attach(
                Arc::new(ChatOnlyModel(ScriptedChatModel::new(
                    tokenizer,
                    4096,
                    Vec::new(),
                ))),
                ModelResidency::host(),
            )
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
            .attach(
                Arc::new(ChatOnlyModel(ScriptedChatModel::new(
                    tokenizer(),
                    4096,
                    Vec::new(),
                ))),
                ModelResidency::host(),
            )
            .unwrap();
        let embedding_id = registry
            .attach(
                Arc::new(ScriptedChatModel::new(tokenizer(), 4096, Vec::new())),
                ModelResidency::host(),
            )
            .unwrap();

        let resolved = match registry.resolve_embedding(Some("text-embedding-3-small")) {
            Resolution::Model(resolved) => resolved,
            _ => panic!("the only embedding model should resolve by default name"),
        };
        assert_eq!(resolved.model.model_id(), embedding_id);
        // `ids` lists both; `embedding_ids` only the one serving embeddings.
        assert_eq!(registry.ids().len(), 2);
        assert_eq!(registry.embedding_ids(), vec![embedding_id.clone()]);
        assert!(resolved.request_lease.is_some());
        assert!(!registry.detach(&embedding_id));
        assert!(registry.detach("live-chat"));
        drop(resolved);
        assert!(registry.detach(&embedding_id));
    }

    fn fixture_for_policy_tests(id: &str) -> Arc<dyn ChatModel> {
        let tokenizer_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tokenizer/tests/fixtures/ChatMLTokenizer");
        let tokenizer =
            MfTokenizer::load_from_dir(&tokenizer_dir).expect("fixture tokenizer should load");
        Arc::new(IdentifiedFixtureModel {
            inner: ScriptedChatModel::new(tokenizer, 4096, Vec::new()),
            id: id.to_string(),
        })
    }

    struct IdentifiedFixtureModel {
        inner: ScriptedChatModel,
        id: String,
    }

    impl ChatModel for IdentifiedFixtureModel {
        fn tokenizer(&self) -> &MfTokenizer {
            self.inner.tokenizer()
        }
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }
        fn max_context(&self) -> u32 {
            self.inner.max_context()
        }
        fn model_id(&self) -> &str {
            &self.id
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
            self.inner.with_producer(f)
        }
    }

    #[test]
    fn attach_refuses_when_the_model_count_cap_is_reached() {
        let registry = LiveRegistry::with_policy(ResidencyPolicy {
            max_models: 2,
            max_bytes: u64::MAX,
        });
        registry
            .attach(
                fixture_for_policy_tests("cap-a.gturbo"),
                ModelResidency::host(),
            )
            .expect("first model fits under the cap");
        registry
            .attach(
                fixture_for_policy_tests("cap-b.gturbo"),
                ModelResidency::host(),
            )
            .expect("second model fills the cap");
        let error = registry
            .attach(
                fixture_for_policy_tests("cap-c.gturbo"),
                ModelResidency::host(),
            )
            .expect_err("the third model must be refused");
        assert_eq!(
            error,
            AttachError::ModelCountExceeded {
                attached: 2,
                max: 2
            }
        );
        assert!(error.to_string().contains("limit of 2"));

        // Detaching frees a slot: the cap is a residency bound, not a
        // lifetime tally.
        assert!(registry.detach("cap-a.gturbo"));
        registry
            .attach(
                fixture_for_policy_tests("cap-c.gturbo"),
                ModelResidency::host(),
            )
            .expect("freed slot admits the model");
    }

    #[test]
    fn attach_refuses_when_the_byte_budget_cannot_cover_the_model() {
        let registry = LiveRegistry::with_policy(ResidencyPolicy {
            max_models: 8,
            max_bytes: 100,
        });
        registry
            .attach(
                fixture_for_policy_tests("bytes-a.gturbo"),
                ModelResidency {
                    bytes: 60,
                    on_demand: false,
                },
            )
            .expect("60 of 100 bytes fits");
        let error = registry
            .attach(
                fixture_for_policy_tests("bytes-b.gturbo"),
                ModelResidency {
                    bytes: 50,
                    on_demand: true,
                },
            )
            .expect_err("60 + 50 exceeds the 100-byte ceiling");
        assert_eq!(
            error,
            AttachError::ByteBudgetExceeded {
                held: 60,
                needs: 50,
                ceiling: 100
            }
        );
        registry
            .attach(
                fixture_for_policy_tests("bytes-c.gturbo"),
                ModelResidency {
                    bytes: 40,
                    on_demand: true,
                },
            )
            .expect("exactly filling the ceiling is admitted");

        assert!(registry.detach("bytes-a.gturbo"));
        registry
            .attach(
                fixture_for_policy_tests("bytes-b.gturbo"),
                ModelResidency {
                    bytes: 50,
                    on_demand: true,
                },
            )
            .expect("detaching releases its bytes to the budget");
    }

    #[test]
    fn a_zero_byte_attach_never_trips_the_byte_budget() {
        let registry = LiveRegistry::with_policy(ResidencyPolicy {
            max_models: 8,
            max_bytes: 0,
        });
        // Models whose cost this registry does not track (embedding
        // encoders, fixtures) carry zero bytes and are bounded by the count
        // cap alone.
        registry
            .attach(
                fixture_for_policy_tests("zero-a.gturbo"),
                ModelResidency::host(),
            )
            .expect("zero-byte attaches are the count cap's business");
    }
}
