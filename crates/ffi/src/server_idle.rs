use std::sync::{Arc, Mutex};
use std::time::Duration;

use turbospark_server::observe::{ServerEvent, ServerObserver};

use crate::server_registry::{EventRing, LiveRegistry};

const SWEEP_INTERVAL: Duration = Duration::from_secs(30);
const DEFAULT_IDLE_TTL: Duration = Duration::from_secs(30 * 60);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct IdlePolicy {
    pub(crate) enabled: bool,
    pub(crate) ttl: Duration,
}

impl IdlePolicy {
    /// `0` disables the sweep; any other value is the idle TTL in seconds.
    pub(crate) fn from_seconds(seconds: u64) -> Self {
        if seconds == 0 {
            Self::default()
        } else {
            Self {
                enabled: true,
                ttl: Duration::from_secs(seconds),
            }
        }
    }
}

impl Default for IdlePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            ttl: DEFAULT_IDLE_TTL,
        }
    }
}

/// One policy and fixed-interval sweep for a running embedded server.
pub(crate) struct IdleSweep {
    registry: Arc<LiveRegistry>,
    events: Arc<EventRing>,
    policy: Mutex<IdlePolicy>,
}

impl IdleSweep {
    pub(crate) fn new(registry: Arc<LiveRegistry>, events: Arc<EventRing>) -> Self {
        Self {
            registry,
            events,
            policy: Mutex::new(IdlePolicy::default()),
        }
    }

    pub(crate) fn set_policy(&self, policy: IdlePolicy) -> Result<(), String> {
        if policy.enabled && policy.ttl.is_zero() {
            return Err("idle-unload duration must be greater than zero".to_string());
        }
        *self.policy.lock().unwrap_or_else(|p| p.into_inner()) = policy;
        Ok(())
    }

    /// Runs one policy evaluation. The registry releases its lock before the
    /// detached events are published to the host observer.
    pub(crate) fn sweep_once(&self) -> Vec<String> {
        let policy = *self.policy.lock().unwrap_or_else(|p| p.into_inner());
        if !policy.enabled {
            return Vec::new();
        }

        let detached = self.registry.detach_idle_expired(policy.ttl);
        for model in &detached {
            ServerObserver::record(
                &*self.events,
                ServerEvent::ModelDetached {
                    at_ms: crate::server::now_ms(),
                    model: model.clone(),
                },
            );
        }
        detached
    }

    /// The task is owned by the server's Tokio runtime and ends when that
    /// runtime exits. It does no model work and never holds the registry lock
    /// while publishing events.
    pub(crate) async fn run(self: Arc<Self>) {
        let mut interval = tokio::time::interval(SWEEP_INTERVAL);
        loop {
            interval.tick().await;
            self.sweep_once();
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn from_seconds_zero_disables_and_nonzero_enables() {
        assert_eq!(IdlePolicy::from_seconds(0), IdlePolicy::default());
        assert!(!IdlePolicy::from_seconds(0).enabled);
        let p = IdlePolicy::from_seconds(90);
        assert!(p.enabled);
        assert_eq!(p.ttl, std::time::Duration::from_secs(90));
    }

    use super::super::server_registry::{EventRing, LiveRegistry};
    use super::{IdlePolicy, IdleSweep};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokenizer::MfTokenizer;
    use turbospark_server::observe::ServerEvent;
    use turbospark_server::registry::{
        LoadOutcome, ModelLoader, ModelLoaderRegistry, ModelRegistry, Resolution,
    };
    use turbospark_server::{ChatModel, GenerationQueue, ScriptedChatModel};

    #[derive(Clone)]
    struct ManualClock {
        base: Instant,
        elapsed_ms: Arc<AtomicUsize>,
    }

    impl ManualClock {
        fn new() -> Self {
            Self {
                base: Instant::now(),
                elapsed_ms: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn advance(&self, duration: Duration) {
            self.elapsed_ms.fetch_add(
                duration
                    .as_millis()
                    .try_into()
                    .expect("test duration fits usize"),
                Ordering::SeqCst,
            );
        }

        fn now(&self) -> Instant {
            self.base + Duration::from_millis(self.elapsed_ms.load(Ordering::SeqCst) as u64)
        }

        fn source(&self) -> Arc<dyn Fn() -> Instant + Send + Sync> {
            let clock = self.clone();
            Arc::new(move || clock.now())
        }
    }

    struct QueuedModel {
        inner: ScriptedChatModel,
        id: String,
        queue: Arc<GenerationQueue>,
    }

    impl ChatModel for QueuedModel {
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

        fn generation_queue(&self) -> Option<Arc<GenerationQueue>> {
            Some(Arc::clone(&self.queue))
        }
    }

    fn fixture_model(id: &str, queue: Option<Arc<GenerationQueue>>) -> Arc<dyn ChatModel> {
        let tokenizer_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tokenizer/tests/fixtures/ChatMLTokenizer");
        let tokenizer =
            MfTokenizer::load_from_dir(&tokenizer_dir).expect("fixture tokenizer should load");
        let inner = ScriptedChatModel::new(tokenizer, 4096, Vec::new());
        match queue {
            Some(queue) => Arc::new(QueuedModel {
                inner,
                id: id.to_string(),
                queue,
            }),
            None => Arc::new(IdentifiedModel {
                inner,
                id: id.to_string(),
            }),
        }
    }

    struct IdentifiedModel {
        inner: ScriptedChatModel,
        id: String,
    }

    impl ChatModel for IdentifiedModel {
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

    fn enabled(ttl: Duration) -> IdlePolicy {
        IdlePolicy { enabled: true, ttl }
    }

    #[test]
    fn default_policy_is_off_and_expired_models_emit_detach_events() {
        let clock = ManualClock::new();
        let registry = Arc::new(LiveRegistry::with_clock(clock.source()));
        let events = Arc::new(EventRing::default());
        let sweep = IdleSweep::new(Arc::clone(&registry), Arc::clone(&events));
        registry
            .attach(fixture_model("idle-model", None))
            .expect("model should attach");

        clock.advance(Duration::from_secs(3_601));
        assert!(sweep.sweep_once().is_empty());
        assert_eq!(registry.ids(), vec!["idle-model"]);

        let ttl = Duration::from_secs(7_200);
        sweep.set_policy(enabled(ttl)).expect("valid idle policy");
        assert!(
            sweep.sweep_once().is_empty(),
            "the model has not reached its TTL"
        );
        clock.advance(Duration::from_secs(3_600) + Duration::from_millis(1));
        assert_eq!(sweep.sweep_once(), vec!["idle-model"]);
        assert!(registry.ids().is_empty());

        let (recorded, _) = events.drain(10);
        assert!(
            matches!(recorded.as_slice(), [ServerEvent::ModelDetached { model, .. }] if model == "idle-model")
        );
    }

    #[test]
    fn zero_ttl_is_rejected_when_enabled_but_kept_the_previous_policy() {
        let clock = ManualClock::new();
        let registry = Arc::new(LiveRegistry::with_clock(clock.source()));
        let events = Arc::new(EventRing::default());
        let sweep = IdleSweep::new(Arc::clone(&registry), Arc::clone(&events));
        sweep
            .set_policy(enabled(Duration::from_secs(30)))
            .expect("valid idle policy");
        registry
            .attach(fixture_model("idle-policy-model", None))
            .expect("model should attach");

        let error = sweep
            .set_policy(enabled(Duration::ZERO))
            .expect_err("an enabled policy with a zero TTL must be rejected");
        assert_eq!(error, "idle-unload duration must be greater than zero");

        clock.advance(Duration::from_secs(31));
        assert_eq!(
            sweep.sweep_once(),
            vec!["idle-policy-model"],
            "the rejected policy must not replace the previous one"
        );

        sweep
            .set_policy(IdlePolicy {
                enabled: false,
                ttl: Duration::ZERO,
            })
            .expect("a disabled policy may carry a zero TTL");
    }

    #[test]
    fn a_live_request_lease_refreshes_use_and_prevents_expiry_detach() {
        let clock = ManualClock::new();
        let registry = Arc::new(LiveRegistry::with_clock(clock.source()));
        let events = Arc::new(EventRing::default());
        let sweep = IdleSweep::new(Arc::clone(&registry), Arc::clone(&events));
        let ttl = Duration::from_secs(30);
        sweep.set_policy(enabled(ttl)).expect("valid idle policy");
        registry
            .attach(fixture_model("leased-model", None))
            .expect("model should attach");
        let lease = match registry.resolve(Some("leased-model")) {
            Resolution::Model(model) => model,
            _ => panic!("expected model resolution"),
        };

        clock.advance(ttl + Duration::from_millis(1));
        assert!(
            sweep.sweep_once().is_empty(),
            "the active lease blocks detach"
        );
        drop(lease);
        assert!(
            sweep.sweep_once().is_empty(),
            "dropping the lease refreshes last use"
        );
        clock.advance(ttl + Duration::from_millis(1));
        assert_eq!(sweep.sweep_once(), vec!["leased-model"]);
    }

    #[tokio::test]
    async fn queued_generation_work_blocks_detach_without_a_request_lease() {
        let clock = ManualClock::new();
        let registry = Arc::new(LiveRegistry::with_clock(clock.source()));
        let events = Arc::new(EventRing::default());
        let sweep = IdleSweep::new(Arc::clone(&registry), Arc::clone(&events));
        let ttl = Duration::from_secs(30);
        sweep.set_policy(enabled(ttl)).expect("valid idle policy");
        let queue = GenerationQueue::shared();
        registry
            .attach(fixture_model("queued-model", Some(Arc::clone(&queue))))
            .expect("model should attach");
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let permit = queue
            .acquire(&cancel)
            .await
            .expect("queue should admit work");

        clock.advance(ttl + Duration::from_millis(1));
        assert!(
            sweep.sweep_once().is_empty(),
            "queued or active work blocks detach"
        );
        drop(permit);
        assert_eq!(sweep.sweep_once(), vec!["queued-model"]);
    }

    #[tokio::test]
    async fn manual_detach_refuses_work_that_outlives_its_request_lease() {
        let registry = LiveRegistry::default();
        let queue = GenerationQueue::shared();
        registry
            .attach(fixture_model("held-model", Some(Arc::clone(&queue))))
            .expect("model should attach");
        let lease = match registry.resolve(Some("held-model")) {
            Resolution::Model(model) => model,
            _ => panic!("expected model resolution"),
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let permit = queue
            .acquire(&cancel)
            .await
            .expect("work should be admitted");
        let blocking_permit = permit.clone();
        drop(lease);
        drop(permit);

        assert!(
            !registry.detach("held-model"),
            "blocking work still owns its permit"
        );
        assert_eq!(registry.ids(), vec!["held-model"]);
        drop(blocking_permit);
        assert!(
            registry.detach("held-model"),
            "work completion allows detach"
        );
    }

    #[tokio::test]
    async fn ffi_model_queue_blocks_idle_detach_after_request_lease_drops() {
        let clock = ManualClock::new();
        let registry = Arc::new(LiveRegistry::with_clock(clock.source()));
        let events = Arc::new(EventRing::default());
        let sweep = IdleSweep::new(Arc::clone(&registry), Arc::clone(&events));
        let ttl = Duration::from_secs(30);
        sweep.set_policy(enabled(ttl)).expect("valid idle policy");

        let tokenizer_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tokenizer/tests/fixtures/ChatMLTokenizer");
        let tokenizer =
            MfTokenizer::load_from_dir(&tokenizer_dir).expect("fixture tokenizer should load");
        let vocab_size = tokenizer.vocab_size;
        let session = crate::testing::session_for_testing(tokenizer, Vec::new(), vocab_size, 4096);
        let model = Arc::new(crate::server_model::FfiChatModel::new(
            session.core(),
            "ffi-idle-model".to_string(),
            turbospark_server::GuardrailConfig::default(),
            None,
            tokenizer::ReasoningEffort::Off,
            Arc::new(AtomicBool::new(false)),
        ));
        let queue = model
            .generation_queue()
            .expect("FFI server models expose their generation queue");
        registry
            .attach(model as Arc<dyn ChatModel>)
            .expect("FFI model should attach");

        let cancel = Arc::new(AtomicBool::new(false));
        let permit = queue
            .acquire(&cancel)
            .await
            .expect("server generation should acquire its queue permit");
        let blocking_permit = permit.clone();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let blocking_body = std::thread::spawn(move || {
            let _permit = blocking_permit;
            started_tx
                .send(())
                .expect("blocking body start signal should be received");
            release_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("blocking body should remain held until released");
        });
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("blocking body should start");
        let request_lease = match registry.resolve(Some("ffi-idle-model")) {
            Resolution::Model(model) => model,
            _ => panic!("expected FFI model resolution"),
        };
        drop(request_lease);
        drop(permit); // The cancelled async waiter no longer owns the queue slot.

        clock.advance(ttl + Duration::from_millis(1));
        assert!(
            sweep.sweep_once().is_empty(),
            "detached blocking generation still owns the FFI model queue"
        );
        assert_eq!(registry.ids(), vec!["ffi-idle-model"]);

        release_tx
            .send(())
            .expect("blocking body should still be waiting");
        blocking_body
            .join()
            .expect("blocking body should exit without panicking");
        assert_eq!(sweep.sweep_once(), vec!["ffi-idle-model"]);
    }

    #[test]
    fn resolution_racing_idle_sweep_has_one_serialized_winner() {
        for _ in 0..20 {
            let clock = ManualClock::new();
            let registry = Arc::new(LiveRegistry::with_clock(clock.source()));
            let events = Arc::new(EventRing::default());
            let sweep = Arc::new(IdleSweep::new(Arc::clone(&registry), events));
            let ttl = Duration::from_secs(1);
            sweep.set_policy(enabled(ttl)).expect("valid idle policy");
            registry
                .attach(fixture_model("racing-model", None))
                .expect("model should attach");
            clock.advance(ttl + Duration::from_millis(1));

            let start = Arc::new(std::sync::Barrier::new(3));
            let resolve_registry = Arc::clone(&registry);
            let resolve_start = Arc::clone(&start);
            let resolve = std::thread::spawn(move || {
                resolve_start.wait();
                resolve_registry.resolve(Some("racing-model"))
            });
            let sweep_start = Arc::clone(&start);
            let sweep = Arc::clone(&sweep);
            let unload = std::thread::spawn(move || {
                sweep_start.wait();
                sweep.sweep_once()
            });

            start.wait();
            let resolution = resolve.join().expect("resolver thread should finish");
            let detached = unload.join().expect("sweep thread should finish");
            match resolution {
                Resolution::Model(resolved) => {
                    assert!(
                        detached.is_empty(),
                        "a leased model cannot also be detached"
                    );
                    assert_eq!(registry.ids(), vec!["racing-model"]);
                    drop(resolved);
                }
                Resolution::Empty => {
                    assert_eq!(detached, vec!["racing-model"]);
                    assert!(registry.ids().is_empty());
                }
                _ => panic!("resolution should either acquire a lease or observe the detach"),
            }
        }
    }

    struct ReloadingLoader {
        registry: Arc<LiveRegistry>,
        model: Arc<dyn ChatModel>,
        loads: AtomicUsize,
    }

    impl ModelLoader for ReloadingLoader {
        fn load(&self, requested: &str) -> LoadOutcome {
            if requested != self.model.model_id() {
                return LoadOutcome::Unavailable {
                    candidates: Vec::new(),
                };
            }
            self.loads.fetch_add(1, Ordering::SeqCst);
            self.registry
                .attach(Arc::clone(&self.model))
                .expect("unloaded model should attach again");
            LoadOutcome::Loaded(Arc::clone(&self.model))
        }

        fn candidates(
            &self,
            _requested: &str,
        ) -> Vec<turbospark_server::registry::ModelSuggestion> {
            Vec::new()
        }
    }

    #[test]
    fn a_request_after_idle_detach_reloads_through_model_loader_registry() {
        let clock = ManualClock::new();
        let registry = Arc::new(LiveRegistry::with_clock(clock.source()));
        let events = Arc::new(EventRing::default());
        let sweep = IdleSweep::new(Arc::clone(&registry), Arc::clone(&events));
        let ttl = Duration::from_secs(15);
        sweep.set_policy(enabled(ttl)).expect("valid idle policy");
        let model = fixture_model("reload-model", None);
        registry
            .attach(Arc::clone(&model))
            .expect("model should attach");
        clock.advance(ttl + Duration::from_millis(1));
        assert_eq!(sweep.sweep_once(), vec!["reload-model"]);

        let loader = Arc::new(ReloadingLoader {
            registry: Arc::clone(&registry),
            model,
            loads: AtomicUsize::new(0),
        });
        let resolver = ModelLoaderRegistry::new(
            Arc::clone(&registry) as Arc<dyn ModelRegistry>,
            loader.clone(),
        );
        match resolver.resolve(Some("reload-model")) {
            Resolution::Model(resolved) => assert_eq!(resolved.model.model_id(), "reload-model"),
            _ => panic!("expected reloaded model"),
        }
        assert_eq!(loader.loads.load(Ordering::SeqCst), 1);
        assert_eq!(registry.ids(), vec!["reload-model"]);
    }
}
