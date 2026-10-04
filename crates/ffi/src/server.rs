//! The in-process HTTP server: a background OS thread hosting its own tokio
//! runtime, serving `turbospark_server::build_router_with_options` over a
//! registry of `FfiChatModel`s, each sharing one of the caller's already-open
//! sessions (`server_model.rs`, `session.rs`'s module doc).
//!
//! **THE MODEL SET IS MUTABLE WHILE THE SERVER RUNS, AND THAT IS WHY THE
//! ROUTER HOLDS A `LiveRegistry` RATHER THAN A MODEL.** A GUI starts a server
//! before its user has chosen anything to load, then attaches and detaches as
//! they go. `server_registry.rs` is the storage; the ROUTING POLICY over it
//! (an exact id wins, one attached model serves any name, several refuse an
//! unknown one) belongs to `turbospark_server::registry` and is not restated
//! here.
//!
//! **THE SERVER OUTLIVES EVERY `TsSession *` IT WAS GIVEN, BY DESIGN.** Each
//! entry holds an `Arc<SessionCore>` clone rather than a borrow, so
//! `ts_session_close` on a session drops only the caller's own reference --
//! the engine stays resident until `ts_server_detach_model` removes that one
//! entry, or `ts_server_stop` (or `Drop`, e.g. process exit) releases the
//! whole set. A caller that wants a model gone must do one of those; this
//! crate does not track that for them, the same way `ts_session_close` does
//! not know whether a caller still holds some other reference of their own.
//!
//! **EVERY SERVER RECORDS, AND A HOST DRAINS.** The router is always given an
//! `EventRing` observer, unlike the standalone `turbospark-server` binary
//! which passes `None`: a host embedding this has a console to show the rows
//! in, and its own stdout is not where they would go. The ring is bounded
//! and reports its own overflow rather than losing rows silently.
//!
//! **`port: 0` MEANS "LET THE OS CHOOSE."** The socket is bound on the
//! BACKGROUND THREAD, inside its own runtime, and the resolved ADDRESS is
//! sent back to `start`'s caller over a plain blocking `std::sync::mpsc`
//! channel before `start` returns -- so a caller can read the actually bound
//! address back from `ServerInfo` immediately, with no race against the
//! thread that will use it.
//!
//! **BOTH HALVES OF THAT ADDRESS COME FROM `local_addr`, INCLUDING THE HOST.**
//! `SocketAddr` travels the channel rather than a bare `u16`, and
//! `ServerInfo.host` is `addr.ip()` rather than the `127.0.0.1` this module
//! interpolated into the bind string one line above. The two are the same
//! today and only one of them is an OBSERVATION: a caller that restates the
//! literal is correct until the day this bind changes, and nothing about the
//! restatement has to change for it to become wrong. The port was already
//! read this way and the host was not, which is the asymmetry that made the
//! second read worth doing.
//!
//! **THE BIND CANNOT HAPPEN ON THE CALLING THREAD, AND THAT IS NOT A STYLE
//! CHOICE.** An earlier version called `rt.block_on(TcpListener::bind(..))`
//! on the thread `ts_server_start` itself runs on, to read the resolved port
//! back before spawning the long-lived thread. That is exactly
//! `tokio::runtime::Runtime::block_on`'s documented panic:
//! "Cannot start a runtime from within a runtime" fires the instant a C
//! host's calling thread happens to already be inside some OTHER Tokio
//! runtime -- which is not a contrived case for a Rust FFI consumer, and it
//! is exactly what `tests/c_surface.rs`'s own `#[tokio::test]` server tests
//! hit immediately. `std::sync::mpsc::Receiver::recv` has no such
//! restriction: it is a plain OS-level blocking wait, legal to call from
//! any thread, async runtime or none.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use turbospark_server::observe::ServerObserver;

use crate::server_idle::{IdlePolicy, IdleSweep};
use crate::server_registry::{EventRing, LiveRegistry};
use crate::session::SessionCore;

const SHUTDOWN_GRACE_PERIOD: Duration = Duration::from_secs(1);

/// What `ts_server_start` writes to `*out`, and what `ts_server_stop` frees.
pub struct Server {
    port: u16,
    /// The IP actually bound, from `local_addr`, never the literal the bind
    /// string was built from (see the module doc).
    host: String,
    auth_enabled: bool,
    /// Resolved ONCE here and handed to every model attached later, so a
    /// server cannot end up serving two models under different rules. The
    /// same process-level scope `turbospark-server --guardrails` has.
    guardrails: turbospark_server::GuardrailConfig,
    default_system: Option<String>,
    default_reasoning: tokenizer::ReasoningEffort,
    started: Instant,
    traffic: Arc<crate::server_transport::Traffic>,
    /// Shared with the router on the background thread. Attaching and
    /// detaching mutate THIS, which is what lets a running server gain and
    /// lose models without rebinding.
    registry: Arc<LiveRegistry>,
    events: Arc<EventRing>,
    idle_sweep: Arc<IdleSweep>,
    image_bridge: Arc<crate::server_image::ImageBridge>,
    /// Set BEFORE the graceful-shutdown signal is sent, and read from every
    /// `FfiChatModel` this server has attached (`attach` clones it in).
    /// Axum's graceful shutdown waits for in-flight connections to finish and
    /// `Drop` blocks the calling thread on `stop`'s own join, so without this
    /// a `ts_server_stop` (or a `Server` going out of scope) called while a
    /// request is decoding blocks for up to that request's whole
    /// `max_tokens` -- on a GUI's main thread, that reads as a frozen window
    /// rather than a stop that took a moment. Composed into the cancel
    /// predicate `run_completion` and `run_with_images` already pass down,
    /// so a request cut this way reports `stopReason: cancelled`, same as
    /// `ts_session_cancel` reports for a direct `ts_generate` call.
    stopping: Arc<AtomicBool>,
    // `Option` so `stop` can be called from both `ts_server_stop` and `Drop`
    // without sending on a closed channel or joining a thread twice.
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Server {
    /// Spawns a background thread that binds `port` (0 for an OS-assigned
    /// one) on loopback or a Tailscale IPv4 address and serves it. Blocks
    /// until the socket is actually bound (or binding fails), never until
    /// the first request is served.
    ///
    /// Starts with NOTHING attached. `ts_server_start` calls
    /// [`Self::attach`] straight after when it was handed a session, which
    /// is what makes a null `session` mean "start empty" rather than being a
    /// second code path.
    pub(crate) fn start(
        port: u16,
        host: Option<String>,
        capture_text: bool,
        api_key: Option<String>,
        guardrails: turbospark_server::GuardrailConfig,
        default_system: Option<String>,
        default_reasoning: tokenizer::ReasoningEffort,
    ) -> Result<Self, String> {
        let host: std::net::IpAddr = host
            .as_deref()
            .unwrap_or("127.0.0.1")
            .parse()
            .map_err(|_| "Host must be a literal IPv4 or IPv6 address".to_string())?;
        let api_key = api_key.filter(|key| !key.trim().is_empty());
        let tailnet = match host {
            std::net::IpAddr::V4(ip) => {
                let octets = ip.octets();
                octets[0] == 100 && (64..=127).contains(&octets[1])
            }
            std::net::IpAddr::V6(_) => false,
        };
        if !host.is_loopback() && !tailnet {
            return Err(
                "Host must be loopback or a Tailscale IPv4 address in 100.64.0.0/10".into(),
            );
        }
        if tailnet && api_key.is_none() {
            return Err("An API key is required for a Tailscale address".into());
        }
        let traffic = Arc::new(crate::server_transport::Traffic::new(capture_text));
        let auth_enabled = api_key.is_some();
        let registry = Arc::new(LiveRegistry::default());
        let events = Arc::new(EventRing::default());
        let stopping = Arc::new(AtomicBool::new(false));
        let loader = Arc::new(HostLoader::new(
            Arc::clone(&registry),
            Arc::clone(&events),
            Arc::clone(&stopping),
            guardrails,
            default_system.clone(),
            default_reasoning,
        ));
        let resolver = Arc::new(turbospark_server::registry::ModelLoaderRegistry::new(
            Arc::clone(&registry) as Arc<dyn turbospark_server::registry::ModelRegistry>,
            loader,
        ));
        let idle_sweep = Arc::new(IdleSweep::new(Arc::clone(&registry), Arc::clone(&events)));
        let idle_sweep_task = Arc::clone(&idle_sweep);

        let image_bridge = Arc::new(crate::server_image::ImageBridge::default());
        let router = turbospark_server::build_router_with_options(
            turbospark_server::ServerState::new(
                resolver as Arc<dyn turbospark_server::registry::ModelRegistry>,
            )
            .with_image_provider(
                Arc::clone(&image_bridge) as Arc<dyn turbospark_server::ImageProvider>
            ),
            turbospark_server::RouterOptions {
                api_key,
                observer: Some(
                    Arc::clone(&events) as Arc<dyn turbospark_server::observe::ServerObserver>
                ),
            },
        );

        let router = router.layer(axum::middleware::from_fn_with_state(
            Arc::clone(&traffic),
            crate::server_transport::observe,
        ));
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        // Plain blocking channel, not `tokio::sync`: `start`'s CALLER may or
        // may not be inside a Tokio runtime of its own, and this is how the
        // bind result gets back to them without ever needing a runtime on
        // their thread (see the module doc).
        let (ready_tx, ready_rx) =
            std::sync::mpsc::channel::<Result<std::net::SocketAddr, String>>();
        let thread = std::thread::Builder::new()
            .name("turbospark-ffi-server".to_string())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ =
                            ready_tx.send(Err(format!("failed to start an async runtime: {e}")));
                        return;
                    }
                };
                rt.block_on(async move {
                    tokio::spawn(idle_sweep_task.run());
                    let addr = std::net::SocketAddr::new(host, port);
                    let listener = match tokio::net::TcpListener::bind(addr).await {
                        Ok(l) => l,
                        Err(e) => {
                            let _ = ready_tx.send(Err(format!("failed to bind {addr}: {e}")));
                            return;
                        }
                    };
                    // The WHOLE address, not just its port: the host half is
                    // an observation too, and this is the only call that can
                    // make it (see the module doc).
                    let resolved = match listener.local_addr() {
                        Ok(a) => a,
                        Err(e) => {
                            let _ = ready_tx
                                .send(Err(format!("failed to read the bound address: {e}")));
                            return;
                        }
                    };
                    // Sent BEFORE `axum::serve`, which does not return until
                    // shutdown -- `start`'s caller is blocked on `ready_rx`
                    // and must hear about the successful bind now, not after
                    // the server has already stopped.
                    if ready_tx.send(Ok(resolved)).is_err() {
                        // The receiving end hung up (e.g. `start` itself
                        // panicked before reaching `recv`). Nothing left to
                        // serve for.
                        return;
                    }
                    // A serve error (e.g. the listener closing unexpectedly)
                    // has no caller left to report to by the time it
                    // happens; the thread simply ends, same as a graceful
                    // shutdown does.
                    let (grace_started_tx, grace_started_rx) = tokio::sync::oneshot::channel();
                    let server = std::future::IntoFuture::into_future(
                        axum::serve(listener, router).with_graceful_shutdown(async {
                            let _ = shutdown_rx.await;
                            let _ = grace_started_tx.send(());
                        }),
                    );
                    tokio::pin!(server);
                    tokio::select! {
                        _ = &mut server => {}
                        _ = async {
                            if grace_started_rx.await.is_ok() {
                                tokio::time::sleep(SHUTDOWN_GRACE_PERIOD).await;
                            } else {
                                std::future::pending::<()>().await;
                            }
                        } => {}
                    }
                });
                shutdown_runtime(rt);
            })
            .map_err(|e| format!("failed to start the server thread: {e}"))?;

        let resolved = match ready_rx.recv() {
            Ok(result) => result?,
            Err(_) => {
                // The thread ended (its runtime failed to build, most
                // likely) without ever sending: join it to surface a panic
                // message if there is one, rather than a bare "disconnected".
                let _ = thread.join();
                return Err("the server thread exited before it finished starting".to_string());
            }
        };

        Ok(Self {
            port: resolved.port(),
            host: resolved.ip().to_string(),
            auth_enabled,
            guardrails,
            default_system,
            default_reasoning,
            started: Instant::now(),
            traffic,
            registry,
            events,
            idle_sweep,
            image_bridge,
            stopping,
            shutdown: Some(shutdown_tx),
            thread: Some(thread),
        })
    }

    /// Adds an open session's model to this running server.
    ///
    /// The `Arc<SessionCore>` clone is what keeps the engine alive
    /// independently of the caller's `TsSession *`; see
    /// `server_registry.rs`'s header for the obligation that creates.
    pub(crate) fn attach(
        &self,
        core: Arc<SessionCore>,
        model_id: String,
    ) -> Result<String, String> {
        let model: Arc<dyn turbospark_server::ChatModel> =
            Arc::new(crate::server_model::FfiChatModel::new(
                core,
                model_id,
                self.guardrails,
                self.default_system.clone(),
                self.default_reasoning,
                Arc::clone(&self.stopping),
            ));
        let id = self.registry.attach(model)?;
        ServerObserver::record(
            &*self.events,
            turbospark_server::observe::ServerEvent::ModelAttached {
                at_ms: now_ms(),
                model: id.clone(),
            },
        );
        Ok(id)
    }

    /// Adds an embedding model to this running server.
    pub(crate) fn attach_embedding_model(&self, model_arg: &str) -> Result<String, String> {
        #[cfg(target_os = "macos")]
        {
            let dir = catalog::resolve_model_arg(model_arg);
            let model = turbospark_server::RealEncoderModel::open(&dir)
                .map_err(|e| format!("failed to open embedding model from {}: {e}", dir.display()))?
                .with_model_id(model_arg.to_string());
            let id = self.registry.attach(Arc::new(model))?;
            ServerObserver::record(
                &*self.events,
                turbospark_server::observe::ServerEvent::ModelAttached {
                    at_ms: now_ms(),
                    model: id.clone(),
                },
            );
            Ok(id)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = model_arg;
            Err("embedding models are supported on macOS only".to_string())
        }
    }

    /// Removes a model by id, releasing this server's reference to it.
    /// Returns whether one was attached under that id.
    pub(crate) fn detach(&self, id: &str) -> bool {
        let removed = self.registry.detach(id);
        if removed {
            ServerObserver::record(
                &*self.events,
                turbospark_server::observe::ServerEvent::ModelDetached {
                    at_ms: now_ms(),
                    model: id.to_string(),
                },
            );
        }
        removed
    }

    pub(crate) fn set_idle_policy(&self, policy: IdlePolicy) -> Result<(), String> {
        self.idle_sweep.set_policy(policy)
    }

    pub(crate) fn image_bridge(&self) -> &crate::server_image::ImageBridge {
        &self.image_bridge
    }

    /// Takes up to `max` buffered events, with however many were dropped
    /// since the last call.
    pub(crate) fn drain_events(&self, max: usize) -> crate::wire::ServerEvents {
        let (events, dropped) = self.events.drain(max);
        crate::wire::ServerEvents { events, dropped }
    }

    pub(crate) fn info(&self) -> crate::wire::ServerInfo {
        let ids = self.registry.ids();
        crate::wire::ServerInfo {
            port: self.port,
            host: self.host.clone(),
            // **THE FIRST ATTACHED MODEL, KEPT FOR THE ONE-MODEL READER.**
            // `models` beside it is the real answer; this stays a bare
            // string so a host written against the pre-registry shape reads
            // something sensible rather than nothing. Empty when the server
            // has nothing attached, which is a state a host can start one in.
            model_id: ids.first().cloned().unwrap_or_default(),
            models: ids,
            image_models: self.image_bridge.model().into_iter().collect(),
            auth_enabled: self.auth_enabled,
            uptime_seconds: self.started.elapsed().as_secs(),
            traffic: self.traffic.snapshot(),
        }
    }

    /// Signals graceful shutdown and blocks until the background thread has
    /// stopped serving, forcing it after a bounded grace period. Idempotent:
    /// a second call is a no-op.
    ///
    /// `stopping` is set BEFORE the shutdown signal is sent, and before the
    /// join below blocks this thread: axum's graceful shutdown waits for
    /// in-flight connections, so a request mid-decode would otherwise hold
    /// this call (and whatever thread called it, e.g. a GUI's main thread via
    /// `ts_server_stop`) for up to that request's whole `max_tokens`. Every
    /// `FfiChatModel` this server attached reads the same flag from its
    /// cancel predicate, so setting it here is what actually ends the
    /// decode the join is waiting on, not the signal to axum by itself. A
    /// connection stalled before model dispatch cannot observe that flag, so
    /// the server task is dropped after `SHUTDOWN_GRACE_PERIOD` as a backstop.
    fn stop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        self.image_bridge.stop();
        if let Some(tx) = self.shutdown.take() {
            // A dropped receiver (the thread already exited on its own,
            // e.g. a bind race elsewhere tore the listener down) means
            // there is nothing left to signal; not an error here.
            let _ = tx.send(());
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Milliseconds since the epoch, for placing an attach or detach on the
/// same timeline the router's own events use.
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn shutdown_runtime(runtime: tokio::runtime::Runtime) {
    // A synchronous model open cannot be interrupted by dropping its HTTP
    // waiter. Its owned Arcs keep it safe to finish after the server stops,
    // but runtime drop must not make the host wait beyond the serve grace.
    runtime.shutdown_timeout(Duration::ZERO);
}

/// Turns a `TsServer *` back into a borrow.
///
/// # Safety
/// `ptr` must be null, or a pointer returned by `ts_server_start` and not
/// yet stopped.
pub(crate) unsafe fn borrow<'a>(ptr: *const Server) -> Result<&'a Server, String> {
    ptr.as_ref()
        .ok_or_else(|| "server must not be null".to_string())
}

struct HostLoader {
    registry: std::sync::Arc<LiveRegistry>,
    events: std::sync::Arc<EventRing>,
    stopping: std::sync::Arc<AtomicBool>,
    guardrails: turbospark_server::GuardrailConfig,
    default_system: Option<String>,
    default_reasoning: tokenizer::ReasoningEffort,
    single_flight: SingleFlight,
}

impl HostLoader {
    fn new(
        registry: std::sync::Arc<LiveRegistry>,
        events: std::sync::Arc<EventRing>,
        stopping: std::sync::Arc<AtomicBool>,
        guardrails: turbospark_server::GuardrailConfig,
        default_system: Option<String>,
        default_reasoning: tokenizer::ReasoningEffort,
    ) -> Self {
        Self {
            registry,
            events,
            stopping,
            guardrails,
            default_system,
            default_reasoning,
            single_flight: SingleFlight::default(),
        }
    }

    fn open_installed(
        &self,
        row: catalog::InstalledModel,
    ) -> turbospark_server::registry::LoadOutcome {
        #[cfg(target_os = "macos")]
        {
            let path = row.path.to_string_lossy().into_owned();
            let session =
                match crate::open::open_with_failure(&path, &crate::wire::OpenOptions::default()) {
                    Ok(session) => session,
                    Err(error) => {
                        return turbospark_server::registry::LoadOutcome::Refused(
                            open_failure_to_refusal(error),
                        )
                    }
                };
            let model_id = std::path::Path::new(&session.info.model_path)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| row.alias.clone());
            let model = match attach_chat_model(
                &self.registry,
                &self.events,
                Arc::clone(&self.stopping),
                self.guardrails,
                self.default_system.clone(),
                self.default_reasoning,
                session.core(),
                model_id,
                vec![row.alias],
            ) {
                Ok(model) => model,
                Err(detail) => {
                    return turbospark_server::registry::LoadOutcome::Refused(
                        turbospark_server::registry::LoadRefusal::OpenFailed { detail },
                    )
                }
            };
            turbospark_server::registry::LoadOutcome::Loaded(model)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = row;
            turbospark_server::registry::LoadOutcome::Refused(
                turbospark_server::registry::LoadRefusal::UnsupportedFamily {
                    family: "the inference runtime is supported on macOS only".to_string(),
                },
            )
        }
    }
}

impl turbospark_server::registry::ModelLoader for HostLoader {
    fn load(&self, id: &str) -> turbospark_server::registry::LoadOutcome {
        use turbospark_server::registry::LoadOutcome;

        if let Some(model) = self.registry.exact_model(id) {
            return LoadOutcome::Loaded(model);
        }

        let row = match catalog::Store::default_store() {
            Ok(store) => match find_installed(&store, id) {
                Ok(row) => row,
                Err(refusal) => return LoadOutcome::Refused(refusal),
            },
            Err(_) => None,
        };
        match row {
            Some(row) => {
                let flight_key = row.alias.clone();
                load_after_miss(&self.registry, &self.single_flight, id, &flight_key, || {
                    self.open_installed(row)
                })
            }
            None => LoadOutcome::Unavailable {
                candidates: self.candidates(id),
            },
        }
    }

    fn candidates(&self, requested: &str) -> Vec<turbospark_server::registry::ModelSuggestion> {
        use turbospark_server::registry::{ModelSuggestion, SuggestionSource};

        let mut candidates = Vec::new();
        if let Ok(store) = catalog::Store::default_store() {
            for (id, model) in store.installed() {
                if model.effective_modality() == catalog::ModelModality::Text
                    && model.kind.as_deref() != Some("vision-tower")
                {
                    candidates.push(ModelSuggestion {
                        label: id.clone(),
                        id,
                        sources: vec![SuggestionSource::Installed],
                    });
                }
            }
        }
        if let Ok(catalog) = catalog::Catalog::embedded() {
            candidates.extend(
                catalog
                    .entries()
                    .filter(|entry| entry.kind == catalog::EntryKind::Model)
                    .map(|entry| ModelSuggestion {
                        id: entry.alias.clone(),
                        label: entry.name.clone(),
                        sources: vec![SuggestionSource::Catalog],
                    }),
            );
        }
        rank_suggestions(requested, candidates)
    }
}

/// Local identity hints only. Merge provenance before applying a bounded,
/// deterministic name ranking; no recommendation here implies model fit.
fn rank_suggestions(
    requested: &str,
    candidates: Vec<turbospark_server::registry::ModelSuggestion>,
) -> Vec<turbospark_server::registry::ModelSuggestion> {
    use turbospark_server::registry::SuggestionSource;

    let needle = requested.trim().to_lowercase();
    if needle.is_empty() {
        return Vec::new();
    }
    let mut merged = std::collections::BTreeMap::new();
    for candidate in candidates {
        let row = merged
            .entry(candidate.id.clone())
            .or_insert(candidate.clone());
        for source in candidate.sources {
            if !row.sources.contains(&source) {
                row.sources.push(source);
            }
        }
    }
    let mut ranked: Vec<_> = merged
        .into_values()
        .filter_map(|row| {
            let id = row.id.to_lowercase();
            let rank = if id == needle {
                0
            } else if id.starts_with(&needle) {
                1
            } else if id.contains(&needle) || row.label.to_lowercase().contains(&needle) {
                2
            } else {
                return None;
            };
            let catalog_only = !row.sources.contains(&SuggestionSource::Installed);
            Some(((rank, catalog_only, id), row))
        })
        .collect();
    ranked.sort_by(|a, b| a.0.cmp(&b.0));
    ranked.into_iter().take(5).map(|(_, row)| row).collect()
}

fn load_after_miss(
    registry: &LiveRegistry,
    single_flight: &SingleFlight,
    requested: &str,
    flight_key: &str,
    load: impl FnOnce() -> turbospark_server::registry::LoadOutcome,
) -> turbospark_server::registry::LoadOutcome {
    use turbospark_server::registry::LoadOutcome;

    single_flight.run(flight_key, || {
        if let Some(model) = registry.exact_model(requested) {
            LoadOutcome::Loaded(model)
        } else {
            load()
        }
    })
}

fn find_installed(
    store: &catalog::Store,
    requested: &str,
) -> Result<Option<catalog::InstalledModel>, turbospark_server::registry::LoadRefusal> {
    find_installed_in(store.installed(), requested)
}

fn find_installed_in(
    mut installed: std::collections::BTreeMap<String, catalog::InstalledModel>,
    requested: &str,
) -> Result<Option<catalog::InstalledModel>, turbospark_server::registry::LoadRefusal> {
    // Store aliases are explicit identities. A coincidentally equal directory
    // basename must never redirect a request to a different installed model.
    if let Some(row) = installed.remove(requested) {
        return Ok(Some(row));
    }
    let mut matches = installed
        .into_values()
        .filter(|row| row.path.file_name().and_then(|name| name.to_str()) == Some(requested));
    let first = matches.next();
    if matches.next().is_some() {
        return Err(turbospark_server::registry::LoadRefusal::OpenFailed {
            detail: format!(
                "installed directory identity '{requested}' is ambiguous; use a store alias"
            ),
        });
    }
    Ok(first)
}

fn attach_chat_model(
    registry: &LiveRegistry,
    events: &EventRing,
    stopping: Arc<AtomicBool>,
    guardrails: turbospark_server::GuardrailConfig,
    default_system: Option<String>,
    default_reasoning: tokenizer::ReasoningEffort,
    core: Arc<SessionCore>,
    model_id: String,
    aliases: Vec<String>,
) -> Result<Arc<dyn turbospark_server::ChatModel>, String> {
    let mut aliases = aliases;
    aliases.retain(|alias| !alias.is_empty() && alias != &model_id);
    let mut unique_aliases = Vec::new();
    for alias in aliases {
        if !unique_aliases.contains(&alias) {
            unique_aliases.push(alias);
        }
    }
    let model: Arc<dyn turbospark_server::ChatModel> =
        Arc::new(crate::server_model::FfiChatModel::new_with_aliases(
            core,
            model_id,
            unique_aliases,
            guardrails,
            default_system,
            default_reasoning,
            stopping,
        ));
    let id = registry.attach(Arc::clone(&model))?;
    ServerObserver::record(
        events,
        turbospark_server::observe::ServerEvent::ModelAttached {
            at_ms: now_ms(),
            model: id,
        },
    );
    Ok(model)
}

#[cfg(target_os = "macos")]
fn open_failure_to_refusal(
    error: crate::open::OpenFailure,
) -> turbospark_server::registry::LoadRefusal {
    use turbospark_server::registry::LoadRefusal;
    match error {
        crate::open::OpenFailure::ContextRefused(runtime::ContextRefused::TooLarge(reason)) => {
            LoadRefusal::DoesNotFit {
                committed_bytes: reason.committed.saturating_add(reason.needs),
                ceiling_bytes: reason.committed.saturating_add(reason.available),
            }
        }
        crate::open::OpenFailure::ContextRefused(runtime::ContextRefused::OverCap(reason)) => {
            LoadRefusal::DoesNotFit {
                committed_bytes: reason.counted,
                ceiling_bytes: reason.cap,
            }
        }
        crate::open::OpenFailure::ContextRefused(reason) => LoadRefusal::OpenFailed {
            detail: reason.to_string(),
        },
        crate::open::OpenFailure::Failed(detail) => LoadRefusal::OpenFailed { detail },
    }
}

#[derive(Default)]
struct SingleFlight {
    flights: std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<LoadFlight>>>,
}

struct LoadFlight {
    outcome: std::sync::Mutex<Option<SharedLoadOutcome>>,
    completed: std::sync::Condvar,
}

enum SharedLoadOutcome {
    Loaded(std::sync::Arc<dyn turbospark_server::ChatModel>),
    Refused(turbospark_server::registry::LoadRefusal),
    Unavailable {
        candidates: Vec<turbospark_server::registry::ModelSuggestion>,
    },
}

impl SharedLoadOutcome {
    fn from_outcome(outcome: &turbospark_server::registry::LoadOutcome) -> Self {
        use turbospark_server::registry::LoadOutcome;
        match outcome {
            LoadOutcome::Loaded(model) => Self::Loaded(std::sync::Arc::clone(model)),
            LoadOutcome::Refused(refusal) => Self::Refused(refusal.clone()),
            LoadOutcome::Unavailable { candidates } => Self::Unavailable {
                candidates: candidates.clone(),
            },
        }
    }

    fn to_outcome(&self) -> turbospark_server::registry::LoadOutcome {
        use turbospark_server::registry::LoadOutcome;
        match self {
            Self::Loaded(model) => LoadOutcome::Loaded(std::sync::Arc::clone(model)),
            Self::Refused(refusal) => LoadOutcome::Refused(refusal.clone()),
            Self::Unavailable { candidates } => LoadOutcome::Unavailable {
                candidates: candidates.clone(),
            },
        }
    }
}

impl LoadFlight {
    fn wait(&self) -> turbospark_server::registry::LoadOutcome {
        let mut outcome = self.outcome.lock().unwrap_or_else(|p| p.into_inner());
        while outcome.is_none() {
            outcome = self
                .completed
                .wait(outcome)
                .unwrap_or_else(|p| p.into_inner());
        }
        outcome
            .as_ref()
            .expect("completed outcome was checked")
            .to_outcome()
    }

    fn finish(&self, outcome: &turbospark_server::registry::LoadOutcome) {
        let mut slot = self.outcome.lock().unwrap_or_else(|p| p.into_inner());
        *slot = Some(SharedLoadOutcome::from_outcome(outcome));
        self.completed.notify_all();
    }
}

impl SingleFlight {
    fn run(
        &self,
        key: &str,
        load: impl FnOnce() -> turbospark_server::registry::LoadOutcome,
    ) -> turbospark_server::registry::LoadOutcome {
        let (flight, leader) = {
            let mut flights = self.flights.lock().unwrap_or_else(|p| p.into_inner());
            match flights.get(key) {
                Some(flight) => (std::sync::Arc::clone(flight), false),
                None => {
                    let flight = std::sync::Arc::new(LoadFlight {
                        outcome: std::sync::Mutex::new(None),
                        completed: std::sync::Condvar::new(),
                    });
                    flights.insert(key.to_string(), std::sync::Arc::clone(&flight));
                    (flight, true)
                }
            }
        };

        if !leader {
            return flight.wait();
        }

        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(load)).unwrap_or_else(|_| {
                turbospark_server::registry::LoadOutcome::Refused(
                    turbospark_server::registry::LoadRefusal::OpenFailed {
                        detail: "the installed model open panicked".to_string(),
                    },
                )
            });
        flight.finish(&outcome);

        let mut flights = self.flights.lock().unwrap_or_else(|p| p.into_inner());
        if flights
            .get(key)
            .is_some_and(|current| std::sync::Arc::ptr_eq(current, &flight))
        {
            flights.remove(key);
        }
        outcome
    }
}

#[cfg(test)]
mod single_flight_tests {
    use super::{load_after_miss, SingleFlight};
    use crate::server_registry::LiveRegistry;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, Condvar, Mutex};
    use std::time::Duration;
    use turbospark_server::registry::{LoadOutcome, LoadRefusal};

    #[test]
    fn runtime_shutdown_returns_while_a_blocking_model_open_is_still_running() {
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (stopped_tx, stopped_rx) = std::sync::mpsc::channel();
        let (finished_tx, finished_rx) = std::sync::mpsc::channel();
        let server_thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .build()
                .unwrap();
            runtime.spawn_blocking(move || {
                started_tx.send(()).unwrap();
                let _ = release_rx.recv();
                finished_tx.send(()).unwrap();
            });
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            super::shutdown_runtime(runtime);
            stopped_tx.send(()).unwrap();
        });
        // Always release the worker before asserting, so a failed mutation
        // cannot leave a blocked test thread behind.
        let stopped = stopped_rx.recv_timeout(Duration::from_secs(5));
        release_tx.send(()).unwrap();
        finished_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        server_thread.join().unwrap();
        assert!(stopped.is_ok(), "runtime shutdown waited for model open");
    }

    #[test]
    fn explicit_installed_alias_wins_over_another_models_directory_name() {
        let rows = installed_rows_for_identity_test();
        let row = super::find_installed_in(rows, "requested.gturbo")
            .unwrap()
            .unwrap();
        assert_eq!(row.alias, "requested.gturbo");
        assert_eq!(row.path, std::path::PathBuf::from("/models/right.gturbo"));
    }

    #[test]
    fn ambiguous_installed_directory_names_are_refused() {
        let mut rows = installed_rows_for_identity_test();
        let mut duplicate = rows.remove("requested.gturbo").unwrap();
        duplicate.alias = "z-another-model".into();
        duplicate.path = "/other/requested.gturbo".into();
        rows.insert(duplicate.alias.clone(), duplicate);
        assert!(matches!(
            super::find_installed_in(rows, "requested.gturbo"),
            Err(LoadRefusal::OpenFailed { .. })
        ));
    }

    fn installed_rows_for_identity_test(
    ) -> std::collections::BTreeMap<String, catalog::InstalledModel> {
        [
            ("a-model", "/models/requested.gturbo"),
            ("requested.gturbo", "/models/right.gturbo"),
        ]
        .into_iter()
        .map(|(alias, path)| {
            let row: catalog::InstalledModel = serde_json::from_value(serde_json::json!({
                "alias": alias,
                "repo": "fixture/model",
                "revision": "fixture",
                "path": path,
                "family": "qwen3",
                "install_bytes": 0,
                "installed_on": "2026-10-03",
                "status": "unlisted"
            }))
            .unwrap();
            (alias.to_string(), row)
        })
        .collect()
    }

    #[test]
    fn suggestions_merge_sources_rank_exact_names_and_limit_results() {
        use turbospark_server::registry::{ModelSuggestion, SuggestionSource};
        let suggestion = |id: &str, source| ModelSuggestion {
            id: id.to_string(),
            label: id.to_string(),
            sources: vec![source],
        };
        let mut candidates: Vec<_> = (0..8)
            .map(|index| suggestion(&format!("qwen-{index}"), SuggestionSource::Catalog))
            .collect();
        candidates.push(suggestion("qwen", SuggestionSource::Catalog));
        candidates.push(suggestion("qwen-7", SuggestionSource::Installed));
        candidates.push(suggestion("unrelated", SuggestionSource::Installed));
        let ranked = super::rank_suggestions("QWEN", candidates);
        assert_eq!(ranked.len(), 5);
        assert_eq!(ranked[0].id, "qwen");
        assert_eq!(ranked[1].id, "qwen-7");
        assert_eq!(
            ranked[1].sources,
            vec![SuggestionSource::Catalog, SuggestionSource::Installed]
        );
        assert!(ranked.iter().all(|row| row.id.starts_with("qwen")));
        assert!(super::rank_suggestions("  ", ranked).is_empty());
    }

    #[test]
    fn host_loader_returns_bundled_catalog_hints_without_opening_models() {
        use turbospark_server::registry::{ModelLoader, SuggestionSource};
        let registry = Arc::new(LiveRegistry::default());
        let events = Arc::new(crate::server_registry::EventRing::default());
        let loader = super::HostLoader::new(
            Arc::clone(&registry),
            Arc::clone(&events),
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            turbospark_server::GuardrailConfig::default(),
            None,
            tokenizer::ReasoningEffort::Off,
        );
        let candidates = loader.candidates("qwen");
        assert!(!candidates.is_empty());
        assert!(candidates
            .iter()
            .any(|row| row.sources.contains(&SuggestionSource::Catalog)));
        assert!(registry.ids().is_empty());
        assert!(events.drain(10).0.is_empty());
    }

    #[test]
    fn concurrent_requests_for_one_id_share_one_open_result() {
        let single_flight = Arc::new(SingleFlight::default());
        let opens = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(3));
        let release = Arc::new((Mutex::new(false), Condvar::new()));

        let request = || {
            let single_flight = Arc::clone(&single_flight);
            let opens = Arc::clone(&opens);
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            std::thread::spawn(move || {
                start.wait();
                single_flight.run("qwen-local", || {
                    opens.fetch_add(1, Ordering::SeqCst);
                    let (lock, changed) = &*release;
                    let mut ready = lock.lock().unwrap();
                    while !*ready {
                        ready = changed.wait(ready).unwrap();
                    }
                    LoadOutcome::Refused(LoadRefusal::OpenFailed {
                        detail: "fixture refusal".to_string(),
                    })
                })
            })
        };

        let first = request();
        let second = request();
        start.wait();
        std::thread::sleep(Duration::from_millis(100));
        {
            let (lock, changed) = &*release;
            *lock.lock().unwrap() = true;
            changed.notify_all();
        }

        let first = first.join().unwrap();
        let second = second.join().unwrap();
        assert_eq!(opens.load(Ordering::SeqCst), 1);
        assert!(matches!(first, LoadOutcome::Refused(_)));
        assert!(matches!(second, LoadOutcome::Refused(_)));
    }

    #[test]
    fn delayed_request_reuses_model_attached_after_its_initial_miss() {
        let registry = LiveRegistry::default();
        let single_flight = SingleFlight::default();
        let opens = AtomicUsize::new(0);

        let first_saw_unloaded = registry.exact_model("scripted").is_none();
        let delayed_saw_unloaded = registry.exact_model("scripted").is_none();
        assert!(first_saw_unloaded);
        assert!(delayed_saw_unloaded);

        let tokenizer_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tokenizer/tests/fixtures/ChatMLTokenizer");
        let tokenizer = tokenizer::MfTokenizer::load_from_dir(&tokenizer_dir)
            .expect("fixture tokenizer should load");
        let model: Arc<dyn turbospark_server::ChatModel> = Arc::new(
            turbospark_server::ScriptedChatModel::new(tokenizer, 4096, Vec::new()),
        );

        let first = load_after_miss(&registry, &single_flight, "scripted", "scripted", || {
            opens.fetch_add(1, Ordering::SeqCst);
            registry
                .attach(Arc::clone(&model))
                .expect("first load should attach");
            LoadOutcome::Loaded(model)
        });
        let delayed = load_after_miss(&registry, &single_flight, "scripted", "scripted", || {
            opens.fetch_add(1, Ordering::SeqCst);
            LoadOutcome::Refused(LoadRefusal::OpenFailed {
                detail: "duplicate open should not run".to_string(),
            })
        });

        let (first, delayed) = match (first, delayed) {
            (LoadOutcome::Loaded(first), LoadOutcome::Loaded(delayed)) => (first, delayed),
            _ => panic!("both requests should resolve to the attached model"),
        };
        assert_eq!(opens.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&first, &delayed));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn context_refusals_map_to_does_not_fit_with_combined_byte_arithmetic() {
        let too_large = super::open_failure_to_refusal(crate::open::OpenFailure::ContextRefused(
            runtime::ContextRefused::TooLarge(runtime::ContextTooLarge {
                requested: 32_768,
                needs: 100,
                available: 60,
                reserve: 10,
                physical: 1_000,
                committed: 40,
                largest_fitting: 16_384,
            }),
        ));
        match too_large {
            LoadRefusal::DoesNotFit {
                committed_bytes,
                ceiling_bytes,
            } => {
                assert_eq!(committed_bytes, 140, "committed 40 + needs 100");
                assert_eq!(ceiling_bytes, 100, "committed 40 + available 60");
            }
            other => panic!("TooLarge must map to DoesNotFit, got {other:?}"),
        }

        // Saturating adds keep the refusal finite when the terms overflow.
        let saturated = super::open_failure_to_refusal(crate::open::OpenFailure::ContextRefused(
            runtime::ContextRefused::TooLarge(runtime::ContextTooLarge {
                requested: 32_768,
                needs: u64::MAX,
                available: 60,
                reserve: 10,
                physical: 1_000,
                committed: 40,
                largest_fitting: 16_384,
            }),
        ));
        match saturated {
            LoadRefusal::DoesNotFit {
                committed_bytes,
                ceiling_bytes,
            } => {
                assert_eq!(committed_bytes, u64::MAX);
                assert_eq!(ceiling_bytes, 100);
            }
            other => panic!("saturated TooLarge must map to DoesNotFit, got {other:?}"),
        }

        let over_cap = super::open_failure_to_refusal(crate::open::OpenFailure::ContextRefused(
            runtime::ContextRefused::OverCap(runtime::ContextOverCap {
                requested: 8_192,
                counted: 900,
                cap: 800,
                kv_bytes: 500,
                slot_cache: 400,
            }),
        ));
        match over_cap {
            LoadRefusal::DoesNotFit {
                committed_bytes,
                ceiling_bytes,
            } => {
                assert_eq!(committed_bytes, 900, "counted is reported verbatim");
                assert_eq!(ceiling_bytes, 800, "cap is reported verbatim");
            }
            other => panic!("OverCap must map to DoesNotFit, got {other:?}"),
        }

        let floor_unmet = super::open_failure_to_refusal(crate::open::OpenFailure::ContextRefused(
            runtime::ContextRefused::FloorUnmet(runtime::ContextFloorUnmet {
                floor: 8_192,
                resolved: 4_096,
                largest_fitting: 4_096,
                capped_by: runtime::ContextCap::Memory,
            }),
        ));
        assert!(matches!(
            floor_unmet,
            LoadRefusal::OpenFailed { detail } if detail.contains("--min-auto-context")
        ));

        let failed = super::open_failure_to_refusal(crate::open::OpenFailure::Failed(
            "weights unreadable".to_string(),
        ));
        assert!(matches!(
            failed,
            LoadRefusal::OpenFailed { ref detail } if detail == "weights unreadable"
        ));
    }

    #[test]
    fn a_panicking_open_refuses_leader_and_follower_then_allows_a_fresh_open() {
        let single_flight = Arc::new(SingleFlight::default());
        let opens = Arc::new(AtomicUsize::new(0));
        let release = Arc::new((Mutex::new(false), Condvar::new()));

        // The panicking body signals once its flight is registered and it is
        // blocked inside the open, so the follower below cannot win the
        // leader slot instead.
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let panicker = {
            let single_flight = Arc::clone(&single_flight);
            let opens = Arc::clone(&opens);
            let release = Arc::clone(&release);
            std::thread::spawn(move || {
                single_flight.run("panic-key", || {
                    opens.fetch_add(1, Ordering::SeqCst);
                    entered_tx.send(()).unwrap();
                    let (lock, changed) = &*release;
                    let mut ready = lock.lock().unwrap();
                    while !*ready {
                        ready = changed.wait(ready).unwrap();
                    }
                    panic!("fixture open failure");
                })
            })
        };
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("panicking open should start");

        let follower = {
            let single_flight = Arc::clone(&single_flight);
            std::thread::spawn(move || {
                single_flight.run("panic-key", || {
                    LoadOutcome::Refused(LoadRefusal::OpenFailed {
                        detail: "follower body must never run".to_string(),
                    })
                })
            })
        };
        // Give the follower time to park on the flight's condvar.
        std::thread::sleep(Duration::from_millis(100));
        {
            let (lock, changed) = &*release;
            *lock.lock().unwrap() = true;
            changed.notify_all();
        }

        for outcome in [panicker.join().unwrap(), follower.join().unwrap()] {
            match outcome {
                LoadOutcome::Refused(LoadRefusal::OpenFailed { detail }) => {
                    assert_eq!(detail, "the installed model open panicked");
                }
                other => {
                    let _ = other;
                    panic!("a panicked open must refuse, got a non-refusal outcome")
                }
            }
        }
        assert_eq!(opens.load(Ordering::SeqCst), 1);

        // The panicked flight is removed, so a later request opens again.
        let retry = single_flight.run("panic-key", || {
            opens.fetch_add(1, Ordering::SeqCst);
            LoadOutcome::Refused(LoadRefusal::OpenFailed {
                detail: "fresh open ran".to_string(),
            })
        });
        assert_eq!(opens.load(Ordering::SeqCst), 2);
        assert!(matches!(
            retry,
            LoadOutcome::Refused(LoadRefusal::OpenFailed { ref detail }) if detail == "fresh open ran"
        ));
    }
}
