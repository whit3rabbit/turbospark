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

use std::sync::Arc;
use std::time::Instant;

use turbospark_server::observe::ServerObserver;

use crate::server_registry::{EventRing, LiveRegistry};
use crate::session::SessionCore;

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
    /// Shared with the router on the background thread. Attaching and
    /// detaching mutate THIS, which is what lets a running server gain and
    /// lose models without rebinding.
    registry: Arc<LiveRegistry>,
    events: Arc<EventRing>,
    // `Option` so `stop` can be called from both `ts_server_stop` and `Drop`
    // without sending on a closed channel or joining a thread twice.
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Server {
    /// Spawns a background thread that binds `port` (0 for an OS-assigned
    /// one) on loopback and serves it. Blocks until the socket is actually
    /// bound (or binding fails), never until the first request is served.
    ///
    /// Starts with NOTHING attached. `ts_server_start` calls
    /// [`Self::attach`] straight after when it was handed a session, which
    /// is what makes a null `session` mean "start empty" rather than being a
    /// second code path.
    pub(crate) fn start(
        port: u16,
        api_key: Option<String>,
        guardrails: turbospark_server::GuardrailConfig,
        default_system: Option<String>,
        default_reasoning: tokenizer::ReasoningEffort,
    ) -> Result<Self, String> {
        let auth_enabled = api_key.is_some();
        let registry = Arc::new(LiveRegistry::default());
        let events = Arc::new(EventRing::default());
        let router = turbospark_server::build_router_with_options(
            Arc::clone(&registry) as Arc<dyn turbospark_server::registry::ModelRegistry>,
            turbospark_server::RouterOptions {
                api_key,
                observer: Some(
                    Arc::clone(&events) as Arc<dyn turbospark_server::observe::ServerObserver>
                ),
            },
        );

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
                    let addr = format!("127.0.0.1:{port}");
                    let listener = match tokio::net::TcpListener::bind(&addr).await {
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
                    let _ = axum::serve(listener, router)
                        .with_graceful_shutdown(async {
                            let _ = shutdown_rx.await;
                        })
                        .await;
                });
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
            registry,
            events,
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
        let model: Arc<dyn turbospark_server::ChatModel> = Arc::new(
            crate::server_model::FfiChatModel::new(
                core,
                model_id,
                self.guardrails,
                self.default_system.clone(),
                self.default_reasoning,
            ),
        );
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
            auth_enabled: self.auth_enabled,
            uptime_seconds: self.started.elapsed().as_secs(),
        }
    }

    /// Signals graceful shutdown and blocks until the background thread has
    /// actually stopped serving. Idempotent: a second call is a no-op.
    fn stop(&mut self) {
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
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
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
