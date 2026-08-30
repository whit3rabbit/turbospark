//! The in-process HTTP server: a background OS thread hosting its own tokio
//! runtime, serving `turbospark_server::build_router_with_options` over an
//! `FfiChatModel` that shares the caller's already-open session
//! (`server_model.rs`, `session.rs`'s module doc).
//!
//! **THE SERVER OUTLIVES THE `TsSession *` IT WAS STARTED FROM, BY DESIGN.**
//! `Server::start` takes an `Arc<SessionCore>` clone rather than a borrow, so
//! `ts_session_close` on the session that started it drops only the caller's
//! own reference -- the engine stays resident until `ts_server_stop` (or
//! `Drop`, e.g. process exit) releases the last one. A caller that wants the
//! model gone must stop the server too; this crate does not track that for
//! them, the same way `ts_session_close` does not know whether a caller
//! still holds some other reference of their own.
//!
//! **`port: 0` MEANS "LET THE OS CHOOSE."** The socket is bound on the
//! BACKGROUND THREAD, inside its own runtime, and the resolved port is sent
//! back to `start`'s caller over a plain blocking `std::sync::mpsc` channel
//! before `start` returns -- so a caller can read the actually bound port
//! back from `ServerInfo` immediately, with no race against the thread that
//! will use it.
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

use crate::session::SessionCore;

/// What `ts_server_start` writes to `*out`, and what `ts_server_stop` frees.
pub struct Server {
    port: u16,
    model_id: String,
    auth_enabled: bool,
    // `Option` so `stop` can be called from both `ts_server_stop` and `Drop`
    // without sending on a closed channel or joining a thread twice.
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Server {
    /// Spawns a background thread that binds `port` (0 for an OS-assigned
    /// one) on loopback and serves it. Blocks until the socket is actually
    /// bound (or binding fails), never until the first request is served.
    pub(crate) fn start(
        core: Arc<SessionCore>,
        model_id: String,
        port: u16,
        api_key: Option<String>,
    ) -> Result<Self, String> {
        let auth_enabled = api_key.is_some();
        let chat_model: Arc<dyn turbospark_server::ChatModel> = Arc::new(
            crate::server_model::FfiChatModel::new(core, model_id.clone()),
        );
        let router = turbospark_server::build_router_with_options(
            chat_model,
            turbospark_server::RouterOptions { api_key },
        );

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        // Plain blocking channel, not `tokio::sync`: `start`'s CALLER may or
        // may not be inside a Tokio runtime of its own, and this is how the
        // bind result gets back to them without ever needing a runtime on
        // their thread (see the module doc).
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<u16, String>>();
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
                    let resolved_port = match listener.local_addr() {
                        Ok(a) => a.port(),
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
                    if ready_tx.send(Ok(resolved_port)).is_err() {
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

        let resolved_port = match ready_rx.recv() {
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
            port: resolved_port,
            model_id,
            auth_enabled,
            shutdown: Some(shutdown_tx),
            thread: Some(thread),
        })
    }

    pub(crate) fn info(&self) -> crate::wire::ServerInfo {
        crate::wire::ServerInfo {
            port: self.port,
            model_id: self.model_id.clone(),
            auth_enabled: self.auth_enabled,
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

/// Turns a `TsServer *` back into a borrow.
///
/// # Safety
/// `ptr` must be null, or a pointer returned by `ts_server_start` and not
/// yet stopped.
pub(crate) unsafe fn borrow<'a>(ptr: *const Server) -> Result<&'a Server, String> {
    ptr.as_ref()
        .ok_or_else(|| "server must not be null".to_string())
}
