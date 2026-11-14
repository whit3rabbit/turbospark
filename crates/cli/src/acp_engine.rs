//! The portable half of the `turbospark acp` engine: a worker thread that owns
//! one chat backend and serves ACP turns to it.
//!
//! Why a thread: the model session (Metal runner, KV cache) is built on one
//! thread, used by one thread, and dropped by that thread, so nothing here
//! needs `Send` on a GPU type and no `unsafe impl Send` exists. The ACP server
//! runs turns on its own worker threads and calls [`Engine::run_turn`] from
//! any of them; they only exchange messages with this thread.
//!
//! Why lazy: the backend is built on the first turn, not at startup. The
//! editor's `initialize` and `session/new` must answer in milliseconds, and a
//! multi-gigabyte model open (or a refusal because another process holds the
//! model, see `acp_lock.rs`) is reported as that first turn's error instead of
//! a hung handshake.
//!
//! Cancellation: the adapter's `TurnHost::cancelled` is polled here every
//! 25 ms and on every delta, and mapped onto a flag the backend hands to the
//! runtime's cancellable decode loop, which checks it between tokens (and
//! between prefill chunks). Cancel latency is therefore one token or one
//! prefill chunk, not one turn.
//!
//! Release: dropping the engine closes the job channel, the worker exits its
//! loop and drops the backend (session, KV, model lock) on its own thread, and
//! the drop joins it. `main` drops the engine before `process::exit`.
//!
//! Tools are deliberately not wired: this backend is text-only. See
//! `docs/ACP.md`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use turbospark_acp::{Engine, Message, StopReason, TurnHost, TurnRequest};

/// How a backend turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    EndTurn,
    MaxTokens,
    Cancelled,
}

/// One loaded model. Built and used on the worker thread only.
pub trait ChatBackend {
    /// Generates the assistant reply to `history` (the new user prompt last),
    /// calling `delta` with each piece of answer text as it is produced. Must
    /// poll `cancel` between tokens. `delta` never receives reasoning text.
    fn generate(
        &mut self,
        history: &[Message],
        cancel: &dyn Fn() -> bool,
        delta: &mut dyn FnMut(&str),
    ) -> Result<Finish, String>;
}

/// Builds the backend on the worker thread. Called again on the next turn if
/// it failed, so a model that was busy can be retried once it is free.
pub type BackendFactory = Box<dyn FnMut() -> Result<Box<dyn ChatBackend>, String> + Send>;

struct Job {
    history: Vec<Message>,
    cancel: Arc<AtomicBool>,
    events: Sender<Event>,
}

enum Event {
    Delta(String),
    Done(Result<Finish, String>),
}

pub struct LocalEngine {
    model_id: String,
    jobs: Option<Sender<Job>>,
    worker: Option<JoinHandle<()>>,
}

impl LocalEngine {
    pub fn new(model_id: String, factory: BackendFactory) -> Self {
        let (tx, rx) = channel::<Job>();
        let worker = std::thread::Builder::new()
            .name("acp-model".to_string())
            .spawn(move || worker_loop(factory, rx))
            .expect("spawning the model worker thread");
        LocalEngine {
            model_id,
            jobs: Some(tx),
            worker: Some(worker),
        }
    }
}

fn worker_loop(mut factory: BackendFactory, jobs: Receiver<Job>) {
    let mut backend: Option<Box<dyn ChatBackend>> = None;
    while let Ok(job) = jobs.recv() {
        if backend.is_none() {
            match catch_unwind(AssertUnwindSafe(&mut factory)) {
                Ok(Ok(b)) => backend = Some(b),
                Ok(Err(e)) => {
                    let _ = job.events.send(Event::Done(Err(e)));
                    continue;
                }
                Err(_) => {
                    let _ = job
                        .events
                        .send(Event::Done(Err("model load panicked".to_string())));
                    continue;
                }
            }
        }
        let Some(b) = backend.as_mut() else { continue };
        let cancel = job.cancel.clone();
        let events = job.events.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            b.generate(
                &job.history,
                &|| cancel.load(Ordering::Relaxed),
                &mut |text| {
                    let _ = events.send(Event::Delta(text.to_string()));
                },
            )
        }));
        let result = match outcome {
            Ok(r) => r,
            Err(_) => {
                // The runner's state after a panic is unknown; drop it (and
                // its model lock) and reload on the next turn.
                backend = None;
                Err("generation panicked; the model will be reloaded".to_string())
            }
        };
        let _ = job.events.send(Event::Done(result));
    }
    // Channel closed: the engine was dropped. `backend` drops here, on this
    // thread, which releases the session and the model lock.
}

impl Engine for LocalEngine {
    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn run_turn(&self, req: &TurnRequest, host: &mut dyn TurnHost) -> Result<StopReason, String> {
        let (events_tx, events_rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let job = Job {
            history: req.history.clone(),
            cancel: cancel.clone(),
            events: events_tx,
        };
        let jobs = self.jobs.as_ref().ok_or("the engine is shut down")?;
        jobs.send(job)
            .map_err(|_| "the model worker is not running".to_string())?;

        loop {
            match events_rx.recv_timeout(Duration::from_millis(25)) {
                Ok(Event::Delta(text)) => {
                    // After a cancel nothing more goes to the editor.
                    if !host.cancelled() {
                        host.text_delta(&text);
                    }
                }
                Ok(Event::Done(result)) => {
                    return result.map(|f| match f {
                        Finish::MaxTokens => StopReason::MaxTokens,
                        // The adapter reports `cancelled` itself.
                        Finish::EndTurn | Finish::Cancelled => StopReason::EndTurn,
                    });
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("the model worker exited mid-turn".to_string());
                }
            }
            if host.cancelled() {
                cancel.store(true, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for LocalEngine {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use turbospark_acp::{Role, ToolCall, ToolOutput};

    struct Host {
        deltas: Vec<String>,
        cancel_after: Option<usize>,
    }
    impl TurnHost for Host {
        fn text_delta(&mut self, text: &str) {
            self.deltas.push(text.to_string());
        }
        fn call_tool(&mut self, _c: ToolCall) -> ToolOutput {
            panic!("the text-only engine never calls tools")
        }
        fn cancelled(&self) -> bool {
            self.cancel_after.is_some_and(|n| self.deltas.len() >= n)
        }
    }

    fn req(prompt: &str) -> TurnRequest {
        TurnRequest {
            session_id: "s".into(),
            cwd: std::path::PathBuf::from("/"),
            history: vec![Message {
                role: Role::User,
                text: prompt.into(),
            }],
        }
    }

    /// Emits one word per "token", polling cancel between tokens like the
    /// runtime does, and records what history it was given.
    struct Fake {
        seen: Arc<Mutex<Vec<String>>>,
        dropped: Arc<AtomicBool>,
        words: usize,
    }
    impl ChatBackend for Fake {
        fn generate(
            &mut self,
            history: &[Message],
            cancel: &dyn Fn() -> bool,
            delta: &mut dyn FnMut(&str),
        ) -> Result<Finish, String> {
            self.seen
                .lock()
                .unwrap()
                .push(history.last().unwrap().text.clone());
            for i in 0..self.words {
                if cancel() {
                    return Ok(Finish::Cancelled);
                }
                delta(&format!("w{i} "));
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(Finish::EndTurn)
        }
    }
    impl Drop for Fake {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[allow(clippy::type_complexity)]
    fn engine(
        words: usize,
    ) -> (
        LocalEngine,
        Arc<Mutex<Vec<String>>>,
        Arc<AtomicBool>,
        Arc<Mutex<u32>>,
    ) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let dropped = Arc::new(AtomicBool::new(false));
        let loads = Arc::new(Mutex::new(0u32));
        let (s, d, l) = (seen.clone(), dropped.clone(), loads.clone());
        let factory: BackendFactory = Box::new(move || {
            *l.lock().unwrap() += 1;
            Ok(Box::new(Fake {
                seen: s.clone(),
                dropped: d.clone(),
                words,
            }) as Box<dyn ChatBackend>)
        });
        (
            LocalEngine::new("fake".into(), factory),
            seen,
            dropped,
            loads,
        )
    }

    #[test]
    fn streams_deltas_in_order_and_loads_once() {
        let (e, seen, _d, loads) = engine(4);
        let mut h = Host {
            deltas: vec![],
            cancel_after: None,
        };
        assert_eq!(e.run_turn(&req("hi"), &mut h), Ok(StopReason::EndTurn));
        assert_eq!(h.deltas, vec!["w0 ", "w1 ", "w2 ", "w3 "]);
        let mut h2 = Host {
            deltas: vec![],
            cancel_after: None,
        };
        e.run_turn(&req("again"), &mut h2).unwrap();
        assert_eq!(*loads.lock().unwrap(), 1, "backend is built once");
        assert_eq!(*seen.lock().unwrap(), vec!["hi", "again"]);
    }

    #[test]
    fn cancel_stops_generation_between_tokens() {
        let (e, _s, _d, _l) = engine(10_000);
        let mut h = Host {
            deltas: vec![],
            cancel_after: Some(2),
        };
        let t = std::time::Instant::now();
        assert_eq!(e.run_turn(&req("x"), &mut h), Ok(StopReason::EndTurn));
        // 10k tokens at 5 ms would be 50 s; a cancel honored between tokens
        // returns in well under a second.
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
        assert!(h.deltas.len() < 100, "{}", h.deltas.len());
    }

    #[test]
    fn dropping_the_engine_releases_the_backend() {
        let (e, _s, dropped, _l) = engine(1);
        let mut h = Host {
            deltas: vec![],
            cancel_after: None,
        };
        e.run_turn(&req("x"), &mut h).unwrap();
        assert!(!dropped.load(Ordering::SeqCst));
        drop(e);
        assert!(
            dropped.load(Ordering::SeqCst),
            "drop joined the worker, which dropped the backend"
        );
    }

    #[test]
    fn load_failure_is_the_turn_error_and_is_retried() {
        let calls = Arc::new(Mutex::new(0u32));
        let c = calls.clone();
        let factory: BackendFactory = Box::new(move || {
            let mut n = c.lock().unwrap();
            *n += 1;
            if *n == 1 {
                Err("model is in use by another process".to_string())
            } else {
                Ok(Box::new(Fake {
                    seen: Arc::new(Mutex::new(vec![])),
                    dropped: Arc::new(AtomicBool::new(false)),
                    words: 1,
                }) as Box<dyn ChatBackend>)
            }
        });
        let e = LocalEngine::new("fake".into(), factory);
        let mut h = Host {
            deltas: vec![],
            cancel_after: None,
        };
        let err = e.run_turn(&req("x"), &mut h).unwrap_err();
        assert!(err.contains("in use"), "{err}");
        assert!(h.deltas.is_empty());
        assert_eq!(e.run_turn(&req("x"), &mut h), Ok(StopReason::EndTurn));
    }

    #[test]
    fn a_backend_panic_is_an_error_not_a_dead_worker() {
        struct Boom;
        impl ChatBackend for Boom {
            fn generate(
                &mut self,
                _h: &[Message],
                _c: &dyn Fn() -> bool,
                _d: &mut dyn FnMut(&str),
            ) -> Result<Finish, String> {
                panic!("boom")
            }
        }
        let factory: BackendFactory = Box::new(|| Ok(Box::new(Boom) as Box<dyn ChatBackend>));
        let e = LocalEngine::new("fake".into(), factory);
        let mut h = Host {
            deltas: vec![],
            cancel_after: None,
        };
        let err = e.run_turn(&req("x"), &mut h).unwrap_err();
        assert!(err.contains("panicked"), "{err}");
        // The worker survived: the next turn gets an answer (another panic,
        // since the factory returns Boom again), not "worker exited".
        let err2 = e.run_turn(&req("x"), &mut h).unwrap_err();
        assert!(err2.contains("panicked"), "{err2}");
    }
}
