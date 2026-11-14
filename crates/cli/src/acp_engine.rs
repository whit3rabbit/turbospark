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
//! Tools: the model is offered the four ACP file tools through the
//! checkpoint's own chat template. A turn is a bounded loop of rounds: generate,
//! and if the reply carried tool calls, run each through `TurnHost::call_tool`
//! (the adapter's gate: containment, size bounds, default-deny writes), append
//! the results and generate again. The caps are [`MAX_TOOL_ROUNDS`] and
//! [`MAX_TOOL_CALLS`]; hitting either ends the turn as `max_turn_requests`.
//! Tool-call ids are assigned here (`call_N`, unique within the turn), not
//! taken from the model.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{json, Value};
use turbospark_acp::{Engine, Message, Role, StopReason, ToolCall, TurnHost, TurnRequest};

/// Most generate-then-run-tools rounds in one turn.
pub const MAX_TOOL_ROUNDS: usize = 8;
/// Most tool calls in one turn. Same figure as the adapter's own
/// `AcpConfig::max_tool_calls_per_turn` default; the adapter's cap still
/// applies if it is lower.
pub const MAX_TOOL_CALLS: usize = 64;
/// A tool result is clipped to this many characters before the model sees it
/// (the editor still gets the full output). Bounds prompt growth per call.
pub const MAX_RESULT_CHARS_TO_MODEL: usize = 16 * 1024;

/// How a backend turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    EndTurn,
    MaxTokens,
    Cancelled,
}

/// A tool call the model asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolReq {
    pub id: String,
    pub name: String,
    pub args: Value,
}

/// One message of the working conversation, tool rounds included.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatMsg {
    User(String),
    Assistant {
        text: String,
        calls: Vec<ToolReq>,
    },
    Tool {
        call_id: String,
        name: String,
        text: String,
    },
}

/// The tool schemas offered to the model: name, description, JSON schema of
/// the arguments. Argument names match `turbospark_acp::tools::plan`.
pub fn tool_schemas() -> Vec<(&'static str, &'static str, Value)> {
    vec![
        (
            "read_file",
            "Read a text file. The path is relative to the workspace root.",
            json!({"type":"object","properties":{
                "path":{"type":"string","description":"File path inside the workspace"},
                "line":{"type":"integer","description":"First line to read (1-based), optional"},
                "limit":{"type":"integer","description":"Most lines to read, optional"}
            },"required":["path"]}),
        ),
        (
            "list_dir",
            "List a directory. The path is relative to the workspace root.",
            json!({"type":"object","properties":{
                "path":{"type":"string","description":"Directory path inside the workspace"}
            },"required":["path"]}),
        ),
        (
            "write_file",
            "Create or overwrite a text file. The user is asked to approve every write and may refuse.",
            json!({"type":"object","properties":{
                "path":{"type":"string","description":"File path inside the workspace"},
                "content":{"type":"string","description":"The whole new file content"}
            },"required":["path","content"]}),
        ),
        (
            "edit_file",
            "Replace one exact piece of text in a file. The user is asked to approve every edit and may refuse.",
            json!({"type":"object","properties":{
                "path":{"type":"string","description":"File path inside the workspace"},
                "old_text":{"type":"string","description":"Exact text to replace (non-empty)"},
                "new_text":{"type":"string","description":"Replacement text"}
            },"required":["path","old_text","new_text"]}),
        ),
    ]
}

/// One loaded model. Built and used on the worker thread only.
pub trait ChatBackend {
    /// Generates the assistant reply to `history` (the new user prompt last),
    /// calling `delta` with each piece of answer text as it is produced and
    /// `call` with each tool call the model made (only when `tools` is true).
    /// Must poll `cancel` between tokens. `delta` never receives reasoning
    /// text.
    fn generate(
        &mut self,
        history: &[ChatMsg],
        tools: bool,
        cancel: &dyn Fn() -> bool,
        delta: &mut dyn FnMut(&str),
        call: &mut dyn FnMut(ToolReq),
    ) -> Result<Finish, String>;
}

/// Builds the backend on the worker thread. Called again on the next turn if
/// it failed, so a model that was busy can be retried once it is free.
pub type BackendFactory = Box<dyn FnMut() -> Result<Box<dyn ChatBackend>, String> + Send>;

struct Job {
    history: Vec<ChatMsg>,
    cancel: Arc<AtomicBool>,
    events: Sender<Event>,
}

enum Event {
    Delta(String),
    Call(ToolReq),
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
        let events2 = job.events.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            b.generate(
                &job.history,
                true,
                &|| cancel.load(Ordering::Relaxed),
                &mut |text| {
                    let _ = events.send(Event::Delta(text.to_string()));
                },
                &mut |call| {
                    let _ = events2.send(Event::Call(call));
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

fn to_chat(history: &[Message]) -> Vec<ChatMsg> {
    history
        .iter()
        .filter_map(|m| match m.role {
            Role::User => Some(ChatMsg::User(m.text.clone())),
            Role::Assistant => Some(ChatMsg::Assistant {
                text: m.text.clone(),
                calls: Vec::new(),
            }),
            // The adapter's stored history is text only: a past turn's tool
            // rounds are not replayed, so a Tool entry has no call to pair
            // with and would break the template's call/result order.
            Role::Tool => None,
        })
        .collect()
}

fn clip_for_model(text: &str) -> String {
    if text.chars().count() <= MAX_RESULT_CHARS_TO_MODEL {
        return text.to_string();
    }
    let mut s: String = text.chars().take(MAX_RESULT_CHARS_TO_MODEL).collect();
    s.push_str("\n[output clipped for the model]");
    s
}

impl LocalEngine {
    /// Runs one generation on the worker and streams its text to the host.
    /// Returns how it finished, the text it produced and the calls it made.
    fn round(
        &self,
        history: &[ChatMsg],
        host: &mut dyn TurnHost,
    ) -> Result<(Finish, String, Vec<ToolReq>), String> {
        let (events_tx, events_rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let job = Job {
            history: history.to_vec(),
            cancel: cancel.clone(),
            events: events_tx,
        };
        let jobs = self.jobs.as_ref().ok_or("the engine is shut down")?;
        jobs.send(job)
            .map_err(|_| "the model worker is not running".to_string())?;

        let mut text = String::new();
        let mut calls = Vec::new();
        loop {
            match events_rx.recv_timeout(Duration::from_millis(25)) {
                Ok(Event::Delta(piece)) => {
                    // After a cancel nothing more goes to the editor.
                    if !host.cancelled() {
                        host.text_delta(&piece);
                        text.push_str(&piece);
                    }
                }
                Ok(Event::Call(c)) => calls.push(c),
                Ok(Event::Done(result)) => return result.map(|f| (f, text, calls)),
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

impl Engine for LocalEngine {
    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn run_turn(&self, req: &TurnRequest, host: &mut dyn TurnHost) -> Result<StopReason, String> {
        let mut msgs = to_chat(&req.history);
        let mut total_calls = 0usize;
        for _round in 0..MAX_TOOL_ROUNDS {
            let (finish, text, mut calls) = self.round(&msgs, host)?;
            // The adapter reports `cancelled` itself.
            if host.cancelled() || finish == Finish::Cancelled {
                return Ok(StopReason::EndTurn);
            }
            if calls.is_empty() {
                return Ok(match finish {
                    Finish::MaxTokens => StopReason::MaxTokens,
                    _ => StopReason::EndTurn,
                });
            }
            // Hard cap on calls per turn: run what fits, then end the turn.
            let room = MAX_TOOL_CALLS.saturating_sub(total_calls);
            let capped = calls.len() > room;
            calls.truncate(room);
            for c in calls.iter_mut() {
                total_calls += 1;
                c.id = format!("call_{total_calls}");
            }
            msgs.push(ChatMsg::Assistant {
                text,
                calls: calls.clone(),
            });
            for c in &calls {
                if host.cancelled() {
                    return Ok(StopReason::EndTurn);
                }
                let out = host.call_tool(ToolCall {
                    id: c.id.clone(),
                    name: c.name.clone(),
                    args: c.args.clone(),
                });
                msgs.push(ChatMsg::Tool {
                    call_id: c.id.clone(),
                    name: c.name.clone(),
                    text: clip_for_model(&out.text),
                });
            }
            if capped || total_calls >= MAX_TOOL_CALLS {
                return Ok(StopReason::MaxTurnRequests);
            }
        }
        // Still asking for tools after the last allowed round.
        Ok(StopReason::MaxTurnRequests)
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
    use turbospark_acp::ToolOutput;

    struct Host {
        deltas: Vec<String>,
        cancel_after: Option<usize>,
        calls: Vec<ToolCall>,
        reply: String,
        cancel_after_calls: Option<usize>,
    }
    impl Host {
        fn new() -> Host {
            Host {
                deltas: vec![],
                cancel_after: None,
                calls: vec![],
                reply: "ok".to_string(),
                cancel_after_calls: None,
            }
        }
    }
    impl TurnHost for Host {
        fn text_delta(&mut self, text: &str) {
            self.deltas.push(text.to_string());
        }
        fn call_tool(&mut self, c: ToolCall) -> ToolOutput {
            self.calls.push(c);
            ToolOutput {
                text: self.reply.clone(),
                is_error: false,
            }
        }
        fn cancelled(&self) -> bool {
            self.cancel_after.is_some_and(|n| self.deltas.len() >= n)
                || self
                    .cancel_after_calls
                    .is_some_and(|n| self.calls.len() >= n)
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

    fn last_user(h: &[ChatMsg]) -> String {
        h.iter()
            .rev()
            .find_map(|m| match m {
                ChatMsg::User(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Emits one word per "token", polling cancel between tokens like the
    /// runtime does, and records what it was given.
    struct Fake {
        seen: Arc<Mutex<Vec<String>>>,
        dropped: Arc<AtomicBool>,
        words: usize,
    }
    impl ChatBackend for Fake {
        fn generate(
            &mut self,
            history: &[ChatMsg],
            _tools: bool,
            cancel: &dyn Fn() -> bool,
            delta: &mut dyn FnMut(&str),
            _call: &mut dyn FnMut(ToolReq),
        ) -> Result<Finish, String> {
            self.seen.lock().unwrap().push(last_user(history));
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
        let mut h = Host::new();
        assert_eq!(e.run_turn(&req("hi"), &mut h), Ok(StopReason::EndTurn));
        assert_eq!(h.deltas, vec!["w0 ", "w1 ", "w2 ", "w3 "]);
        let mut h2 = Host::new();
        e.run_turn(&req("again"), &mut h2).unwrap();
        assert_eq!(*loads.lock().unwrap(), 1, "backend is built once");
        assert_eq!(*seen.lock().unwrap(), vec!["hi", "again"]);
    }

    #[test]
    fn cancel_stops_generation_between_tokens() {
        let (e, _s, _d, _l) = engine(10_000);
        let mut h = Host::new();
        h.cancel_after = Some(2);
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
        let mut h = Host::new();
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
        let mut h = Host::new();
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
                _h: &[ChatMsg],
                _t: bool,
                _c: &dyn Fn() -> bool,
                _d: &mut dyn FnMut(&str),
                _k: &mut dyn FnMut(ToolReq),
            ) -> Result<Finish, String> {
                panic!("boom")
            }
        }
        let factory: BackendFactory = Box::new(|| Ok(Box::new(Boom) as Box<dyn ChatBackend>));
        let e = LocalEngine::new("fake".into(), factory);
        let mut h = Host::new();
        let err = e.run_turn(&req("x"), &mut h).unwrap_err();
        assert!(err.contains("panicked"), "{err}");
        // The worker survived: the next turn gets an answer (another panic,
        // since the factory returns Boom again), not "worker exited".
        let err2 = e.run_turn(&req("x"), &mut h).unwrap_err();
        assert!(err2.contains("panicked"), "{err2}");
    }

    // ---- tool rounds ----

    type Script = dyn FnMut(&[ChatMsg], &mut dyn FnMut(&str), &mut dyn FnMut(ToolReq)) + Send;

    /// A backend driven by a closure, recording each round's history.
    struct Scripted {
        script: Box<Script>,
        rounds: Arc<Mutex<Vec<Vec<ChatMsg>>>>,
    }
    impl ChatBackend for Scripted {
        fn generate(
            &mut self,
            history: &[ChatMsg],
            tools: bool,
            _cancel: &dyn Fn() -> bool,
            delta: &mut dyn FnMut(&str),
            call: &mut dyn FnMut(ToolReq),
        ) -> Result<Finish, String> {
            assert!(tools, "the engine always offers the tools");
            self.rounds.lock().unwrap().push(history.to_vec());
            (self.script)(history, delta, call);
            Ok(Finish::EndTurn)
        }
    }

    fn scripted(
        script: impl FnMut(&[ChatMsg], &mut dyn FnMut(&str), &mut dyn FnMut(ToolReq)) + Send + 'static,
    ) -> (LocalEngine, Arc<Mutex<Vec<Vec<ChatMsg>>>>) {
        let rounds = Arc::new(Mutex::new(Vec::new()));
        let r = rounds.clone();
        let mut script = Some(Box::new(script) as Box<Script>);
        let factory: BackendFactory = Box::new(move || {
            Ok(Box::new(Scripted {
                script: script.take().expect("loaded once"),
                rounds: r.clone(),
            }) as Box<dyn ChatBackend>)
        });
        (LocalEngine::new("fake".into(), factory), rounds)
    }

    fn call(name: &str, args: Value) -> ToolReq {
        ToolReq {
            // The model's own id is ignored: the engine assigns call_N.
            id: "model-chosen".into(),
            name: name.into(),
            args,
        }
    }

    #[test]
    fn a_tool_call_runs_through_the_host_and_its_result_reaches_the_model() {
        let (e, rounds) = scripted(|h, delta, c| {
            if h.iter().any(|m| matches!(m, ChatMsg::Tool { .. })) {
                delta("it says hello");
            } else {
                delta("let me look. ");
                c(call("read_file", json!({"path":"a.txt"})));
            }
        });
        let mut h = Host::new();
        h.reply = "hello".into();
        assert_eq!(
            e.run_turn(&req("read a.txt"), &mut h),
            Ok(StopReason::EndTurn)
        );
        assert_eq!(h.calls.len(), 1);
        assert_eq!(h.calls[0].name, "read_file");
        assert_eq!(h.calls[0].id, "call_1", "ids are engine-assigned");
        assert_eq!(h.calls[0].args, json!({"path":"a.txt"}));
        assert_eq!(h.deltas.concat(), "let me look. it says hello");
        let rounds = rounds.lock().unwrap();
        assert_eq!(rounds.len(), 2);
        // Round 2 sees: the user turn, the assistant turn WITH its call, and
        // the tool result paired by id.
        match (&rounds[1][1], &rounds[1][2]) {
            (
                ChatMsg::Assistant { text, calls },
                ChatMsg::Tool {
                    call_id,
                    name,
                    text: result,
                },
            ) => {
                assert_eq!(text, "let me look. ");
                assert_eq!(calls[0].id, "call_1");
                assert_eq!((call_id.as_str(), name.as_str()), ("call_1", "read_file"));
                assert_eq!(result, "hello");
            }
            other => panic!("unexpected round-2 history: {other:?}"),
        }
    }

    #[test]
    fn a_model_that_never_stops_calling_tools_hits_the_round_cap() {
        let (e, rounds) = scripted(|_h, _d, c| c(call("list_dir", json!({"path":"."}))));
        let mut h = Host::new();
        assert_eq!(
            e.run_turn(&req("x"), &mut h),
            Ok(StopReason::MaxTurnRequests)
        );
        assert_eq!(rounds.lock().unwrap().len(), MAX_TOOL_ROUNDS);
        assert_eq!(h.calls.len(), MAX_TOOL_ROUNDS);
    }

    #[test]
    fn calls_in_one_reply_are_capped_at_the_turn_limit() {
        let (e, _r) = scripted(|_h, _d, c| {
            for _ in 0..(MAX_TOOL_CALLS + 20) {
                c(call("list_dir", json!({"path":"."})));
            }
        });
        let mut h = Host::new();
        assert_eq!(
            e.run_turn(&req("x"), &mut h),
            Ok(StopReason::MaxTurnRequests)
        );
        assert_eq!(h.calls.len(), MAX_TOOL_CALLS, "no call past the cap runs");
        let ids: std::collections::HashSet<_> = h.calls.iter().map(|c| c.id.clone()).collect();
        assert_eq!(ids.len(), MAX_TOOL_CALLS, "ids are unique within the turn");
    }

    #[test]
    fn cancel_between_tool_calls_stops_the_turn() {
        let (e, _r) = scripted(|_h, _d, c| {
            for _ in 0..5 {
                c(call("list_dir", json!({"path":"."})));
            }
        });
        let mut h = Host::new();
        h.cancel_after_calls = Some(2);
        assert_eq!(e.run_turn(&req("x"), &mut h), Ok(StopReason::EndTurn));
        assert_eq!(h.calls.len(), 2, "no tool runs after the cancel");
    }

    #[test]
    fn a_huge_tool_result_is_clipped_for_the_model_only() {
        let (e, rounds) = scripted(|h, _d, c| {
            if !h.iter().any(|m| matches!(m, ChatMsg::Tool { .. })) {
                c(call("read_file", json!({"path":"big"})));
            }
        });
        let mut h = Host::new();
        h.reply = "x".repeat(MAX_RESULT_CHARS_TO_MODEL * 3);
        e.run_turn(&req("x"), &mut h).unwrap();
        let rounds = rounds.lock().unwrap();
        let ChatMsg::Tool { text, .. } = &rounds[1][2] else {
            panic!("no tool message")
        };
        assert!(text.chars().count() < MAX_RESULT_CHARS_TO_MODEL + 100);
        assert!(text.ends_with("[output clipped for the model]"));
    }

    #[test]
    fn stored_tool_history_is_not_replayed() {
        let (e, rounds) = scripted(|_h, d, _c| d("ok"));
        let mut r = req("now");
        r.history.insert(
            0,
            Message {
                role: Role::Tool,
                text: "orphan".into(),
            },
        );
        let mut h = Host::new();
        e.run_turn(&r, &mut h).unwrap();
        assert_eq!(rounds.lock().unwrap()[0], vec![ChatMsg::User("now".into())]);
    }

    #[test]
    fn schemas_name_exactly_the_adapter_tools_and_its_argument_names() {
        let names: Vec<_> = tool_schemas().iter().map(|t| t.0).collect();
        assert_eq!(names, ["read_file", "list_dir", "write_file", "edit_file"]);
        let edit = &tool_schemas()[3].2;
        assert!(edit["properties"]["old_text"].is_object());
        assert!(edit["properties"]["new_text"].is_object());
        for (_, _, schema) in tool_schemas() {
            assert!(schema["properties"]["path"].is_object());
        }
    }

    // ---- through the adapter: gate, containment and default-deny ----

    mod through_serve {
        use super::*;
        use std::io::{BufReader, Read, Write};
        use turbospark_acp::{serve, AcpConfig};

        struct ChanRead {
            rx: Receiver<Vec<u8>>,
            buf: Vec<u8>,
            pos: usize,
        }
        impl Read for ChanRead {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                while self.pos >= self.buf.len() {
                    match self.rx.recv() {
                        Ok(b) => {
                            self.buf = b;
                            self.pos = 0;
                        }
                        Err(_) => return Ok(0),
                    }
                }
                let n = (self.buf.len() - self.pos).min(out.len());
                out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
                self.pos += n;
                Ok(n)
            }
        }
        struct ChanWrite(Sender<String>, Vec<u8>);
        impl Write for ChanWrite {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.1.extend_from_slice(b);
                while let Some(i) = self.1.iter().position(|&c| c == b'\n') {
                    let line: Vec<u8> = self.1.drain(..=i).collect();
                    let _ = self
                        .0
                        .send(String::from_utf8_lossy(&line).trim().to_string());
                }
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        const WAIT: Duration = Duration::from_secs(20);

        /// Runs one turn whose model asks for a write and an escaping read,
        /// then echoes what it was told. `answer` is the client's reply to a
        /// permission request (`None` = stay silent). Returns the streamed
        /// text, the number of permission requests, and the workspace.
        fn run(answer: Option<&'static str>) -> (String, usize, std::path::PathBuf) {
            let ws = std::env::temp_dir().join(format!(
                "tsacp-eng-{}-{}",
                std::process::id(),
                answer.unwrap_or("silent")
            ));
            let _ = std::fs::remove_dir_all(&ws);
            std::fs::create_dir_all(&ws).unwrap();
            let ws = std::fs::canonicalize(&ws).unwrap();

            let (engine, _rounds) = scripted(|h, delta, c| {
                let told: Vec<String> = h
                    .iter()
                    .filter_map(|m| match m {
                        ChatMsg::Tool { name, text, .. } => Some(format!("[{name}: {text}]")),
                        _ => None,
                    })
                    .collect();
                if told.is_empty() {
                    c(call(
                        "write_file",
                        json!({"path":"notes.txt","content":"from the model"}),
                    ));
                    c(call("read_file", json!({"path":"../escape.txt"})));
                } else {
                    delta(&told.join(" "));
                }
            });
            let engine: &'static LocalEngine = Box::leak(Box::new(engine));
            let cfg: &'static AcpConfig = Box::leak(Box::new(AcpConfig {
                permission_timeout: Duration::from_millis(300),
                ..AcpConfig::default()
            }));
            let (in_tx, in_rx) = channel::<Vec<u8>>();
            let (out_tx, out_rx) = channel::<String>();
            let server = std::thread::spawn(move || {
                serve(
                    engine,
                    cfg,
                    BufReader::new(ChanRead {
                        rx: in_rx,
                        buf: vec![],
                        pos: 0,
                    }),
                    ChanWrite(out_tx, vec![]),
                )
            });
            let send = |v: Value| in_tx.send(format!("{v}\n").into_bytes()).unwrap();
            let next = || -> Value {
                serde_json::from_str(&out_rx.recv_timeout(WAIT).expect("agent went quiet")).unwrap()
            };
            send(
                json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}),
            );
            assert_eq!(next()["id"], 1);
            send(
                json!({"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":ws,"mcpServers":[]}}),
            );
            let sid = next()["result"]["sessionId"].as_str().unwrap().to_string();
            send(json!({"jsonrpc":"2.0","id":3,"method":"session/prompt",
                "params":{"sessionId":sid,"prompt":[{"type":"text","text":"go"}]}}));
            let (mut text, mut asked) = (String::new(), 0usize);
            loop {
                let v = next();
                if v["method"] == "session/update"
                    && v["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
                {
                    text.push_str(v["params"]["update"]["content"]["text"].as_str().unwrap());
                } else if v["method"] == "session/request_permission" {
                    asked += 1;
                    if let Some(opt) = answer {
                        send(json!({"jsonrpc":"2.0","id":v["id"],
                            "result":{"outcome":{"outcome":"selected","optionId":opt}}}));
                    }
                } else if v["id"] == 3 {
                    assert_eq!(v["result"]["stopReason"], "end_turn", "{v}");
                    break;
                }
            }
            drop(in_tx);
            server.join().unwrap().unwrap();
            (text, asked, ws)
        }

        #[test]
        fn a_silent_client_denies_the_model_write_and_the_model_is_told() {
            let (text, asked, ws) = run(None);
            assert_eq!(asked, 1, "the write asks; the escaping read never does");
            assert!(
                !ws.join("notes.txt").exists(),
                "default-deny: nothing written"
            );
            assert!(text.contains("[write_file: permission denied"), "{text}");
            assert!(text.contains("outside the workspace root"), "{text}");
            let _ = std::fs::remove_dir_all(ws);
        }

        #[test]
        fn only_an_explicit_allow_lets_the_model_write() {
            let (text, asked, ws) = run(Some("allow_once"));
            assert_eq!(asked, 1);
            assert_eq!(
                std::fs::read_to_string(ws.join("notes.txt")).unwrap(),
                "from the model"
            );
            assert!(text.contains("outside the workspace root"), "{text}");
            let _ = std::fs::remove_dir_all(ws);
        }

        #[test]
        fn a_reject_leaves_the_disk_untouched() {
            let (_text, asked, ws) = run(Some("reject_once"));
            assert_eq!(asked, 1);
            assert!(!ws.join("notes.txt").exists());
            let _ = std::fs::remove_dir_all(ws);
        }
    }
}
