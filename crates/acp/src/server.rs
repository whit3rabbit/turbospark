//! The stdio JSON-RPC loop: dispatch, session registry, update fan-out,
//! permission forwarding, cancellation.
//!
//! Threading: the calling thread reads frames and dispatches. Each
//! `session/prompt` runs on its own scoped worker so the reader stays free to
//! deliver `session/cancel` and the client's replies to our permission and fs
//! requests. All writes go through one mutex so frames never interleave.
//!
//! stdout carries protocol only. Diagnostics go to stderr and never include
//! message bodies, file contents, or prompts.

use crate::engine::{Engine, Message, Role, ToolCall, ToolOutput, TurnHost, TurnRequest};
use crate::framing::{read_frame, Frame};
use crate::protocol::{self as p, Inbound, RpcError};
use crate::sessions::{self, Session, Store};
use crate::tools::{self, Action, DiskBackend, FileBackend, Limits, Plan, Risk};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct AcpConfig {
    /// Largest accepted inbound frame (one JSON line), in bytes.
    pub max_frame_bytes: usize,
    /// How long to wait for the editor to answer a permission request before
    /// treating silence as a refusal.
    pub permission_timeout: Duration,
    /// How long to wait for an editor `fs/*` reply.
    pub fs_timeout: Duration,
    /// When non-empty, a session's cwd must lie inside one of these.
    pub allowed_roots: Vec<PathBuf>,
    /// Where session records go. `None` keeps sessions in memory only.
    pub state_dir: Option<PathBuf>,
    pub max_sessions: usize,
    pub max_tool_calls_per_turn: usize,
    pub limits: Limits,
    pub agent_version: String,
}

impl Default for AcpConfig {
    fn default() -> Self {
        AcpConfig {
            max_frame_bytes: 1024 * 1024,
            permission_timeout: Duration::from_secs(60),
            fs_timeout: Duration::from_secs(30),
            allowed_roots: Vec::new(),
            state_dir: None,
            max_sessions: 32,
            max_tool_calls_per_turn: 64,
            limits: Limits::default(),
            agent_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

fn log(msg: &str) {
    eprintln!("turbospark-acp: {msg}");
}

enum ReqErr {
    Timeout,
    Cancelled,
    Closed,
    Rpc(RpcError),
}

type Reply = Result<Value, RpcError>;

struct Shared<'a> {
    cfg: &'a AcpConfig,
    engine: &'a dyn Engine,
    store: Store,
    out: Mutex<Box<dyn Write + Send + 'a>>,
    pending: Mutex<HashMap<String, Sender<Reply>>>,
    next_id: AtomicU64,
    sessions: Mutex<HashMap<String, Arc<Session>>>,
    fs_caps: Mutex<p::FsCapabilities>,
    initialized: AtomicBool,
    closed: AtomicBool,
}

/// Runs the adapter until `input` reaches EOF. Returns after every in-flight
/// turn has stopped. EOF cancels running generations, so an editor that exits
/// does not leave an orphan process holding GPU memory.
pub fn serve<'a, R, W>(
    engine: &'a dyn Engine,
    cfg: &'a AcpConfig,
    mut input: R,
    output: W,
) -> io::Result<()>
where
    R: BufRead,
    W: Write + Send + 'a,
{
    let shared = Shared {
        cfg,
        engine,
        store: Store::new(cfg.state_dir.clone()),
        out: Mutex::new(Box::new(output)),
        pending: Mutex::new(HashMap::new()),
        next_id: AtomicU64::new(1),
        sessions: Mutex::new(HashMap::new()),
        fs_caps: Mutex::new(p::FsCapabilities::default()),
        initialized: AtomicBool::new(false),
        closed: AtomicBool::new(false),
    };
    let mut read_err: Option<io::Error> = None;
    std::thread::scope(|scope| {
        loop {
            if shared.closed.load(Ordering::SeqCst) {
                break;
            }
            match read_frame(&mut input, cfg.max_frame_bytes) {
                Ok(Frame::Eof) => break,
                Ok(Frame::Oversized) => {
                    log("dropped an oversized frame");
                    shared.send(&p::response_err(
                        &Value::Null,
                        &RpcError::new(
                            p::INVALID_REQUEST,
                            format!("frame exceeds {} bytes", cfg.max_frame_bytes),
                        ),
                    ));
                }
                Ok(Frame::Line(bytes)) => match p::parse_inbound(&bytes) {
                    Ok(msg) => shared.handle(scope, msg),
                    Err((id, e)) => {
                        shared.send(&p::response_err(&id, &e));
                    }
                },
                Err(e) => {
                    read_err = Some(e);
                    break;
                }
            }
        }
        // Input is gone: stop generation and release every waiter, then the
        // scope joins the workers.
        shared.begin_shutdown();
    });
    match read_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

impl<'a> Shared<'a> {
    fn send(&self, v: &Value) -> bool {
        let line = match serde_json::to_string(v) {
            Ok(l) => l,
            Err(_) => return false,
        };
        let mut out = match self.out.lock() {
            Ok(o) => o,
            Err(_) => return false,
        };
        let ok = out
            .write_all(line.as_bytes())
            .and_then(|_| out.write_all(b"\n"))
            .and_then(|_| out.flush())
            .is_ok();
        if !ok {
            self.closed.store(true, Ordering::SeqCst);
        }
        ok
    }

    fn begin_shutdown(&self) {
        self.closed.store(true, Ordering::SeqCst);
        if let Ok(map) = self.sessions.lock() {
            for s in map.values() {
                s.cancel.store(true, Ordering::SeqCst);
            }
        }
    }

    fn handle<'scope, 'env>(
        &'scope self,
        scope: &'scope std::thread::Scope<'scope, 'env>,
        msg: Inbound,
    ) where
        'a: 'scope,
    {
        match msg {
            Inbound::Response { id, result } => {
                let tx = self
                    .pending
                    .lock()
                    .ok()
                    .and_then(|mut m| m.remove(&id.to_string()));
                match tx {
                    Some(tx) => {
                        let _ = tx.send(result);
                    }
                    None => log("reply to an unknown or expired request ignored"),
                }
            }
            Inbound::Notification { method, params } => {
                if method == "session/cancel" {
                    match serde_json::from_value::<p::CancelParams>(params) {
                        Ok(c) => {
                            if let Some(s) = self.session(&c.session_id) {
                                s.cancel.store(true, Ordering::SeqCst);
                            }
                        }
                        Err(_) => log("malformed session/cancel ignored"),
                    }
                }
                // Other notifications (including `$/` extensions) are ignored.
            }
            Inbound::Request { id, method, params } => match method.as_str() {
                "initialize" => {
                    let r = self.initialize(params);
                    self.reply(&id, r);
                }
                "session/new" | "session/prompt" if !self.initialized.load(Ordering::SeqCst) => {
                    self.reply(
                        &id,
                        Err(RpcError::new(p::NOT_INITIALIZED, "initialize first")),
                    );
                }
                "session/new" => {
                    let r = self.new_session(params);
                    self.reply(&id, r);
                }
                "session/prompt" => match self.start_prompt(params) {
                    Ok((sess, text)) => {
                        scope.spawn(move || self.run_prompt(id, sess, text));
                    }
                    Err(e) => self.reply(&id, Err(e)),
                },
                _ => self.reply(
                    &id,
                    Err(RpcError::new(p::METHOD_NOT_FOUND, "method not found")),
                ),
            },
        }
    }

    fn reply(&self, id: &Value, r: Result<Value, RpcError>) {
        let v = match r {
            Ok(v) => p::response_ok(id, v),
            Err(e) => p::response_err(id, &e),
        };
        self.send(&v);
    }

    fn session(&self, id: &str) -> Option<Arc<Session>> {
        self.sessions.lock().ok().and_then(|m| m.get(id).cloned())
    }

    fn initialize(&self, params: Value) -> Result<Value, RpcError> {
        let ip: p::InitializeParams = serde_json::from_value(params)
            .map_err(|_| RpcError::new(p::INVALID_PARAMS, "invalid initialize params"))?;
        if let Ok(mut c) = self.fs_caps.lock() {
            *c = ip.client_capabilities.fs;
        }
        self.initialized.store(true, Ordering::SeqCst);
        // The agent answers with the version it supports (1), whatever the
        // client asked for; the client decides whether it can proceed.
        Ok(p::initialize_result(&self.cfg.agent_version))
    }

    fn new_session(&self, params: Value) -> Result<Value, RpcError> {
        let np: p::NewSessionParams = serde_json::from_value(params)
            .map_err(|_| RpcError::new(p::INVALID_PARAMS, "session/new needs an absolute cwd"))?;
        let root = crate::paths::validate_root(&np.cwd, &self.cfg.allowed_roots)
            .map_err(|e| RpcError::new(p::INVALID_PARAMS, e.to_string()))?;
        let mut map = self
            .sessions
            .lock()
            .map_err(|_| RpcError::new(p::INTERNAL_ERROR, "session registry unavailable"))?;
        if map.len() >= self.cfg.max_sessions {
            return Err(RpcError::new(p::INVALID_PARAMS, "too many open sessions"));
        }
        let id = sessions::new_session_id();
        let s = Arc::new(Session::new(id.clone(), root));
        map.insert(id.clone(), s.clone());
        drop(map);
        if let Err(e) = self.store.save(&s, &self.engine.model_id()) {
            log(&format!("session record not written: {}", e.kind()));
        }
        Ok(json!({"sessionId": id}))
    }

    fn start_prompt(&self, params: Value) -> Result<(Arc<Session>, String), RpcError> {
        let pp: p::PromptParams = serde_json::from_value(params)
            .map_err(|_| RpcError::new(p::INVALID_PARAMS, "invalid session/prompt params"))?;
        let sess = self
            .session(&pp.session_id)
            .ok_or_else(|| RpcError::new(p::INVALID_PARAMS, "unknown session"))?;
        let text = p::prompt_to_text(&pp.prompt)?;
        if sess
            .busy
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(RpcError::new(
                p::SESSION_BUSY,
                "a prompt is already running in this session",
            ));
        }
        // Reset here, on the reader thread, not in the worker: a cancel that
        // arrives right behind this prompt is dispatched after this line and
        // must not be erased by a worker that starts late.
        sess.cancel
            .store(self.closed.load(Ordering::SeqCst), Ordering::SeqCst);
        Ok((sess, text))
    }

    fn run_prompt(&self, id: Value, sess: Arc<Session>, text: String) {
        let history = {
            let mut st = match sess.state.lock() {
                Ok(s) => s,
                Err(_) => {
                    sess.busy.store(false, Ordering::SeqCst);
                    self.reply(
                        &id,
                        Err(RpcError::new(p::INTERNAL_ERROR, "session state poisoned")),
                    );
                    return;
                }
            };
            sessions::push_message(
                &mut st,
                Message {
                    role: Role::User,
                    text,
                },
            );
            st.history.clone()
        };
        let req = TurnRequest {
            session_id: sess.id.clone(),
            cwd: sess.root.clone(),
            history,
        };
        let mut host = Host {
            shared: self,
            session: &sess,
            assistant: String::new(),
            tool_calls: 0,
        };
        let result = catch_unwind(AssertUnwindSafe(|| self.engine.run_turn(&req, &mut host)));
        let assistant = std::mem::take(&mut host.assistant);
        if !assistant.is_empty() {
            if let Ok(mut st) = sess.state.lock() {
                sessions::push_message(
                    &mut st,
                    Message {
                        role: Role::Assistant,
                        text: assistant,
                    },
                );
            }
        }
        if let Err(e) = self.store.save(&sess, &self.engine.model_id()) {
            log(&format!("session record not written: {}", e.kind()));
        }
        let cancelled = sess.cancel.load(Ordering::SeqCst);
        let reply = match result {
            // The editor asked to stop: always a clean `cancelled`, never an
            // error, whatever the engine returned.
            _ if cancelled => Ok(json!({"stopReason": "cancelled"})),
            Ok(Ok(reason)) => Ok(json!({"stopReason": reason.as_acp()})),
            Ok(Err(msg)) => Err(RpcError::new(p::INTERNAL_ERROR, msg)),
            Err(_) => {
                log("engine panicked during a turn");
                Err(RpcError::new(
                    p::INTERNAL_ERROR,
                    "the engine failed unexpectedly",
                ))
            }
        };
        // Free the session BEFORE answering so a client that re-prompts the
        // instant it sees the response is never told the session is busy.
        sess.busy.store(false, Ordering::SeqCst);
        self.reply(&id, reply);
    }

    /// One agent-to-client request, waited on with cancel/shutdown/timeout.
    fn request(
        &self,
        method: &str,
        params: Value,
        sess: &Session,
        timeout: Duration,
    ) -> Result<Value, ReqErr> {
        let n = self.next_id.fetch_add(1, Ordering::SeqCst);
        let key = json!(n).to_string();
        let (tx, rx) = channel();
        if let Ok(mut m) = self.pending.lock() {
            m.insert(key.clone(), tx);
        }
        if !self.send(&p::request(n, method, params)) {
            self.forget(&key);
            return Err(ReqErr::Closed);
        }
        let deadline = Instant::now() + timeout;
        loop {
            match rx.recv_timeout(Duration::from_millis(25)) {
                Ok(r) => return r.map_err(ReqErr::Rpc),
                Err(RecvTimeoutError::Timeout) => {
                    if self.closed.load(Ordering::SeqCst) {
                        self.forget(&key);
                        return Err(ReqErr::Closed);
                    }
                    if sess.cancel.load(Ordering::SeqCst) {
                        self.forget(&key);
                        return Err(ReqErr::Cancelled);
                    }
                    if Instant::now() >= deadline {
                        self.forget(&key);
                        return Err(ReqErr::Timeout);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return Err(ReqErr::Closed),
            }
        }
    }

    fn forget(&self, key: &str) {
        if let Ok(mut m) = self.pending.lock() {
            m.remove(key);
        }
    }

    /// The permission gate. Ok means the editor approved; Err is the
    /// refusal text the model will see. Every path that is not an explicit,
    /// recognized approval is a refusal.
    fn authorize(&self, sess: &Session, tool_call: &Value, perm: &str) -> Result<(), String> {
        let deny = |why: &str| {
            format!("permission denied: '{perm}' was not approved ({why}). Nothing was executed.")
        };
        if let Some(g) = sess
            .state
            .lock()
            .ok()
            .and_then(|s| s.grants.get(perm).copied())
        {
            return if g {
                Ok(())
            } else {
                Err(deny("rejected earlier in this session"))
            };
        }
        let params = p::permission_params(&sess.id, tool_call);
        match self.request(
            "session/request_permission",
            params,
            sess,
            self.cfg.permission_timeout,
        ) {
            Ok(v) => match p::selected_option(&v) {
                Some(p::OPT_ALLOW_ONCE) => Ok(()),
                Some(p::OPT_ALLOW_ALWAYS) => {
                    if let Ok(mut s) = sess.state.lock() {
                        s.grants.insert(perm.to_string(), true);
                    }
                    Ok(())
                }
                Some(p::OPT_REJECT_ALWAYS) => {
                    if let Ok(mut s) = sess.state.lock() {
                        s.grants.insert(perm.to_string(), false);
                    }
                    Err(deny("rejected by the editor"))
                }
                Some(_) => Err(deny("rejected by the editor")),
                None => Err(deny("the editor gave no valid selection")),
            },
            Err(ReqErr::Timeout) => Err(deny(&format!(
                "no answer from the editor within {}s",
                self.cfg.permission_timeout.as_secs()
            ))),
            Err(ReqErr::Cancelled) => Err(deny("the turn was cancelled")),
            Err(ReqErr::Closed) => Err(deny("the editor connection closed")),
            Err(ReqErr::Rpc(e)) => Err(deny(&format!(
                "the editor does not support or failed the permission request, code {}",
                e.code
            ))),
        }
    }

    fn run_tool(&self, sess: &Session, call: ToolCall) -> ToolOutput {
        let fail = |text: String| ToolOutput {
            text,
            is_error: true,
        };
        let plan = match tools::plan(&sess.root, &call.name, &call.args, &self.cfg.limits) {
            Ok(p) => p,
            Err(e) => {
                self.send(&p::tool_call_update(
                    &sess.id,
                    &json!({
                        "toolCallId": call.id, "title": clip(&call.name, 64),
                        "kind": "other", "status": "pending"
                    }),
                ));
                self.send(&p::tool_call_status_update(
                    &sess.id, &call.id, "failed", &e,
                ));
                return fail(e);
            }
        };
        let tc = tool_call_json(&call, &plan);
        self.send(&p::tool_call_update(&sess.id, &tc));

        if plan.risk == Risk::Write {
            if let Err(msg) = self.authorize(sess, &tc, plan.risk.permission()) {
                self.send(&p::tool_call_status_update(
                    &sess.id, &call.id, "failed", &msg,
                ));
                return fail(msg);
            }
        }
        if sess.cancel.load(Ordering::SeqCst) {
            let m = "the turn was cancelled before the tool ran".to_string();
            self.send(&p::tool_call_status_update(
                &sess.id, &call.id, "failed", &m,
            ));
            return fail(m);
        }
        let caps = self.fs_caps.lock().map(|c| c.clone()).unwrap_or_default();
        let backend = EditorFs {
            shared: self,
            session: sess,
            caps,
        };
        let result = match &plan.action {
            // Directory listing has no ACP fs method: always disk, inside root.
            Action::ListDir => tools::execute(&plan, &DiskBackend, &self.cfg.limits),
            _ => tools::execute(&plan, &backend, &self.cfg.limits),
        };
        match result {
            Ok(text) => {
                self.send(&p::tool_call_status_update(
                    &sess.id,
                    &call.id,
                    "completed",
                    &text,
                ));
                ToolOutput {
                    text,
                    is_error: false,
                }
            }
            Err(e) => {
                self.send(&p::tool_call_status_update(
                    &sess.id, &call.id, "failed", &e,
                ));
                fail(e)
            }
        }
    }
}

fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn tool_call_json(call: &ToolCall, plan: &Plan) -> Value {
    let mut v = json!({
        "toolCallId": call.id,
        "title": plan.title,
        "kind": plan.risk.acp_kind(),
        "status": "pending",
        "locations": [{"path": plan.path.to_string_lossy()}],
    });
    match &plan.action {
        Action::WriteFile { content } => {
            v["rawInput"] = json!({"path": plan.path.to_string_lossy(), "bytes": content.len()});
            // Show the proposed file in the editor's diff view when it is small
            // enough to be a reasonable frame.
            if content.len() <= 64 * 1024 {
                v["content"] = json!([{
                    "type": "diff", "path": plan.path.to_string_lossy(),
                    "oldText": Value::Null, "newText": content
                }]);
            }
        }
        Action::EditFile { old, new } => {
            v["rawInput"] = json!({
                "path": plan.path.to_string_lossy(),
                "old_text": clip(old, 4096), "new_text": clip(new, 4096)
            });
        }
        _ => {
            v["rawInput"] = json!({"path": plan.path.to_string_lossy()});
        }
    }
    v
}

struct Host<'h, 'a> {
    shared: &'h Shared<'a>,
    session: &'h Arc<Session>,
    assistant: String,
    tool_calls: usize,
}

impl TurnHost for Host<'_, '_> {
    fn text_delta(&mut self, text: &str) {
        if self.cancelled() || text.is_empty() {
            return;
        }
        self.assistant.push_str(text);
        self.shared
            .send(&p::text_chunk_update(&self.session.id, text));
    }

    fn call_tool(&mut self, call: ToolCall) -> ToolOutput {
        self.tool_calls += 1;
        let out = if self.tool_calls > self.shared.cfg.max_tool_calls_per_turn {
            ToolOutput {
                text: "refused: tool call limit for this turn reached".to_string(),
                is_error: true,
            }
        } else if self.cancelled() {
            ToolOutput {
                text: "the turn was cancelled".to_string(),
                is_error: true,
            }
        } else {
            self.shared.run_tool(self.session, call)
        };
        if let Ok(mut st) = self.session.state.lock() {
            sessions::push_message(
                &mut st,
                Message {
                    role: Role::Tool,
                    text: out.text.clone(),
                },
            );
        }
        out
    }

    fn cancelled(&self) -> bool {
        self.session.cancel.load(Ordering::SeqCst) || self.shared.closed.load(Ordering::SeqCst)
    }
}

/// File access that prefers the editor's own buffers when the client
/// advertised `fs.readTextFile` / `fs.writeTextFile`, else the disk. The
/// path handed to either is already resolved inside the workspace root.
struct EditorFs<'h, 'a> {
    shared: &'h Shared<'a>,
    session: &'h Session,
    caps: p::FsCapabilities,
}

impl FileBackend for EditorFs<'_, '_> {
    fn read_text(&self, path: &Path, max_bytes: usize) -> Result<(String, bool), String> {
        if !self.caps.read_text_file {
            return DiskBackend.read_text(path, max_bytes);
        }
        let v = self
            .shared
            .request(
                "fs/read_text_file",
                json!({"sessionId": self.session.id, "path": path.to_string_lossy()}),
                self.session,
                self.shared.cfg.fs_timeout,
            )
            .map_err(fs_err)?;
        let content = v
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| "the editor returned no file content".to_string())?;
        if content.len() <= max_bytes {
            return Ok((content.to_string(), false));
        }
        let mut cut = max_bytes;
        while !content.is_char_boundary(cut) {
            cut -= 1;
        }
        Ok((content[..cut].to_string(), true))
    }

    fn write_text(&self, path: &Path, content: &str) -> Result<(), String> {
        if !self.caps.write_text_file {
            return DiskBackend.write_text(path, content);
        }
        self.shared
            .request(
                "fs/write_text_file",
                json!({"sessionId": self.session.id, "path": path.to_string_lossy(), "content": content}),
                self.session,
                self.shared.cfg.fs_timeout,
            )
            .map(|_| ())
            .map_err(fs_err)
    }
}

fn fs_err(e: ReqErr) -> String {
    match e {
        ReqErr::Timeout => "the editor did not answer the file request in time".to_string(),
        ReqErr::Cancelled => "the turn was cancelled".to_string(),
        ReqErr::Closed => "the editor connection closed".to_string(),
        ReqErr::Rpc(e) => format!("the editor refused the file request, code {}", e.code),
    }
}
