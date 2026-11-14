//! A scripted ACP client, written from the Agent Client Protocol text and
//! JSON-RPC 2.0, NOT from `turbospark_acp`'s own types: it shares no code with
//! the agent under test, so a mistake mirrored on both sides cannot hide. It
//! speaks line-delimited JSON over a child's pipes.
//!
//! This is a scripted client only. It is not Zed or any editor, and a pass
//! here says nothing about how a real editor behaves.

#![allow(dead_code)]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

/// What the client does when the agent sends `session/request_permission`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Perm {
    /// Never answer (a client that ignores the request).
    Silent,
    /// Pick the first option whose `kind` is `allow_once`.
    AllowOnce,
    /// Pick the first option whose `kind` is `reject_once`.
    RejectOnce,
    /// Answer with the `cancelled` outcome.
    Cancelled,
    /// Answer with a JSON-RPC "method not found" error.
    MethodNotFound,
}

pub struct Wire {
    pub child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    next_id: i64,
    /// Every `session/update` `update` object received, in order.
    pub updates: Vec<Value>,
    /// Every agent-to-client request received, in order.
    pub agent_requests: Vec<Value>,
    pub perm: Perm,
    /// When the last `send` of a request happened (for latency readings).
    pub last_sent: Instant,
}

impl Wire {
    pub fn spawn(mut cmd: Command) -> Wire {
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn agent");
        let stdin = child.stdin.take();
        let out = child.stdout.take().unwrap();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                match line {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            return;
                        }
                    }
                    Err(_) => return,
                }
            }
        });
        Wire {
            child,
            stdin,
            lines: rx,
            next_id: 1,
            updates: vec![],
            agent_requests: vec![],
            perm: Perm::Silent,
            last_sent: Instant::now(),
        }
    }

    fn write(&mut self, v: &Value) {
        let s = self.stdin.as_mut().expect("stdin already closed");
        writeln!(s, "{v}").unwrap();
        s.flush().unwrap();
    }

    pub fn send_raw_line(&mut self, line: &str) {
        let s = self.stdin.as_mut().expect("stdin already closed");
        writeln!(s, "{line}").unwrap();
        s.flush().unwrap();
    }

    pub fn request(&mut self, method: &str, params: Value) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        self.last_sent = Instant::now();
        self.write(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
        id
    }

    pub fn notify(&mut self, method: &str, params: Value) {
        self.write(&json!({"jsonrpc":"2.0","method":method,"params":params}));
    }

    /// Closes the agent's stdin (an editor going away).
    pub fn close_stdin(&mut self) {
        self.stdin = None;
    }

    /// Next stdout line as JSON, or `None` on timeout or EOF. A line that is
    /// not a JSON-RPC 2.0 object fails the test: stdout is protocol only.
    pub fn next(&mut self, wait: Duration) -> Option<Value> {
        let line = match self.lines.recv_timeout(wait) {
            Ok(l) => l,
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => return None,
        };
        let v: Value = serde_json::from_str(&line)
            .unwrap_or_else(|_| panic!("non-JSON stdout line: {line:?}"));
        assert_eq!(v["jsonrpc"], "2.0", "not a JSON-RPC 2.0 message: {line}");
        Some(v)
    }

    /// Routes one received message: updates are recorded, agent requests get
    /// the configured permission behavior (or method-not-found), and a
    /// matching response is returned.
    fn route(&mut self, v: Value, want: Option<i64>) -> Option<Value> {
        let has_method = v.get("method").is_some();
        let has_id = v.get("id").is_some();
        if has_method && !has_id {
            // A notification. Only session/update is expected from an agent.
            if v["method"] == "session/update" {
                assert!(v["params"]["sessionId"].is_string(), "{v}");
                self.updates.push(v["params"]["update"].clone());
            }
            return None;
        }
        if has_method && has_id {
            self.agent_requests.push(v.clone());
            let id = v["id"].clone();
            if v["method"] == "session/request_permission" {
                self.answer_permission(&v);
            } else {
                self.write(&json!({"jsonrpc":"2.0","id":id,
                    "error":{"code":-32601,"message":"method not found"}}));
            }
            return None;
        }
        // A response: exactly one of result or error.
        assert!(
            v.get("result").is_some() ^ v.get("error").is_some(),
            "response must carry exactly one of result/error: {v}"
        );
        if let Some(w) = want {
            if v["id"] == w {
                return Some(v);
            }
        }
        None
    }

    fn answer_permission(&mut self, req: &Value) {
        let id = req["id"].clone();
        let pick = |kind: &str| -> Option<String> {
            req["params"]["options"].as_array().and_then(|os| {
                os.iter()
                    .find(|o| o["kind"] == kind)
                    .and_then(|o| o["optionId"].as_str())
                    .map(String::from)
            })
        };
        match self.perm {
            Perm::Silent => {}
            Perm::AllowOnce => {
                let o = pick("allow_once").expect("an allow_once option");
                self.write(&json!({"jsonrpc":"2.0","id":id,
                    "result":{"outcome":{"outcome":"selected","optionId":o}}}));
            }
            Perm::RejectOnce => {
                let o = pick("reject_once").expect("a reject_once option");
                self.write(&json!({"jsonrpc":"2.0","id":id,
                    "result":{"outcome":{"outcome":"selected","optionId":o}}}));
            }
            Perm::Cancelled => self.write(&json!({"jsonrpc":"2.0","id":id,
                "result":{"outcome":{"outcome":"cancelled"}}})),
            Perm::MethodNotFound => self.write(&json!({"jsonrpc":"2.0","id":id,
                "error":{"code":-32601,"message":"method not found"}})),
        }
    }

    /// Reads until the response to `id` (or `wait` runs out), recording
    /// updates and serving agent requests on the way.
    pub fn response(&mut self, id: i64, wait: Duration) -> Value {
        let deadline = Instant::now() + wait;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let v = self
                .next(left)
                .unwrap_or_else(|| panic!("no response to request {id} within {wait:?}"));
            if let Some(r) = self.route(v, Some(id)) {
                return r;
            }
        }
    }

    /// Reads until an update that arrives AFTER this call satisfies `pred`.
    /// Earlier updates never count: a caller waiting for "the next chunk"
    /// must not be satisfied by the previous turn's.
    pub fn until_update(&mut self, wait: Duration, pred: impl Fn(&Value) -> bool) {
        let deadline = Instant::now() + wait;
        let mut seen = self.updates.len();
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let v = self.next(left).expect("timed out waiting for an update");
            self.route(v, None);
            if self.updates.len() > seen {
                if self.updates[seen..].iter().any(&pred) {
                    return;
                }
                seen = self.updates.len();
            }
        }
    }

    /// Reads until an agent request with `method` arrives.
    pub fn until_agent_request(&mut self, wait: Duration, method: &str) -> Value {
        let deadline = Instant::now() + wait;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let v = self
                .next(left)
                .unwrap_or_else(|| panic!("no {method} request within {wait:?}"));
            let before = self.agent_requests.len();
            self.route(v, None);
            if self.agent_requests.len() > before {
                let r = self.agent_requests.last().unwrap().clone();
                if r["method"] == method {
                    return r;
                }
            }
        }
    }

    // ---- protocol steps ----

    pub fn initialize(&mut self) -> Value {
        let id = self.request(
            "initialize",
            json!({"protocolVersion":1,
                   "clientCapabilities":{"fs":{"readTextFile":false,"writeTextFile":false}}}),
        );
        let r = self.response(id, Duration::from_secs(20));
        assert_eq!(r["result"]["protocolVersion"], 1, "{r}");
        assert!(r["result"]["agentCapabilities"].is_object(), "{r}");
        r
    }

    pub fn new_session(&mut self, cwd: &std::path::Path) -> String {
        let id = self.request("session/new", json!({"cwd":cwd,"mcpServers":[]}));
        let r = self.response(id, Duration::from_secs(20));
        let sid = r["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();
        assert!(!sid.is_empty());
        sid
    }

    pub fn prompt_request(&mut self, sid: &str, text: &str) -> i64 {
        self.request(
            "session/prompt",
            json!({"sessionId":sid,"prompt":[{"type":"text","text":text}]}),
        )
    }

    /// All streamed answer text recorded so far.
    pub fn message_text(&self) -> String {
        self.updates
            .iter()
            .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
            .map(|u| {
                assert_eq!(u["content"]["type"], "text", "{u}");
                u["content"]["text"].as_str().expect("chunk text")
            })
            .collect()
    }

    /// Waits for the child to exit, killing it (and failing) on timeout.
    pub fn wait_exit(&mut self, wait: Duration) -> std::process::ExitStatus {
        let deadline = Instant::now() + wait;
        loop {
            if let Some(s) = self.child.try_wait().unwrap() {
                return s;
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                panic!("agent did not exit within {wait:?}");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Wire {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// A fresh directory per call: tests in one process run in parallel, and a
/// shared workspace would let one test's write satisfy another's assertion.
pub fn scratch(tag: &str) -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("tsacp-scripted-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    std::fs::canonicalize(&p).unwrap()
}
