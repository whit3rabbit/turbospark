//! In-process ACP client harness: the adapter runs on a thread, wired to the
//! test through channels standing in for stdin and stdout.

#![allow(dead_code)]

use serde_json::{json, Value};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;
use turbospark_acp::{serve, AcpConfig, Engine, StopReason, TurnHost, TurnRequest};

pub const WAIT: Duration = Duration::from_secs(10);

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

struct ChanWrite(Sender<Vec<u8>>);

impl Write for ChanWrite {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0
            .send(b.to_vec())
            .map(|_| b.len())
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "closed"))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub type TurnFn =
    dyn Fn(&TurnRequest, &mut dyn TurnHost) -> Result<StopReason, String> + Send + Sync;

pub struct FakeEngine(pub Box<TurnFn>);

impl Engine for FakeEngine {
    fn model_id(&self) -> String {
        "fake".into()
    }
    fn run_turn(&self, req: &TurnRequest, host: &mut dyn TurnHost) -> Result<StopReason, String> {
        (self.0)(req, host)
    }
}

pub struct Client {
    to_agent: Option<Sender<Vec<u8>>>,
    from_agent: Receiver<Value>,
    pub server: Option<std::thread::JoinHandle<std::io::Result<()>>>,
    next_id: u64,
    /// Notifications and requests seen while waiting for something else.
    pub backlog: Vec<Value>,
}

pub fn start(
    cfg: AcpConfig,
    f: impl Fn(&TurnRequest, &mut dyn TurnHost) -> Result<StopReason, String> + Send + Sync + 'static,
) -> Client {
    let engine: &'static FakeEngine = Box::leak(Box::new(FakeEngine(Box::new(f))));
    let cfg: &'static AcpConfig = Box::leak(Box::new(cfg));
    let (in_tx, in_rx) = channel::<Vec<u8>>();
    let (out_tx, out_rx) = channel::<Vec<u8>>();
    let (msg_tx, msg_rx) = channel::<Value>();
    // Reassemble the agent's byte stream into one JSON value per line. Any
    // line that is not valid JSON would fail the test: stdout is protocol only.
    std::thread::spawn(move || {
        let mut acc: Vec<u8> = Vec::new();
        while let Ok(chunk) = out_rx.recv() {
            acc.extend_from_slice(&chunk);
            while let Some(i) = acc.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = acc.drain(..=i).collect();
                let v: Value = serde_json::from_slice(&line[..line.len() - 1])
                    .expect("agent wrote a non-JSON line to stdout");
                if msg_tx.send(v).is_err() {
                    return;
                }
            }
        }
    });
    let server = std::thread::spawn(move || {
        serve(
            engine,
            cfg,
            BufReader::new(ChanRead {
                rx: in_rx,
                buf: vec![],
                pos: 0,
            }),
            ChanWrite(out_tx),
        )
    });
    Client {
        to_agent: Some(in_tx),
        from_agent: msg_rx,
        server: Some(server),
        next_id: 1,
        backlog: Vec::new(),
    }
}

pub fn cfg_in(dir: &Path) -> AcpConfig {
    AcpConfig {
        permission_timeout: Duration::from_secs(5),
        fs_timeout: Duration::from_secs(5),
        state_dir: Some(dir.join("state")),
        ..AcpConfig::default()
    }
}

impl Client {
    pub fn send_raw(&self, bytes: &[u8]) {
        let _ = self.to_agent.as_ref().unwrap().send(bytes.to_vec());
    }

    pub fn send(&self, v: &Value) {
        let mut b = serde_json::to_vec(v).unwrap();
        b.push(b'\n');
        self.send_raw(&b);
    }

    /// Sends a request and returns its id.
    pub fn request(&mut self, method: &str, params: Value) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
        id
    }

    pub fn notify(&self, method: &str, params: Value) {
        self.send(&json!({"jsonrpc":"2.0","method":method,"params":params}));
    }

    pub fn reply(&self, id: &Value, result: Value) {
        self.send(&json!({"jsonrpc":"2.0","id":id,"result":result}));
    }

    pub fn reply_err(&self, id: &Value, code: i64, msg: &str) {
        self.send(&json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":msg}}));
    }

    /// Next message from the agent, or None on timeout.
    pub fn next(&mut self, wait: Duration) -> Option<Value> {
        if !self.backlog.is_empty() {
            return Some(self.backlog.remove(0));
        }
        match self.from_agent.recv_timeout(wait) {
            Ok(v) => Some(v),
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => None,
        }
    }

    /// Waits for the response to `id`, stashing everything else in `backlog`.
    pub fn response(&mut self, id: u64) -> Value {
        if let Some(i) = self
            .backlog
            .iter()
            .position(|m| m.get("id") == Some(&json!(id)) && m.get("method").is_none())
        {
            return self.backlog.remove(i);
        }
        let mut stash = Vec::new();
        loop {
            let m = match self.from_agent.recv_timeout(WAIT) {
                Ok(m) => m,
                Err(_) => panic!("no response to request {id}; saw {stash:?}"),
            };
            if m.get("id") == Some(&json!(id)) && m.get("method").is_none() {
                self.backlog.extend(stash);
                return m;
            }
            stash.push(m);
        }
    }

    /// Next agent-to-client request with this method, stashing the rest.
    pub fn expect_request(&mut self, method: &str) -> Value {
        if let Some(i) = self.backlog.iter().position(|m| {
            m.get("method").and_then(Value::as_str) == Some(method) && m.get("id").is_some()
        }) {
            return self.backlog.remove(i);
        }
        let mut stash = Vec::new();
        loop {
            let m = match self.from_agent.recv_timeout(WAIT) {
                Ok(m) => m,
                Err(_) => panic!("no {method} request; saw {stash:?}"),
            };
            if m.get("method").and_then(Value::as_str) == Some(method) && m.get("id").is_some() {
                self.backlog.extend(stash);
                return m;
            }
            stash.push(m);
        }
    }

    /// True when no agent-to-client request with this method arrives in `wait`.
    pub fn no_request(&mut self, method: &str, wait: Duration) -> bool {
        let end = std::time::Instant::now() + wait;
        while let Some(left) = end.checked_duration_since(std::time::Instant::now()) {
            match self.from_agent.recv_timeout(left) {
                Ok(m) => {
                    if m.get("method").and_then(Value::as_str) == Some(method)
                        && m.get("id").is_some()
                    {
                        return false;
                    }
                    self.backlog.push(m);
                }
                Err(_) => break,
            }
        }
        true
    }

    pub fn initialize(&mut self, fs_read: bool, fs_write: bool) -> Value {
        let id = self.request(
            "initialize",
            json!({"protocolVersion":1,"clientCapabilities":{"fs":{"readTextFile":fs_read,"writeTextFile":fs_write}}}),
        );
        self.response(id)
    }

    pub fn new_session(&mut self, cwd: &str) -> String {
        let id = self.request("session/new", json!({"cwd": cwd, "mcpServers": []}));
        let r = self.response(id);
        r["result"]["sessionId"]
            .as_str()
            .unwrap_or_else(|| panic!("session/new failed: {r}"))
            .to_string()
    }

    pub fn prompt(&mut self, session: &str, text: &str) -> u64 {
        self.request(
            "session/prompt",
            json!({"sessionId": session, "prompt":[{"type":"text","text":text}]}),
        )
    }

    /// Every `session/update` update object seen so far (backlog included),
    /// waiting briefly for stragglers.
    pub fn updates(&mut self) -> Vec<Value> {
        // Read the channel directly: `next` drains the backlog first, so
        // looping on it would pop and re-push the same message forever.
        while let Ok(m) = self.from_agent.recv_timeout(Duration::from_millis(100)) {
            self.backlog.push(m);
        }
        self.backlog
            .iter()
            .filter(|m| m["method"] == "session/update")
            .map(|m| m["params"]["update"].clone())
            .collect()
    }

    /// Closes stdin and waits for the adapter to return.
    pub fn close(&mut self) -> std::io::Result<()> {
        self.to_agent = None;
        let h = self.server.take().expect("already closed");
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(h.join());
        });
        match rx.recv_timeout(WAIT) {
            Ok(r) => r.expect("adapter thread panicked"),
            Err(_) => panic!("adapter did not exit after stdin EOF"),
        }
    }
}

pub struct Tmp(pub PathBuf);
impl Tmp {
    pub fn new(tag: &str) -> Tmp {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "tsacp-it-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        Tmp(std::fs::canonicalize(p).unwrap())
    }
    pub fn ws(&self) -> PathBuf {
        let w = self.0.join("ws");
        std::fs::create_dir_all(&w).unwrap();
        w
    }
}
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
