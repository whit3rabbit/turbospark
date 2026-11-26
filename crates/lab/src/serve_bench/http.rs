//! A small HTTP/1.1 and SSE client for loopback servers.
//!
//! Hand-rolled over `TcpStream` so the lab adds no dependency, and so that
//! every timestamp is taken by this code on its own monotonic clock. It
//! refuses anything but a loopback address and never retries: a request that
//! fails is a result, not something to hide.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};

/// How often a blocked read wakes to look at its deadline and abort flag.
const POLL: Duration = Duration::from_millis(100);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// A header block or chunk-size line longer than this is not a server.
const MAX_LINE: u64 = 64 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct Server {
    pub addr: SocketAddr,
}

impl Server {
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

/// `http://HOST:PORT`, loopback only. Every address the name resolves to
/// must be loopback, so a hosts-file entry cannot send prompts off the
/// machine.
pub fn parse_loopback_url(url: &str) -> Result<Server, String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("{url:?}: only http://HOST:PORT on loopback"))?;
    let authority = rest.strip_suffix('/').unwrap_or(rest);
    if authority.contains('/') || authority.contains('@') {
        return Err(format!("{url:?}: expected http://HOST:PORT with no path"));
    }
    let addrs: Vec<SocketAddr> = authority
        .to_socket_addrs()
        .map_err(|e| format!("{url:?}: {e}"))?
        .collect();
    if let Some(a) = addrs.iter().find(|a| !a.ip().is_loopback()) {
        return Err(format!("{url:?}: {} is not a loopback address", a.ip()));
    }
    // IPv4 first: the server binds 127.0.0.1.
    let addr = addrs
        .iter()
        .find(|a| a.is_ipv4())
        .or_else(|| addrs.first())
        .copied()
        .ok_or_else(|| format!("{url:?}: no address"))?;
    Ok(Server { addr })
}

/// A socket read bounded by one absolute deadline and an abort flag, so a
/// server that stops sending cannot park a request forever.
struct DeadlineReader {
    stream: TcpStream,
    deadline: Instant,
    abort: Option<Arc<AtomicBool>>,
}

impl Read for DeadlineReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            if self
                .abort
                .as_ref()
                .is_some_and(|a| a.load(Ordering::Relaxed))
            {
                return Err(io::Error::other("aborted by the harness"));
            }
            if Instant::now() >= self.deadline {
                return Err(io::Error::other("request timed out"));
            }
            match self.stream.read(buf) {
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::Interrupted
                    ) => {}
                other => return other,
            }
        }
    }
}

fn read_line<R: BufRead>(r: &mut R) -> io::Result<String> {
    let mut line = String::new();
    r.take(MAX_LINE).read_line(&mut line)?;
    if !line.ends_with('\n') {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "connection closed mid-line",
        ));
    }
    line.truncate(line.trim_end_matches(['\r', '\n']).len());
    Ok(line)
}

enum Framing {
    /// Bytes left in the current chunk; `done` after the zero chunk.
    Chunked {
        left: u64,
        done: bool,
    },
    Length(u64),
    UntilClose,
}

struct Body {
    inner: BufReader<DeadlineReader>,
    framing: Framing,
}

impl Read for Body {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let eof = || io::Error::new(io::ErrorKind::UnexpectedEof, "truncated body");
        match &mut self.framing {
            Framing::UntilClose => self.inner.read(buf),
            Framing::Length(left) => {
                if *left == 0 {
                    return Ok(0);
                }
                let n = buf.len().min(*left as usize);
                let n = self.inner.read(&mut buf[..n])?;
                if n == 0 {
                    return Err(eof());
                }
                *left -= n as u64;
                Ok(n)
            }
            Framing::Chunked { left, done } => {
                if *done {
                    return Ok(0);
                }
                if *left == 0 {
                    let line = read_line(&mut self.inner)?;
                    let size = line.split(';').next().unwrap_or("").trim();
                    *left = u64::from_str_radix(size, 16)
                        .map_err(|_| io::Error::other(format!("bad chunk size {line:?}")))?;
                    if *left == 0 {
                        *done = true;
                        return Ok(0);
                    }
                }
                let n = buf.len().min(*left as usize);
                let n = self.inner.read(&mut buf[..n])?;
                if n == 0 {
                    return Err(eof());
                }
                *left -= n as u64;
                if *left == 0 {
                    read_line(&mut self.inner)?;
                }
                Ok(n)
            }
        }
    }
}

struct Response {
    status: u16,
    body: Body,
}

fn send(
    server: &Server,
    method: &str,
    path: &str,
    body: Option<&Value>,
    deadline: Instant,
    abort: Option<Arc<AtomicBool>>,
) -> Result<(Response, TcpStream), String> {
    let io_err = |e: io::Error| format!("{method} {path}: {e}");
    let mut stream = TcpStream::connect_timeout(&server.addr, CONNECT_TIMEOUT).map_err(io_err)?;
    stream.set_read_timeout(Some(POLL)).map_err(io_err)?;
    stream
        .set_write_timeout(Some(CONNECT_TIMEOUT))
        .map_err(io_err)?;
    let _ = stream.set_nodelay(true);
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        server.addr,
        payload.len()
    );
    stream.write_all(request.as_bytes()).map_err(io_err)?;
    let handle = stream.try_clone().map_err(io_err)?;
    let mut reader = BufReader::new(DeadlineReader {
        stream,
        deadline,
        abort,
    });
    let status_line = read_line(&mut reader).map_err(io_err)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("{method} {path}: bad status line {status_line:?}"))?;
    let mut framing = Framing::UntilClose;
    loop {
        let line = read_line(&mut reader).map_err(io_err)?;
        if line.is_empty() {
            break;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let v = v.trim();
        if k.eq_ignore_ascii_case("transfer-encoding") && v.eq_ignore_ascii_case("chunked") {
            framing = Framing::Chunked {
                left: 0,
                done: false,
            };
        } else if k.eq_ignore_ascii_case("content-length")
            && !matches!(framing, Framing::Chunked { .. })
        {
            framing = Framing::Length(
                v.parse()
                    .map_err(|_| format!("{method} {path}: content-length {v:?}"))?,
            );
        }
    }
    let response = Response {
        status,
        body: Body {
            inner: reader,
            framing,
        },
    };
    Ok((response, handle))
}

/// One non-streaming request: status and parsed JSON body (null when the
/// body is not JSON).
pub fn json_request(
    server: &Server,
    method: &str,
    path: &str,
    body: Option<&Value>,
    timeout: Duration,
) -> Result<(u16, Value), String> {
    let (mut r, _) = send(server, method, path, body, Instant::now() + timeout, None)?;
    let mut text = String::new();
    (&mut r.body)
        .take(8 << 20)
        .read_to_string(&mut text)
        .map_err(|e| format!("{method} {path}: {e}"))?;
    Ok((r.status, serde_json::from_str(&text).unwrap_or(Value::Null)))
}

pub fn health_ok(server: &Server) -> bool {
    json_request(server, "GET", "/health", None, Duration::from_secs(5))
        .is_ok_and(|(status, _)| status == 200)
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub seed: u64,
    pub timeout: Duration,
    /// Drop the connection after this many output deltas (a client that
    /// walks away mid-stream).
    pub cancel_after_deltas: Option<usize>,
}

impl ChatRequest {
    pub fn body(&self) -> Value {
        json!({
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "seed": self.seed,
            "stream": true,
            "stream_options": {"include_usage": true},
        })
    }
}

/// What one streamed request did. Latencies are the client's monotonic
/// clock from just before connect; counts and rates are only what the
/// server said.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Outcome {
    pub status: u16,
    pub error: Option<String>,
    pub disconnected_by_client: bool,
    /// First delta carrying output of any kind (reasoning, text or tool).
    /// The role-only opening chunk is not output.
    pub ttft_s: Option<f64>,
    /// First answer delta (text or tool). Stays null for a turn that hit
    /// the cap while still reasoning.
    pub first_answer_s: Option<f64>,
    pub total_s: f64,
    pub finish_reason: Option<String>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    /// The server's own `timings` object, when it sends one.
    pub timings: Option<Value>,
    pub output_deltas: usize,
}

impl Outcome {
    pub fn completed(&self) -> bool {
        self.error.is_none() && !self.disconnected_by_client
    }
}

fn absorb(out: &mut Outcome, chunk: &Value, at: f64) {
    if let Some(choice) = chunk["choices"].get(0) {
        let delta = &choice["delta"];
        let non_empty = |k: &str| delta[k].as_str().is_some_and(|s| !s.is_empty());
        let tool = delta["tool_calls"]
            .as_array()
            .is_some_and(|a| !a.is_empty());
        let answer = non_empty("content") || tool;
        if answer || non_empty("reasoning_content") {
            out.output_deltas += 1;
            out.ttft_s.get_or_insert(at);
        }
        if answer {
            out.first_answer_s.get_or_insert(at);
        }
        if let Some(f) = choice["finish_reason"].as_str() {
            out.finish_reason = Some(f.to_string());
        }
    }
    let usage = &chunk["usage"];
    if usage.is_object() {
        out.prompt_tokens = usage["prompt_tokens"].as_u64();
        out.completion_tokens = usage["completion_tokens"].as_u64();
        out.total_tokens = usage["total_tokens"].as_u64();
        out.cached_tokens = usage["prompt_tokens_details"]["cached_tokens"].as_u64();
    }
    if chunk["timings"].is_object() {
        out.timings = Some(chunk["timings"].clone());
        if out.cached_tokens.is_none() {
            out.cached_tokens = chunk["timings"]["cache_n"].as_u64();
        }
    }
}

fn stream(
    server: &Server,
    req: &ChatRequest,
    start: Instant,
    abort: Option<Arc<AtomicBool>>,
    out: &mut Outcome,
) -> Result<(), String> {
    let (r, socket) = send(
        server,
        "POST",
        "/v1/chat/completions",
        Some(&req.body()),
        start + req.timeout,
        abort,
    )?;
    out.status = r.status;
    let mut lines = BufReader::new(r.body);
    if out.status != 200 {
        let mut text = String::new();
        let _ = lines.take(4096).read_to_string(&mut text);
        return Err(format!("HTTP {}: {}", out.status, text.trim()));
    }
    let (mut event, mut data) = (String::new(), String::new());
    loop {
        // A stream that ends without [DONE] did not finish.
        let line = read_line(&mut lines).map_err(|e| format!("stream ended early: {e}"))?;
        let at = start.elapsed().as_secs_f64();
        if line.is_empty() {
            if event == "error" {
                return Err(format!("server error event: {data}"));
            }
            if data == "[DONE]" {
                return Ok(());
            }
            if !data.is_empty() {
                let chunk: Value =
                    serde_json::from_str(&data).map_err(|e| format!("bad chunk: {e}"))?;
                if !chunk["error"].is_null() {
                    return Err(format!("server error chunk: {}", chunk["error"]));
                }
                absorb(out, &chunk, at);
                if req
                    .cancel_after_deltas
                    .is_some_and(|n| out.output_deltas >= n)
                {
                    out.disconnected_by_client = true;
                    let _ = socket.shutdown(Shutdown::Both);
                    return Ok(());
                }
            }
            event.clear();
            data.clear();
        } else if let Some(v) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(v.strip_prefix(' ').unwrap_or(v));
        } else if let Some(v) = line.strip_prefix("event:") {
            event = v.trim().to_string();
        }
        // Anything else, keep-alive comments included, is ignored.
    }
}

/// Send one streaming chat completion. Never retried.
pub fn chat(server: &Server, req: &ChatRequest, abort: Option<Arc<AtomicBool>>) -> Outcome {
    let start = Instant::now();
    let mut out = Outcome::default();
    if let Err(e) = stream(server, req, start, abort, &mut out) {
        out.error = Some(e);
    }
    out.total_s = start.elapsed().as_secs_f64();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loopback_http_urls() {
        assert_eq!(
            parse_loopback_url("http://127.0.0.1:8080/").unwrap().url(),
            "http://127.0.0.1:8080"
        );
        assert!(parse_loopback_url("http://localhost:1").is_ok());
        assert!(parse_loopback_url("http://[::1]:1").is_ok());
        for bad in [
            "https://127.0.0.1:1",
            "http://10.0.0.1:80",
            "http://0.0.0.0:80",
            "http://192.168.1.5:8080",
            "http://127.0.0.1:1/v1",
            "http://127.0.0.1@10.0.0.1:1",
            "127.0.0.1:1",
        ] {
            assert!(parse_loopback_url(bad).is_err(), "{bad} accepted");
        }
    }

    #[test]
    fn role_chunk_is_not_output_and_reasoning_is_not_answer() {
        let mut o = Outcome::default();
        absorb(
            &mut o,
            &json!({"choices":[{"delta":{"role":"assistant","content":""}}]}),
            0.1,
        );
        assert_eq!(o.ttft_s, None);
        absorb(
            &mut o,
            &json!({"choices":[{"delta":{"reasoning_content":"hm"}}]}),
            0.2,
        );
        assert_eq!((o.ttft_s, o.first_answer_s), (Some(0.2), None));
        absorb(&mut o, &json!({"choices":[{"delta":{"content":"a"}}]}), 0.3);
        assert_eq!((o.ttft_s, o.first_answer_s), (Some(0.2), Some(0.3)));
        absorb(
            &mut o,
            &json!({"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,
                    "prompt_tokens_details":{"cached_tokens":3}}}),
            0.4,
        );
        assert_eq!(
            (o.prompt_tokens, o.completion_tokens, o.cached_tokens),
            (Some(5), Some(2), Some(3))
        );
    }
}
