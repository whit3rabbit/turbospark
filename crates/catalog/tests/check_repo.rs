//! `check_repo` against a local fixture Hub: every outcome is a state the UI
//! can render, none is an `Err`. The GGUF header read is covered by the probe
//! tests (`evaluate_gguf` over synthetic headers), so these exercise the
//! listing, auth and safetensors paths that need only small JSON bodies.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use turbospark_catalog::{check_repo, CheckState, Client, RepoRef};

static ENDPOINT_LOCK: Mutex<()> = Mutex::new(());

struct EndpointReset;

impl Drop for EndpointReset {
    fn drop(&mut self) {
        turbospark_catalog::set_hf_endpoint_override(None);
    }
}

struct Reply {
    status: u16,
    body: String,
}

fn ok(body: impl Into<String>) -> Reply {
    Reply {
        status: 200,
        body: body.into(),
    }
}

fn status(status: u16) -> Reply {
    Reply {
        status,
        body: String::new(),
    }
}

/// Serves `route(method, path)` until the returned guard drops.
struct Fixture {
    endpoint: String,
    stop: Arc<AtomicBool>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Unblock the accept loop.
        let _ = std::net::TcpStream::connect(self.endpoint.trim_start_matches("http://"));
    }
}

fn serve(route: impl Fn(&str, &str) -> Reply + Send + 'static) -> Fixture {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = stop.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            if stop_flag.load(Ordering::SeqCst) {
                break;
            }
            let Ok(mut stream) = stream else { continue };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() {
                continue;
            }
            // Drain headers so the client sees a clean response.
            loop {
                let mut header = String::new();
                if reader.read_line(&mut header).unwrap_or(0) <= 2 {
                    break;
                }
            }
            let mut parts = line.split_whitespace();
            let (method, path) = (
                parts.next().unwrap_or("").to_string(),
                parts.next().unwrap_or("").to_string(),
            );
            let reply = route(&method, &path);
            let body = if method == "HEAD" { "" } else { &reply.body };
            let _ = write!(
                stream,
                "HTTP/1.1 {} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply.status,
                reply.body.len(),
                body
            );
        }
    });
    Fixture { endpoint, stop }
}

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

fn listing(files: &[(&str, u64)]) -> String {
    serde_json::json!({
        "id": "o/m",
        "sha": SHA,
        "siblings": files.iter().map(|(name, size)| serde_json::json!({
            "rfilename": name, "size": size
        })).collect::<Vec<_>>(),
    })
    .to_string()
}

fn qwen2_config() -> String {
    serde_json::json!({
        "model_type": "qwen2",
        "quantization": {"bits": 4, "group_size": 64},
        "hidden_size": 3584, "intermediate_size": 18944, "num_hidden_layers": 28,
        "num_attention_heads": 28, "num_key_value_heads": 4, "vocab_size": 152064,
        "max_position_embeddings": 32768, "rms_norm_eps": 1e-6, "rope_theta": 1e6,
        "hidden_act": "silu", "tie_word_embeddings": false, "use_sliding_window": false
    })
    .to_string()
}

fn run(route: impl Fn(&str, &str) -> Reply + Send + 'static) -> turbospark_catalog::RepoCheck {
    let _lock = ENDPOINT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let fixture = serve(route);
    turbospark_catalog::set_hf_endpoint_override(Some(fixture.endpoint.clone()));
    let _reset = EndpointReset;
    let client = Client::with_timeout(Duration::from_secs(3));
    check_repo(&client, &RepoRef::new("o/m", "main"), None, None)
}

#[test]
fn a_gated_repository_is_needs_auth_not_a_refusal() {
    let check = run(|_, _| status(401));
    assert_eq!(check.state, CheckState::NeedsAuth);
    assert!(check.report.is_none());
    assert!(check.reason.as_deref().unwrap().contains("401"));
}

#[test]
fn a_missing_repository_is_unavailable_and_retryable() {
    let check = run(|_, _| status(404));
    assert_eq!(check.state, CheckState::Unavailable);
}

#[test]
fn a_repository_with_no_weights_is_refused_with_a_reason() {
    let body = listing(&[("README.md", 10), ("config.json", 10)]);
    let check = run(move |_, _| ok(body.clone()));
    assert_eq!(check.state, CheckState::Refused);
    assert!(check
        .reason
        .as_deref()
        .unwrap()
        .contains("neither a .gguf nor a .safetensors"));
}

#[test]
fn a_sharded_only_gguf_repository_is_refused_not_an_empty_picker() {
    let body = listing(&[
        ("m-Q4_K_M-00001-of-00002.gguf", 5),
        ("m-Q4_K_M-00002-of-00002.gguf", 5),
    ]);
    let check = run(move |_, _| ok(body.clone()));
    assert_eq!(check.state, CheckState::Refused);
    assert!(check.reason.as_deref().unwrap().contains("sharded"));
}

#[test]
fn an_architecture_the_port_cannot_install_is_refused_with_the_probes_words() {
    let siblings = listing(&[
        ("model.safetensors", 100),
        ("config.json", 10),
        ("tokenizer.json", 10),
    ]);
    let check = run(move |method, path| {
        if path.contains("/resolve/main/config.json") {
            ok(r#"{"model_type":"minimax_m2"}"#)
        } else if method == "HEAD" {
            ok("")
        } else if path.contains("/api/models/") {
            ok(siblings.clone())
        } else {
            status(404)
        }
    });
    assert_eq!(check.state, CheckState::Refused);
    assert!(check.reason.as_deref().unwrap().contains("minimax_m2"));
    assert!(
        check.report.is_some(),
        "the refusal still carries its report"
    );
}

#[test]
fn a_supported_mlx_repository_is_runnable() {
    let siblings = listing(&[
        ("model.safetensors", 4_000_000),
        ("config.json", 10),
        ("tokenizer.json", 10),
        ("tokenizer_config.json", 10),
        ("chat_template.jinja", 10),
    ]);
    let config = qwen2_config();
    let check = run(move |method, path| {
        if path.contains("/resolve/main/config.json") {
            ok(config.clone())
        } else if method == "HEAD" {
            ok("x".repeat(4))
        } else if path.contains("/api/models/") {
            ok(siblings.clone())
        } else {
            status(404)
        }
    });
    assert_eq!(check.state, CheckState::Runnable, "{:?}", check.reason);
    let report = check.report.expect("a runnable check carries its report");
    assert_eq!(report.architecture.as_deref(), Some("qwen2"));
    assert!(check.variants.is_empty(), "safetensors has no quant picker");
    assert!(check.sidecar_repo.is_none());
}
