//! `serve_bench` and `soak` end to end against a fake loopback HTTP/SSE
//! server: no model, so what is held here is the client, the protocol rules
//! and the failure verdicts.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use turbospark_lab::args::Args;
use turbospark_lab::ledger::Ledger;
use turbospark_lab::schema::CmdOutput;
use turbospark_lab::serve_bench::{
    serve_bench_cmd, soak_cmd, METHODOLOGY_SERVE_BENCH, METHODOLOGY_SOAK,
};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Normal,
    /// An SSE `error` event after the first delta.
    ErrorEvent,
    /// HTTP 429, as a full queue answers.
    QueueFull,
    /// The connection closes without `[DONE]`.
    Truncated,
    /// Completes `n` requests, then only keep-alives. `/health` stays 200.
    StallAfter(usize),
}

struct Fake {
    mode: Mode,
    /// Service time per generation, taken under one lock like the server's
    /// one-at-a-time queue.
    service: Duration,
    timings: bool,
    gate: Mutex<()>,
    chats: AtomicUsize,
    in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
}

fn start(mode: Mode, service_ms: u64, timings: bool) -> (String, Arc<Fake>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Fake {
        mode,
        service: Duration::from_millis(service_ms),
        timings,
        gate: Mutex::new(()),
        chats: AtomicUsize::new(0),
        in_flight: AtomicUsize::new(0),
        peak_in_flight: AtomicUsize::new(0),
    });
    let shared = fake.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let fake = shared.clone();
            std::thread::spawn(move || {
                let _ = serve(&fake, stream);
            });
        }
    });
    (url, fake)
}

fn respond(stream: &mut TcpStream, status: &str, body: &Value) -> std::io::Result<()> {
    let body = body.to_string();
    write!(
        stream,
        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
}

/// One chunk of a chunked body.
fn chunk(stream: &mut TcpStream, text: &str) -> std::io::Result<()> {
    write!(stream, "{:x}\r\n{text}\r\n", text.len())?;
    stream.flush()
}

/// One SSE event, split across two HTTP chunks so the client has to
/// reassemble lines.
fn event(stream: &mut TcpStream, data: &Value) -> std::io::Result<()> {
    let text = format!("data: {data}\n\n");
    let half = text.len() / 2;
    chunk(stream, &text[..half])?;
    chunk(stream, &text[half..])
}

fn delta(d: Value, finish: Option<&str>) -> Value {
    json!({"choices": [{"index": 0, "delta": d, "finish_reason": finish}]})
}

fn words(body: &Value) -> usize {
    body["messages"][0]["content"]
        .as_str()
        .unwrap_or("")
        .split_whitespace()
        .count()
}

fn serve(fake: &Fake, mut stream: TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request = String::new();
    reader.read_line(&mut request)?;
    let mut length = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = v.trim().parse().unwrap();
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let path = request.split_whitespace().nth(1).unwrap_or("");
    match path {
        "/health" => respond(&mut stream, "200 OK", &json!({"status": "ok"})),
        "/v1/models" => respond(&mut stream, "200 OK", &json!({"data": [{"id": "fake"}]})),
        "/v1/messages/count_tokens" => respond(
            &mut stream,
            "200 OK",
            &json!({"input_tokens": words(&body)}),
        ),
        "/v1/chat/completions" => chat(fake, &mut stream, &body),
        _ => respond(&mut stream, "404 Not Found", &json!({})),
    }
}

fn chat(fake: &Fake, stream: &mut TcpStream, body: &Value) -> std::io::Result<()> {
    let nth = fake.chats.fetch_add(1, Ordering::SeqCst);
    if fake.mode == Mode::QueueFull {
        return respond(
            stream,
            "429 Too Many Requests",
            &json!({"error": "queue full"}),
        );
    }
    let now = fake.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
    fake.peak_in_flight.fetch_max(now, Ordering::SeqCst);
    let result = generate(fake, stream, body, nth);
    fake.in_flight.fetch_sub(1, Ordering::SeqCst);
    result
}

fn generate(fake: &Fake, stream: &mut TcpStream, body: &Value, nth: usize) -> std::io::Result<()> {
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    let max = body["max_tokens"].as_u64().unwrap();
    write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
    )?;
    event(stream, &delta(json!({"role": "assistant"}), None))?;
    chunk(stream, ":\n\n")?;
    if matches!(fake.mode, Mode::StallAfter(n) if nth >= n) {
        // Alive but never finishing, until the client hangs up.
        loop {
            std::thread::sleep(Duration::from_millis(50));
            chunk(stream, ":\n\n")?;
        }
    }
    let _turn = fake.gate.lock().unwrap_or_else(|p| p.into_inner());
    std::thread::sleep(fake.service);
    // A cap of 8 or less is hit while still reasoning.
    let capped = max <= 8;
    event(stream, &delta(json!({"reasoning_content": "think"}), None))?;
    match fake.mode {
        Mode::ErrorEvent => {
            return chunk(
                stream,
                "event: error\ndata: {\"error\":{\"message\":\"boom\"}}\n\n",
            )
        }
        Mode::Truncated => return Ok(()),
        _ => {}
    }
    if !capped {
        for _ in 0..4 {
            event(stream, &delta(json!({"content": "word "}), None))?;
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    let completion = if capped { max } else { 5 };
    let finish = if capped { "length" } else { "stop" };
    event(stream, &delta(json!({}), Some(finish)))?;
    let prompt = words(body) as u64;
    let mut usage = json!({"choices": [], "usage": {
        "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }});
    if fake.timings {
        usage["timings"] = json!({
            "predicted_n": completion, "predicted_per_second": 100.0,
            "predicted_ms": completion as f64 * 10.0,
            "prompt_per_second": 1234.0, "cache_n": 7,
        });
    }
    event(stream, &usage)?;
    chunk(stream, "data: [DONE]\n\n")?;
    chunk(stream, "")
}

fn lab(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tslab-serve-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn args(url: &str, lab: &std::path::Path, extra: &str) -> Args {
    let mut v = vec![
        "--server".to_string(),
        url.to_string(),
        "--lab".to_string(),
        lab.display().to_string(),
        "--allow-battery".to_string(),
    ];
    v.extend(extra.split_whitespace().map(str::to_string));
    Args::new(v)
}

fn only_record(lab: &std::path::Path, experiment: &str, kind: &str) -> Value {
    let mut all = Ledger::open(&lab.join(experiment))
        .unwrap()
        .of_kind(kind)
        .unwrap();
    assert_eq!(all.len(), 1);
    all.remove(0)
}

fn bench(tag: &str, url: &str, extra: &str) -> (CmdOutput, Value, PathBuf) {
    let dir = lab(tag);
    let out = serve_bench_cmd(args(url, &dir, extra)).unwrap();
    let record = only_record(&dir, "serve", "serve_bench");
    (out, record, dir)
}

#[test]
fn matrix_and_burst_record_what_the_server_said() {
    let (url, fake) = start(Mode::Normal, 1, true);
    let (out, r, dir) = bench(
        "matrix",
        &url,
        "--sizes short,64,1k --seeds 2 --max-tokens 32 --burst 4",
    );
    assert_eq!(out.code, 0, "{:?}", out.warnings);
    assert_eq!(r["methodology"], METHODOLOGY_SERVE_BENCH);
    // 1 warmup, 3 sizes x 2 seeds, 4 in the burst.
    assert_eq!(fake.chats.load(Ordering::SeqCst), 11);
    assert_eq!(r["counts"]["requests"], 11);
    assert_eq!(r["counts"]["accounting_mismatch"], 0);
    assert_eq!(r["warmup"].as_array().unwrap().len(), 1);
    assert_eq!(r["requests"].as_array().unwrap().len(), 6);
    let first = &r["requests"][0];
    assert_eq!(first["completion_tokens"], 5);
    assert_eq!(first["cached_tokens"], 7);
    assert_eq!(first["finish_reason"], "stop");
    assert_eq!(first["accounting"]["rate_checked"], true);
    assert_eq!(first["server_decode_tok_s"], 100.0);
    assert!(first["first_answer_s"].as_f64() >= first["ttft_s"].as_f64());
    // The short prompt is too small for a prefill rate; 1k is not.
    assert_eq!(r["by_size"][0]["server_prefill_tok_s"], Value::Null);
    assert_eq!(r["by_size"][2]["server_prefill_tok_s"]["median"], 1234.0);
    // Calibrated against the server's own counter.
    let long = r["by_size"][2]["prompt_tokens"]["median"].as_f64().unwrap();
    assert!((long - 1024.0).abs() < 32.0, "{long}");
    let burst = &r["burst"];
    assert_eq!(burst["completion_tokens"], 20);
    let wall = burst["wall_s"].as_f64().unwrap();
    let rate = burst["aggregate_tok_s"].as_f64().unwrap();
    assert!((rate - 20.0 / wall).abs() < 1e-9);
    assert!(fake.peak_in_flight.load(Ordering::SeqCst) > 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_cap_hit_while_reasoning_has_no_first_answer_and_no_invented_rate() {
    let (url, _) = start(Mode::Normal, 1, false);
    let (out, r, dir) = bench(
        "capped",
        &url,
        "--sizes short --seeds 2 --max-tokens 8 --burst 0 --warmup 0",
    );
    assert_eq!(out.code, 0);
    assert_eq!(r["counts"]["capped"], 2);
    let first = &r["requests"][0];
    assert!(first["ttft_s"].is_number());
    assert_eq!(first["first_answer_s"], Value::Null);
    assert_eq!(first["server_decode_tok_s"], Value::Null);
    assert_eq!(first["cached_tokens"], Value::Null);
    assert!(out.warnings.iter().any(|w| w.contains("no timings")));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn every_kind_of_failure_is_counted_once_and_never_retried() {
    for (tag, mode, expect) in [
        ("errev", Mode::ErrorEvent, "error event"),
        ("full", Mode::QueueFull, "HTTP 429"),
        ("trunc", Mode::Truncated, "ended early"),
    ] {
        let (url, fake) = start(mode, 1, false);
        let (out, r, dir) = bench(tag, &url, "--sizes short --seeds 3 --burst 0 --warmup 0");
        assert_eq!(out.code, 3, "{tag}");
        assert_eq!(fake.chats.load(Ordering::SeqCst), 3, "{tag} retried");
        assert_eq!(r["counts"]["failed"], 3, "{tag}");
        assert_eq!(r["eligible"], false, "{tag}");
        let error = r["requests"][0]["error"].as_str().unwrap();
        assert!(error.contains(expect), "{tag}: {error}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn open_loop_keeps_sending_and_judges_from_arrival() {
    // Serial 150 ms service, arrivals every 25 ms: the queue grows.
    let (url, fake) = start(Mode::Normal, 150, false);
    let (_, r, dir) = bench(
        "open",
        &url,
        "--sizes short --arrival periodic --rate 40 --requests 6 --warmup 0 \
         --slo-first-output-s 0.4",
    );
    assert!(
        fake.peak_in_flight.load(Ordering::SeqCst) >= 4,
        "a slow server held back later sends: {}",
        fake.peak_in_flight.load(Ordering::SeqCst)
    );
    let round = &r["rounds"][0];
    let samples = round["samples"].as_array().unwrap();
    for (i, s) in samples.iter().enumerate() {
        assert!((s["arrival_s"].as_f64().unwrap() - i as f64 * 0.025).abs() < 1e-9);
    }
    // 6 x 150 ms of service: the tail waits past the SLO, the head does not.
    let met = round["met_slo"].as_u64().unwrap();
    assert!((1..6).contains(&met), "met {met}");
    let wall = round["wall_s"].as_f64().unwrap();
    let goodput = r["goodput_req_s"]["median"].as_f64().unwrap();
    assert!((goodput - met as f64 / wall).abs() < 1e-9);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn closed_loop_never_exceeds_its_concurrency_and_reports_rounds() {
    let (url, fake) = start(Mode::Normal, 20, false);
    let (_, r, dir) = bench(
        "closed",
        &url,
        "--sizes short --arrival closed --concurrency 2 --requests 6 --rounds 3 --warmup 0 \
         --slo-duration-s 30",
    );
    assert_eq!(fake.peak_in_flight.load(Ordering::SeqCst), 2);
    assert_eq!(r["rounds"].as_array().unwrap().len(), 3);
    assert_eq!(r["goodput_req_s"]["n"], 3);
    assert!(r["goodput_req_s"]["min"].as_f64() <= r["goodput_req_s"]["max"].as_f64());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bad_targets_and_flags_are_refused_before_any_request() {
    let (url, fake) = start(Mode::Normal, 1, false);
    let dir = lab("refuse");
    for (url, extra) in [
        ("http://10.1.2.3:80", ""),
        (url.as_str(), "--model m"),
        (url.as_str(), "--typo 1"),
        (url.as_str(), "--rate 2"),
        (url.as_str(), "--arrival poisson"),
        (url.as_str(), "--server-bin /bin/true"),
    ] {
        let result = serve_bench_cmd(args(url, &dir, extra));
        // `--rate` without `--arrival` is a leftover argument.
        assert!(result.is_err(), "{url} {extra} accepted");
    }
    assert_eq!(fake.chats.load(Ordering::SeqCst), 0);
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn a_spawned_server_that_exits_is_reported_with_its_log() {
    let dir = lab("spawn");
    let argv = [
        "--server-bin",
        "/usr/bin/false",
        "--model",
        "none",
        "--lab",
        &dir.display().to_string(),
    ];
    let err = serve_bench_cmd(Args::new(argv.map(str::to_string).to_vec())).unwrap_err();
    assert!(
        err.contains("before listening") && err.contains("server-"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

fn soak(tag: &str, url: &str, extra: &str) -> (CmdOutput, Value) {
    let dir = lab(tag);
    // This process stands in for the server pid.
    let extra = format!(
        "--pid {} --long-prompt-tokens 0 {extra}",
        std::process::id()
    );
    let out = soak_cmd(args(url, &dir, &extra)).unwrap();
    let record = only_record(&dir, "soak", "soak");
    std::fs::remove_dir_all(dir).unwrap();
    (out, record)
}

#[test]
fn a_healthy_soak_passes_and_counts_walkouts_apart() {
    let (url, fake) = start(Mode::Normal, 5, false);
    let dir = lab("soak-ok");
    let extra = format!(
        "--pid {} --duration 2.2 --bucket-s 1 --sample-s 0.2 --concurrency 3 \
         --disconnect-pct 40 --long-prompt-tokens 600",
        std::process::id()
    );
    let out = soak_cmd(args(&url, &dir, &extra)).unwrap();
    let r = only_record(&dir, "soak", "soak");
    assert_eq!(out.code, 0, "{}", r["failure"]);
    assert_eq!(r["methodology"], METHODOLOGY_SOAK);
    assert_eq!(r["passed"], true);
    let (done, gone) = (
        r["completed"].as_u64().unwrap(),
        r["disconnected"].as_u64().unwrap(),
    );
    assert!(done > 0 && gone > 0, "{done} completed, {gone} walked out");
    assert_eq!(r["failed"], 0);
    // Every request the server saw is accounted for, plus the drain probe.
    assert_eq!(fake.chats.load(Ordering::SeqCst) as u64, done + gone + 1);
    let buckets = r["per_bucket"].as_array().unwrap();
    assert!(buckets.len() >= 2);
    let summed: u64 = buckets
        .iter()
        .map(|b| b["completed"].as_u64().unwrap())
        .sum();
    assert_eq!(summed, done);
    assert!(r["long_prefill"]["prompt_tokens"].as_u64().unwrap() > 400);
    assert_eq!(r["drain_probe"]["error"], Value::Null);
    if cfg!(target_os = "macos") {
        assert!(r["process"]["samples"].as_u64().unwrap() > 3);
        assert!(r["process"]["footprint_slope_bytes_per_min"].is_number());
        assert!(r["process"]["threads"]["max"].as_u64().unwrap() > 1);
        assert!(r["process"]["fds"]["max"].as_u64().unwrap() > 1);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn completions_stopping_while_health_is_ok_fails_the_soak() {
    let (url, _) = start(Mode::StallAfter(3), 1, false);
    let started = std::time::Instant::now();
    let (out, r) = soak(
        "soak-stall",
        &url,
        "--duration 60 --stall-s 0.8 --sample-s 0.2 --concurrency 2 --disconnect-pct 0",
    );
    assert_eq!(out.code, 3);
    let why = r["failure"].as_str().unwrap();
    assert!(why.contains("while /health is OK"), "{why}");
    // The stuck requests were cut off, not waited out, and are not counted
    // as server errors.
    assert!(started.elapsed() < Duration::from_secs(30));
    assert_eq!(
        (r["completed"].as_u64(), r["failed"].as_u64()),
        (Some(3), Some(0))
    );
}

#[test]
fn one_failed_request_fails_the_soak() {
    let (url, _) = start(Mode::QueueFull, 1, false);
    let (out, r) = soak(
        "soak-429",
        &url,
        "--duration 60 --concurrency 1 --disconnect-pct 0",
    );
    assert_eq!(out.code, 3);
    assert!(r["failure"].as_str().unwrap().contains("failed"));
    assert!(r["errors"][0].as_str().unwrap().contains("HTTP 429"));
}
