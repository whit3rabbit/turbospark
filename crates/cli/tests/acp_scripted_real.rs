//! The independent scripted ACP client (`crates/acp/tests/scripted/client.rs`,
//! hand-written from the spec, no `turbospark_acp` types) against the real
//! `turbospark-acp-agent` with a real model. `#[ignore]`d: needs a Metal
//! device, an installed chat model in `TURBOSPARK_ACP_SMOKE_MODEL` and a quiet
//! host. The CI-safe variant of the same flows (fake engine) is
//! `crates/acp/tests/scripted_client.rs`.
//!
//! Scripted client only: this is not Zed or any other editor.

#[path = "../../acp/tests/scripted/client.rs"]
mod client;

use client::{scratch, Wire};
use serde_json::json;
use std::process::Command;
use std::time::{Duration, Instant};

/// Model load plus first token can take a while on a cold 5 GB install.
const LOAD: Duration = Duration::from_secs(600);

#[test]
#[ignore = "needs a Metal device and an installed chat model (TURBOSPARK_ACP_SMOKE_MODEL)"]
fn scripted_client_against_a_real_model() {
    let model = std::env::var("TURBOSPARK_ACP_SMOKE_MODEL")
        .expect("set TURBOSPARK_ACP_SMOKE_MODEL to an installed chat model");
    let ws = scratch("real-ws");
    let state = scratch("real-state");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_turbospark-acp-agent"));
    cmd.args(["--model", &model, "--state-dir", state.to_str().unwrap()]);
    let mut w = Wire::spawn(cmd);
    w.initialize();
    let sid = w.new_session(&ws);

    // Turn 1: streams text and ends. The model loads inside this turn, so the
    // first-chunk time includes the load; the second turn's does not.
    let id = w.prompt_request(&sid, "Reply with the single word: pong");
    let t0 = Instant::now();
    w.until_update(LOAD, |u| u["sessionUpdate"] == "agent_message_chunk");
    let first_cold = t0.elapsed();
    let r = w.response(id, LOAD);
    assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");
    let text = w.message_text();
    assert!(!text.trim().is_empty(), "no streamed text");
    eprintln!(
        "turn 1: first chunk {:?} after the request (includes model load), {} updates, text {text:?}",
        first_cold,
        w.updates.len()
    );

    // Turn 2 on the loaded model: first-chunk latency, then cancel mid-stream.
    let n = w.updates.len();
    let id = w.prompt_request(&sid, "Count from 1 to 500, one number per line.");
    let t1 = Instant::now();
    w.until_update(LOAD, |u| u["sessionUpdate"] == "agent_message_chunk");
    eprintln!(
        "turn 2: first chunk {:?} (model already loaded)",
        t1.elapsed()
    );
    w.notify("session/cancel", json!({"sessionId":sid}));
    let tc = Instant::now();
    let r = w.response(id, Duration::from_secs(60));
    assert_eq!(r["result"]["stopReason"], "cancelled", "{r}");
    eprintln!("cancel landed in {:?}", tc.elapsed());
    assert!(w.updates.len() > n);

    // The session still answers after a cancel.
    let id = w.prompt_request(&sid, "Reply with the single word: ok");
    assert_eq!(w.response(id, LOAD)["result"]["stopReason"], "end_turn");

    // Closing stdin unloads the model and exits cleanly.
    w.close_stdin();
    let t = Instant::now();
    let status = w.wait_exit(Duration::from_secs(60));
    eprintln!("exit after EOF in {:?}: {status:?}", t.elapsed());
    assert!(status.success());
    let _ = std::fs::remove_dir_all(ws);
    let _ = std::fs::remove_dir_all(state);
}
