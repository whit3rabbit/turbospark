//! Scripted-client interop: an independent client (`scripted/client.rs`,
//! hand-written from the ACP spec, sharing no code with `turbospark_acp`)
//! drives the real `serve` loop as a child process over pipes, with a
//! scripted fake engine (`examples/scripted_agent.rs`). No model, no GPU.
//!
//! Scripted client only: this proves nothing about Zed or any other editor.

#[path = "scripted/client.rs"]
mod client;

use client::{scratch, Perm, Wire};
use serde_json::json;
use std::process::Command;
use std::time::Duration;

/// `cargo test` builds examples into target/<profile>/examples. The test
/// binary's own depth under <profile> depends on the cargo version: the
/// classic layout puts it at <profile>/deps/<test>, the newer build-dir
/// layout at <profile>/build/<pkg>/<hash>/out/<test> (CI's floating stable
/// switched to it). Walk up from the test binary instead of assuming a depth.
fn agent_cmd() -> Command {
    let exe = std::env::current_exe().unwrap();
    let path = exe
        .ancestors()
        .skip(1)
        .map(|dir| dir.join("examples").join("scripted_agent"))
        .find(|candidate| candidate.exists());
    let path = path.unwrap_or_else(|| panic!("build examples first: none above {exe:?}"));
    Command::new(path)
}

const T: Duration = Duration::from_secs(20);

fn start() -> (Wire, std::path::PathBuf, String) {
    let ws = scratch("ws");
    let mut w = Wire::spawn(agent_cmd());
    w.initialize();
    let sid = w.new_session(&ws);
    (w, ws, sid)
}

#[test]
fn prompt_streams_ordered_chunks_before_the_response() {
    let (mut w, ws, sid) = start();
    let id = w.prompt_request(&sid, "stream");
    let r = w.response(id, T);
    assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");
    let expected: String = (0..10).map(|i| format!("word{i} ")).collect();
    assert_eq!(w.message_text(), expected);
    assert!(w.updates.len() >= 2, "text arrives incrementally");
    // A second turn on the same session works.
    let id = w.prompt_request(&sid, "other");
    w.response(id, T);
    assert!(w.message_text().ends_with("a fixed sentence"));
    w.close_stdin();
    assert!(w.wait_exit(T).success());
    let _ = std::fs::remove_dir_all(ws);
}

#[test]
fn cancel_mid_stream_ends_the_turn_as_cancelled_and_session_survives() {
    let (mut w, ws, sid) = start();
    let id = w.prompt_request(&sid, "endless");
    w.until_update(T, |u| u["sessionUpdate"] == "agent_message_chunk");
    w.notify("session/cancel", json!({"sessionId":sid}));
    let t = std::time::Instant::now();
    let r = w.response(id, T);
    assert_eq!(r["result"]["stopReason"], "cancelled", "{r}");
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
    // Nothing is streamed after the cancelled response.
    let n = w.updates.len();
    assert!(w.next(Duration::from_millis(200)).is_none());
    assert_eq!(w.updates.len(), n);
    // The session still answers.
    let id = w.prompt_request(&sid, "stream");
    assert_eq!(w.response(id, T)["result"]["stopReason"], "end_turn");
    w.close_stdin();
    assert!(w.wait_exit(T).success());
    let _ = std::fs::remove_dir_all(ws);
}

#[test]
fn unanswered_permission_request_is_denied_and_nothing_is_written() {
    let (mut w, ws, sid) = start();
    w.perm = Perm::Silent;
    let id = w.prompt_request(&sid, "write");
    let req = w.until_agent_request(T, "session/request_permission");
    // Shape the spec requires of a permission request.
    assert_eq!(req["params"]["sessionId"], sid.as_str());
    assert!(req["params"]["toolCall"]["toolCallId"].is_string(), "{req}");
    let opts = req["params"]["options"].as_array().expect("options");
    assert!(opts.iter().any(|o| o["kind"] == "allow_once"));
    assert!(opts.iter().any(|o| o["kind"] == "reject_once"));
    for o in opts {
        assert!(o["optionId"].is_string() && o["name"].is_string(), "{o}");
    }
    // We stay silent; the agent's timeout must deny, finish the turn, and
    // tell the model.
    let r = w.response(id, T);
    assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");
    assert!(
        w.message_text().contains("permission denied"),
        "{}",
        w.message_text()
    );
    assert!(
        !ws.join("notes.txt").exists(),
        "a silent client must not write"
    );
    w.close_stdin();
    assert!(w.wait_exit(T).success());
    let _ = std::fs::remove_dir_all(ws);
}

#[test]
fn rejecting_cancelling_or_erroring_the_permission_all_deny() {
    for perm in [Perm::RejectOnce, Perm::Cancelled, Perm::MethodNotFound] {
        let (mut w, ws, sid) = start();
        w.perm = perm;
        let id = w.prompt_request(&sid, "write");
        let r = w.response(id, T);
        assert_eq!(r["result"]["stopReason"], "end_turn", "{perm:?}: {r}");
        assert!(
            w.message_text().contains("permission denied"),
            "{perm:?}: {}",
            w.message_text()
        );
        assert!(!ws.join("notes.txt").exists(), "{perm:?} must not write");
        w.close_stdin();
        assert!(w.wait_exit(T).success());
        let _ = std::fs::remove_dir_all(ws);
    }
}

#[test]
fn explicit_allow_once_is_the_only_way_a_write_lands() {
    let (mut w, ws, sid) = start();
    w.perm = Perm::AllowOnce;
    let id = w.prompt_request(&sid, "write");
    w.response(id, T);
    assert_eq!(std::fs::read_to_string(ws.join("notes.txt")).unwrap(), "x");
    // Allow once: a second write asks again.
    let before = w.agent_requests.len();
    let id = w.prompt_request(&sid, "write");
    w.response(id, T);
    assert!(w.agent_requests.len() > before, "allow_once is not sticky");
    w.close_stdin();
    assert!(w.wait_exit(T).success());
    let _ = std::fs::remove_dir_all(ws);
}

#[test]
fn reads_inside_the_workspace_run_without_asking() {
    let (mut w, ws, sid) = start();
    std::fs::write(ws.join("hello.txt"), "hi there").unwrap();
    let id = w.prompt_request(&sid, "read");
    w.response(id, T);
    assert!(w.agent_requests.is_empty(), "reads must not prompt");
    assert!(
        w.message_text().contains("hi there"),
        "{}",
        w.message_text()
    );
    w.close_stdin();
    assert!(w.wait_exit(T).success());
    let _ = std::fs::remove_dir_all(ws);
}

#[test]
fn bad_input_gets_json_rpc_errors_and_the_stream_stays_aligned() {
    let (mut w, ws, _sid) = start();
    let id = w.request("no/such/method", json!({}));
    let r = w.response(id, T);
    assert_eq!(r["error"]["code"], -32601, "{r}");
    w.send_raw_line("{ this is not json");
    let e = w.next(T).expect("an error for the garbage line");
    assert!(e["error"].is_object(), "{e}");
    // Still aligned afterwards.
    let id = w.request("session/new", json!({"cwd": ws, "mcpServers": []}));
    assert!(w.response(id, T)["result"]["sessionId"].is_string());
    w.close_stdin();
    assert!(w.wait_exit(T).success());
    let _ = std::fs::remove_dir_all(ws);
}

#[test]
fn a_relative_or_missing_cwd_is_refused() {
    let mut w = Wire::spawn(agent_cmd());
    w.initialize();
    for cwd in ["relative/dir", "/nonexistent/tsacp/xyz", "/"] {
        let id = w.request("session/new", json!({"cwd":cwd,"mcpServers":[]}));
        let r = w.response(id, T);
        assert!(r["error"].is_object(), "{cwd}: {r}");
    }
    w.close_stdin();
    assert!(w.wait_exit(T).success());
}

#[test]
fn stdin_eof_mid_turn_cancels_and_exits() {
    let (mut w, ws, sid) = start();
    let _id = w.prompt_request(&sid, "endless");
    w.until_update(T, |u| u["sessionUpdate"] == "agent_message_chunk");
    w.close_stdin();
    let s = w.wait_exit(Duration::from_secs(10));
    assert!(s.success(), "{s:?}");
    let _ = std::fs::remove_dir_all(ws);
}
