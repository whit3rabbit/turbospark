//! Protocol and session behavior against a fake engine. No model involved.

mod common;
use common::*;
use serde_json::{json, Value};
use std::time::Duration;
use turbospark_acp::{StopReason, ToolCall};

fn end_turn() -> impl Fn(
    &turbospark_acp::TurnRequest,
    &mut dyn turbospark_acp::TurnHost,
) -> Result<StopReason, String>
       + Send
       + Sync {
    |_, h| {
        h.text_delta("hi");
        Ok(StopReason::EndTurn)
    }
}

#[test]
fn initialize_negotiates_and_session_new_requires_it() {
    let t = Tmp::new("init");
    let mut c = start(cfg_in(&t.0), end_turn());
    // Before initialize.
    let id = c.request("session/new", json!({"cwd": t.ws()}));
    assert_eq!(c.response(id)["error"]["code"], -32002);
    let r = c.initialize(false, false);
    assert_eq!(r["result"]["protocolVersion"], 1);
    assert_eq!(r["result"]["agentCapabilities"]["loadSession"], false);
    assert_eq!(r["result"]["authMethods"], json!([]));
    // Authentication is not offered.
    let id = c.request("authenticate", json!({"methodId": "x"}));
    assert_eq!(c.response(id)["error"]["code"], -32601);
    // Unknown method.
    let id = c.request("terminal/create", json!({}));
    assert_eq!(c.response(id)["error"]["code"], -32601);
    c.close().unwrap();
}

#[test]
fn session_new_validates_the_workspace_root() {
    let t = Tmp::new("root");
    let mut c = start(cfg_in(&t.0), end_turn());
    c.initialize(false, false);
    for bad in ["relative/dir", "/", "", "/definitely/not/here/xyz"] {
        let id = c.request("session/new", json!({"cwd": bad}));
        assert_eq!(c.response(id)["error"]["code"], -32602, "cwd {bad:?}");
    }
    let id = c.request("session/new", json!({}));
    assert_eq!(c.response(id)["error"]["code"], -32602);
    let sid = c.new_session(t.ws().to_str().unwrap());
    assert!(sid.starts_with("sess_"));
    c.close().unwrap();
}

#[test]
fn allow_list_confines_session_roots() {
    let t = Tmp::new("allow");
    let mut cfg = cfg_in(&t.0);
    let ok = t.0.join("ok");
    let other = t.0.join("other");
    std::fs::create_dir_all(&ok).unwrap();
    std::fs::create_dir_all(&other).unwrap();
    cfg.allowed_roots = vec![ok.clone()];
    let mut c = start(cfg, end_turn());
    c.initialize(false, false);
    let id = c.request("session/new", json!({"cwd": other}));
    assert_eq!(c.response(id)["error"]["code"], -32602);
    c.new_session(ok.to_str().unwrap());
    c.close().unwrap();
}

#[test]
fn prompt_streams_deltas_in_order_and_persists_the_record() {
    let t = Tmp::new("stream");
    let state = t.0.join("state");
    let mut c = start(cfg_in(&t.0), |_, h| {
        for part in ["al", "pha", "!"] {
            h.text_delta(part);
        }
        Ok(StopReason::EndTurn)
    });
    c.initialize(false, false);
    let sid = c.new_session(t.ws().to_str().unwrap());
    let id = c.prompt(&sid, "hello");
    let r = c.response(id);
    assert_eq!(r["result"]["stopReason"], "end_turn");
    let chunks: Vec<String> = c
        .updates()
        .iter()
        .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
        .map(|u| u["content"]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(chunks, ["al", "pha", "!"]);
    c.close().unwrap();
    let rec: Value =
        serde_json::from_slice(&std::fs::read(state.join(format!("{sid}.json"))).unwrap()).unwrap();
    assert_eq!(rec["history"][0]["role"], "user");
    assert_eq!(rec["history"][0]["text"], "hello");
    assert_eq!(rec["history"][1]["role"], "assistant");
    assert_eq!(rec["history"][1]["text"], "alpha!");
}

#[test]
fn history_carries_across_turns() {
    let t = Tmp::new("hist");
    let mut c = start(cfg_in(&t.0), |req, h| {
        h.text_delta(&format!("n={}", req.history.len()));
        Ok(StopReason::EndTurn)
    });
    c.initialize(false, false);
    let sid = c.new_session(t.ws().to_str().unwrap());
    let a = c.prompt(&sid, "one");
    c.response(a);
    let b = c.prompt(&sid, "two");
    c.response(b);
    let texts: Vec<String> = c
        .updates()
        .iter()
        .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
        .map(|u| u["content"]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(texts, ["n=1", "n=3"]);
    c.close().unwrap();
}

#[test]
fn bad_prompts_are_refused() {
    let t = Tmp::new("badp");
    let mut c = start(cfg_in(&t.0), end_turn());
    c.initialize(false, false);
    let sid = c.new_session(t.ws().to_str().unwrap());
    let id = c.request(
        "session/prompt",
        json!({"sessionId": sid, "prompt":[{"type":"image","data":"AA==","mimeType":"image/png"}]}),
    );
    assert_eq!(c.response(id)["error"]["code"], -32602);
    let id = c.request(
        "session/prompt",
        json!({"sessionId":"sess_nope","prompt":[]}),
    );
    assert_eq!(c.response(id)["error"]["code"], -32602);
    c.close().unwrap();
}

#[test]
fn engine_error_and_panic_become_rpc_errors_and_the_session_survives() {
    let t = Tmp::new("eerr");
    let mut c = start(cfg_in(&t.0), |req, _| {
        match req.history.last().unwrap().text.as_str() {
            "panic" => panic!("boom"),
            _ => Err("model failed".to_string()),
        }
    });
    c.initialize(false, false);
    let sid = c.new_session(t.ws().to_str().unwrap());
    let a = c.prompt(&sid, "x");
    let r = c.response(a);
    assert_eq!(r["error"]["code"], -32603);
    assert_eq!(r["error"]["message"], "model failed");
    let b = c.prompt(&sid, "panic");
    let r = c.response(b);
    assert_eq!(r["error"]["code"], -32603);
    // Busy flag was released both times.
    let d = c.prompt(&sid, "again");
    assert_eq!(c.response(d)["error"]["code"], -32603);
    c.close().unwrap();
}

#[test]
fn second_prompt_while_running_is_busy() {
    let t = Tmp::new("busy");
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let release_rx = std::sync::Mutex::new(release_rx);
    let mut c = start(cfg_in(&t.0), move |_, h| {
        let _ = release_rx
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5));
        h.text_delta("done");
        Ok(StopReason::EndTurn)
    });
    c.initialize(false, false);
    let sid = c.new_session(t.ws().to_str().unwrap());
    let first = c.prompt(&sid, "a");
    // Let the worker start before the second prompt arrives.
    std::thread::sleep(Duration::from_millis(150));
    let second = c.prompt(&sid, "b");
    assert_eq!(c.response(second)["error"]["code"], -32000);
    release_tx.send(()).unwrap();
    assert_eq!(c.response(first)["result"]["stopReason"], "end_turn");
    c.close().unwrap();
}

#[test]
fn cancel_stops_the_turn_with_stop_reason_cancelled() {
    let t = Tmp::new("cancel");
    let mut c = start(cfg_in(&t.0), |_, h| {
        h.text_delta("start");
        let end = std::time::Instant::now() + Duration::from_secs(8);
        while !h.cancelled() && std::time::Instant::now() < end {
            std::thread::sleep(Duration::from_millis(10));
        }
        // An engine that wrongly reports an error after cancel must still
        // surface as `cancelled`.
        Err("interrupted".to_string())
    });
    c.initialize(false, false);
    let sid = c.new_session(t.ws().to_str().unwrap());
    let id = c.prompt(&sid, "go");
    std::thread::sleep(Duration::from_millis(150));
    c.notify("session/cancel", json!({"sessionId": sid}));
    let r = c.response(id);
    assert_eq!(r["result"]["stopReason"], "cancelled");
    // The session is usable afterwards and the cancel flag was reset.
    let id2 = c.prompt(&sid, "again");
    c.notify("session/cancel", json!({"sessionId": sid}));
    assert_eq!(c.response(id2)["result"]["stopReason"], "cancelled");
    c.close().unwrap();
}

#[test]
fn text_after_cancel_is_not_streamed() {
    let t = Tmp::new("cancel2");
    let mut c = start(cfg_in(&t.0), |_, h| {
        while !h.cancelled() {
            std::thread::sleep(Duration::from_millis(10));
        }
        h.text_delta("late");
        Ok(StopReason::EndTurn)
    });
    c.initialize(false, false);
    let sid = c.new_session(t.ws().to_str().unwrap());
    let id = c.prompt(&sid, "go");
    std::thread::sleep(Duration::from_millis(100));
    c.notify("session/cancel", json!({"sessionId": sid}));
    c.response(id);
    assert!(c
        .updates()
        .iter()
        .all(|u| u["sessionUpdate"] != "agent_message_chunk"));
    c.close().unwrap();
}

#[test]
fn stdin_eof_cancels_a_running_turn_and_exits() {
    let t = Tmp::new("eof");
    let mut c = start(cfg_in(&t.0), |_, h| {
        let end = std::time::Instant::now() + Duration::from_secs(30);
        while !h.cancelled() && std::time::Instant::now() < end {
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(StopReason::EndTurn)
    });
    c.initialize(false, false);
    let sid = c.new_session(t.ws().to_str().unwrap());
    c.prompt(&sid, "go");
    std::thread::sleep(Duration::from_millis(100));
    // close() panics if the adapter does not exit within WAIT (10 s), well
    // under the engine's 30 s loop: EOF must be what stopped it.
    c.close().unwrap();
}

#[test]
fn malformed_and_oversized_frames_do_not_break_the_stream() {
    let t = Tmp::new("frames");
    let mut cfg = cfg_in(&t.0);
    cfg.max_frame_bytes = 512;
    let mut c = start(cfg, end_turn());
    c.send_raw(b"this is not json\n");
    let m = c.next(WAIT).unwrap();
    assert_eq!(m["error"]["code"], -32700);
    assert!(m["id"].is_null());
    c.send_raw(b"[1,2,3]\n");
    assert_eq!(c.next(WAIT).unwrap()["error"]["code"], -32600);
    let mut big = vec![b'a'; 100_000];
    big.push(b'\n');
    c.send_raw(&big);
    let m = c.next(WAIT).unwrap();
    assert_eq!(m["error"]["code"], -32600);
    assert!(m["error"]["message"].as_str().unwrap().contains("512"));
    // Still aligned: a normal request works.
    let r = c.initialize(false, false);
    assert_eq!(r["result"]["protocolVersion"], 1);
    c.close().unwrap();
}

#[test]
fn engine_tool_call_limit_is_enforced() {
    let t = Tmp::new("limit");
    let mut cfg = cfg_in(&t.0);
    cfg.max_tool_calls_per_turn = 2;
    let ws = t.ws();
    std::fs::write(ws.join("f"), "x").unwrap();
    let mut c = start(cfg, |_, h| {
        let mut errs = 0;
        for i in 0..4 {
            let o = h.call_tool(ToolCall {
                id: format!("t{i}"),
                name: "read_file".into(),
                args: json!({"path": "f"}),
            });
            if o.is_error {
                errs += 1;
            }
        }
        h.text_delta(&format!("errs={errs}"));
        Ok(StopReason::EndTurn)
    });
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "go");
    c.response(id);
    let texts: Vec<_> = c
        .updates()
        .iter()
        .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
        .map(|u| u["content"]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(texts, ["errs=2"]);
    c.close().unwrap();
}
