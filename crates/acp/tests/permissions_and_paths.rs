//! Permission gate, default-deny, workspace confinement, editor fs bridge.

mod common;
use common::*;
use serde_json::{json, Value};
use std::time::Duration;
use turbospark_acp::{StopReason, ToolCall};

/// Engine that performs one tool call per prompt, named by the prompt text
/// ("write", "read", "edit", "list", or a raw path for write), and reports the
/// tool outcome as assistant text so the test can see what the model saw.
fn tool_engine() -> impl Fn(
    &turbospark_acp::TurnRequest,
    &mut dyn turbospark_acp::TurnHost,
) -> Result<StopReason, String>
       + Send
       + Sync {
    |req, h| {
        let last = req.history.last().unwrap().text.clone();
        let (name, args) = match last.as_str() {
            "write" => ("write_file", json!({"path":"out.txt","content":"data"})),
            "write2" => ("write_file", json!({"path":"out2.txt","content":"data2"})),
            "read" => ("read_file", json!({"path":"in.txt"})),
            "edit" => (
                "edit_file",
                json!({"path":"in.txt","old_text":"hello","new_text":"bye"}),
            ),
            "list" => ("list_dir", json!({"path":"."})),
            other => ("write_file", json!({"path": other, "content":"pwned"})),
        };
        let o = h.call_tool(ToolCall {
            id: "call1".into(),
            name: name.into(),
            args,
        });
        h.text_delta(&format!(
            "{}|{}",
            if o.is_error { "ERR" } else { "OK" },
            o.text
        ));
        Ok(StopReason::EndTurn)
    }
}

fn tool_result_text(c: &mut Client) -> String {
    c.updates()
        .iter()
        .rev()
        .find(|u| u["sessionUpdate"] == "agent_message_chunk")
        .map(|u| u["content"]["text"].as_str().unwrap().to_string())
        .expect("no assistant text")
}

fn select(c: &Client, req: &Value, option: &str) {
    c.reply(
        &req["id"],
        json!({"outcome":{"outcome":"selected","optionId":option}}),
    );
}

#[test]
fn write_asks_the_editor_and_allow_once_executes() {
    let t = Tmp::new("allow1");
    let ws = t.ws();
    let mut c = start(cfg_in(&t.0), tool_engine());
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "write");
    let req = c.expect_request("session/request_permission");
    assert_eq!(req["params"]["sessionId"], sid);
    assert_eq!(req["params"]["toolCall"]["kind"], "edit");
    assert_eq!(req["params"]["toolCall"]["toolCallId"], "call1");
    let kinds: Vec<&str> = req["params"]["options"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["kind"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        ["allow_once", "allow_always", "reject_once", "reject_always"]
    );
    // Nothing has run while the question is open.
    assert!(!ws.join("out.txt").exists());
    select(&c, &req, "allow_once");
    c.response(id);
    assert_eq!(std::fs::read_to_string(ws.join("out.txt")).unwrap(), "data");
    assert!(tool_result_text(&mut c).starts_with("OK|"));
    let statuses: Vec<String> = c
        .updates()
        .iter()
        .filter(|u| u["toolCallId"] == "call1")
        .map(|u| u["status"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(statuses, ["pending", "completed"]);
    // allow_once is not remembered: a second write asks again.
    let id2 = c.prompt(&sid, "write2");
    let req2 = c.expect_request("session/request_permission");
    select(&c, &req2, "reject_once");
    c.response(id2);
    assert!(!ws.join("out2.txt").exists());
    c.close().unwrap();
}

#[test]
fn allow_always_is_reused_for_the_session_only() {
    let t = Tmp::new("always");
    let ws = t.ws();
    let mut c = start(cfg_in(&t.0), tool_engine());
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "write");
    let req = c.expect_request("session/request_permission");
    select(&c, &req, "allow_always");
    c.response(id);
    let id2 = c.prompt(&sid, "write2");
    assert!(c.no_request("session/request_permission", Duration::from_millis(300)));
    c.response(id2);
    assert!(ws.join("out2.txt").exists());
    // A different session starts with no grant.
    let sid2 = c.new_session(ws.to_str().unwrap());
    let id3 = c.prompt(&sid2, "write");
    let req3 = c.expect_request("session/request_permission");
    select(&c, &req3, "reject_once");
    c.response(id3);
    c.close().unwrap();
}

#[test]
fn reject_always_blocks_later_writes_without_asking() {
    let t = Tmp::new("rejalways");
    let ws = t.ws();
    let mut c = start(cfg_in(&t.0), tool_engine());
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "write");
    let req = c.expect_request("session/request_permission");
    select(&c, &req, "reject_always");
    c.response(id);
    let id2 = c.prompt(&sid, "write2");
    assert!(c.no_request("session/request_permission", Duration::from_millis(300)));
    c.response(id2);
    assert!(!ws.join("out.txt").exists());
    assert!(!ws.join("out2.txt").exists());
    let text = tool_result_text(&mut c);
    assert!(text.starts_with("ERR|permission denied: 'edit'"), "{text}");
    c.close().unwrap();
}

/// CUJ 3 and the default-deny mutation target: a client that never answers.
#[test]
fn headless_client_that_never_answers_is_denied_by_timeout() {
    let t = Tmp::new("headless");
    let ws = t.ws();
    let mut cfg = cfg_in(&t.0);
    cfg.permission_timeout = Duration::from_millis(300);
    let mut c = start(cfg, tool_engine());
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "write");
    // The request is sent; the client says nothing.
    let _req = c.expect_request("session/request_permission");
    let r = c.response(id);
    assert_eq!(r["result"]["stopReason"], "end_turn");
    assert!(!ws.join("out.txt").exists(), "write ran without approval");
    let text = tool_result_text(&mut c);
    assert!(text.starts_with("ERR|permission denied: 'edit'"), "{text}");
    assert!(text.contains("Nothing was executed"));
    c.close().unwrap();
}

#[test]
fn every_non_approval_answer_is_a_denial() {
    let answers: Vec<(&str, Value)> = vec![
        (
            "cancelled outcome",
            json!({"outcome":{"outcome":"cancelled"}}),
        ),
        (
            "unknown option id",
            json!({"outcome":{"outcome":"selected","optionId":"allow_forever"}}),
        ),
        ("missing outcome", json!({})),
        ("garbage", json!("yes")),
        (
            "approval shaped wrong",
            json!({"outcome":"selected","optionId":"allow_once"}),
        ),
    ];
    for (label, answer) in answers {
        let t = Tmp::new("deny");
        let ws = t.ws();
        let mut c = start(cfg_in(&t.0), tool_engine());
        c.initialize(false, false);
        let sid = c.new_session(ws.to_str().unwrap());
        let id = c.prompt(&sid, "write");
        let req = c.expect_request("session/request_permission");
        c.reply(&req["id"], answer);
        c.response(id);
        assert!(!ws.join("out.txt").exists(), "{label}: write ran");
        c.close().unwrap();
    }
}

#[test]
fn a_client_that_rejects_the_method_is_a_denial() {
    let t = Tmp::new("nomethod");
    let ws = t.ws();
    let mut c = start(cfg_in(&t.0), tool_engine());
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "write");
    let req = c.expect_request("session/request_permission");
    c.reply_err(&req["id"], -32601, "Method not found");
    c.response(id);
    assert!(!ws.join("out.txt").exists());
    assert!(tool_result_text(&mut c).contains("does not support"));
    c.close().unwrap();
}

#[test]
fn cancel_during_a_pending_permission_denies_and_reports_cancelled() {
    let t = Tmp::new("cancelperm");
    let ws = t.ws();
    let mut c = start(cfg_in(&t.0), tool_engine());
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "write");
    let _req = c.expect_request("session/request_permission");
    c.notify("session/cancel", json!({"sessionId": sid}));
    let r = c.response(id);
    assert_eq!(r["result"]["stopReason"], "cancelled");
    assert!(!ws.join("out.txt").exists());
    c.close().unwrap();
}

#[test]
fn stdin_eof_during_a_pending_permission_exits_without_writing() {
    let t = Tmp::new("eofperm");
    let ws = t.ws();
    let mut c = start(cfg_in(&t.0), tool_engine());
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    c.prompt(&sid, "write");
    let _req = c.expect_request("session/request_permission");
    c.close().unwrap();
    assert!(!ws.join("out.txt").exists());
}

#[test]
fn reads_do_not_prompt_and_edits_do() {
    let t = Tmp::new("readedit");
    let ws = t.ws();
    std::fs::write(ws.join("in.txt"), "hello world").unwrap();
    let mut c = start(cfg_in(&t.0), tool_engine());
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "read");
    assert!(c.no_request("session/request_permission", Duration::from_millis(200)));
    c.response(id);
    assert_eq!(tool_result_text(&mut c), "OK|hello world");
    let id = c.prompt(&sid, "list");
    c.response(id);
    assert!(tool_result_text(&mut c).contains("in.txt"));
    let id = c.prompt(&sid, "edit");
    let req = c.expect_request("session/request_permission");
    assert_eq!(req["params"]["toolCall"]["kind"], "edit");
    select(&c, &req, "allow_once");
    c.response(id);
    assert_eq!(
        std::fs::read_to_string(ws.join("in.txt")).unwrap(),
        "bye world"
    );
    c.close().unwrap();
}

#[test]
fn paths_outside_the_root_are_refused_before_any_prompt() {
    let t = Tmp::new("escape");
    let ws = t.ws();
    let outside = t.0.join("outside.txt");
    let mut c = start(cfg_in(&t.0), tool_engine());
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    for raw in [
        "../outside.txt".to_string(),
        outside.to_string_lossy().into_owned(),
        "sub/../../outside.txt".to_string(),
    ] {
        let id = c.prompt(&sid, &raw);
        assert!(
            c.no_request("session/request_permission", Duration::from_millis(150)),
            "asked permission for escaping path {raw}"
        );
        c.response(id);
        assert!(!outside.exists(), "wrote outside the root via {raw}");
        assert!(tool_result_text(&mut c).contains("outside the workspace root"));
    }
    c.close().unwrap();
}

#[cfg(unix)]
#[test]
fn symlink_that_leaves_the_root_is_refused_for_reads_and_writes() {
    let t = Tmp::new("symlink");
    let ws = t.ws();
    let outside = t.0.join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("secret.txt"), "secret").unwrap();
    std::os::unix::fs::symlink(&outside, ws.join("link")).unwrap();
    let mut c = start(cfg_in(&t.0), |req, h| {
        let last = req.history.last().unwrap().text.clone();
        let (name, args) = if last == "r" {
            ("read_file", json!({"path":"link/secret.txt"}))
        } else {
            ("write_file", json!({"path":"link/new.txt","content":"x"}))
        };
        let o = h.call_tool(ToolCall {
            id: "c".into(),
            name: name.into(),
            args,
        });
        h.text_delta(&format!("{}|{}", o.is_error, o.text));
        Ok(StopReason::EndTurn)
    });
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "r");
    c.response(id);
    assert!(tool_result_text(&mut c).starts_with("true|refused"));
    let id = c.prompt(&sid, "w");
    assert!(c.no_request("session/request_permission", Duration::from_millis(150)));
    c.response(id);
    assert!(!outside.join("new.txt").exists());
    c.close().unwrap();
}

#[test]
fn unknown_tool_names_do_not_exist() {
    let t = Tmp::new("unknowntool");
    let mut c = start(cfg_in(&t.0), |_, h| {
        let o = h.call_tool(ToolCall {
            id: "c".into(),
            name: "run_shell".into(),
            args: json!({"path":".","command":"id"}),
        });
        h.text_delta(&format!("{}|{}", o.is_error, o.text));
        Ok(StopReason::EndTurn)
    });
    c.initialize(false, false);
    let sid = c.new_session(t.ws().to_str().unwrap());
    let id = c.prompt(&sid, "x");
    c.response(id);
    assert!(tool_result_text(&mut c).starts_with("true|unknown tool"));
    c.close().unwrap();
}

#[test]
fn editor_fs_bridge_is_used_when_advertised() {
    let t = Tmp::new("fsbridge");
    let ws = t.ws();
    // On disk the file says "disk"; the editor buffer says "buffer".
    std::fs::write(ws.join("in.txt"), "disk").unwrap();
    let mut c = start(cfg_in(&t.0), tool_engine());
    c.initialize(true, true);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "read");
    let rd = c.expect_request("fs/read_text_file");
    assert_eq!(rd["params"]["sessionId"], sid);
    assert_eq!(
        rd["params"]["path"],
        ws.join("in.txt").to_string_lossy().as_ref()
    );
    c.reply(&rd["id"], json!({"content":"buffer"}));
    c.response(id);
    assert_eq!(tool_result_text(&mut c), "OK|buffer");
    // A write goes through permission, then to the editor, not the disk.
    let id = c.prompt(&sid, "write");
    let perm = c.expect_request("session/request_permission");
    select(&c, &perm, "allow_once");
    let wr = c.expect_request("fs/write_text_file");
    assert_eq!(wr["params"]["content"], "data");
    c.reply(&wr["id"], json!(null));
    c.response(id);
    assert!(
        !ws.join("out.txt").exists(),
        "wrote to disk instead of the editor"
    );
    c.close().unwrap();
}

#[test]
fn editor_fs_failure_fails_the_tool_instead_of_falling_back_to_disk() {
    let t = Tmp::new("fsfail");
    let ws = t.ws();
    std::fs::write(ws.join("in.txt"), "disk").unwrap();
    let mut c = start(cfg_in(&t.0), tool_engine());
    c.initialize(true, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "read");
    let rd = c.expect_request("fs/read_text_file");
    c.reply_err(&rd["id"], -32603, "nope");
    c.response(id);
    assert!(tool_result_text(&mut c).starts_with("ERR|"));
    c.close().unwrap();
}

#[test]
fn tool_output_is_bounded() {
    let t = Tmp::new("bound");
    let ws = t.ws();
    std::fs::write(ws.join("in.txt"), "x".repeat(10_000)).unwrap();
    let mut cfg = cfg_in(&t.0);
    cfg.limits.max_output_bytes = 100;
    let mut c = start(cfg, tool_engine());
    c.initialize(false, false);
    let sid = c.new_session(ws.to_str().unwrap());
    let id = c.prompt(&sid, "read");
    c.response(id);
    let text = tool_result_text(&mut c);
    assert!(text.len() < 200, "unbounded read: {} bytes", text.len());
    assert!(text.contains("[truncated at 100 bytes]"));
    c.close().unwrap();
}
