//! The `turbospark-acp-agent` process over real pipes (`turbospark acp`).
//!
//! The first three tests need no model and no GPU. The smoke test at the end
//! loads a real model and is `#[ignore]`d: it needs a Metal device, an
//! installed chat model named by `TURBOSPARK_ACP_SMOKE_MODEL`, and a quiet
//! host, so CI never runs it.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

struct Agent {
    child: Child,
    stdin: Option<ChildStdin>,
    out: BufReader<std::process::ChildStdout>,
    updates: Vec<Value>,
}

impl Agent {
    fn start(args: &[&str], home: &std::path::Path) -> Agent {
        let mut child = Command::new(env!("CARGO_BIN_EXE_turbospark-acp-agent"))
            .args(args)
            .env("TURBOSPARK_HOME", home)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let out = BufReader::new(child.stdout.take().unwrap());
        Agent {
            child,
            stdin,
            out,
            updates: vec![],
        }
    }

    fn send(&mut self, v: Value) {
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(stdin, "{v}").unwrap();
        stdin.flush().unwrap();
    }

    /// Reads until the response with `id`, keeping `session/update`s. Every
    /// stdout line must be JSON: stdout is the protocol channel.
    fn response(&mut self, id: i64) -> Value {
        loop {
            let mut l = String::new();
            assert!(
                self.out.read_line(&mut l).unwrap() > 0,
                "stdout closed early"
            );
            let v: Value =
                serde_json::from_str(&l).unwrap_or_else(|_| panic!("non-JSON stdout line: {l:?}"));
            if v["method"] == "session/update" {
                self.updates.push(v["params"]["update"].clone());
            } else if v["id"] == id {
                return v;
            }
        }
    }

    fn handshake(&mut self, ws: &std::path::Path) -> String {
        self.send(
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}),
        );
        assert_eq!(self.response(1)["result"]["protocolVersion"], 1);
        self.send(json!({"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":ws,"mcpServers":[]}}));
        self.response(2)["result"]["sessionId"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn prompt(&mut self, sid: &str, text: &str, id: i64) -> Value {
        self.send(json!({"jsonrpc":"2.0","id":id,"method":"session/prompt",
            "params":{"sessionId":sid,"prompt":[{"type":"text","text":text}]}}));
        self.response(id)
    }

    fn finish(mut self) -> std::process::ExitStatus {
        drop(self.stdin.take());
        self.child.wait().unwrap()
    }
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("tsacp-agent-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    std::fs::canonicalize(&p).unwrap()
}

#[test]
fn model_flag_is_required_and_stdout_stays_empty() {
    let o = Command::new(env!("CARGO_BIN_EXE_turbospark-acp-agent"))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    assert!(o.stdout.is_empty());
    let o = Command::new(env!("CARGO_BIN_EXE_turbospark-acp-agent"))
        .args(["--model", "m", "--listen"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    assert!(o.stdout.is_empty());
}

#[test]
fn unloadable_model_fails_the_turn_honestly_and_eof_exits_clean() {
    let home = tmp("home");
    let ws = tmp("ws");
    let state = home.join("state");
    let mut a = Agent::start(
        &[
            "--model",
            "/nonexistent/model.gturbo",
            "--state-dir",
            state.to_str().unwrap(),
        ],
        &home,
    );
    // The handshake does not touch the model: it must answer without a load.
    let sid = a.handshake(&ws);
    let r = a.prompt(&sid, "hi", 3);
    assert!(
        r["error"].is_object(),
        "a model that cannot load is a turn error: {r}"
    );
    assert!(a
        .updates
        .iter()
        .all(|u| u["sessionUpdate"] != "agent_message_chunk"));
    assert!(a.finish().success());
    let _ = std::fs::remove_dir_all(&home);
    let _ = std::fs::remove_dir_all(&ws);
}

#[test]
fn front_door_lists_acp() {
    let o = Command::new(env!("CARGO_BIN_EXE_turbospark"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&o.stdout).contains("acp"));
}

/// Real model, real Metal. Run locally on a quiet host:
///   TURBOSPARK_ACP_SMOKE_MODEL=<alias or path> cargo test -p turbospark-cli \
///     --test acp_agent -- --ignored --nocapture
#[test]
#[ignore = "needs a Metal device and an installed chat model (TURBOSPARK_ACP_SMOKE_MODEL)"]
fn real_model_streams_a_reply_and_honors_cancel() {
    let model = std::env::var("TURBOSPARK_ACP_SMOKE_MODEL")
        .expect("set TURBOSPARK_ACP_SMOKE_MODEL to an installed chat model");
    let ws = tmp("smoke-ws");
    // The real TURBOSPARK_HOME (the model store), not a scratch one; only the
    // session records go to a scratch dir.
    let home = std::env::var_os("TURBOSPARK_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".turbospark"))
        })
        .unwrap();
    let state = tmp("smoke-state");
    let mut a = Agent::start(
        &["--model", &model, "--state-dir", state.to_str().unwrap()],
        &home,
    );
    let sid = a.handshake(&ws);
    let r = a.prompt(&sid, "Reply with the single word: pong", 3);
    assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");
    let text: String = a
        .updates
        .iter()
        .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
        .map(|u| u["content"]["text"].as_str().unwrap_or(""))
        .collect();
    assert!(!text.trim().is_empty(), "no streamed text");
    eprintln!("streamed {} chunks: {text:?}", a.updates.len());

    // Cancel mid-turn: ask for a long answer, cancel after the first chunk.
    let before = a.updates.len();
    a.send(json!({"jsonrpc":"2.0","id":4,"method":"session/prompt",
        "params":{"sessionId":sid,"prompt":[{"type":"text","text":"Count from 1 to 500, one number per line."}]}}));
    loop {
        let mut l = String::new();
        a.out.read_line(&mut l).unwrap();
        let v: Value = serde_json::from_str(&l).unwrap();
        if v["method"] == "session/update" {
            a.updates.push(v["params"]["update"].clone());
            break;
        }
    }
    a.send(json!({"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":sid}}));
    let r = a.response(4);
    assert_eq!(r["result"]["stopReason"], "cancelled", "{r}");
    assert!(a.updates.len() > before);
    assert!(a.finish().success());
    let _ = std::fs::remove_dir_all(&ws);
    let _ = std::fs::remove_dir_all(&state);
}
