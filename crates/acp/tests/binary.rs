//! The real `turbospark-acp` process over real pipes: stdout is protocol only,
//! EOF exits cleanly, and the unwired engine fails a turn honestly.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

#[test]
fn process_speaks_protocol_on_stdout_and_exits_on_eof() {
    let state = std::env::temp_dir().join(format!("tsacp-bin-{}", std::process::id()));
    let ws = std::env::temp_dir().join(format!("tsacp-bin-ws-{}", std::process::id()));
    std::fs::create_dir_all(&ws).unwrap();
    let ws = std::fs::canonicalize(&ws).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_turbospark-acp"))
        .args(["--state-dir", state.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut send = |v: Value| {
        writeln!(stdin, "{v}").unwrap();
        stdin.flush().unwrap();
    };
    let mut read = || -> Value {
        let mut l = String::new();
        out.read_line(&mut l).unwrap();
        serde_json::from_str(&l).unwrap_or_else(|_| panic!("non-JSON stdout line: {l:?}"))
    };

    send(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}));
    assert_eq!(read()["result"]["protocolVersion"], 1);
    send(
        json!({"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":ws,"mcpServers":[]}}),
    );
    let r = read();
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();
    send(json!({"jsonrpc":"2.0","id":3,"method":"session/prompt",
        "params":{"sessionId":sid,"prompt":[{"type":"text","text":"hi"}]}}));
    let r = read();
    assert_eq!(r["id"], 3);
    assert!(r["error"]["message"]
        .as_str()
        .unwrap()
        .contains("no local model"));
    drop(stdin);
    let status = child.wait().unwrap();
    assert!(status.success());
    let _ = std::fs::remove_dir_all(&state);
    let _ = std::fs::remove_dir_all(&ws);
}

#[test]
fn bad_flag_exits_2() {
    let s = Command::new(env!("CARGO_BIN_EXE_turbospark-acp"))
        .arg("--listen")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(s.status.code(), Some(2));
    assert!(s.stdout.is_empty());
}
