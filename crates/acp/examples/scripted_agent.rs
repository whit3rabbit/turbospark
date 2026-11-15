//! A scripted fake engine served over stdio, used only by the scripted-client
//! interop test (`tests/scripted_client.rs`). It is an example, not a bin, so
//! it is never installed or published as a command. It exists so the real
//! `serve` loop can be driven as a child process, off-device, with a model
//! whose behavior the test controls through the prompt text:
//!
//!   stream   emits ten words, 5 ms apart, then ends the turn
//!   endless  emits a word every 5 ms until the editor cancels
//!   write    asks for `write_file notes.txt` through the permission gate
//!   read     reads `hello.txt` (no prompt) and echoes its text
//!   (other)  one fixed sentence
//!
//! `TS_SCRIPTED_PERM_MS` shortens the permission timeout (default 300 ms here
//! so the silence-means-deny test does not wait 60 s).

use serde_json::json;
use std::io::BufReader;
use std::time::Duration;
use turbospark_acp::{serve, AcpConfig, Engine, Role, StopReason, ToolCall, TurnHost, TurnRequest};

struct Scripted;

impl Engine for Scripted {
    fn model_id(&self) -> String {
        "scripted".into()
    }
    fn run_turn(&self, req: &TurnRequest, host: &mut dyn TurnHost) -> Result<StopReason, String> {
        let prompt = req
            .history
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.text.clone())
            .unwrap_or_default();
        match prompt.trim() {
            "stream" => {
                for i in 0..10 {
                    if host.cancelled() {
                        break;
                    }
                    host.text_delta(&format!("word{i} "));
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            "endless" => {
                let mut i = 0u64;
                while !host.cancelled() {
                    host.text_delta(&format!("tok{i} "));
                    i += 1;
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            "write" => {
                let out = host.call_tool(ToolCall {
                    id: "t1".into(),
                    name: "write_file".into(),
                    args: json!({"path":"notes.txt","content":"x"}),
                });
                host.text_delta(&format!("write result: {}", out.text));
            }
            "read" => {
                let out = host.call_tool(ToolCall {
                    id: "t1".into(),
                    name: "read_file".into(),
                    args: json!({"path":"hello.txt"}),
                });
                host.text_delta(&format!("read result: {}", out.text));
            }
            _ => host.text_delta("a fixed sentence"),
        }
        Ok(StopReason::EndTurn)
    }
}

fn main() {
    let perm_ms = std::env::var("TS_SCRIPTED_PERM_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let cfg = AcpConfig {
        permission_timeout: Duration::from_millis(perm_ms),
        state_dir: None,
        ..AcpConfig::default()
    };
    let stdin = std::io::stdin();
    let code = match serve(
        &Scripted,
        &cfg,
        BufReader::new(stdin.lock()),
        std::io::stdout(),
    ) {
        Ok(()) => 0,
        Err(_) => 1,
    };
    std::process::exit(code);
}
