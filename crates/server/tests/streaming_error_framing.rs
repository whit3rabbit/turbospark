//! F18: a failed generation on a streaming endpoint must send a FRAMED
//! `event: error` SSE event, and never the `[DONE]` sentinel after it --
//! `[DONE]` is the sentinel that says the stream finished normally, and a
//! bare `data:` line with no `event:` line is indistinguishable from a
//! chunk a client does not recognise.

use std::path::PathBuf;
use std::sync::Arc;

use runtime::{LogitProducer, RawDecodeResult, RuntimeError};
use tokenizer::MfTokenizer;
use turbospark_server::{build_router, ChatModel};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

/// Fails on the very first call, prefill included, so the whole generation
/// fails immediately rather than after any content has streamed.
struct FailingProducer;

impl LogitProducer for FailingProducer {
    fn reset(&mut self) {}

    fn produce(
        &mut self,
        _token: i32,
        _position: usize,
        _logits: &mut [foundation::LogitValue],
    ) -> Result<(), String> {
        Err("boom".to_string())
    }
}

struct FailingModel {
    tokenizer: MfTokenizer,
}

impl ChatModel for FailingModel {
    fn tokenizer(&self) -> &MfTokenizer {
        &self.tokenizer
    }
    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size
    }
    fn max_context(&self) -> u32 {
        4096
    }
    fn model_id(&self) -> &str {
        "failing"
    }
    fn with_producer(
        &self,
        f: &mut dyn FnMut(&mut dyn LogitProducer) -> Result<RawDecodeResult, RuntimeError>,
    ) -> Result<RawDecodeResult, RuntimeError> {
        let mut producer = FailingProducer;
        f(&mut producer)
    }
}

async fn serve() -> String {
    let model: Arc<dyn ChatModel> = Arc::new(FailingModel {
        tokenizer: load_tokenizer(),
    });
    let router = build_router(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

fn event_names(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|l| l.strip_prefix("event:"))
        .map(|n| n.trim().to_string())
        .collect()
}

#[tokio::test]
async fn a_failed_chat_completions_stream_sends_a_framed_error_and_no_done() {
    let base = serve().await;
    let body = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m", "max_tokens": 10, "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert_eq!(event_names(&body), vec!["error"], "{body}");
    assert!(body.contains("boom"), "{body}");
    assert!(!body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn a_failed_completions_stream_sends_a_framed_error_and_no_done() {
    let base = serve().await;
    let body = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "m", "prompt": "hi", "max_tokens": 10, "stream": true
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert_eq!(event_names(&body), vec!["error"], "{body}");
    assert!(body.contains("boom"), "{body}");
    assert!(!body.contains("[DONE]"), "{body}");
}

/// The BUFFERED path (tools present, guardrails on by default) shares the
/// same fix in a separate code path -- see `crates/server/CLAUDE.md`
/// Gotcha 18.
#[tokio::test]
async fn a_failed_buffered_tool_stream_sends_a_framed_error_and_no_done() {
    let base = serve().await;
    let body = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m", "max_tokens": 10, "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}
            }]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    assert_eq!(event_names(&body), vec!["error"], "{body}");
    assert!(!body.contains("[DONE]"), "{body}");
}
