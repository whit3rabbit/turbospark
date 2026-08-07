//! End-to-end tests for the Anthropic `POST /v1/messages` endpoint, the
//! `GET /v1/models` route, and the OpenAI request shapes the switch to
//! `anyllm_translate`'s wire types newly accepts.
//!
//! Same harness as `chat_completions.rs`: a real axum server on an ephemeral
//! loopback port, real tokenizer, scripted "model". Nothing here asserts on
//! generated TEXT (the scripted producer's output is meaningless); the
//! assertions are on envelope shape, event structure, and stop reason.

use std::path::PathBuf;
use std::sync::Arc;

use mrefrust_server::{build_router, ScriptedChatModel};
use tokenizer::MfTokenizer;

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

async fn spawn_server(steps: Vec<Vec<foundation::LogitValue>>) -> String {
    let tok = load_tokenizer();
    let model: Arc<dyn mrefrust_server::ChatModel> =
        Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let router = build_router(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

fn h_steps(tok: &MfTokenizer, count: usize) -> Vec<Vec<foundation::LogitValue>> {
    let h_id = tok.token_to_id("h").unwrap() as usize;
    (0..count).map(|_| one_hot(tok.vocab_size, h_id)).collect()
}

/// The `event:` name of each SSE frame, in order. Anthropic clients dispatch
/// on these, so their order is the contract being tested.
fn event_names(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|l| l.strip_prefix("event:"))
        .map(|n| n.trim().to_string())
        .collect()
}

#[tokio::test]
async fn non_streaming_messages_returns_an_anthropic_envelope() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{base}/v1/messages"))
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 3,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["content"][0]["type"], "text");
    // Capped by max_tokens, not stopped by the model.
    assert_eq!(body["stop_reason"], "max_tokens");
    assert_eq!(body["usage"]["output_tokens"], 3);
    // The model name the CLIENT asked for comes back, not the backend's.
    assert_eq!(body["model"], "claude-sonnet-4-6");
}

#[tokio::test]
async fn a_system_prompt_is_accepted() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 80)).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{base}/v1/messages"))
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 2,
            "temperature": 0.0,
            "system": "You are terse.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["type"], "message");
}

#[tokio::test]
async fn streaming_messages_emits_anthropic_events_in_order() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{base}/v1/messages"))
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();
    let names = event_names(&body);

    assert_eq!(names.first().map(String::as_str), Some("message_start"));
    assert_eq!(names.last().map(String::as_str), Some("message_stop"));
    assert!(names.iter().any(|n| n == "content_block_start"));
    assert!(names.iter().any(|n| n == "content_block_delta"));
    assert!(names.iter().any(|n| n == "content_block_stop"));
    // `[DONE]` is an OpenAI sentinel; an Anthropic stream ends at message_stop.
    assert!(!body.contains("[DONE]"));
}

#[tokio::test]
async fn messages_rejects_a_malformed_body() {
    let base = spawn_server(Vec::new()).await;
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{base}/v1/messages"))
        .header("content-type", "application/json")
        .body("not json")
        .send()
        .await
        .unwrap();
    assert!(response.status().is_client_error());
}

#[tokio::test]
async fn models_lists_the_loaded_backend() {
    let base = spawn_server(Vec::new()).await;
    let client = reqwest::Client::new();
    let response = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "scripted");
    assert_eq!(body["data"][0]["object"], "model");
}

/// The three OpenAI request shapes the hand-rolled envelope used to reject:
/// `stop` as a bare string, `max_completion_tokens` instead of `max_tokens`,
/// and `content` as an array of parts.
#[tokio::test]
async fn chat_completions_accepts_the_wider_openai_request_shapes() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 80)).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "scripted",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hi"}]
            }],
            "max_completion_tokens": 3,
            "stop": "zzz",
            "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 3);
}
