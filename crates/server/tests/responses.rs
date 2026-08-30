//! End-to-end tests for the OpenAI `POST /v1/responses` endpoint.
//!
//! Same harness as `chat_completions.rs`: a real axum server on an ephemeral
//! loopback port, real tokenizer, scripted "model". The streaming tests
//! assert on the exact `event:` name SEQUENCE, which is the actual contract
//! a Responses client's state machine depends on -- the same reason
//! `messages.rs`'s streaming test checks Anthropic event order rather than
//! just "the response was 200".

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::MfTokenizer;
use turbospark_server::{build_router, ScriptedChatModel};

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
    let model: Arc<dyn turbospark_server::ChatModel> =
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

fn event_names(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|l| l.strip_prefix("event:"))
        .map(|n| n.trim().to_string())
        .collect()
}

#[tokio::test]
async fn non_streaming_response_returns_a_message_output_item() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "max_output_tokens": 3, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["type"], "response");
    assert_eq!(body["status"], "incomplete"); // capped by max_output_tokens
    let output = body["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["role"], "assistant");
    let text = output[0]["content"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty());
    assert_eq!(body["usage"]["output_tokens"], 3);
}

#[tokio::test]
async fn an_items_array_input_is_accepted() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "max_output_tokens": 2, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn previous_response_id_is_refused() {
    let base = spawn_server(Vec::new()).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "previous_response_id": "resp_1"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let text = response.text().await.unwrap();
    assert!(text.contains("previous_response_id"), "{text}");
}

#[tokio::test]
async fn an_unknown_input_item_type_is_refused() {
    let base = spawn_server(Vec::new()).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": [{"type": "reasoning", "summary": []}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn store_true_is_reported_on_the_degradation_header() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "max_output_tokens": 2, "temperature": 0.0, "store": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let header = response
        .headers()
        .get("x-anyllm-degradation")
        .expect("store: true must be reported")
        .to_str()
        .unwrap()
        .to_string();
    assert!(header.contains("store"), "{header}");
}

#[tokio::test]
async fn a_plain_request_carries_no_degradation_header() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "max_output_tokens": 2, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.headers().get("x-anyllm-degradation").is_none());
}

/// **THE POINT OF THIS TEST**: a Responses client's state machine dispatches
/// on the `event:` line, so the SEQUENCE is the actual contract, not just
/// "some events arrived". `response.created` opens the stream and
/// `response.completed` closes it; a text turn opens and closes exactly one
/// message item and one content part in between.
#[tokio::test]
async fn streaming_response_emits_events_in_the_documented_order() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "max_output_tokens": 2, "temperature": 0.0,
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();
    let names = event_names(&body);

    assert_eq!(names.first().map(String::as_str), Some("response.created"));
    assert_eq!(names.last().map(String::as_str), Some("response.completed"));
    assert_eq!(
        names,
        vec![
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ],
        "{names:?}"
    );
    // `[DONE]` is a Chat-Completions-ism; a Responses stream ends at
    // response.completed, the same convention the Anthropic endpoint uses.
    assert!(!body.contains("[DONE]"));
}

#[tokio::test]
async fn streaming_response_carries_the_degradation_header() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .json(&serde_json::json!({
            "model": "m", "input": "hi", "max_output_tokens": 2, "temperature": 0.0,
            "stream": true, "store": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let header = response
        .headers()
        .get("x-anyllm-degradation")
        .expect("store: true must be reported on the streaming path too")
        .to_str()
        .unwrap()
        .to_string();
    assert!(header.contains("store"), "{header}");
}

#[tokio::test]
async fn malformed_request_body_is_rejected() {
    let base = spawn_server(Vec::new()).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/responses"))
        .header("content-type", "application/json")
        .body("not json")
        .send()
        .await
        .unwrap();
    assert!(response.status().is_client_error());
}
