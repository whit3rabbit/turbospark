//! End-to-end tests: binds the real axum server to an ephemeral loopback
//! port and hits it with a blocking HTTP client, driving generation through
//! `ScriptedChatModel` (real tokenizer, real HTTP, scripted "model").

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

/// Spawns the real server on an OS-assigned loopback port and returns its
/// base URL. The server runs for the lifetime of the current tokio runtime.
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

/// Enough scripted one-hot steps (all producing 'h') to cover any plausible
/// prefill length plus a few decode steps before max_tokens caps it.
fn h_steps(tok: &MfTokenizer, count: usize) -> Vec<Vec<foundation::LogitValue>> {
    let h_id = tok.token_to_id("h").unwrap() as usize;
    (0..count).map(|_| one_hot(tok.vocab_size, h_id)).collect()
}

#[tokio::test]
async fn non_streaming_chat_completion_returns_generated_text() {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let base = spawn_server(steps).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "scripted",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 3);
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.is_empty());
}

#[tokio::test]
async fn streaming_chat_completion_emits_sse_chunks_and_done() {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let base = spawn_server(steps).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "scripted",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();
    assert!(body.contains("chat.completion.chunk"));
    assert!(body.contains("[DONE]"));
}

#[tokio::test]
async fn malformed_request_body_is_rejected() {
    let base = spawn_server(Vec::new()).await;
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body("not json")
        .send()
        .await
        .unwrap();
    assert!(response.status().is_client_error());
}

#[tokio::test]
async fn health_endpoint_reports_ready_without_a_body() {
    let base = spawn_server(Vec::new()).await;
    let client = reqwest::Client::new();
    let response = client.get(format!("{base}/health")).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["model"], "scripted");
    assert_eq!(body["state"], "ready");
    assert!(body["version"].is_string());
}

/// **THE DISCRIMINATING HALF**: a request that asks for nothing this server
/// declines must carry no header at all, or the signal below means nothing.
#[tokio::test]
async fn a_plain_request_carries_no_degradation_header() {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let base = spawn_server(steps).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m", "max_tokens": 2, "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.headers().get("x-anyllm-degradation").is_none());
}

/// A non-text `response_format`, `n > 1`, and both penalty fields are
/// accepted rather than rejected -- OpenAI clients that always send them
/// must not 400 -- but every one of them is reported, since none is
/// actually honoured.
#[tokio::test]
async fn unsupported_openai_fields_are_reported_on_the_degradation_header() {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let base = spawn_server(steps).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m", "max_tokens": 2, "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
            "n": 2,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let header = response
        .headers()
        .get("x-anyllm-degradation")
        .expect("unsupported fields must be reported")
        .to_str()
        .unwrap()
        .to_string();
    // `n` is checked as a whole comma-separated item rather than a
    // substring: every other label here contains the letter `n`.
    let items: Vec<&str> = header.split(", ").collect();
    assert!(header.contains("response_format"), "{header}");
    assert!(items.contains(&"n"), "{header}");
    assert!(header.contains("presence_penalty"), "{header}");
    assert!(header.contains("frequency_penalty"), "{header}");
}

/// `n: 1` is the field's own default and this server already returns one
/// choice, so it must not be reported -- the discriminating half of the
/// case above.
#[tokio::test]
async fn n_equal_to_one_is_not_reported() {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let base = spawn_server(steps).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m", "max_tokens": 2, "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}],
            "n": 1
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.headers().get("x-anyllm-degradation").is_none());
}

/// The degradation header applies identically on the streaming path, not
/// just the non-streaming one it was first wired against.
#[tokio::test]
async fn streaming_responses_also_carry_the_degradation_header() {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let base = spawn_server(steps).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m", "max_tokens": 2, "temperature": 0.0, "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
            "presence_penalty": 0.2
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let header = response
        .headers()
        .get("x-anyllm-degradation")
        .expect("a streaming response must carry the header too")
        .to_str()
        .unwrap()
        .to_string();
    assert!(header.contains("presence_penalty"), "{header}");
    // The header is set on the response before the body streams, so it says
    // nothing about whether generation itself succeeded; check that too.
    let body = response.text().await.unwrap();
    assert!(body.contains("[DONE]"), "{body}");
    assert!(!body.contains("server_error"), "{body}");
}

#[tokio::test]
async fn get_models_list_and_model_detail_endpoint() {
    let base = spawn_server(Vec::new()).await;
    let client = reqwest::Client::new();

    // GET /v1/models
    let res = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let body: serde_json::Value = res.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "scripted");

    // GET /v1/models/scripted (existing)
    let res = client
        .get(format!("{base}/v1/models/scripted"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let detail: serde_json::Value = res.json().await.unwrap();
    assert_eq!(detail["id"], "scripted");
    assert_eq!(detail["object"], "model");

    // GET /v1/models/nonexistent (not found)
    let res = client
        .get(format!("{base}/v1/models/nonexistent"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 404);
}

#[tokio::test]
async fn streaming_chat_completion_with_include_usage_emits_usage_chunk() {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let base = spawn_server(steps).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "scripted",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": true,
            "stream_options": { "include_usage": true }
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();
    assert!(body.contains("chat.completion.chunk"));
    assert!(body.contains("\"usage\":"));
    assert!(body.contains("[DONE]"));
}

#[tokio::test]
async fn chat_completion_with_custom_shaping_and_tool_choice_none() {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let base = spawn_server(steps).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "scripted",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "temperature": 0.5,
            "top_k": 32,
            "repetition_penalty": 1.1,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather"
                }
            }],
            "tool_choice": "none"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "length");
}
