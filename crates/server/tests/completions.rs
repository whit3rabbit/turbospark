//! End-to-end tests for the legacy `POST /v1/completions` endpoint.
//!
//! Same harness as `chat_completions.rs`: a real axum server on an ephemeral
//! loopback port, real tokenizer, scripted "model".

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

#[tokio::test]
async fn non_streaming_completion_returns_generated_text() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "m", "prompt": "hi", "max_tokens": 3, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 3);
    let text = body["choices"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty());
}

#[tokio::test]
async fn a_one_element_prompt_array_is_accepted() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "m", "prompt": ["hi"], "max_tokens": 2, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn a_multi_element_prompt_array_is_refused() {
    let base = spawn_server(Vec::new()).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({"model": "m", "prompt": ["a", "b"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let text = response.text().await.unwrap();
    assert!(text.contains("one prompt"), "{text}");
}

#[tokio::test]
async fn suffix_is_refused() {
    let base = spawn_server(Vec::new()).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({"model": "m", "prompt": "hi", "suffix": " there"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let text = response.text().await.unwrap();
    assert!(text.contains("suffix"), "{text}");
}

#[tokio::test]
async fn n_greater_than_one_is_refused() {
    let base = spawn_server(Vec::new()).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({"model": "m", "prompt": "hi", "n": 2}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn max_tokens_defaults_to_sixteen() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({"model": "m", "prompt": "hi", "temperature": 0.0}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["usage"]["completion_tokens"], 16);
}

#[tokio::test]
async fn streaming_completion_emits_text_completion_chunks_and_done() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "m", "prompt": "hi", "max_tokens": 2, "temperature": 0.0, "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body = response.text().await.unwrap();
    assert!(body.contains("text_completion"), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
    assert!(!body.contains("server_error"), "{body}");
}

#[tokio::test]
async fn malformed_request_body_is_rejected() {
    let base = spawn_server(Vec::new()).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .header("content-type", "application/json")
        .body("not json")
        .send()
        .await
        .unwrap();
    assert!(response.status().is_client_error());
}

#[tokio::test]
async fn logprobs_best_of_and_echo_are_reported_on_the_degradation_header() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "m", "prompt": "hi", "max_tokens": 2, "temperature": 0.0,
            "logprobs": 3, "best_of": 2, "echo": true
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
    assert!(header.contains("logprobs"), "{header}");
    assert!(header.contains("best_of"), "{header}");
    assert!(header.contains("echo"), "{header}");
}

#[tokio::test]
async fn a_plain_completion_request_carries_no_degradation_header() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 50)).await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "m", "prompt": "hi", "max_tokens": 2, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.headers().get("x-anyllm-degradation").is_none());
}

/// **THE POINT OF THIS ENDPOINT**: unlike `/v1/chat/completions`, the prompt
/// reaches the model with no chat template wrapped around it. The scripted
/// backend replays fixed logits regardless of content, so the only way to
/// observe the template is indirectly, through how many tokens prefill
/// consumed for the SAME text: a ChatML-wrapped prompt carries
/// `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` on top of the
/// content, so the templated count must come out higher for identical text.
#[tokio::test]
async fn the_prompt_is_not_chat_templated() {
    let tok = load_tokenizer();
    let base = spawn_server(h_steps(&tok, 200)).await;
    let client = reqwest::Client::new();

    let raw: serde_json::Value = client
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "model": "m", "prompt": "hi", "max_tokens": 1, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let templated: serde_json::Value = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1, "temperature": 0.0
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    let raw_prompt_tokens = raw["usage"]["prompt_tokens"].as_u64().unwrap();
    let templated_prompt_tokens = templated["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(
        raw_prompt_tokens < templated_prompt_tokens,
        "raw={raw_prompt_tokens} templated={templated_prompt_tokens}: \
         the raw prompt must carry fewer tokens than the templated one for \
         identical text, or the template is leaking into /v1/completions"
    );
}
