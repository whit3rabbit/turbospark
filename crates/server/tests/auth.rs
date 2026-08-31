//! End-to-end tests for `build_router_with_options`'s `--api-key` layer.
//!
//! Same harness as `chat_completions.rs`: a real axum server on an ephemeral
//! loopback port, real tokenizer, scripted "model".

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::MfTokenizer;
use turbospark_server::{
    build_router, build_router_with_options, RouterOptions, ScriptedChatModel,
};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

fn h_steps(tok: &MfTokenizer, count: usize) -> Vec<Vec<foundation::LogitValue>> {
    let h_id = tok.token_to_id("h").unwrap() as usize;
    (0..count).map(|_| one_hot(tok.vocab_size, h_id)).collect()
}

async fn spawn_protected(api_key: &str) -> String {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let model: Arc<dyn turbospark_server::ChatModel> =
        Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let router = build_router_with_options(
        model,
        RouterOptions {
            api_key: Some(api_key.to_string()),
            ..Default::default()
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

fn chat_body() -> serde_json::Value {
    serde_json::json!({
        "model": "m", "max_tokens": 2, "temperature": 0.0,
        "messages": [{"role": "user", "content": "hi"}]
    })
}

#[tokio::test]
async fn a_request_with_no_credential_is_refused() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], "authentication_error");
}

/// **SAME LENGTH AS THE CORRECT KEY, DELIBERATELY.** `constant_time_eq`
/// refuses a length mismatch on its own fast path before ever touching the
/// byte-comparison loop; a wrong key of a DIFFERENT length would pass this
/// test without that loop doing anything, which is exactly the gap that let
/// a first version of this test survive `constant_time_eq`'s loop being
/// deleted outright.
#[tokio::test]
async fn a_wrong_key_of_the_same_length_is_refused() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-api-key", "sk-wrongxx")
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
}

#[tokio::test]
async fn a_wrong_key_of_a_different_length_is_refused() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-api-key", "short")
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
}

#[tokio::test]
async fn x_api_key_with_the_right_value_is_accepted() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .header("x-api-key", "sk-correct")
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn authorization_bearer_with_the_right_value_is_accepted() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .header("authorization", "Bearer sk-correct")
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

/// **THE ONE ROUTE THE LAYER MUST NEVER REACH.** A liveness probe answering
/// 401 to an unauthenticated caller is indistinguishable from the process
/// being down, which is the one thing it must never be wrong about.
#[tokio::test]
async fn health_is_exempt_from_the_api_key_requirement() {
    let base = spawn_protected("sk-correct").await;
    let response = reqwest::Client::new()
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

/// **`build_router` (no options) is untouched.** Every existing caller --
/// every other integration test file in this crate, and the FFI's future
/// in-process server -- must keep the exact unauthenticated behavior it had
/// before `--api-key` existed.
#[tokio::test]
async fn build_router_with_no_options_stays_unauthenticated() {
    let tok = load_tokenizer();
    let steps = h_steps(&tok, 50);
    let model: Arc<dyn turbospark_server::ChatModel> =
        Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let router = build_router(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let base = format!("http://{addr}");

    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}
