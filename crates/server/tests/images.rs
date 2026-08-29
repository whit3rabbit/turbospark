//! Image content parts on both endpoints (ROADMAP M-V8).
//!
//! Driven by `ScriptedChatModel`, which has NO vision tower -- so every case
//! here exercises the text-only half of the contract: a remote URL refused, a
//! malformed data URL refused, and an image this server cannot serve
//! REPORTED rather than silently dropped.
//!
//! **The serving half needs a real install and is `real_backend.rs`'s.** That
//! split is deliberate and it is the lesson M-V7 paid for: the injection map
//! bug that shipped in M-V5 was invisible to every test that did not run the
//! real generation loop, so a shape check here would prove nothing about
//! whether a picture reaches the model.

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

async fn spawn_server() -> String {
    let tok = load_tokenizer();
    let h = tok.token_to_id("h").unwrap() as usize;
    let steps = vec![one_hot(tok.vocab_size, h); 512];
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

/// A 1x1 PNG, as the smallest thing that really decodes.
const TINY_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

// ---------------------------------------------------------------------------
// A remote URL is refused rather than fetched
// ---------------------------------------------------------------------------

/// **The refusal that keeps this server from becoming an HTTP client driven
/// by request content.** Fetching would bring an SSRF surface, a timeout
/// budget and a redirect policy into a local inference server.
#[tokio::test]
async fn a_remote_image_url_is_refused_on_the_openai_endpoint() {
    let base = spawn_server().await;
    let body = serde_json::json!({
        "model": "m", "max_tokens": 4,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            {"type": "text", "text": "what is this?"}
        ]}]
    });
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let text = response.text().await.unwrap();
    assert!(text.contains("does not fetch remote"), "{text}");
}

/// The SAME refusal through the Anthropic endpoint, which reaches it by a
/// different route: `translate_request` maps a `url` image source onto the
/// OpenAI `image_url` shape, so one decoder serves both.
#[tokio::test]
async fn a_remote_image_url_is_refused_on_the_anthropic_endpoint() {
    let base = spawn_server().await;
    let body = serde_json::json!({
        "model": "claude-sonnet-4-6", "max_tokens": 4,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
            {"type": "text", "text": "what is this?"}
        ]}]
    });
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let text = response.text().await.unwrap();
    assert!(text.contains("does not fetch remote"), "{text}");
}

/// A percent-encoded data URL is a DIFFERENT decoding; reading it as base64
/// yields garbage pixels rather than an error.
#[tokio::test]
async fn a_non_base64_data_url_is_refused() {
    let base = spawn_server().await;
    let body = serde_json::json!({
        "model": "m", "max_tokens": 4,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png,%89PNG"}},
            {"type": "text", "text": "hi"}
        ]}]
    });
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    assert!(response.text().await.unwrap().contains("not base64"));
}

// ---------------------------------------------------------------------------
// A text-only backend REPORTS the drop
// ---------------------------------------------------------------------------

/// **The silence M-V8 exists to end.** Before this, a client sending a
/// picture to a model with no tower got a fluent text answer and no signal --
/// indistinguishable from a model that looked and had nothing to say.
#[tokio::test]
async fn a_text_only_backend_reports_the_dropped_image_on_openai() {
    let base = spawn_server().await;
    let body = serde_json::json!({
        "model": "m", "max_tokens": 4,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{TINY_PNG}")}},
            {"type": "text", "text": "what is this?"}
        ]}]
    });
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    // The turn still SUCCEEDS: the text half of the request is answerable and
    // refusing it would break every client that sends an incidental image.
    assert_eq!(response.status(), 200);
    let header = response
        .headers()
        .get("x-anyllm-degradation")
        .expect("a dropped image must be reported")
        .to_str()
        .unwrap()
        .to_string();
    assert!(header.contains("no vision tower"), "{header}");
}

#[tokio::test]
async fn a_text_only_backend_reports_the_dropped_image_on_anthropic() {
    let base = spawn_server().await;
    let body = serde_json::json!({
        "model": "claude-sonnet-4-6", "max_tokens": 4,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": TINY_PNG}},
            {"type": "text", "text": "what is this?"}
        ]}]
    });
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let header = response
        .headers()
        .get("x-anyllm-degradation")
        .expect("a dropped image must be reported")
        .to_str()
        .unwrap()
        .to_string();
    assert!(header.contains("no vision tower"), "{header}");
}

/// A TEXT-ONLY request must not acquire the header, or the signal means
/// nothing. This is the discriminating half of the pair above.
#[tokio::test]
async fn a_text_only_request_carries_no_degradation_note_about_images() {
    let base = spawn_server().await;
    let body = serde_json::json!({
        "model": "m", "max_tokens": 4,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let header = response
        .headers()
        .get("x-anyllm-degradation")
        .map(|v| v.to_str().unwrap().to_string())
        .unwrap_or_default();
    assert!(
        !header.contains("vision"),
        "a text-only request was reported as degraded: {header}"
    );
}
