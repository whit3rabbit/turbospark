//! The Ollama-compatible routes, and specifically their FRAMING.
//!
//! `/api/chat` and `/api/generate` stream NDJSON -- one bare JSON object per
//! line, no `event:` lines, no `[DONE]` sentinel, and the last object
//! carrying `"done": true`. That is the one non-SSE stream in this crate, so
//! a shape-only assertion (200, plausible JSON somewhere in the body) would
//! pass against an SSE response and prove nothing.

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::MfTokenizer;
use turbospark_server::{build_router, ChatModel, ScriptedChatModel};

fn load_tokenizer() -> MfTokenizer {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer should load")
}

fn one_hot(vocab_size: usize, index: usize) -> Vec<foundation::LogitValue> {
    let mut v = vec![foundation::LogitValue::from_f32(0.0); vocab_size];
    v[index] = foundation::LogitValue::from_f32(1.0);
    v
}

async fn serve() -> String {
    let tok = load_tokenizer();
    let vocab = tok.vocab_size;
    let steps = (0..64).map(|_| one_hot(vocab, 5)).collect();
    let model: Arc<dyn ChatModel> = Arc::new(ScriptedChatModel::new(tok, 4096, steps));
    let router = build_router(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn tags_lists_every_attached_model() {
    let base = serve().await;
    let body: serde_json::Value = reqwest::get(format!("{base}/api/tags"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    // Ollama clients read `name`; some read `model`. Both are the id.
    assert_eq!(models[0]["name"], "scripted");
    assert_eq!(models[0]["model"], "scripted");
    // Reported as zero rather than estimated: this server downloaded
    // nothing, and an install's on-disk bytes are not what it committed.
    assert_eq!(models[0]["size"], 0);
}

#[tokio::test]
async fn version_reports_this_crates_version_not_an_ollama_one() {
    let base = serve().await;
    let body: serde_json::Value = reqwest::get(format!("{base}/api/version"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn show_answers_for_an_attached_model_and_404s_otherwise() {
    let base = serve().await;
    let client = reqwest::Client::new();

    let found = client
        .post(format!("{base}/api/show"))
        .json(&serde_json::json!({"model": "scripted"}))
        .send()
        .await
        .unwrap();
    assert_eq!(found.status().as_u16(), 200);

    let missing = client
        .post(format!("{base}/api/show"))
        .json(&serde_json::json!({"model": "nope"}))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status().as_u16(), 404);
}

#[tokio::test]
async fn a_non_streaming_chat_returns_one_object_marked_done() {
    let base = serve().await;
    let body: serde_json::Value = reqwest::Client::new()
        .post(format!("{base}/api/chat"))
        .json(&serde_json::json!({
            "model": "scripted",
            "stream": false,
            "options": {"num_predict": 3},
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(body["done"], true);
    assert_eq!(body["message"]["role"], "assistant");
    assert!(body["message"]["content"].is_string());
    // Off `RawDecodeResult`, which is what an Ollama client divides to show
    // a rate. `num_predict` maps onto `max_tokens`.
    assert_eq!(body["eval_count"], 3);
    assert!(body["prompt_eval_count"].as_u64().unwrap() > 0);
}

/// **THE FRAMING CASE.** Asserted line by line rather than by looking for
/// substrings: an SSE response also contains these JSON objects, prefixed
/// with `data: `, and every one of those lines would fail to parse here.
///
/// Mutation check: returning the stream through `Sse::new` instead of
/// `ndjson` reddens this and leaves the non-streaming cases green.
#[tokio::test]
async fn a_streaming_chat_is_ndjson_ending_in_a_done_object() {
    let base = serve().await;
    let response = reqwest::Client::new()
        .post(format!("{base}/api/chat"))
        .json(&serde_json::json!({
            // `stream` omitted on purpose: Ollama's default is TRUE, unlike
            // every OpenAI-shaped endpoint in this crate.
            "model": "scripted",
            "options": {"num_predict": 3},
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), 200);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/x-ndjson")
    );

    let text = response.text().await.unwrap();
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    assert!(!lines.is_empty(), "the stream sent nothing");

    // EVERY line is a complete JSON object on its own. This is what an SSE
    // body cannot satisfy.
    let objects: Vec<serde_json::Value> = lines
        .iter()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("line {l:?} is not JSON: {e}")))
        .collect();

    // No sentinel: the last object IS the terminator.
    let (last, rest) = objects.split_last().unwrap();
    assert_eq!(last["done"], true);
    assert_eq!(last["eval_count"], 3);
    for object in rest {
        assert_eq!(object["done"], false, "only the last object is done");
        assert_eq!(object["message"]["role"], "assistant");
    }
}

/// `/api/generate` puts the text flat on `response` where `/api/chat` nests
/// it under `message.content`. Same core, two shapes, and a client of one
/// cannot read the other.
#[tokio::test]
async fn generate_uses_the_flat_response_field() {
    let base = serve().await;
    let body: serde_json::Value = reqwest::Client::new()
        .post(format!("{base}/api/generate"))
        .json(&serde_json::json!({
            "model": "scripted",
            "stream": false,
            "options": {"num_predict": 2},
            "prompt": "why is the sky blue",
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(body["done"], true);
    assert!(body["response"].is_string());
    assert!(
        body["message"].is_null(),
        "generate must not carry chat's nested shape"
    );
    assert_eq!(body["eval_count"], 2);
}
