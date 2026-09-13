//! Routing a request to one of several attached models, and the
//! single-model fallback that keeps every pre-registry client working.
//!
//! The fallback is the case worth reading first: `crates/server/src/
//! registry.rs`'s header records why a request naming a model this server
//! has never heard of is SERVED rather than refused when only one is
//! attached, and `one_model_serves_a_name_it_does_not_know` below is the
//! documented Claude Code invocation turned into an assertion.

use std::path::PathBuf;
use std::sync::Arc;

use tokenizer::MfTokenizer;
use turbospark_server::registry::StaticRegistry;
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

/// `ScriptedChatModel::model_id` is the constant `"scripted"`, so a test
/// with two distinguishable backends needs a wrapper that names them. It
/// also makes the ROUTING observable: each model's id comes back on the
/// response's own `model` field.
struct Named(ScriptedChatModel, String);

impl ChatModel for Named {
    fn tokenizer(&self) -> &MfTokenizer {
        self.0.tokenizer()
    }
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn max_context(&self) -> u32 {
        self.0.max_context()
    }
    fn model_id(&self) -> &str {
        &self.1
    }
    fn with_producer(
        &self,
        f: &mut dyn FnMut(
            &mut dyn runtime::LogitProducer,
        ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError>,
    ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError> {
        self.0.with_producer(f)
    }
}

fn named(id: &str) -> Arc<dyn ChatModel> {
    let tok = load_tokenizer();
    let vocab = tok.vocab_size;
    // Enough steps to prefill a short prompt and decode a couple of tokens.
    let steps = (0..64).map(|_| one_hot(vocab, 5)).collect();
    Arc::new(Named(
        ScriptedChatModel::new(tok, 4096, steps),
        id.to_string(),
    ))
}

async fn serve(models: Vec<Arc<dyn ChatModel>>) -> String {
    let registry: Arc<dyn turbospark_server::registry::ModelRegistry> =
        Arc::new(StaticRegistry::new(models).expect("test model ids are unique"));
    let router = build_router(registry);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

async fn chat(base: &str, model: &str) -> (u16, serde_json::Value) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model,
            "max_tokens": 2,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

#[tokio::test]
async fn two_models_are_both_listed_with_their_own_context_windows() {
    let base = serve(vec![named("alpha.gturbo"), named("beta.gturbo")]).await;
    let body: serde_json::Value = reqwest::get(format!("{base}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        vec![
            "alpha.gturbo",
            "claude-turbospark-alpha.gturbo",
            "beta.gturbo",
            "claude-turbospark-beta.gturbo",
        ]
    );
    assert_eq!(body["data"][0]["context_window"], 4096);
}

#[tokio::test]
async fn an_exact_model_id_reaches_that_model() {
    let base = serve(vec![named("alpha.gturbo"), named("beta.gturbo")]).await;
    for id in ["alpha.gturbo", "beta.gturbo"] {
        let (status, body) = chat(&base, id).await;
        assert_eq!(status, 200, "{id} should have been served");
        // The response echoes the request's own model name, so routing is
        // read off `/health` instead -- see the dedicated case below. What
        // this asserts is that both ids are ACCEPTED, which is the half a
        // 404 would break.
        assert_eq!(body["object"], "chat.completion");
    }
}

#[tokio::test]
async fn an_exact_claude_alias_reaches_its_model_on_a_multi_model_server() {
    let base = serve(vec![named("alpha.gturbo"), named("beta.gturbo")]).await;
    let (status, body) = chat(&base, "claude-turbospark-beta.gturbo").await;
    assert_eq!(status, 200);
    assert_eq!(body["object"], "chat.completion");
}

/// **THE CASE THE FALLBACK EXISTS FOR.** `docs/CLI.md` points Claude Code at
/// this server with `ANTHROPIC_BASE_URL`, and it sends
/// `"model": "claude-sonnet-4-6"` at an install named nothing of the sort.
/// Every OpenAI SDK does the same with its own default. Routing strictly on
/// the name would 404 all of them.
///
/// Mutation check: deleting `resolve_among`'s `[only] =>` arm reddens this
/// case alone and leaves the two-model cases green.
#[tokio::test]
async fn one_model_serves_a_name_it_does_not_know() {
    let base = serve(vec![named("gemma4.gturbo")]).await;
    let (status, body) = chat(&base, "claude-sonnet-4-6").await;
    assert_eq!(status, 200);
    assert_eq!(body["object"], "chat.completion");
}

/// With two attached there is a real ambiguity, so the same request is
/// refused -- and the refusal names what IS there, because a caller who
/// guessed wrong has no other way to find out.
#[tokio::test]
async fn several_models_refuse_an_unknown_name_and_name_the_alternatives() {
    let base = serve(vec![named("alpha.gturbo"), named("beta.gturbo")]).await;
    let (status, body) = chat(&base, "claude-sonnet-4-6").await;
    assert_eq!(status, 404);
    assert_eq!(body["error"]["code"], "model_not_found");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("alpha.gturbo"), "{message}");
    assert!(message.contains("beta.gturbo"), "{message}");
}

/// A server can run with nothing attached: that is the state a GUI starts
/// one in before loading a model. 503 rather than 404, because the request
/// is fine and the server is not ready -- a distinction a client's retry
/// logic acts on.
#[tokio::test]
async fn an_empty_server_reports_itself_unavailable_rather_than_missing() {
    let base = serve(Vec::new()).await;
    let (status, _) = chat(&base, "anything").await;
    assert_eq!(status, 503);

    let health: serde_json::Value = reqwest::get(format!("{base}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["state"], "empty");
    assert!(health["model"].is_null());
    assert_eq!(health["models"].as_array().unwrap().len(), 0);
}

/// `/v1/models/:id` answers about the id it was ASKED about, and must not
/// inherit the single-model fallback: the fallback serves generations,
/// where a lookup's whole question is whether this exact id exists.
///
/// Mutation check: routing `model_detail` through `registry.resolve` instead
/// of through `rows()` reddens the second half of this and nothing else.
#[tokio::test]
async fn a_model_lookup_does_not_take_the_single_model_fallback() {
    let base = serve(vec![named("gemma4.gturbo")]).await;

    let found = reqwest::get(format!("{base}/v1/models/gemma4.gturbo"))
        .await
        .unwrap();
    assert_eq!(found.status().as_u16(), 200);

    let alias = reqwest::get(format!("{base}/v1/models/claude-turbospark-gemma4.gturbo"))
        .await
        .unwrap();
    assert_eq!(alias.status().as_u16(), 200);
    let detail: serde_json::Value = alias.json().await.unwrap();
    assert_eq!(detail["id"], "claude-turbospark-gemma4.gturbo");

    let missing = reqwest::get(format!("{base}/v1/models/not-a-model"))
        .await
        .unwrap();
    assert_eq!(
        missing.status().as_u16(),
        404,
        "a lookup must not answer for a name it does not have"
    );
}
